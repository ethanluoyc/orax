import absl.app
import absl.flags
import acme
import d4rl  # noqa: F401
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import reverb
import tensorflow as tf
import tensorflow_datasets as tfds
import tree
from absl import flags
from acme import datasets as acme_datasets
from acme import types
from acme import wrappers
from acme.adders import reverb as adders_reverb
from acme.agents.jax import actor_core
from acme.agents.jax import actors
from acme.agents.jax import sac
from acme.jax import utils as jax_utils
from acme.jax import variable_utils
from acme.utils import counting
from ml_collections import config_flags

from orax.agents import calql
from orax.agents.calql.adder import SparseReward
from orax.baselines import experiment_utils
from orax.baselines.d4rl import d4rl_evaluation

_CONFIG = config_flags.DEFINE_config_file(
    "config", "configs/calql_antmaze.py", lock_config=False
)
_LOG_TO_WANDB = flags.DEFINE_bool("log_to_wandb", False, "Log to wandb")


@tf.function
def compute_return_to_go(rewards, discounts, gamma):
    rewards = tf.convert_to_tensor(rewards)
    discounts = tf.convert_to_tensor(discounts)

    def discounted_return_fn(acc, reward_discount):
        reward, discount = reward_discount
        return acc * discount * gamma + reward

    return tf.scan(
        fn=discounted_return_fn,
        elems=(rewards, discounts),
        reverse=True,
        initializer=tf.constant(0.0, dtype=rewards.dtype),
    )


@tf.function
def preprocess_episode(episode, reward_scale, reward_bias, gamma):
    steps = episode["steps"].batch(episode["steps"].cardinality()).get_single_element()

    observations = steps["observation"][:-1]
    next_observations = steps["observation"][1:]
    rewards = tf.cast(steps["reward"][:-1], tf.float64)
    actions = steps["action"][:-1]
    discounts = tf.cast(steps["discount"][:-1], tf.float64)

    rewards = rewards * reward_scale + reward_bias
    reward_negative = 0.0 * reward_scale + reward_bias
    gamma = tf.convert_to_tensor(gamma, dtype=rewards.dtype)

    if tf.reduce_all(rewards == reward_negative):
        return_to_go = tf.ones_like(rewards) * (reward_negative / (1 - gamma))
    else:
        return_to_go = compute_return_to_go(rewards, discounts, gamma)

    return types.Transition(
        observation=observations,
        action=actions,
        discount=discounts,
        reward=tf.cast(rewards, tf.float32),
        next_observation=next_observations,
        extras={"mc_return": return_to_go},
    )


def get_transitions_dataset(reward_scale, reward_bias, gamma):
    dataset = tfds.load("d4rl_antmaze/medium-diverse-v2", split="train")
    tf_transitions = []
    for episode in dataset:
        converted_transitions = preprocess_episode(
            episode, reward_scale, reward_bias, gamma
        )
        tf_transitions.append(converted_transitions)

    transitions = tf.data.Dataset.from_tensor_slices(
        tree.map_structure(lambda *x: tf.concat(x, axis=0), *tf_transitions)
    )
    return transitions


def main(argv):
    del argv
    config = _CONFIG.value
    transitions = get_transitions_dataset(
        config.reward_scale, config.reward_bias, config.cql.discount
    )

    def make_offline_iterator(batch_size):
        from orax.datasets.tfds import JaxInMemoryRandomSampleIterator

        return JaxInMemoryRandomSampleIterator(
            transitions, jax.random.PRNGKey(0), batch_size
        )

    env = gym.make(config.env)
    env = wrappers.GymWrapper(env)
    env = wrappers.SinglePrecisionWrapper(env)
    env_spec = acme.make_environment_spec(env)

    networks = calql.make_networks(
        env_spec,
        policy_hidden_sizes=(256, 256),
        critic_hidden_sizes=(256, 256, 256, 256),
    )

    # if config.cql.target_entropy >= 0.0:
    #     config.cql.target_entropy = -np.prod(env_spec.actions.shape).item()
    target_entropy = -np.prod(env_spec.actions.shape).item()

    parent_counter = counting.Counter(time_delta=0.0)

    logger_factory = experiment_utils.LoggerFactory(
        workdir=None,
        log_to_wandb=_LOG_TO_WANDB.value,
        evaluator_time_delta=0.001,
    )

    offline_iterator = make_offline_iterator(config.batch_size)
    offline_iterator = jax_utils.prefetch(offline_iterator)

    learner_logger = logger_factory("learner", "learner_steps", 0)
    learner_counter = counting.Counter(parent_counter, "learner", time_delta=0.0)

    offline_learner = calql.CalQLLearner(
        config.batch_size,
        networks,
        jax.random.PRNGKey(0),
        offline_iterator,
        policy_optimizer=optax.adam(config.cql.policy_lr),
        critic_optimizer=optax.adam(config.cql.qf_lr),
        tau=config.cql.soft_target_update_rate,
        cql_lagrange_threshold=config.cql.cql_target_action_gap,
        cql_num_samples=config.cql.cql_n_actions,
        logger=learner_logger,
        num_sgd_steps_per_step=1,
        reward_scale=1.0,
        discount=config.cql.discount,
        target_entropy=target_entropy,
        num_bc_iters=0,
        max_target_backup=config.cql.cql_max_target_backup,
        use_calql=config.enable_calql,
        counter=learner_counter,
    )

    offline_eval_actor = actors.GenericActor(
        actor_core.batched_feed_forward_to_actor_core(
            sac.apply_policy_and_sample(networks, eval_mode=True)
        ),
        jax.random.PRNGKey(0),
        variable_utils.VariableClient(
            offline_learner, "policy", update_period=1, device="cpu"
        ),
        per_episode_update=True,
    )

    evaluator_counter = counting.Counter(parent_counter, "evaluator", time_delta=0.0)
    evaluator_logger = logger_factory("evaluator", "evaluator_steps", 0)

    offline_evaluator = d4rl_evaluation.D4RLEvaluator(
        lambda: env,
        offline_eval_actor,
        logger=evaluator_logger,
        counter=evaluator_counter,
    )

    num_offline_steps = config.n_train_step_per_epoch_offline * config.n_pretrain_epochs
    for step in range(num_offline_steps):
        offline_learner.step()
        if (step + 1) % int(
            config.offline_eval_every_n_epoch * config.n_train_step_per_epoch_offline
        ) == 0:
            offline_evaluator.run(num_episodes=20)

    # mix offline and online buffer
    assert config.mixing_ratio >= 0.0
    batch_size_offline = int(config.mixing_ratio * config.batch_size)
    batch_size_online = config.batch_size - batch_size_offline

    reverb_tables = [
        reverb.Table(
            name="priority_table",
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=int(1e6),
            rate_limiter=reverb.rate_limiters.MinSize(1),
            signature=adders_reverb.NStepTransitionAdder.signature(
                env_spec,
                extras_spec={"mc_return": tf.TensorSpec(shape=(), dtype=tf.float32)},
            ),
        )
    ]

    replay_server = reverb.Server(reverb_tables, port=None)
    replay_client = replay_server.localhost_client()

    train_env = gym.make(config.env)
    train_env = wrappers.GymWrapper(train_env)
    train_env = wrappers.SinglePrecisionWrapper(train_env)
    online_logger = logger_factory("actor", "actor_steps", 0)
    online_counter = counting.Counter(parent_counter, "actor", time_delta=0.0)

    num_steps = 0
    episode_length = 0
    episode_return = 0
    timestep = train_env.reset()
    initial_num_steps = 5000
    eval_every = config.online_eval_every_n_env_steps
    eval_episodes = config.eval_n_trajs

    del offline_iterator

    def make_online_iterator():
        offline_iterator = make_offline_iterator(batch_size_offline)
        online_dataset = acme_datasets.make_reverb_dataset(
            table="priority_table",
            server_address=replay_client.server_address,
            num_parallel_calls=4,
            batch_size=batch_size_online,
            prefetch_size=1,
        ).as_numpy_iterator()

        while True:
            offline_batch = next(offline_iterator)
            offline_transitions = jax.device_put(offline_batch)
            online_transitions = jax.device_put(next(online_dataset).data)

            yield tree.map_structure(
                lambda x, y: jnp.concatenate([x, y]),
                offline_transitions,
                online_transitions,
            )

    adder = calql.CalQLAdder(
        adders_reverb.NStepTransitionAdder(
            replay_client, 1, discount=config.cql.discount
        ),
        config.cql.discount,
        reward_config=SparseReward(
            reward_scale=config.reward_scale,
            reward_bias=config.reward_bias,
            positive_reward=1,
            negative_reward=0,
        ),
    )

    online_learner = calql.CalQLLearner(
        config.batch_size,
        networks,
        jax.random.PRNGKey(3),
        make_online_iterator(),
        policy_optimizer=optax.adam(config.cql.policy_lr),
        critic_optimizer=optax.adam(config.cql.qf_lr),
        tau=config.cql.soft_target_update_rate,
        cql_lagrange_threshold=config.cql.cql_target_action_gap,
        cql_num_samples=config.cql.cql_n_actions,
        logger=learner_logger,
        num_sgd_steps_per_step=1,
        reward_scale=1.0,
        discount=config.cql.discount,
        target_entropy=target_entropy,
        num_bc_iters=0,
        max_target_backup=config.cql.cql_max_target_backup,
        use_calql=config.enable_calql,
        counter=learner_counter,
    )
    online_learner.restore(offline_learner.save())

    eval_actor = actors.GenericActor(
        actor_core.batched_feed_forward_to_actor_core(
            sac.apply_policy_and_sample(networks, eval_mode=True)
        ),
        jax.random.PRNGKey(42),
        variable_utils.VariableClient(
            online_learner, "policy", update_period=1, device="cpu"
        ),
        per_episode_update=True,
    )

    online_evaluator = d4rl_evaluation.D4RLEvaluator(
        lambda: env,
        eval_actor,
        logger=evaluator_logger,
        counter=evaluator_counter,
    )

    online_actor = actors.GenericActor(
        actor_core.batched_feed_forward_to_actor_core(
            sac.apply_policy_and_sample(networks, eval_mode=False)
        ),
        jax.random.PRNGKey(0),
        variable_utils.VariableClient(
            online_learner, "policy", update_period=1, device="cpu"
        ),
        adder=adder,
        per_episode_update=True,
    )

    online_actor.observe_first(timestep)
    online_actor.update()

    while True:
        action = online_actor.select_action(timestep.observation)
        next_timestep = train_env.step(action)
        num_steps += 1
        episode_return += next_timestep.reward
        episode_length += 1
        if num_steps >= int(config.max_online_env_steps):
            break
        if num_steps >= initial_num_steps:
            for _ in range(config.online_utd_ratio):
                online_learner.step()
        if num_steps % eval_every == 0:
            online_evaluator.run(num_episodes=eval_episodes)

        online_actor.observe(action, next_timestep)
        online_actor.update()

        if next_timestep.last():
            counts = online_counter.increment(episodes=1, steps=episode_length)
            online_logger.write({**counts, "episode_return": episode_return})
            episode_return = 0
            episode_length = 0
            timestep = train_env.reset()
            online_actor.observe_first(timestep)
        else:
            timestep = next_timestep


if __name__ == "__main__":
    absl.app.run(main)
