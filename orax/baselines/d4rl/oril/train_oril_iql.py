import acme
import dataset_utils
import evaluation
import jax
import numpy as np
import optax
from absl import app
from absl import flags
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import actors
from acme.jax import variable_utils
from acme.utils import counting
from ml_collections import config_flags

from orax.agents import iql
from orax.agents import oril

_CONFIG = config_flags.DEFINE_config_file("config", None)
_WORKDIR = flags.DEFINE_string("workdir", "/tmp/offline_otil", "")


def get_demonstration_dataset(config):
    expert_dataset_name = config.expert_dataset_name
    offline_dataset_name = config.offline_dataset_name
    # Load expert demonstrations
    offline_traj = dataset_utils.load_trajectories(expert_dataset_name)
    if "antmaze" in offline_dataset_name:
        # 1/Distance (from the bottom-right corner) times return
        returns = [
            sum([t.reward for t in traj])
            / (1e-4 + np.linalg.norm(traj[0].observation[:2]))
            for traj in offline_traj
        ]
    else:
        returns = [sum([t.reward for t in traj]) for traj in offline_traj]
    idx = np.argpartition(returns, -config.k)[-config.k :]
    demo_returns = [returns[i] for i in idx]
    print(f"demo returns {demo_returns}, mean {np.mean(demo_returns)}")
    expert_demo = [offline_traj[i] for i in idx]
    expert_transitions = dataset_utils.merge_trajectories(expert_demo)
    offline_transitions = dataset_utils.merge_trajectories(
        dataset_utils.load_trajectories(offline_dataset_name)
    )
    key = jax.random.PRNGKey(config.seed)
    expert_key, offline_key = jax.random.split(key)
    expert_iterator = dataset_utils.JaxInMemorySampler(
        expert_transitions, expert_key, config.batch_size
    )
    unlabeled_iterator = dataset_utils.JaxInMemorySampler(
        offline_transitions, offline_key, config.batch_size
    )
    while True:
        # pylint: disable=stop-iteration-return
        # TODO(yl): Set up dataset aggregation in the way described in
        # https://arxiv.org/pdf/2011.13885.pdf, A.1
        expert_sample = next(expert_iterator)
        unlabeled_sample = next(unlabeled_iterator)
        offline_rl_sample = next(unlabeled_iterator)
        yield oril.ORILSample(expert_sample, unlabeled_sample, offline_rl_sample)


def main(_):
    config = _CONFIG.value
    offline_dataset_name = config.offline_dataset_name

    {
        "project": config.wandb_project,
        "entity": config.wandb_entity,
        "config": config.to_dict(),
    }

    iterator = get_demonstration_dataset(config)

    # Create dataset iterator for the relabeled dataset
    key = jax.random.PRNGKey(config.seed)
    key_learner, _, key = jax.random.split(key, 3)

    # Create an environment and grab the spec.
    environment = dataset_utils.make_environment(offline_dataset_name, seed=config.seed)
    # Create the networks to optimize.
    spec = acme.make_environment_spec(environment)
    networks = iql.make_networks(
        spec, hidden_dims=config.hidden_dims, dropout_rate=config.dropout_rate
    )

    counter = counting.Counter(time_delta=0.0)

    key_oril_learner, key_iql_learner = jax.random.split(key_learner)
    reward_network = oril.make_reward_network(spec)

    def make_iql_learner(iterator):
        # Create the learner.
        if "antmaze" in config.offline_dataset_name:
            iterator = (
                sample._replace(reward=sample.reward - 2) for sample in iterator
            )

        if config.opt_decay_schedule == "cosine":
            schedule_fn = optax.cosine_decay_schedule(
                -config.actor_lr, config.max_steps
            )
            policy_optimizer = optax.chain(
                optax.scale_by_adam(), optax.scale_by_schedule(schedule_fn)
            )
        else:
            policy_optimizer = optax.adam(config.actor_lr)

        learner = iql.IQLLearner(
            networks=networks,
            random_key=key_iql_learner,
            dataset=iterator,
            policy_optimizer=policy_optimizer,
            critic_optimizer=optax.adam(config.critic_lr),
            value_optimizer=optax.adam(config.value_lr),
            **config.iql_kwargs,
            # logger=experiment_utils.make_experiment_logger(
            #     "offline_learner", "offline_learner_steps", 0
            # ),
            counter=counting.Counter(counter, "offline_learner", time_delta=0.0),
        )
        return learner

    learner = oril.ORILLearner(
        iterator,
        make_iql_learner,
        reward_network,
        loss_fn=oril.oril_pu_loss,
        random_key=key_oril_learner,
        optimizer=optax.adam(3e-4),
        counter=counting.Counter(counter, prefix="reward_learner", time_delta=0.0),
    )

    def evaluator_network(params, key, observation):
        del key
        action_distribution = networks.policy_network.apply(
            params, observation, is_training=False
        )
        return action_distribution.mode()

    eval_actor = actors.GenericActor(
        actor_core_lib.batched_feed_forward_to_actor_core(evaluator_network),
        random_key=key,
        variable_client=variable_utils.VariableClient(learner, "policy", device="cpu"),
        backend="cpu",
    )

    eval_loop = evaluation.D4RLEvalLoop(
        environment,
        eval_actor,
        counter=counting.Counter(counter, "eval_loop", time_delta=0.0),
    )

    # Run the environment loop.
    steps = 0
    while steps < config.max_steps:
        for _ in range(config.evaluate_every):
            learner.step()
        steps += config.evaluate_every
        eval_loop.run(config.evaluation_episodes)


if __name__ == "__main__":
    app.run(main)
