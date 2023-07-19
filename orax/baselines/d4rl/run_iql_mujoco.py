"""Run Implicit Q-Learning (IQL) on D4RL MuJoCo."""

import numpy as np
import rlds
import tensorflow as tf
from absl import app
from absl import flags
from acme import types
from acme import wrappers
from acme.jax import experiments

from orax.agents import iql
from orax.baselines import experiment_utils
from orax.baselines.d4rl import d4rl_evaluation
from orax.datasets import tfds

FLAGS = flags.FLAGS
flags.DEFINE_string("workdir", "/tmp/orax", "")
# flags.DEFINE_string("env", "hopper-medium-v2", "OpenAI gym environment name")
flags.DEFINE_integer("seed", 0, "seed")
flags.DEFINE_integer("eval_freq", int(5e4), "evaluation frequency")
flags.DEFINE_integer("batch_size", int(256), "evaluation frequency")
flags.DEFINE_integer("eval_episodes", int(10), "number of evaluation episodes")
flags.DEFINE_integer("max_timesteps", int(1e6), "maximum number of steps")
flags.DEFINE_integer("num_sgd_steps_per_step", 1, "maximum number of steps")
flags.DEFINE_bool("log_to_wandb", False, "whether to use W&B")


def make_environment(name, seed):
    import d4rl  # noqa: F401
    import gym

    environment = gym.make(name)
    environment = wrappers.GymWrapper(environment)
    environment = wrappers.SinglePrecisionWrapper(environment)
    return environment


@tf.function
def add_episode_return(episode):
    episode_length = episode["steps"].cardinality()
    steps = episode["steps"].batch(episode_length).get_single_element()
    episode_return = tf.reduce_sum(steps["reward"])
    return {**episode, "episode_return": episode_return}


def _batched_step_to_transition(step: rlds.Step) -> types.Transition:
    return types.Transition(
        observation=tf.nest.map_structure(lambda x: x[0], step[rlds.OBSERVATION]),
        action=tf.nest.map_structure(lambda x: x[0], step[rlds.ACTION]),
        reward=tf.nest.map_structure(lambda x: x[0], step[rlds.REWARD]),
        discount=1.0 - tf.cast(step[rlds.IS_TERMINAL][1], dtype=tf.float32),
        # If next step is terminal, then the observation may be arbitrary.
        next_observation=tf.nest.map_structure(lambda x: x[1], step[rlds.OBSERVATION]),
    )


def _batch_steps(episode: rlds.Episode) -> tf.data.Dataset:
    return rlds.transformations.batch(
        episode[rlds.STEPS], size=2, shift=1, drop_remainder=True
    )


def transform_transitions_dataset(episode_dataset, reward_scale):
    batched_steps = episode_dataset.flat_map(_batch_steps)
    transitions = rlds.transformations.map_steps(
        batched_steps, _batched_step_to_transition
    )
    return transitions.map(
        lambda transition: transition._replace(
            reward=(transition.reward * reward_scale)
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def _get_demonstration_dataset_factory(name, batch_size):
    dataset = tfds.load_tfds_dataset(name)
    num_episodes = dataset.cardinality()
    dataset = dataset.map(
        add_episode_return,
        num_parallel_calls=tf.data.AUTOTUNE,
    ).cache()
    episode_returns = dataset.map(
        lambda episode: episode["episode_return"],
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    episode_returns = episode_returns.batch(num_episodes).get_single_element().numpy()
    max_episode_return = np.max(episode_returns)
    min_episode_return = np.min(episode_returns)
    reward_scale = 1000.0 / (max_episode_return - min_episode_return)

    def make_demonstrations(key):
        transitions = transform_transitions_dataset(dataset, reward_scale)
        iterator = tfds.JaxInMemoryRandomSampleIterator(transitions, key, batch_size)
        yield from iterator

    return make_demonstrations


def main(_):
    # Disable TF GPU
    dataset_name = "d4rl_mujoco_halfcheetah/v2-expert"
    env_name = "halfcheetah-expert-v2"
    tf.config.set_visible_devices([], "GPU")

    builder = iql.IQLBuilder(
        config=iql.IQLConfig(
            learning_rate=3e-4,
            discount=0.99,
            expectile=0.7,  # The actual tau for expectiles.
            temperature=3.0,
        )
    )

    assert FLAGS.max_timesteps % FLAGS.num_sgd_steps_per_step == 0
    assert FLAGS.eval_freq % FLAGS.num_sgd_steps_per_step == 0
    demonstration_dataset_factory = _get_demonstration_dataset_factory(
        dataset_name,
        FLAGS.batch_size * FLAGS.num_sgd_steps_per_step,
    )

    environment_factory = lambda seed: make_environment(env_name, seed)
    network_factory = iql.make_networks

    config = experiments.OfflineExperimentConfig(
        builder,
        network_factory=network_factory,
        demonstration_dataset_factory=demonstration_dataset_factory,
        environment_factory=environment_factory,
        max_num_learner_steps=FLAGS.max_timesteps // FLAGS.num_sgd_steps_per_step,
        seed=FLAGS.seed,
        checkpointing=None,
        logger_factory=experiment_utils.LoggerFactory(
            log_to_wandb=FLAGS.log_to_wandb,
            evaluator_time_delta=0.0,
        ),
        observers=[d4rl_evaluation.D4RLScoreObserver(env_name)],
    )
    experiments.run_offline_experiment(
        config,
        eval_every=FLAGS.eval_freq // FLAGS.num_sgd_steps_per_step,
        num_eval_episodes=10,
    )


if __name__ == "__main__":
    app.run(main)
