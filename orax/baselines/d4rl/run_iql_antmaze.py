"""Run TD3-BC on D4RL."""

import tensorflow as tf
from absl import app
from absl import flags
from acme import wrappers
from acme.jax import experiments

from orax.agents import iql
from orax.baselines import experiment_utils
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


def _get_demonstration_dataset_factory(name, batch_size):
    def make_demonstrations(key):
        transitions = tfds.get_tfds_dataset(name)
        # IQL antmaze subtracts 1 from the reward.
        transitions = transitions.map(
            lambda t: t._replace(reward=t.reward - 1),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        iterator = tfds.JaxInMemoryRandomSampleIterator(transitions, key, batch_size)
        yield from iterator

    return make_demonstrations


def main(_):
    # Disable TF GPU
    dataset_name = "d4rl_antmaze/medium-play-v2"
    env_name = "antmaze-medium-play-v2"
    tf.config.set_visible_devices([], "GPU")

    builder = iql.IQLBuilder(
        config=iql.IQLConfig(
            learning_rate=3e-4,
            discount=0.99,
            expectile=0.9,  # The actual tau for expectiles.
            temperature=10.0,
        )
    )

    assert FLAGS.max_timesteps % FLAGS.num_sgd_steps_per_step == 0
    assert FLAGS.eval_freq % FLAGS.num_sgd_steps_per_step == 0
    demonstration_dataset_factory = _get_demonstration_dataset_factory(
        dataset_name,
        FLAGS.batch_size * FLAGS.num_sgd_steps_per_step,
    )
    environment_factory = lambda seed: make_environment(env_name, seed)

    config = experiments.OfflineExperimentConfig(
        builder,
        network_factory=iql.make_networks,
        demonstration_dataset_factory=demonstration_dataset_factory,
        environment_factory=environment_factory,
        max_num_learner_steps=FLAGS.max_timesteps // FLAGS.num_sgd_steps_per_step,
        seed=FLAGS.seed,
        checkpointing=None,
        logger_factory=experiment_utils.LoggerFactory(
            log_to_wandb=FLAGS.log_to_wandb,
            evaluator_time_delta=0.0,
        ),
    )
    experiments.run_offline_experiment(
        config,
        eval_every=FLAGS.eval_freq // FLAGS.num_sgd_steps_per_step,
        num_eval_episodes=20,
    )


if __name__ == "__main__":
    app.run(main)
