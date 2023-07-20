"""Run Implicit Q-Learning (IQL) on D4RL MuJoCo."""

import numpy as np
import rlds
import tensorflow as tf
from absl import app
from absl import flags
from acme import types
from acme.jax import experiments
from ml_collections import config_flags

from orax.agents import iql
from orax.baselines import experiment_utils
from orax.baselines.d4rl import d4rl_evaluation
from orax.baselines.d4rl import d4rl_utils
from orax.datasets import tfds

_WORKDIR = flags.DEFINE_string("workdir", "/tmp/orax", "")
_WANDB_PROJECT = flags.DEFINE_string("wandb_project", None, "")
_WANDB_ENTITY = flags.DEFINE_string("wandb_entity", None, "")
_CONFIG = config_flags.DEFINE_config_file("config", None)


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


def transform_transitions_dataset(episode_dataset, reward_scale, reward_bias):
    batched_steps = episode_dataset.flat_map(_batch_steps)
    transitions = rlds.transformations.map_steps(
        batched_steps, _batched_step_to_transition
    )
    return transitions.map(
        lambda transition: transition._replace(
            reward=(transition.reward * reward_scale + reward_bias)
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def _get_demonstration_dataset_factory(d4rl_name, batch_size):
    tfds_name = d4rl_utils.get_tfds_name(d4rl_name)
    dataset = tfds.load_tfds_dataset(tfds_name)
    if "antmaze" in d4rl_name:
        reward_scale = 1.0
        reward_bias = -1.0
    else:
        num_episodes = dataset.cardinality()
        dataset = dataset.map(
            add_episode_return, num_parallel_calls=tf.data.AUTOTUNE
        ).cache()
        episode_returns = dataset.map(
            lambda episode: episode["episode_return"],
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        episode_returns = (
            episode_returns.batch(num_episodes).get_single_element().numpy()
        )
        max_episode_return = np.max(episode_returns)
        min_episode_return = np.min(episode_returns)
        reward_scale = 1000.0 / (max_episode_return - min_episode_return)
        reward_bias = 0.0

    def make_demonstrations(key):
        transitions = transform_transitions_dataset(dataset, reward_scale, reward_bias)
        iterator = tfds.JaxInMemoryRandomSampleIterator(transitions, key, batch_size)
        yield from iterator

    return make_demonstrations


def main(_):
    # Disable TF GPU
    tf.config.set_visible_devices([], "GPU")

    workdir = _WORKDIR.value
    config = _CONFIG.value
    dataset_name = config.dataset_name
    max_num_learner_steps = config.max_num_learner_steps
    batch_size = config.batch_size
    seed = config.seed
    log_to_wandb = config.log_to_wandb
    eval_every = config.eval_every
    add_uid = True

    builder = iql.IQLBuilder(config=iql.IQLConfig(**config.iql_config))

    demonstration_dataset_factory = _get_demonstration_dataset_factory(
        dataset_name, batch_size
    )

    environment_factory = lambda seed: d4rl_utils.make_environment(dataset_name, seed)
    network_factory = iql.make_networks
    logger_factory = experiment_utils.LoggerFactory(
        workdir=workdir,
        log_to_wandb=log_to_wandb,
        evaluator_time_delta=0.0,
        async_learner_logger=True,
        add_uid=add_uid,
        wandb_kwargs={
            "config": config.to_dict(),
            "project": _WANDB_PROJECT.value,
            "entity": _WANDB_ENTITY.value,
            "tags": [dataset_name, "iql"],
        },
    )

    checkpoint_config = None
    if config.checkpoint:
        checkpoint_config = experiments.CheckpointingConfig(
            directory=workdir, add_uid=add_uid, **config.checkpoint_kwargs
        )

    experiment_config = experiments.OfflineExperimentConfig(
        builder,
        network_factory=network_factory,
        demonstration_dataset_factory=demonstration_dataset_factory,
        environment_factory=environment_factory,
        max_num_learner_steps=max_num_learner_steps,
        seed=seed,
        checkpointing=checkpoint_config,
        logger_factory=logger_factory,
        observers=[d4rl_evaluation.D4RLScoreObserver(dataset_name)],
    )

    experiments.run_offline_experiment(
        experiment_config,
        eval_every=eval_every,
        num_eval_episodes=config.num_eval_episodes,
    )


if __name__ == "__main__":
    flags.mark_flag_as_required("config")
    app.run(main)
