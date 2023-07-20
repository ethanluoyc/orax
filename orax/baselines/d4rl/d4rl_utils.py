import re
from typing import Tuple

import rlds
import tensorflow as tf
from acme import types


def make_environment(name: str, seed: int):
    import d4rl  # noqa: F401
    import gym
    from acme import wrappers

    environment = gym.make(name)
    environment.seed(seed)
    environment = wrappers.GymWrapper(environment)
    environment = wrappers.CanonicalSpecWrapper(environment, clip=True)
    environment = wrappers.SinglePrecisionWrapper(environment)

    return environment


def parse_d4rl_dataset_name(dataset_name: str) -> Tuple[str, str, str]:
    match = re.match(
        r"(?P<env>[a-z]+)-(?P<dataset>[a-z\-]+)-(?P<version>v\d)", dataset_name
    )
    if not match:
        raise ValueError(f"Invalid D4RL dataset name: {dataset_name}")

    return match.group("env"), match.group("dataset"), match.group("version")


def get_tfds_name(d4rl_name: str) -> str:
    """Return the corresponding TFDS name for a given D4RL dataset name."""

    env, dataset, version = parse_d4rl_dataset_name(d4rl_name)
    if env in ["halfcheetah", "hopper", "walker2d", "ant"]:
        return f"d4rl_mujoco_{env}/{version}-{dataset}"
    elif env in ["antmaze"]:
        return f"d4rl_antmaze/{dataset}-{version}"
    elif env in ["pen", "door", "hammer", "relocate"]:
        return f"d4rl_adroit_{env}/{version}-{dataset}"
    else:
        raise ValueError(f"Unknown D4RL environment: {env}")


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


def transform_transitions_dataset(
    episode_dataset,
    reward_scale: float = 1.0,
    reward_bias: float = 0.0,
):
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


@tf.function
def add_episode_return(episode):
    episode_length = episode["steps"].cardinality()
    steps = episode["steps"].batch(episode_length).get_single_element()
    episode_return = tf.reduce_sum(steps["reward"])
    return {**episode, "episode_return": episode_return}
