import re
from typing import Tuple


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


def make_environment(name, seed):
    import d4rl  # noqa: F401
    import gym
    from acme import wrappers

    environment = gym.make(name)
    environment.seed(seed)
    environment = wrappers.GymWrapper(environment)
    environment = wrappers.SinglePrecisionWrapper(environment)
    return environment
