import gym
import jax
from absl import app
from absl import flags
from acme import wrappers
from acme.jax import experiments
from ml_collections import config_flags

from orax.agents import redq
from orax.baselines import experiment_utils
from orax.datasets import tfds

# Somehow importing d4rl earlier causes segfault in singularity
import d4rl  # noqa: F401 isort:skip

_WORKDIR = flags.DEFINE_string("workdir", "/tmp/rlpd", "")
_CONFIG = config_flags.DEFINE_config_file(
    "config", None, "File path to the training configuration."
)


def make_environment(name, rescale_actions=True):
    env = gym.make(name)
    env = wrappers.GymWrapper(env)

    if rescale_actions:
        env = wrappers.CanonicalSpecWrapper(env, clip=True)

    env = wrappers.SinglePrecisionWrapper(env)

    return env


qualities = ["medium", "random", "expert", "medium-replay", "random", "medium-expert"]
tasks = ["halfcheetah", "hopper", "walker2d", "ant"]

# Mujoco
DATASET_NAME_TO_TFDS_MAP = {}
for quality in qualities:
    for task in tasks:
        d4rl_name = f"{task}-{quality}-v2"
        tfds_name = f"d4rl_mujoco_{task}/v2-{quality}"
        DATASET_NAME_TO_TFDS_MAP[d4rl_name] = tfds_name

# Antmaze
for dset in [
    "large-diverse",
    "large-play",
    "medium-play",
    "medium-diverse",
    "umaze",
    "umaze-diverse",
]:
    for version in ["v0", "v2"]:
        # V2 added recently in nightly only
        # https://github.com/tensorflow/datasets/pull/5008
        d4rl_name = f"antmaze-{dset}-{version}"
        tfds_name = f"d4rl_antmaze/{dset}-{version}"
        DATASET_NAME_TO_TFDS_MAP[d4rl_name] = tfds_name


def _get_demonstration_dataset_factory(name, seed):
    def make_demonstrations(batch_size):
        transitions = tfds.get_tfds_dataset(name)
        with jax.default_device(jax.devices("cpu")[0]):
            # NOTE(yl): The yield from is necessary to
            # for some reason returning the iterator directly causes
            # sampling to be much slower.
            yield from tfds.JaxInMemoryRandomSampleIterator(
                transitions, jax.random.PRNGKey(seed), batch_size
            )

    return make_demonstrations


def main(_):
    config = _CONFIG.value
    env_name = config.env_name
    dataset_name = DATASET_NAME_TO_TFDS_MAP[env_name]

    seed = config.seed

    network_factory = lambda spec: redq.make_networks(
        spec,
        hidden_sizes=config.hidden_dims,
        num_qs=config.num_qs,
        num_min_qs=config.num_min_qs,
        critic_layer_norm=config.critic_layer_norm,
    )

    redq_config = redq.REDQConfig(
        actor_learning_rate=config.actor_lr,
        critic_learning_rate=config.critic_lr,
        temperature_learning_rate=config.temp_lr,
        init_temperature=config.init_temperature,
        backup_entropy=config.backup_entropy,
        discount=config.discount,
        n_step=1,
        target_entropy=None,  # Automatic entropy tuning.
        # Target smoothing coefficient.
        tau=config.tau,
        max_replay_size=config.max_steps,
        batch_size=config.batch_size,
        min_replay_size=config.start_training,
        # Convert from UTD to SPI
        # In the pure online setting, SPI = batch_size is equivalent to UTD = 1
        # For RLPD, SPI = online_batch_size * UTD = (1 - offline_ratio) * batch_size * UTD
        samples_per_insert=(
            config.utd_ratio * config.batch_size * (1 - config.offline_ratio)
        ),
        # Effectively equivalent to UTD
        num_sgd_steps_per_step=config.utd_ratio,
        offline_fraction=config.offline_ratio,
        reward_bias=-1 if "antmaze" in env_name else 0,
    )

    builder = redq.REDQBuilder(
        redq_config,
        make_demonstrations=_get_demonstration_dataset_factory(dataset_name, seed=seed),
    )

    experiment = experiments.ExperimentConfig(
        builder=builder,
        environment_factory=lambda _: make_environment(env_name),
        network_factory=network_factory,
        seed=seed,
        max_num_actor_steps=config.max_steps,
        logger_factory=experiment_utils.LoggerFactory(
            workdir=_WORKDIR.value,
            log_to_wandb=config.log_to_wandb,
            wandb_kwargs={
                "project": "rlpd",
                "config": config,
            },
            evaluator_time_delta=0.01,
        ),
        checkpointing=None,
    )
    experiments.run_experiment(
        experiment=experiment,
        eval_every=config.eval_interval,
        num_eval_episodes=config.eval_episodes,
    )


if __name__ == "__main__":
    app.run(main)
