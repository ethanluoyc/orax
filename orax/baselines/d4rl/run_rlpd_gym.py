import gym
import jax
from absl import app
from absl import flags
from acme import wrappers
from acme.jax import experiments
from ml_collections import config_flags

from orax.agents import rlpd
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


def _get_demonstration_dataset_factory(name, seed):
    def make_demonstrations(batch_size):
        transitions = tfds.get_tfds_dataset(name)
        with jax.default_device(jax.devices("cpu")[0]):
            return tfds.JaxInMemoryRandomSampleIterator(
                transitions, jax.random.PRNGKey(seed), batch_size
            )

    return make_demonstrations


qualities = ["medium", "random", "expert", "medium-replay", "random", "medium-expert"]
tasks = ["halfcheetah", "hopper", "walker2d", "ant"]

DATASET_NAME_TO_TFDS_MAP = {}
DATASET_NAME_TO_GYM_NAME = {}
for quality in qualities:
    for task in tasks:
        d4rl_name = f"{task}-{quality}-v2"
        tfds_name = f"d4rl_mujoco_{task}/v2-{quality}"
        DATASET_NAME_TO_TFDS_MAP[d4rl_name] = tfds_name
        DATASET_NAME_TO_GYM_NAME[d4rl_name] = {
            "halfcheetah": "HalfCheetah-v2",
            "hopper": "Hopper-v2",
            "walker2d": "Walker2d-v2",
            "ant": "Ant-v2",
        }[task]


def main(_):
    config = _CONFIG.value
    env_name = config.env_name
    dataset_name = DATASET_NAME_TO_TFDS_MAP[env_name]

    seed = config.seed

    network_factory = lambda spec: rlpd.make_networks(
        spec,
        hidden_dims=config.hidden_dims,
        num_qs=config.num_qs,
        num_min_qs=config.num_min_qs,
        critic_layer_norm=config.critic_layer_norm,
        use_critic_resnet=False,
        # critic_dropout_rate=config.critic_dropout_rate,
        # use_pnorm=config.use_pnorm,
    )

    sac_config = rlpd.REDQConfig(
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        temp_lr=config.temp_lr,
        critic_weight_decay=config.critic_weight_decay,
        init_temperature=config.init_temperature,
        backup_entropy=config.backup_entropy,
        discount=config.discount,
        n_step=1,
        target_entropy=None,  # Automatic entropy tuning.
        # Target smoothing coefficient.
        tau=config.tau,
        # Replay options
        max_replay_size=config.max_steps,
        #
        batch_size=config.batch_size,
        min_replay_size=config.start_training,
        samples_per_insert=(
            config.utd_ratio * config.batch_size * (1 - config.offline_ratio)
        ),
        num_sgd_steps_per_step=config.utd_ratio,
        offline_ratio=config.offline_ratio,
    )

    builder = rlpd.REDQBuilder(
        sac_config,
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
