import absl
import jax
import ml_collections
import tensorflow as tf
from acme.jax import utils as jax_utils
from ml_collections import config_flags

from orax import experiments
from orax.agents import decision_transformer
from orax.agents.decision_transformer import dataset as dataset_lib
from orax.baselines import experiment_utils
from orax.baselines.d4rl import d4rl_evaluation
from orax.baselines.d4rl import d4rl_utils
from orax.datasets import tfds


def get_config():
    config = ml_collections.ConfigDict()
    config.dataset = "medium"
    config.env = "hopper"
    config.mode = "normal"
    config.K = 20
    config.pct_traj = 1.0
    config.batch_size = 64
    config.network_config = dict(
        hidden_size=128,
        num_layers=3,
        num_heads=1,
        dropout_rate=0.1,
    )
    config.seed = 0
    config.learning_rate = 1e-4
    config.weight_decay = 1e-4
    config.warmup_steps = 10000
    config.num_eval_episodes = 10
    config.num_steps = int(1e5)
    config.eval_every = int(1e4)
    config.log_to_wandb = False
    config.max_ep_len = 1000
    config.scale = 1000.0
    return config


_CONFIG = config_flags.DEFINE_config_dict("config", get_config())


def main(_):
    config = _CONFIG.value

    env_name = config["env"]
    max_ep_len = config.max_ep_len
    scale = config.scale  # normalization for rewards/returns
    seed = config.seed

    if env_name == "hopper":
        eval_env_name = "Hopper-v3"
        target_return = 3600  # evaluation conditioning targets
    elif env_name == "halfcheetah":
        eval_env_name = "HalfCheetah-v3"
        target_return = 12000
    elif env_name == "walker2d":
        eval_env_name = "Walker2d-v3"
        target_return = 5000
    else:
        raise NotImplementedError

    dataset_name = f"{env_name}-{config['dataset']}-v2"
    env = d4rl_utils.make_environment(eval_env_name, seed)

    K = config.K
    batch_size = config.batch_size
    num_eval_episodes = config.num_eval_episodes
    max_num_learner_steps = config.num_steps
    eval_every = config.eval_every
    episode_dataset = tfds.load_tfds_dataset(d4rl_utils.get_tfds_name(dataset_name))
    observation_mean_std = dataset_lib.get_observation_mean_std(episode_dataset)

    def make_dataset_iterator(key):
        del key
        dataset = (
            dataset_lib.transform_decision_transformer_input(
                episode_dataset,
                sequence_length=K,
                scale=scale,
                observation_mean_std=observation_mean_std,
            )
            .shuffle(int(1e6))
            .repeat()
            .batch(batch_size)
            .as_numpy_iterator()
        )
        return jax_utils.device_put(dataset, jax.local_devices()[0])

    network_factory = lambda spec: decision_transformer.make_gym_networks(
        spec=spec, episode_length=max_ep_len, **config.network_config
    )

    dt_config = decision_transformer.DecisionTransformerConfig(
        context_length=K,
        target_return=target_return,
        return_scale=scale,
        mode=config.mode,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        grad_norm_clipping=0.25,
        weight_decay=config.weight_decay,
    )

    builder = decision_transformer.DecisionTransformerBuilder(
        dt_config,
        observation_mean_std,
        max_num_learner_steps=max_num_learner_steps,
        actor_device="gpu",
    )

    logger_factory = experiment_utils.LoggerFactory(log_to_wandb=config.log_to_wandb)

    experiment_config = experiments.OfflineExperimentConfig(
        builder,
        network_factory=network_factory,
        demonstration_dataset_factory=make_dataset_iterator,
        environment_factory=lambda seed: env,
        max_num_learner_steps=max_num_learner_steps,
        seed=0,
        checkpointing=None,
        logger_factory=logger_factory,
        observers=[d4rl_evaluation.D4RLScoreObserver(dataset_name)],
    )

    experiments.run_offline_experiment(
        experiment_config,
        eval_every=eval_every,
        num_eval_episodes=num_eval_episodes,
    )


if __name__ == "__main__":
    tf.config.set_visible_devices([], "GPU")
    absl.app.run(main)
