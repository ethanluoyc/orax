from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()
    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr = 3e-4

    config.hidden_dims = (256, 256, 256)
    config.num_qs = 10
    config.num_min_qs = 1
    config.discount = 0.99

    config.tau = 0.005

    config.critic_layer_norm = True

    config.init_temperature = 1.0

    config.backup_entropy = False

    config.env_name = "antmaze-umaze-v2"
    config.offline_ratio = 0.5
    config.seed = 42
    config.eval_episodes = 10
    config.eval_interval = 10000
    config.batch_size = 256
    config.max_steps = int(300000)

    config.utd_ratio = 20
    config.start_training = int(5000)

    config.log_to_wandb = False

    return config


_TASKS = ["walker2d", "hopper", "halfcheetah", "ant"]
_DATASET_QUALITIES = ["medium", "medium-replay"]
_DS_VERSION = "v2"
_NUM_SEEDS = 5

datasets = []
for task in _TASKS:
    datasets.extend(
        [f"{task}-{quality}-{_DS_VERSION}" for quality in _DATASET_QUALITIES]
    )


def get_sweep():
    sweep = []
    for seed in range(_NUM_SEEDS):
        for dataset in datasets:
            sweep.append({"seed": seed, "env_name": dataset})
    return sweep
