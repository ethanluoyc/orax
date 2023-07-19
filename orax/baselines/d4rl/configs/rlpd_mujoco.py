from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()
    config.actor_lr = 3e-4
    config.critic_lr = 3e-4

    config.hidden_dims = (256, 256)

    config.discount = 0.99

    config.num_qs = 2

    config.tau = 0.005

    config.critic_layer_norm = False

    config.temp_lr = 3e-4

    config.init_temperature = 1.0
    config.target_entropy = config_dict.placeholder(float)

    config.backup_entropy = True
    config.critic_weight_decay = config_dict.placeholder(float)

    config.num_qs = 10
    config.num_min_qs = 2
    config.critic_layer_norm = True

    config.log_to_wandb = False
    config.env_name = "halfcheetah-expert-v2"
    config.offline_ratio = 0.5
    config.seed = 42
    config.eval_episodes = 10
    config.eval_interval = 10000
    config.batch_size = 256
    config.max_steps = int(250000)
    config.utd_ratio = 20
    config.start_training = int(1e4)

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
