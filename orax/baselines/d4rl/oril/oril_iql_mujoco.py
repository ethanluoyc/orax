from ml_collections import config_dict

_NUM_SEEDS = 10


def get_config():
    config = config_dict.ConfigDict()
    config.batch_size = 256
    config.max_steps = int(1e6)
    config.evaluate_every = int(1e4)
    config.evaluation_episodes = 10
    config.seed = 0
    config.use_dataset_reward = False
    config.wandb_project = "offline_otil"
    config.wandb_entity = None
    config.expert_dataset_name = "walker2d-medium-expert-v2"
    config.offline_dataset_name = "walker2d-medium-expert-v2"
    config.k = 1

    # IQL config
    config.opt_decay_schedule = "cosine"
    config.dropout_rate = None
    config.actor_lr = 3e-4
    config.value_lr = 3e-4
    config.critic_lr = 3e-4
    config.hidden_dims = (256, 256)
    config.iql_kwargs = dict(
        discount=0.99, expectile=0.7, temperature=3.0  # The actual tau for expectiles.
    )
    config.log_to_wandb = False
    return config


def get_sweep(h):
    del h
    params = []
    for seed in range(_NUM_SEEDS):
        for task in ["walker2d", "hopper", "halfcheetah"]:
            for quality in ["medium", "medium-replay", "medium-expert"]:
                for num_demos in [1, 10]:
                    params.append(
                        {
                            "config.expert_dataset_name": f"{task}-{quality}-v2",
                            "config.k": num_demos,
                            "config.offline_dataset_name": f"{task}-{quality}-v2",
                            "config.seed": seed,
                        }
                    )
    return params
