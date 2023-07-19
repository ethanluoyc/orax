from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()

    # General configuration
    config.dataset_name = "halfcheetah-medium-v2"
    config.seed = 0
    config.batch_size = 256
    config.eval_every = int(5e4)
    config.num_eval_episodes = 10
    config.max_num_learner_steps = int(1e6)

    config.iql_config = config_dict.ConfigDict()
    config.iql_config.learning_rate = 3e-4
    config.iql_config.discount = 0.99
    config.iql_config.expectile = 0.7
    config.iql_config.temperature = 3.0

    config.log_to_wandb = False
    config.checkpoint = False
    config.checkpoint_kwargs = config_dict.ConfigDict()

    return config
