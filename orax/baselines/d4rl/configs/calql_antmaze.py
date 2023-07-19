import ml_collections
import numpy as np


def get_default_cql_config():
    config = ml_collections.ConfigDict()
    config.discount = 0.99
    config.alpha_multiplier = 1.0
    config.use_automatic_entropy_tuning = True
    config.backup_entropy = False
    config.target_entropy = 0.0
    config.policy_lr = 1e-4
    config.qf_lr = 3e-4
    config.optimizer_type = "adam"
    config.soft_target_update_rate = 5e-3
    config.cql_n_actions = 10
    config.cql_importance_sample = True
    config.cql_lagrange = False
    config.cql_target_action_gap = 1.0
    config.cql_temp = 1.0
    config.cql_max_target_backup = True
    config.cql_clip_diff_min = -np.inf
    config.cql_clip_diff_max = np.inf
    return config


def get_base_config():
    return ml_collections.ConfigDict(
        dict(
            env="antmaze-medium-diverse-v2",
            seed=42,
            save_model=False,
            batch_size=256,
            reward_scale=1.0,
            reward_bias=0.0,
            clip_action=0.99999,
            policy_arch="256-256",
            qf_arch="256-256",
            orthogonal_init=True,
            policy_log_std_multiplier=1.0,
            policy_log_std_offset=-1.0,
            # Total grad_steps of offline pretrain will be (n_train_step_per_epoch_offline * n_pretrain_epochs)
            n_train_step_per_epoch_offline=1000,
            n_pretrain_epochs=1000,
            offline_eval_every_n_epoch=10,
            max_online_env_steps=1e6,
            online_eval_every_n_env_steps=1000,
            eval_n_trajs=5,
            replay_buffer_size=1000000,
            mixing_ratio=-1.0,
            use_cql=True,
            online_use_cql=True,
            cql_min_q_weight=5.0,
            cql_min_q_weight_online=-1.0,
            enable_calql=True,  # Turn on for Cal-QL
            n_online_traj_per_epoch=1,
            online_utd_ratio=1,
            cql=get_default_cql_config(),
        )
    )


def get_config():
    config = get_base_config()
    config.env = "antmaze-medium-diverse-v2"
    config.seed = 0
    config.cql_min_q_weight = 5.0
    config.cql.cql_target_action_gap = 0.8
    config.cql.cql_lagrange = True
    config.policy_arch = "256-256"
    config.qf_arch = "256-256-256-256"
    config.offline_eval_every_n_epoch = 50
    config.online_eval_every_n_env_steps = 2000
    config.eval_n_trajs = 20
    config.n_train_step_per_epoch_offline = 1000
    config.n_pretrain_epochs = 1000
    config.max_online_env_steps = 1e6
    config.mixing_ratio = 0.5
    config.reward_scale = 10.0
    config.reward_bias = -5
    config.enable_calql = True
    return config
