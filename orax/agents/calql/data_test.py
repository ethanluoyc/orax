# ruff: noqa: E501
"""Tests that we reproduce the same behavior as the data processing in Cal-QL."""
import collections

import gym
import numpy as np

from orax.agents.calql import adder
from orax.datasets.adroit_binary.adroit_binary import load_bc_data
from orax.datasets.adroit_binary.adroit_binary import load_expert_data

# Code adapted from https://raw.githubusercontent.com/nakamotoo/Cal-QL/main/JaxCQL/replay_buffer.py

ENV_CONFIG = {
    "antmaze": {
        "reward_pos": 1.0,
        "reward_neg": 0.0,
    },
    "adroit-binary": {
        "reward_pos": 0.0,
        "reward_neg": -1.0,
    },
}


def qlearning_dataset_and_calc_mc(
    env,
    reward_scale,
    reward_bias,
    clip_action,
    gamma,
    dataset=None,
    terminate_on_end=False,
    is_sparse_reward=True,
    **kwargs,
):
    dataset = env.get_dataset(**kwargs)
    N = dataset["rewards"].shape[0]
    data_ = collections.defaultdict(list)
    episodes_dict_list = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if "timeouts" in dataset:
        use_timeouts = True

    # first process by traj
    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset["terminals"][i])
        if use_timeouts:
            final_timestep = dataset["timeouts"][i]
        else:
            final_timestep = episode_step == env._max_episode_steps - 1
        if (not terminate_on_end) and final_timestep or i == N - 1:
            # Skip this transition and don't apply terminals on the last step of an episode
            pass
        else:
            for k in dataset:
                if k in [
                    "actions",
                    "next_observations",
                    "observations",
                    "rewards",
                    "terminals",
                    "timeouts",
                ]:
                    data_[k].append(dataset[k][i])
            if "next_observations" not in dataset.keys():
                data_["next_observations"].append(dataset["observations"][i + 1])
            episode_step += 1

        if (done_bool or final_timestep) and episode_step > 0:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])

            episode_data["rewards"] = (
                episode_data["rewards"] * reward_scale + reward_bias
            )
            episode_data["mc_returns"] = calc_return_to_go(
                env.spec.name,
                episode_data["rewards"],
                episode_data["terminals"],
                gamma,
                reward_scale,
                reward_bias,
                is_sparse_reward,
            )
            episode_data["actions"] = np.clip(
                episode_data["actions"], -clip_action, clip_action
            )
            episodes_dict_list.append(episode_data)
            data_ = collections.defaultdict(list)

    return episodes_dict_list


def get_d4rl_dataset_with_mc_calculation(
    env, reward_scale, reward_bias, clip_action, gamma
):
    if "antmaze" in env:
        is_sparse_reward = True
    else:
        raise NotImplementedError
    dataset = qlearning_dataset_and_calc_mc(
        gym.make(env).unwrapped,
        reward_scale,
        reward_bias,
        clip_action,
        gamma,
        is_sparse_reward=is_sparse_reward,
    )

    return dict(
        observations=dataset["observations"],
        actions=dataset["actions"],
        next_observations=dataset["next_observations"],
        rewards=dataset["rewards"],
        dones=dataset["terminals"].astype(np.float32),
        mc_returns=dataset["mc_returns"],
    )


def get_hand_dataset_with_mc_calculation(
    env_name,
    gamma,
    add_expert_demos=True,
    add_bc_demos=True,
    reward_scale=1.0,
    reward_bias=0.0,
    pos_ind=-1,
    clip_action=None,
    data_dir="demonstraionts/offpolicy_hand_data",
):
    assert env_name in [
        "pen-binary-v0",
        "door-binary-v0",
        "relocate-binary-v0",
        "pen-binary",
        "door-binary",
        "relocate-binary",
    ]

    expert_demo_paths = {
        "pen-binary-v0": f"{data_dir}/pen2_sparse.npy",
        "door-binary-v0": f"{data_dir}/door2_sparse.npy",
        "relocate-binary-v0": f"{data_dir}/relocate2_sparse.npy",
    }

    bc_demo_paths = {
        "pen-binary-v0": f"{data_dir}/pen_bc_sparse4.npy",
        "door-binary-v0": f"{data_dir}/door_bc_sparse4.npy",
        "relocate-binary-v0": f"{data_dir}/relocate_bc_sparse4.npy",
    }

    def truncate_traj(
        env_name,
        dataset,
        i,
        reward_scale,
        reward_bias,
        gamma,
        start_index=None,
        end_index=None,
    ):
        """
        This function truncates the i'th trajectory in dataset from start_index to end_index.
        Since in Adroit-binary datasets, we have trajectories like [-1, -1, -1, -1, 0, 0, 0, -1, -1] which transit from neg -> pos -> neg,
        we truncate the trajcotry from the beginning to the last positive reward, i.e., [-1, -1, -1, -1, 0, 0, 0]
        """
        reward_pos = ENV_CONFIG["adroit-binary"]["reward_pos"]

        observations = np.array(dataset[i]["observations"])[start_index:end_index]
        next_observations = np.array(dataset[i]["next_observations"])[
            start_index:end_index
        ]
        rewards = dataset[i]["rewards"][start_index:end_index]
        dones = rewards == reward_pos
        rewards = rewards * reward_scale + reward_bias
        actions = np.array(dataset[i]["actions"])[start_index:end_index]
        mc_returns = calc_return_to_go(
            env_name,
            rewards,
            dones,
            gamma,
            reward_scale,
            reward_bias,
            is_sparse_reward=True,
        )

        return dict(
            observations=observations,
            next_observations=next_observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            mc_returns=mc_returns,
        )

    dataset_list = []
    dataset_bc_list = []
    if add_expert_demos:
        print("loading expert demos from:", expert_demo_paths[env_name])
        dataset = np.load(expert_demo_paths[env_name], allow_pickle=True)

        for i in range(len(dataset)):
            N = len(dataset[i]["observations"])
            for j in range(len(dataset[i]["observations"])):
                dataset[i]["observations"][j] = dataset[i]["observations"][j][
                    "state_observation"
                ]
                dataset[i]["next_observations"][j] = dataset[i]["next_observations"][j][
                    "state_observation"
                ]
            if (
                np.array(dataset[i]["rewards"]).shape
                != np.array(dataset[i]["terminals"]).shape
            ):
                dataset[i]["rewards"] = dataset[i]["rewards"][:N]

            if clip_action:
                dataset[i]["actions"] = np.clip(
                    dataset[i]["actions"], -clip_action, clip_action
                )

            assert (
                np.array(dataset[i]["rewards"]).shape
                == np.array(dataset[i]["terminals"]).shape
            )
            dataset[i].pop("terminals", None)

            if not (0 in dataset[i]["rewards"]):  # noqa
                continue

            trunc_ind = np.where(dataset[i]["rewards"] == 0)[0][pos_ind] + 1
            d_pos = truncate_traj(
                env_name,
                dataset,
                i,
                reward_scale,
                reward_bias,
                gamma,
                start_index=None,
                end_index=trunc_ind,
            )
            dataset_list.append(d_pos)

    if add_bc_demos:
        print("loading BC demos from:", bc_demo_paths[env_name])
        dataset_bc = np.load(bc_demo_paths[env_name], allow_pickle=True)
        for i in range(len(dataset_bc)):
            dataset_bc[i]["rewards"] = dataset_bc[i]["rewards"].squeeze()
            dataset_bc[i]["dones"] = dataset_bc[i]["terminals"].squeeze()
            dataset_bc[i].pop("terminals", None)
            if clip_action:
                dataset_bc[i]["actions"] = np.clip(
                    dataset_bc[i]["actions"], -clip_action, clip_action
                )

            if not (0 in dataset_bc[i]["rewards"]):  # noqa
                continue
            trunc_ind = np.where(dataset_bc[i]["rewards"] == 0)[0][pos_ind] + 1
            d_pos = truncate_traj(
                env_name,
                dataset_bc,
                i,
                reward_scale,
                reward_bias,
                gamma,
                start_index=None,
                end_index=trunc_ind,
            )
            dataset_bc_list.append(d_pos)

    # dataset = np.concatenate([dataset_list, dataset_bc_list])
    return dataset_list + dataset_bc_list


def calc_return_to_go(
    env_name, rewards, terminals, gamma, reward_scale, reward_bias, is_sparse_reward
):
    """
    A config dict for getting the default high/low rewrd values for each envs
    This is used in calc_return_to_go func in sampler.py and replay_buffer.py
    """
    if len(rewards) == 0:
        return np.array([])

    if "antmaze" in env_name:
        reward_neg = ENV_CONFIG["antmaze"]["reward_neg"] * reward_scale + reward_bias
    elif env_name in [
        "pen-binary-v0",
        "door-binary-v0",
        "relocate-binary-v0",
        "pen-binary",
        "door-binary",
        "relocate-binary",
    ]:
        reward_neg = (
            ENV_CONFIG["adroit-binary"]["reward_neg"] * reward_scale + reward_bias
        )
    else:
        assert (
            not is_sparse_reward
        ), "If you want to try on a sparse reward env, please add the reward_neg value in the ENV_CONFIG dict."

    if is_sparse_reward and np.all(np.array(rewards) == reward_neg):
        """
        If the env has sparse reward and the trajectory is all negative rewards,
        we use r / (1-gamma) as return to go.
        For exapmle, if gamma = 0.99 and the rewards = [-1, -1, -1],
        then return_to_go = [-100, -100, -100]
        """
        # assuming failure reward is negative
        # use r / (1-gamma) for negative trajctory
        return_to_go = [float(reward_neg / (1 - gamma))] * len(rewards)
    else:
        return_to_go = [0] * len(rewards)
        prev_return = 0
        for i in range(len(rewards)):
            return_to_go[-i - 1] = rewards[-i - 1] + gamma * prev_return * (
                1 - terminals[-i - 1]
            )
            prev_return = return_to_go[-i - 1]

    return np.array(return_to_go, dtype=np.float32)


def transform_adroit_episodes(
    rlds_episodes,
    gamma,
    positive_reward,
    negative_reward,
    reward_scale=1.0,
    reward_bias=0.0,
):
    def truncate_traj(episode):
        """
        This function truncates the i'th trajectory in dataset from start_index to end_index.
        Since in Adroit-binary datasets, we have trajectories like [-1, -1, -1, -1, 0, 0, 0, -1, -1] which transit from neg -> pos -> neg,
        we truncate the trajcotry from the beginning to the last positive reward, i.e., [-1, -1, -1, -1, 0, 0, 0]
        """
        reward_pos = ENV_CONFIG["adroit-binary"]["reward_pos"]

        rewards = episode["reward"][:-1]
        observations = episode["observation"][:-1]
        next_observations = episode["observation"][1:]
        actions = episode["action"][:-1]
        start_index = 0
        trunc_ind = np.where(rewards == 0)[0][-1] + 1
        end_index = trunc_ind

        observations = observations[start_index:end_index]
        next_observations = next_observations[start_index:end_index]
        rewards = rewards[start_index:end_index]
        dones = rewards == reward_pos
        rewards = rewards * reward_scale + reward_bias
        actions = actions[start_index:end_index]

        mc_returns = adder.compute_return_to_go(
            rewards,
            dones,
            gamma=gamma,
            reward_scale=reward_scale,
            reward_bias=reward_bias,
            is_sparse_reward=True,
            negative_reward=negative_reward,
        )

        return dict(
            observations=observations,
            next_observations=next_observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            mc_returns=mc_returns,
        )

    episodes = []
    for episode in rlds_episodes:
        # Filter episodes without any positive reward (0)
        if not ((episode["reward"][:-1] == positive_reward).any()):  # noqa
            continue
        episodes.append(truncate_traj(episode))

    return episodes


def main():
    data_dir = "/home/yicheng/data/offpolicy_hand_data"
    for is_expert in [True, False]:
        for env in ["pen-binary-v0", "door-binary-v0", "relocate-binary-v0"]:
            expected_episodes = get_hand_dataset_with_mc_calculation(
                env,
                gamma=0.99,
                add_expert_demos=is_expert,
                add_bc_demos=not is_expert,
                reward_scale=10.0,
                reward_bias=5.0,
                data_dir=data_dir,
            )

            def load_data(env_name, is_expert):
                expert_demo_paths = {
                    "pen-binary-v0": f"{data_dir}/pen2_sparse.npy",
                    "door-binary-v0": f"{data_dir}/door2_sparse.npy",
                    "relocate-binary-v0": f"{data_dir}/relocate2_sparse.npy",
                }

                bc_demo_paths = {
                    "pen-binary-v0": f"{data_dir}/pen_bc_sparse4.npy",
                    "door-binary-v0": f"{data_dir}/door_bc_sparse4.npy",
                    "relocate-binary-v0": f"{data_dir}/relocate_bc_sparse4.npy",
                }

                if is_expert:
                    path = expert_demo_paths[env_name]
                    return load_expert_data(path)
                else:
                    path = bc_demo_paths[env_name]
                    return load_bc_data(path)

            episodes = transform_adroit_episodes(
                load_data(env, is_expert=is_expert),
                positive_reward=ENV_CONFIG["adroit-binary"]["reward_pos"],
                negative_reward=ENV_CONFIG["adroit-binary"]["reward_neg"],
                reward_scale=10.0,
                reward_bias=5.0,
                gamma=0.99,
            )

            assert len(expected_episodes) == len(episodes)
            for expected_episode, episode in zip(expected_episodes, episodes):
                np.testing.assert_allclose(
                    expected_episode["observations"], episode["observations"]
                )
                np.testing.assert_allclose(
                    expected_episode["actions"], episode["actions"]
                )
                np.testing.assert_allclose(
                    expected_episode["rewards"], episode["rewards"]
                )
                np.testing.assert_allclose(expected_episode["dones"], episode["dones"])
                np.testing.assert_allclose(
                    expected_episode["next_observations"], episode["next_observations"]
                )


if __name__ == "__main__":
    main()
