from typing import Sequence

import haiku as hk
import jax.numpy as jnp
from acme import specs
from acme.jax import networks as networks_lib
from acme.jax import utils


def make_reward_network(
    spec: specs.EnvironmentSpec, hidden_layer_sizes: Sequence[int] = (256, 256)
):
    def _reward_fn(obs):
        network = hk.nets.MLP(tuple(hidden_layer_sizes) + (1,))
        reward = network(obs)
        return jnp.squeeze(reward)

    reward_network = hk.without_apply_rng(hk.transform(_reward_fn))
    dummy_obs = utils.add_batch_dim(utils.zeros_like(spec.observations))
    return networks_lib.FeedForwardNetwork(
        init=lambda key: reward_network.init(key, dummy_obs),
        apply=reward_network.apply,
    )
