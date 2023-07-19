import dataclasses
import functools
from typing import Any, Callable, Optional, Sequence, Type

import flax.linen as nn
import jax
import jax.numpy as jnp
import tensorflow_probability
from acme import specs

tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions
tfb = tfp.bijectors

ModuleDef = Any

default_init = nn.initializers.xavier_uniform


class TanhDeterministic(nn.Module):
    base_cls: Type[nn.Module]
    action_dim: int

    @nn.compact
    def __call__(self, inputs, *args, **kwargs) -> jnp.ndarray:
        x = self.base_cls()(inputs, *args, **kwargs)

        means = nn.Dense(
            self.action_dim, kernel_init=default_init(), name="OutputDenseMean"
        )(x)

        means = nn.tanh(means)

        return means


class TanhTransformedDistribution(tfd.TransformedDistribution):
    def __init__(self, distribution: tfd.Distribution, validate_args: bool = False):
        super().__init__(
            distribution=distribution, bijector=tfb.Tanh(), validate_args=validate_args
        )

    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())

    @classmethod
    def _parameter_properties(cls, dtype: Optional[Any], num_classes=None):
        td_properties = super()._parameter_properties(dtype, num_classes=num_classes)
        del td_properties["bijector"]
        return td_properties


class Normal(nn.Module):
    base_cls: Type[nn.Module]
    action_dim: int
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    state_dependent_std: bool = True
    squash_tanh: bool = False

    @nn.compact
    def __call__(self, inputs, *args, **kwargs) -> tfd.Distribution:
        x = self.base_cls()(inputs, *args, **kwargs)

        means = nn.Dense(
            self.action_dim, kernel_init=default_init(), name="OutputDenseMean"
        )(x)
        if self.state_dependent_std:
            log_stds = nn.Dense(
                self.action_dim, kernel_init=default_init(), name="OutputDenseLogStd"
            )(x)
        else:
            log_stds = self.param(
                "OutpuLogStd", nn.initializers.zeros, (self.action_dim,), jnp.float32
            )

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = tfd.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds)
        )

        if self.squash_tanh:
            return TanhTransformedDistribution(distribution)
        else:
            return distribution


TanhNormal = functools.partial(Normal, squash_tanh=True)


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: bool = False
    use_layer_norm: bool = False
    scale_final: Optional[float] = None
    dropout_rate: Optional[float] = None
    use_pnorm: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            if i + 1 == len(self.hidden_dims) and self.scale_final is not None:
                x = nn.Dense(size, kernel_init=default_init(self.scale_final))(x)
            else:
                x = nn.Dense(size, kernel_init=default_init())(x)

            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training
                    )
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
        if self.use_pnorm:
            x /= jnp.linalg.norm(x, axis=-1, keepdims=True).clip(1e-10)
        return x


class StateActionValue(nn.Module):
    base_cls: nn.Module

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray, *args, **kwargs
    ) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], axis=-1)
        outputs = self.base_cls()(inputs, *args, **kwargs)

        value = nn.Dense(1, kernel_init=default_init())(outputs)

        return jnp.squeeze(value, -1)


class Ensemble(nn.Module):
    net_cls: Type[nn.Module]
    num: int = 2

    @nn.compact
    def __call__(self, *args):
        ensemble = nn.vmap(
            self.net_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num,
        )
        return ensemble()(*args)


def subsample_ensemble(key: jax.random.PRNGKey, params, num_sample: int, num_qs: int):
    if num_sample is not None:
        all_indx = jnp.arange(0, num_qs)
        indx = jax.random.choice(key, a=all_indx, shape=(num_sample,), replace=False)

        if "Ensemble_0" in params:
            ens_params = jax.tree_util.tree_map(
                lambda param: param[indx], params["Ensemble_0"]
            )
            params = params.copy(add_or_replace={"Ensemble_0": ens_params})
        else:
            params = jax.tree_util.tree_map(lambda param: param[indx], params)
    return params


class MLPResNetV2Block(nn.Module):
    """MLPResNet block."""

    features: int
    act: Callable

    @nn.compact
    def __call__(self, x):
        residual = x
        y = nn.LayerNorm()(x)
        y = self.act(y)
        y = nn.Dense(self.features)(y)
        y = nn.LayerNorm()(y)
        y = self.act(y)
        y = nn.Dense(self.features)(y)

        if residual.shape != y.shape:
            residual = nn.Dense(self.features)(residual)

        return residual + y


class MLPResNetV2(nn.Module):
    """MLPResNetV2."""

    num_blocks: int
    features: int = 256
    dtype: Any = jnp.float32
    act: Callable = nn.relu

    @nn.compact
    def __call__(self, x, training=False):
        x = nn.Dense(self.features)(x)
        for _ in range(self.num_blocks):
            x = MLPResNetV2Block(self.features, act=self.act)(x)
        x = nn.LayerNorm()(x)
        x = self.act(x)
        return x


class Temperature(nn.Module):
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param(
            "log_temp",
            init_fn=lambda key: jnp.full((), jnp.log(self.initial_temperature)),
        )
        return jnp.exp(log_temp)


@dataclasses.dataclass
class REDQNetworks:
    actor: nn.Module
    critic: nn.Module
    target_critic: nn.Module
    num_qs: int
    num_min_qs: int


def make_networks(
    spec: specs.EnvironmentSpec,
    hidden_dims: Sequence[int] = (256, 256),
    num_qs: int = 2,
    num_min_qs: Optional[int] = None,
    critic_dropout_rate: Optional[float] = None,
    use_pnorm: bool = False,
    use_critic_resnet: bool = False,
    critic_layer_norm: bool = True,
):
    action_dim = spec.actions.shape[0]

    actor_base_cls = functools.partial(
        MLP, hidden_dims=hidden_dims, activate_final=True, use_pnorm=use_pnorm
    )
    actor_def = TanhNormal(actor_base_cls, action_dim)

    if use_critic_resnet:
        critic_base_cls = functools.partial(
            MLPResNetV2,
            num_blocks=1,
        )
    else:
        critic_base_cls = functools.partial(
            MLP,
            hidden_dims=hidden_dims,
            activate_final=True,
            dropout_rate=critic_dropout_rate,
            use_layer_norm=critic_layer_norm,
            use_pnorm=use_pnorm,
        )

    critic_cls = functools.partial(StateActionValue, base_cls=critic_base_cls)
    critic_def = Ensemble(critic_cls, num=num_qs)
    target_critic_def = Ensemble(critic_cls, num=num_min_qs or num_qs)

    return REDQNetworks(actor_def, critic_def, target_critic_def, num_qs, num_min_qs)


def apply_policy_and_sample(networks, evaluation: bool):
    def policy_network(params, key, observations):
        dist = networks.actor.apply({"params": params}, observations)
        if evaluation:
            return dist.mode()
        return dist.sample(seed=key)

    return policy_network
