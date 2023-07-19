"""Implementations of algorithms for continuous control."""

from functools import partial
from typing import Dict, Optional, Tuple

import flax
import jax
import jax.numpy as jnp
import optax
from acme import core
from acme import types
from acme.utils import counting
from acme.utils import loggers
from dm_env import specs
from flax import struct
from flax.training.train_state import TrainState

from orax.agents.rlpd.networks import Temperature
from orax.agents.rlpd.networks import subsample_ensemble


# From https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification_flax.ipynb#scrollTo=ap-zaOyKJDXM
def decay_mask_fn(params):
    flat_params = flax.traverse_util.flatten_dict(params)
    flat_mask = {path: path[-1] != "bias" for path in flat_params}
    return flax.core.FrozenDict(flax.traverse_util.unflatten_dict(flat_mask))


class REDQLearnerCore(struct.PyTreeNode):
    actor: TrainState
    rng: jax.random.PRNGKeyArray
    critic: TrainState
    target_critic: TrainState
    temp: TrainState
    tau: float
    discount: float
    target_entropy: float
    num_qs: int = struct.field(pytree_node=False)
    num_min_qs: Optional[int] = struct.field(
        pytree_node=False
    )  # See M in RedQ https://arxiv.org/abs/2101.05982
    backup_entropy: bool = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        networks,
        key: int,
        observation_spec: specs.Array,
        action_spec: specs.Array,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        discount: float = 0.99,
        tau: float = 0.005,
        critic_weight_decay: Optional[float] = None,
        target_entropy: Optional[float] = None,
        init_temperature: float = 1.0,
        backup_entropy: bool = True,
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        action_dim = action_spec.shape[-1]
        observations = observation_spec.generate_value()
        actions = action_spec.generate_value()

        if target_entropy is None:
            target_entropy = -action_dim / 2

        rng = key
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        actor_params = networks.actor.init(actor_key, observations)["params"]
        actor = TrainState.create(
            apply_fn=networks.actor.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr),
        )

        critic_params = networks.critic.init(critic_key, observations, actions)[
            "params"
        ]
        if critic_weight_decay is not None:
            tx = optax.adamw(
                learning_rate=critic_lr,
                weight_decay=critic_weight_decay,
                mask=decay_mask_fn,
            )
        else:
            tx = optax.adam(learning_rate=critic_lr)
        critic = TrainState.create(
            apply_fn=networks.critic.apply,
            params=critic_params,
            tx=tx,
        )
        target_critic = TrainState.create(
            apply_fn=networks.target_critic.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=optax.adam(learning_rate=temp_lr),
        )

        return cls(
            rng=rng,
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            temp=temp,
            target_entropy=target_entropy,
            tau=tau,
            discount=discount,
            num_qs=networks.num_qs,
            num_min_qs=networks.num_min_qs,
            backup_entropy=backup_entropy,
        )

    def update_actor(
        self, batch: types.Transition
    ) -> Tuple["REDQLearnerCore", Dict[str, float]]:
        key, rng = jax.random.split(self.rng)
        key2, rng = jax.random.split(rng)

        def actor_loss_fn(actor_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            dist = self.actor.apply_fn({"params": actor_params}, batch.observation)
            actions = dist.sample(seed=key)
            log_probs = dist.log_prob(actions)
            qs = self.critic.apply_fn(
                {"params": self.critic.params},
                batch.observation,
                actions,
                True,
                rngs={"dropout": key2},
            )  # training=True
            q = qs.mean(axis=0)
            actor_loss = (
                log_probs * self.temp.apply_fn({"params": self.temp.params}) - q
            ).mean()
            return actor_loss, {"actor_loss": actor_loss, "entropy": -log_probs.mean()}

        grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(self.actor.params)
        actor = self.actor.apply_gradients(grads=grads)

        return self.replace(actor=actor, rng=rng), actor_info

    def update_temperature(
        self, entropy: float
    ) -> Tuple["REDQLearnerCore", Dict[str, float]]:
        def temperature_loss_fn(temp_params):
            temperature = self.temp.apply_fn({"params": temp_params})
            temp_loss = temperature * (entropy - self.target_entropy).mean()
            return temp_loss, {
                "temperature": temperature,
                "temperature_loss": temp_loss,
            }

        grads, temp_info = jax.grad(temperature_loss_fn, has_aux=True)(self.temp.params)
        temp = self.temp.apply_gradients(grads=grads)

        return self.replace(temp=temp), temp_info

    def update_critic(
        self, batch: types.Transition
    ) -> Tuple[TrainState, Dict[str, float]]:
        dist = self.actor.apply_fn(
            {"params": self.actor.params}, batch.next_observation
        )

        rng = self.rng

        key, rng = jax.random.split(rng)
        next_actions = dist.sample(seed=key)

        # Used only for REDQ.
        key, rng = jax.random.split(rng)
        target_params = subsample_ensemble(
            key, self.target_critic.params, self.num_min_qs, self.num_qs
        )

        key, rng = jax.random.split(rng)
        next_qs = self.target_critic.apply_fn(
            {"params": target_params},
            batch.next_observation,
            next_actions,
            True,
            rngs={"dropout": key},
        )  # training=True
        next_q = next_qs.min(axis=0)

        target_q = batch.reward + self.discount * batch.discount * next_q

        if self.backup_entropy:
            next_log_probs = dist.log_prob(next_actions)
            target_q -= (
                self.discount
                * batch.discount
                * self.temp.apply_fn({"params": self.temp.params})
                * next_log_probs
            )

        key, rng = jax.random.split(rng)

        def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qs = self.critic.apply_fn(
                {"params": critic_params},
                batch.observation,
                batch.action,
                True,
                rngs={"dropout": key},
            )  # training=True
            critic_loss = ((qs - target_q) ** 2).mean()
            return critic_loss, {"critic_loss": critic_loss, "q": qs.mean()}

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(self.critic.params)
        critic = self.critic.apply_gradients(grads=grads)

        target_critic_params = optax.incremental_update(
            critic.params, self.target_critic.params, self.tau
        )
        target_critic = self.target_critic.replace(params=target_critic_params)

        return self.replace(critic=critic, target_critic=target_critic, rng=rng), info

    @partial(jax.jit, static_argnames="utd_ratio")
    def update(self, batch: types.Transition, utd_ratio: int):
        new_agent = self
        for i in range(utd_ratio):

            def slice(x):
                assert x.shape[0] % utd_ratio == 0
                batch_size = x.shape[0] // utd_ratio
                return x[batch_size * i : batch_size * (i + 1)]

            mini_batch = jax.tree_util.tree_map(slice, batch)
            new_agent, critic_info = new_agent.update_critic(mini_batch)

        new_agent, actor_info = new_agent.update_actor(mini_batch)
        new_agent, temp_info = new_agent.update_temperature(actor_info["entropy"])

        return new_agent, {**actor_info, **critic_info, **temp_info}


class REDQLearner(core.Learner):
    def __init__(
        self,
        networks,
        key,
        observation_spec: specs.Array,
        action_spec: specs.Array,
        iterator,
        *,
        utd_ratio: int = 1,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        discount: float = 0.99,
        tau: float = 0.005,
        critic_weight_decay: Optional[float] = None,
        target_entropy: Optional[float] = None,
        init_temperature: float = 1.0,
        backup_entropy: bool = True,
        logger=None,
        counter=None,
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """
        self._learner_core = REDQLearnerCore.create(
            networks,
            key,
            observation_spec=observation_spec,
            action_spec=action_spec,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            temp_lr=temp_lr,
            discount=discount,
            tau=tau,
            critic_weight_decay=critic_weight_decay,
            target_entropy=target_entropy,
            init_temperature=init_temperature,
            backup_entropy=backup_entropy,
        )
        self._utd_ratio = utd_ratio
        self._iterator = iterator
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger("learner", save_data=False)

    def step(self):
        samples = next(self._iterator)
        batch = types.Transition(*samples.data)
        self._learner_core, info = self._learner_core.update(
            batch, utd_ratio=self._utd_ratio
        )
        counts = self._counter.increment(steps=1)
        self._logger.write({**info, **counts})

    def save(self):
        return self._learner_core

    def load(self, state):
        self._learner_core = state

    def get_variables(self, names):
        variables = {
            "policy": self._learner_core.actor.params,
            "critic": self._learner_core.critic.params,
        }
        return [variables[k] for k in names]
