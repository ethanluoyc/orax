"""DrQ-v2 builder"""
from typing import Iterator, List, Optional

import jax
import optax
import reverb
from acme import adders
from acme import core
from acme import datasets
from acme import specs
from acme.adders import reverb as adders_reverb
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import actors
from acme.agents.jax import builders
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.jax import variable_utils
from acme.utils import counting
from acme.utils import loggers
from reverb import rate_limiters

from orax.agents.drq_v2 import config as drq_v2_config
from orax.agents.drq_v2 import learning as learning_lib
from orax.agents.drq_v2 import networks as drq_v2_networks


class DrQV2Builder(builders.ActorLearnerBuilder):
    """DrQ-v2 Builder."""

    def __init__(self, config: drq_v2_config.DrQV2Config):
        self._config = config

    def make_replay_tables(
        self,
        environment_spec: specs.EnvironmentSpec,
        policy: drq_v2_networks.DrQV2PolicyNetwork,
    ) -> List[reverb.Table]:
        """Create tables to insert data into."""
        del policy
        samples_per_insert_tolerance = (
            self._config.samples_per_insert_tolerance_rate
            * self._config.samples_per_insert
        )
        error_buffer = self._config.min_replay_size * samples_per_insert_tolerance
        limiter = rate_limiters.SampleToInsertRatio(
            min_size_to_sample=self._config.min_replay_size,
            samples_per_insert=self._config.samples_per_insert,
            error_buffer=error_buffer,
        )
        replay_table = reverb.Table(
            name=self._config.replay_table_name,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=self._config.max_replay_size,
            rate_limiter=limiter,
            signature=adders_reverb.NStepTransitionAdder.signature(
                environment_spec=environment_spec
            ),
        )
        return [replay_table]

    def make_dataset_iterator(
        self, replay_client: reverb.Client
    ) -> Iterator[reverb.ReplaySample]:
        """Create a dataset iterator to use for learning/updating the agent."""
        dataset = datasets.make_reverb_dataset(
            table=self._config.replay_table_name,
            server_address=replay_client.server_address,
            batch_size=(self._config.batch_size * self._config.num_sgd_steps_per_step),
            num_parallel_calls=max(12, 4 * jax.local_device_count()),
        )
        iterator = utils.device_put(dataset.as_numpy_iterator(), jax.devices()[0])
        return iterator

    def make_adder(
        self,
        replay_client: reverb.Client,
        environment_spec: specs.EnvironmentSpec,
        policy: builders.Policy,
    ) -> Optional[adders.Adder]:
        return adders_reverb.NStepTransitionAdder(
            client=replay_client,
            n_step=self._config.n_step,
            discount=self._config.discount,
        )

    def make_actor(
        self,
        random_key: networks_lib.PRNGKey,
        policy: drq_v2_networks.DrQV2PolicyNetwork,
        environment_spec: specs.EnvironmentSpec,
        variable_source: Optional[core.VariableSource] = None,
        adder: Optional[adders.Adder] = None,
    ) -> actors.GenericActor:
        del environment_spec
        assert variable_source is not None
        device = "cpu"
        variable_client = variable_utils.VariableClient(
            variable_source,
            "policy",
            device=device,
            update_period=self._config.variable_update_period,
        )

        actor_core = actor_core_lib.batched_feed_forward_to_actor_core(policy)
        return actors.GenericActor(
            actor_core, random_key, variable_client, adder, backend=device
        )

    def make_learner(
        self,
        random_key: networks_lib.PRNGKey,
        networks: drq_v2_networks.DrQV2Networks,
        dataset: Iterator[reverb.ReplaySample],
        logger_fn: loggers.LoggerFactory,
        environment_spec: specs.EnvironmentSpec,
        replay_client: Optional[reverb.Client] = None,
        counter: Optional[counting.Counter] = None,
    ) -> learning_lib.DrQV2Learner:
        del replay_client, environment_spec
        config = self._config
        critic_optimizer = optax.adam(config.learning_rate)
        policy_optimizer = optax.adam(config.learning_rate)
        encoder_optimizer = optax.adam(config.learning_rate)
        if self._config.grad_norm_clip is not None:
            critic_optimizer = optax.chain(
                optax.clip_by_global_norm(self._config.grad_norm_clip), critic_optimizer
            )
            policy_optimizer = optax.chain(
                optax.clip_by_global_norm(self._config.grad_norm_clip), policy_optimizer
            )
            encoder_optimizer = optax.chain(
                optax.clip_by_global_norm(self._config.grad_norm_clip),
                encoder_optimizer,
            )

        return learning_lib.DrQV2Learner(
            random_key=random_key,
            dataset=dataset,
            networks=networks,
            sigma=config.sigma,
            policy_optimizer=policy_optimizer,
            critic_optimizer=critic_optimizer,
            encoder_optimizer=encoder_optimizer,
            augmentation=config.augmentation,
            critic_soft_update_rate=config.critic_q_soft_update_rate,
            discount=config.discount,
            noise_clip=config.noise_clip,
            num_sgd_steps_per_step=config.num_sgd_steps_per_step,
            bc_alpha=config.bc_alpha,
            logger=logger_fn("learner"),
            counter=counter,
        )

    def make_policy(
        self,
        networks,
        environment_spec: specs.EnvironmentSpec,
        evaluation: bool = False,
    ) -> actor_core_lib.FeedForwardPolicy:
        """Construct the policy."""
        sigma = 0 if evaluation else self._config.sigma
        return drq_v2_networks.apply_policy_and_sample(
            networks, environment_spec.actions, sigma
        )
