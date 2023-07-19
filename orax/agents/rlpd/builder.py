"""SAC Builder."""
import dataclasses
from typing import Any, Callable, Iterator, List, Optional

import acme
import jax
import numpy as np
import reverb
from acme import adders
from acme import core
from acme import specs
from acme import types
from acme.adders import reverb as adders_reverb
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import actors
from acme.agents.jax import builders
from acme.datasets import reverb as datasets
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.jax import variable_utils
from acme.utils import counting
from acme.utils import loggers
from reverb import rate_limiters

from orax.agents.rlpd import learning
from orax.agents.rlpd import networks as rlpd_networks

SACNetworks = Any


@dataclasses.dataclass
class REDQConfig:
    """Configuration options for SAC."""

    # Loss options
    batch_size: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    temp_lr: float = 3e-4

    critic_weight_decay: Optional[float] = None

    init_temperature: float = 1.0
    backup_entropy: bool = True
    discount: float = 0.99
    n_step: int = 1
    target_entropy: Optional[float] = None
    # Target smoothing coefficient.
    tau: float = 0.005

    # Replay options
    min_replay_size: int = 10000
    max_replay_size: int = 1000000
    replay_table_name: str = adders_reverb.DEFAULT_PRIORITY_TABLE
    prefetch_size: int = 4
    samples_per_insert: float = 256
    # Rate to be used for the SampleToInsertRatio rate limitter tolerance.
    # See a formula in make_replay_tables for more details.
    samples_per_insert_tolerance_rate: float = 0.1

    offline_ratio: float = 0.5

    # How many gradient updates to perform per step.
    num_sgd_steps_per_step: int = 1


def _generate_rlpd_samples(
    demonstration_iterator: Iterator[types.Transition],
    replay_iterator: Iterator[reverb.ReplaySample],
) -> Iterator[reverb.ReplaySample]:
    """Generator which creates the sample iterator for RLPD.

    Args:
      demonstration_iterator: Iterator of demonstrations.
      replay_iterator: Replay buffer sample iterator.

    Yields:
      Samples having a mix of offline and online data.
    """

    def combine(one, other):
        def combine_leaf(a, b):
            tmp = np.empty((a.shape[0] + b.shape[0], *a.shape[1:]), dtype=a.dtype)
            tmp[0::2] = a
            tmp[1::2] = b
            return tmp

        return jax.tree_map(combine_leaf, one, other)

    for demonstrations, replay_sample in zip(demonstration_iterator, replay_iterator):
        replay_transitions = types.Transition(*replay_sample.data)
        combined = combine(demonstrations, replay_transitions)
        yield reverb.ReplaySample(data=combined, info=replay_sample.info)


class REDQBuilder(builders.ActorLearnerBuilder):
    """REDQ Builder."""

    def __init__(
        self,
        config: REDQConfig,
        make_demonstrations: Callable[[int], Iterator[types.Transition]],
    ):
        """Creates a SAC learner, a behavior policy and an eval actor.

        Args:
          config: a config with SAC hps
        """
        self._config = config
        self._make_demonstrations = make_demonstrations

    def make_learner(
        self,
        random_key: networks_lib.PRNGKey,
        networks: SACNetworks,
        dataset: Iterator[reverb.ReplaySample],
        logger_fn: loggers.LoggerFactory,
        environment_spec: specs.EnvironmentSpec,
        replay_client: Optional[reverb.Client] = None,
        counter: Optional[counting.Counter] = None,
    ) -> core.Learner:
        del replay_client

        return learning.REDQLearner(
            networks,
            key=random_key,
            observation_spec=environment_spec.observations,
            action_spec=environment_spec.actions,
            iterator=dataset,
            utd_ratio=self._config.num_sgd_steps_per_step,
            actor_lr=self._config.actor_lr,
            critic_lr=self._config.critic_lr,
            temp_lr=self._config.temp_lr,
            discount=self._config.discount,
            tau=self._config.tau,
            critic_weight_decay=self._config.critic_weight_decay,
            target_entropy=self._config.target_entropy,
            init_temperature=self._config.init_temperature,
            backup_entropy=self._config.backup_entropy,
            logger=logger_fn("learner"),
            counter=counter,
        )

    def make_actor(
        self,
        random_key: networks_lib.PRNGKey,
        policy: actor_core_lib.FeedForwardPolicy,
        environment_spec: specs.EnvironmentSpec,
        variable_source: Optional[core.VariableSource] = None,
        adder: Optional[adders.Adder] = None,
    ) -> acme.Actor:
        del environment_spec
        assert variable_source is not None
        actor_core = actor_core_lib.batched_feed_forward_to_actor_core(policy)
        variable_client = variable_utils.VariableClient(
            variable_source, "policy", device="cpu"
        )
        return actors.GenericActor(
            actor_core, random_key, variable_client, adder, backend="cpu"
        )

    def make_replay_tables(
        self,
        environment_spec: specs.EnvironmentSpec,
        policy: actor_core_lib.FeedForwardPolicy,
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
        return [
            reverb.Table(
                name=self._config.replay_table_name,
                sampler=reverb.selectors.Uniform(),
                remover=reverb.selectors.Fifo(),
                max_size=self._config.max_replay_size,
                rate_limiter=limiter,
                signature=adders_reverb.NStepTransitionAdder.signature(
                    environment_spec
                ),
            )
        ]

    def make_dataset_iterator(
        self, replay_client: reverb.Client
    ) -> Iterator[reverb.ReplaySample]:
        """Create a dataset iterator to use for learning/updating the agent."""
        total_batch_size = self._config.batch_size * self._config.num_sgd_steps_per_step

        offline_batch_size = int(total_batch_size * self._config.offline_ratio)
        online_batch_size = total_batch_size - offline_batch_size

        replay_iterator = datasets.make_reverb_dataset(
            table=self._config.replay_table_name,
            server_address=replay_client.server_address,
            batch_size=online_batch_size,
            prefetch_size=self._config.prefetch_size,
        ).as_numpy_iterator()

        offline_iterator = self._make_demonstrations(offline_batch_size)
        iterator = _generate_rlpd_samples(offline_iterator, replay_iterator)
        return utils.device_put(iterator, jax.devices()[0])

    def make_adder(
        self,
        replay_client: reverb.Client,
        environment_spec: Optional[specs.EnvironmentSpec],
        policy: Optional[actor_core_lib.FeedForwardPolicy],
    ) -> Optional[adders.Adder]:
        """Create an adder which records data generated by the actor/environment."""
        del environment_spec, policy
        return adders_reverb.NStepTransitionAdder(
            priority_fns={self._config.replay_table_name: None},
            client=replay_client,
            n_step=self._config.n_step,
            discount=self._config.discount,
        )

    def make_policy(
        self,
        networks: SACNetworks,
        environment_spec: specs.EnvironmentSpec,
        evaluation: bool = False,
    ) -> actor_core_lib.FeedForwardPolicy:
        """Construct the policy."""
        del environment_spec

        return rlpd_networks.apply_policy_and_sample(networks, evaluation)
