"""Tests for the RLPD agent."""

import jax
from absl.testing import absltest
from acme import specs
from acme.testing import fakes
from acme.utils import loggers

from orax.agents import rlpd


class RLPDTest(absltest.TestCase):
    def test_train(self):
        seed = 0
        num_iterations = 2
        batch_size = 64

        # Create a fake environment to test with.
        environment = fakes.ContinuousEnvironment(
            episode_length=10, bounded=True, action_dim=6
        )
        spec = specs.make_environment_spec(environment)

        # Construct the agent.
        networks = rlpd.make_networks(
            spec,
            hidden_dims=(8, 8),
        )
        dataset = (
            fakes.transition_dataset(environment).batch(batch_size).as_numpy_iterator()
        )
        key = jax.random.PRNGKey(seed)
        learner = rlpd.REDQLearner(
            networks,
            key=key,
            observation_spec=spec.observations,
            action_spec=spec.actions,
            iterator=dataset,
            utd_ratio=1,
            actor_lr=1e-4,
            critic_lr=1e-4,
            temp_lr=1e-4,
            discount=0.99,
            tau=5e-3,
            critic_weight_decay=0.0,
            target_entropy=-1,
            init_temperature=1.0,
            backup_entropy=False,
            logger=loggers.TerminalLogger("learner"),
        )

        # Train the agent
        for _ in range(num_iterations):
            learner.step()


if __name__ == "__main__":
    absltest.main()
