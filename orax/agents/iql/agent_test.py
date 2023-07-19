"""Tests for the IQL agent."""

import jax
import optax
from absl.testing import absltest
from acme import specs
from acme.testing import fakes
from acme.utils import loggers

from orax.agents import iql


class IQLTest(absltest.TestCase):
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
        networks = iql.make_networks(
            spec,
            hidden_dims=(8, 8),
        )
        dataset = fakes.transition_iterator(environment)(batch_size)
        key = jax.random.PRNGKey(seed)
        learner = iql.IQLLearner(
            key,
            networks,
            dataset,
            optax.adam(1e-4),
            optax.adam(1e-4),
            optax.adam(1e-4),
            logger=loggers.make_default_logger("learner", save_data=False),
        )

        # Train the agent
        for _ in range(num_iterations):
            learner.step()


if __name__ == "__main__":
    absltest.main()
