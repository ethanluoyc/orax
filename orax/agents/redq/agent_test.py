"""Tests for the REDQ agent."""

import acme
import jax
import optax
from absl.testing import absltest
from acme.testing import fakes

from orax.agents import redq


class AgentTest(absltest.TestCase):
    def test_train(self):
        env = fakes.ContinuousEnvironment(bounded=True)
        env_spec = acme.make_environment_spec(env)
        networks = redq.make_networks(
            env_spec, hidden_sizes=(8, 8), num_qs=5, num_min_qs=2
        )
        utd_ratio = 2
        batch_size = 2
        learner_core = redq.REDQLearnerCore(
            networks,
            policy_optimizer=optax.adam(1e-4),
            critic_optimizer=optax.adam(1e-4),
            temperature_optimizer=optax.adam(1e-4),
            target_entropy=redq.target_entropy_from_spec(env_spec.actions),
        )
        iterator = fakes.transition_iterator(env)(batch_size * utd_ratio)
        key = jax.random.PRNGKey(0)
        state = learner_core.init(key)
        state, _ = learner_core.update(state, next(iterator))


if __name__ == "__main__":
    absltest.main()
