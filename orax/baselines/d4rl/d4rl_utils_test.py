from absl.testing import absltest
from absl.testing import parameterized

from orax.baselines.d4rl import d4rl_utils


class D4RLUtilsTest(parameterized.TestCase):
    @parameterized.parameters(
        ("walker2d-random-v0", "d4rl_mujoco_walker2d/v0-random"),
        ("walker2d-medium-replay-v0", "d4rl_mujoco_walker2d/v0-mixed"),
        ("halfcheetah-random-v0", "d4rl_mujoco_halfcheetah/v0-random"),
        ("halfcheetah-medium-v0", "d4rl_mujoco_halfcheetah/v0-medium"),
        ("halfcheetah-medium-expert-v0", "d4rl_mujoco_halfcheetah/v0-medium-expert"),
        ("antmaze-medium-diverse-v0", "d4rl_antmaze/medium-diverse-v0"),
        ("pen-human-v0", "d4rl_adroit_pen/v0-human"),
    )
    def test_get_tfds_name(self, d4rl_name, expected_tfds_name):
        tfds_name = d4rl_utils.get_tfds_name(d4rl_name)
        self.assertEqual(tfds_name, expected_tfds_name)

    @parameterized.parameters(
        ("walker2d-random-v0", ("walker2d", "random", "v0")),
        ("halfcheetah-medium-v0", ("halfcheetah", "medium", "v0")),
        ("halfcheetah-medium-replay-v0", ("halfcheetah", "medium-replay", "v0")),
    )
    def test_parse_d4rl_dataset_name(self, d4rl_name, expected):
        result = d4rl_utils.parse_d4rl_dataset_name(d4rl_name)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    absltest.main()
