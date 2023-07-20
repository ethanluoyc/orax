import absl
import acme
import gym
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import tensorflow_datasets as tfds
import tree
from acme import wrappers
from acme.jax import variable_utils
from acme.utils import counting

import wandb
from orax.agents import decision_transformer
from orax.agents.decision_transformer import dataset as dataset_lib
from orax.baselines.d4rl import d4rl_evaluation


def get_config():
    config = ml_collections.ConfigDict()
    config.dataset = "medium"
    config.env = "hopper"
    config.mode = "normal"
    config.K = 20
    config.pct_traj = 1.0
    config.batch_size = 64
    config.network_config = dict(
        hidden_size=128,
        num_layers=3,
        num_heads=1,
        dropout_rate=0.1,
    )
    config.learning_rate = 1e-4
    config.weight_decay = 1e-4
    config.warmup_steps = 10000
    config.num_eval_episodes = 10
    config.num_steps = int(1e5)
    config.eval_every = int(1e4)
    # config.log_every = 500
    # config.log_to_wandb = False
    return config


def exclude_bias_and_normalizers(params):
    def predicate(path, value: jnp.ndarray) -> jnp.ndarray:
        del value
        return path[-1] == "b" or "norm" in path[-2] or path[-1] == "embeddings"

    return tree.map_structure_with_path(predicate, params)


def make_environment(name, seed=None):
    env = gym.make(name)
    env.reset(seed=seed)
    env = wrappers.GymWrapper(env)
    return wrappers.SinglePrecisionWrapper(env)


def main(_):
    np.random.seed(0)
    config = get_config()
    log_to_wandb = config.get("log_to_wandb", False)

    if log_to_wandb:
        wandb.init(project="dt_jax", entity="ethanluoyc")

    env_name = config["env"]

    if env_name == "hopper":
        env = make_environment("Hopper-v3")
        max_ep_len = 1000
        target_return = 3600  # evaluation conditioning targets
        scale = 1000.0  # normalization for rewards/returns
    elif env_name == "halfcheetah":
        env = make_environment("HalfCheetah-v3")
        max_ep_len = 1000
        target_return = 12000
        scale = 1000.0
    elif env_name == "walker2d":
        env = make_environment("Walker2d-v3")
        max_ep_len = 1000
        target_return = 5000
        scale = 1000.0
    else:
        raise NotImplementedError

    env.seed(0)

    K = config.K
    batch_size = config.batch_size
    num_eval_episodes = config.num_eval_episodes
    max_num_learner_steps = config.num_steps
    eval_every = config.eval_every
    dataset_name = f"{env_name}-{config['dataset']}-v2"
    episode_dataset = tfds.load(
        f"d4rl_mujoco_{env_name}/v2-{config['dataset']}", split="train"
    )
    observation_mean_std = dataset_lib.get_observation_mean_std(episode_dataset)
    dataset_iterator = (
        dataset_lib.transform_decision_transformer_input(
            episode_dataset,
            sequence_length=K,
            scale=scale,
            observation_mean_std=observation_mean_std,
        )
        .shuffle(int(1e6))
        .repeat()
        .batch(batch_size)
        .as_numpy_iterator()
    )

    spec = acme.make_environment_spec(env)

    forward = decision_transformer.make_gym_networks(
        spec=spec, episode_length=max_ep_len, **config.network_config
    )

    forward_fn = forward.apply
    key = jax.random.PRNGKey(0)

    warmup_steps = config.warmup_steps
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.25),
        optax.adamw(
            learning_rate=optax.warmup_cosine_decay_schedule(
                0,
                config.learning_rate,
                warmup_steps=warmup_steps,
                decay_steps=max_num_learner_steps,
            ),
            weight_decay=config.weight_decay,
            mask=exclude_bias_and_normalizers,
        ),
    )

    key = jax.random.PRNGKey(0)
    parent_counter = counting.Counter(time_delta=0.0)
    learner = decision_transformer.DecisionTransformerLearner(
        model=forward,
        key=key,
        dataset=dataset_iterator,
        optimizer=optimizer,
        counter=counting.Counter(parent_counter, "learner", time_delta=0.0),
    )

    eval_actor = decision_transformer.DecisionTransformerActor(
        spec,
        random_key=key,
        forward_fn=forward_fn,
        context_length=K,
        target_return=target_return,
        return_scale=scale,
        variable_client=variable_utils.VariableClient(
            learner, "model", device=jax.devices()[0]
        ),
        observation_mean_std=observation_mean_std,
        mode="normal",
    )

    eval_loop = acme.EnvironmentLoop(
        environment=env,
        actor=eval_actor,
        counter=counting.Counter(parent_counter, "eval_loop", time_delta=0.0),
        # Update manually at the beginning of eval is faster.
        should_update=False,
        observers=(d4rl_evaluation.D4RLScoreObserver(dataset_name),),
    )

    # Run the environment loop.
    steps = 0
    while steps < max_num_learner_steps:
        for _ in range(eval_every):
            learner.step()
        steps += eval_every
        eval_actor.update(wait=True)
        eval_loop.run(num_episodes=num_eval_episodes)


if __name__ == "__main__":
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    absl.app.run(main)
