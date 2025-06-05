from functools import partial
from itertools import cycle
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from typing import NamedTuple, Any, Dict, Tuple
from flax.training.train_state import TrainState
import flax.training.checkpoints as checkpoints  # For checkpointing
import distrax
import wandb
import os
from gymnax.gymnax.environments.medical import SmallBowel, SmallBowelParams
from navigator.config import Config, parse_args
from navigator.dataset import SmallBowelDataset
from tqdm import tqdm
from torch.utils.data import Subset


class ConvBlock(nnx.Module):
    """
    A simple convolutional block with Conv3D, GroupNorm, and GELU activation.
    """

    def __init__(
        self,
        rngs,
        in_channels,
        out_channels,
        kernel_size: tuple[int, ...] = (3, 3, 3),
        padding=1,
        num_groups=8,
    ):
        super().__init__()
        self.conv = nnx.Conv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            use_bias=False,
            rngs=rngs,
        )
        self.norm = nnx.GroupNorm(num_groups=num_groups, num_features=out_channels, rngs=rngs)

    def forward(self, x):
        return nnx.gelu(self.norm(self.conv(x)))


class Actor(nnx.Module):
    def __init__(self, rngs, input_channels, eps=1e-5, patch_size=(32, 32, 32)):
        super().__init__()
        # Define the layers of the actor network

        self.conv1 = ConvBlock(
            rngs, input_channels, 16, kernel_size=(3, 3, 3), padding=1, num_groups=8
        )
        self.pool1 = nnx.Conv(
            16, 16, kernel_size=(2, 2, 2), stride=2, padding=0, use_bias=False, rngs=rngs
        )
        self.conv2 = ConvBlock(rngs, 16, 32, kernel_size=(3, 3, 3), padding=1, num_groups=16)
        self.pool2 = nnx.Conv(
            32, 32, kernel_size=(2, 2, 2), stride=2, padding=0, use_bias=False, rngs=rngs
        )
        self.conv3 = ConvBlock(rngs, 32, 64, kernel_size=(3, 3, 3), padding=1, num_groups=32)
        self.pool3 = nnx.Conv(
            64, 64, kernel_size=(2, 2, 2), stride=2, padding=0, use_bias=False, rngs=rngs
        )

        self.linear = nnx.Linear(
            64 * (patch_size[0] // 8) * (patch_size[1] // 8) * (patch_size[2] // 8), 256, rngs=rngs
        )
        self.gn = nnx.GroupNorm(32, 256, rngs=rngs)

        # Output layer for alpha/beta parameters (6 values = 3 dimensions × 2 params)
        self.alpha = nnx.Linear(256, 3, rngs=rngs)
        self.beta = nnx.Linear(256, 3, rngs=rngs)

    def __call__(self, x):
        # Forward pass through the actor network
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        # Flatten and pass through linear layers
        x = x.reshape(x.shape[0], -1)  # Flatten the output
        x = nnx.gelu(self.gn(self.linear(x)))
        # Output alpha and beta parameters
        return self.alpha(x), self.beta(x)

    def get_action_dist(self, x):
        """
        Returns a distrax distribution for the actor's output.
        """
        alpha, beta = self(x)
        return distrax.Independent(
            distrax.Beta(alpha, beta),
            reinterpreted_batch_ndims=1,
        )


class Critic(nnx.Module):
    def __init__(self, rngs, input_channels, patch_size=(32, 32, 32)):
        super().__init__()
        # Define the layers of the critic network
        self.conv1 = ConvBlock(
            rngs, input_channels, 16, kernel_size=(3, 3, 3), padding=1, num_groups=8
        )
        self.pool1 = nnx.Conv(
            16, 16, kernel_size=(2, 2, 2), stride=2, padding=0, use_bias=False, rngs=rngs
        )
        self.conv2 = ConvBlock(rngs, 16, 32, kernel_size=(3, 3, 3), padding=1, num_groups=16)
        self.pool2 = nnx.Conv(
            32, 32, kernel_size=(2, 2, 2), stride=2, padding=0, use_bias=False, rngs=rngs
        )
        self.conv3 = ConvBlock(rngs, 32, 64, kernel_size=(3, 3, 3), padding=1, num_groups=32)
        self.pool3 = nnx.Conv(
            64, 64, kernel_size=(2, 2, 2), stride=2, padding=0, use_bias=False, rngs=rngs
        )

        self.linear = nnx.Linear(
            64 * (patch_size[0] // 8) * (patch_size[1] // 8) * (patch_size[2] // 8), 256, rngs=rngs
        )
        self.gn = nnx.GroupNorm(32, 256, rngs=rngs)

        # Output layer for value prediction
        self.value = nnx.Linear(256, 1, rngs=rngs)

    def __call__(self, x):
        # Forward pass through the critic network
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)

        # Flatten and pass through linear layers
        x = x.reshape(x.shape[0], -1)  # Flatten the output
        x = nnx.gelu(self.gn(self.linear(x)))
        return self.value(x)


class ActorCritic(nnx.Module):
    def __init__(self, rngs, input_channels, patch_size=(32, 32, 32)):
        super().__init__()
        self.actor = Actor(rngs, input_channels, patch_size=patch_size)
        self.critic = Critic(rngs, input_channels, patch_size=patch_size)

    def __call__(self, x):
        dist = self.actor.get_action_dist(x[:3])
        value = self.critic(x[3:])
        return dist, value

    def get_action_dist(self, x):
        return self.actor.get_action_dist(x[:3])


# --- Transition NamedTuple ---
class TransitionJax(NamedTuple):
    obs: jnp.ndarray  # Shape: (3+4, feature_dim (actor + critic))
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    value: jnp.ndarray
    log_prob: jnp.ndarray
    info: Dict


# --- WandB Logging Helper ---
def log_wandb(data: dict, step: int):
    if wandb.run is not None:
        processed_data = {
            k: np.array(v) if isinstance(v, jax.Array) else v for k, v in data.items()
        }
        processed_data = {
            k: v.item() if isinstance(v, np.ndarray) and v.size == 1 else v
            for k, v in processed_data.items()
        }
        wandb.log(processed_data, step=step)
    else:
        print(f"WandB not initialized. Step {step}: {data}")


@jax.jit
def ct_preprocessor(
    image: jnp.ndarray, clip_low: float = -150, clip_high: float = 250
) -> jnp.ndarray:
    """
    Preprocesses a CT image by clipping and normalizing.
    """
    image = jnp.clip(image, clip_low, clip_high)
    image = (image - clip_low) / (clip_high - clip_low)
    return image


@jax.jit
def normalize(image: jnp.ndarray) -> jnp.ndarray:
    mean = jnp.mean(image)
    std = jnp.std(image)
    return (image - mean) / (std + 1e-8)


def dataset_sample_to_envparams(sample: Dict[str, Any], config: Config):
    """
    Converts a dataset sample to SmallBowelParams for the environment.
    """
    seg = jnp.asarray(sample["seg"])
    img = normalize(ct_preprocessor(jnp.asarray(sample["image"])))
    wall_map = ct_preprocessor(jnp.asarray(sample["wall_map"]), 0, 0.1)
    return SmallBowelParams(
        max_steps_in_episode=config.max_episode_steps,
        image=img,
        seg=seg,
        wall_map=wall_map,
        gt_path_vol=jnp.zeros((1, 1, 1)),
        gdt_start=jnp.asarray(sample["gdt_start"]),
        gdt_end=jnp.asarray(sample["gdt_end"]),
        start_coord=jnp.asarray(sample["start_coord"]),
        end_coord=jnp.asarray(sample["end_coord"]),
        seg_volume=seg.sum(),
        image_shape=jnp.asarray(img.shape),
    )


class RolloutWrapperSmallBowel:
    """Wrapper to define batch evaluation for generation parameters."""

    def __init__(
        self,
        model: Actor = None,
        num_env_steps: int | None = None,
        env_params: SmallBowelParams | None = None,
    ):
        self.env = SmallBowel()
        self.env_params = env_params if env_params is not None else self.env.default_params
        self.model = model
        if num_env_steps is None:
            self.num_env_steps = self.env_params.max_steps_in_episode
        else:
            self.num_env_steps = num_env_steps

    @partial(jax.jit, static_argnames=("self",))
    def population_rollout(self, key_eval):
        """Reshape parameter vector and evaluate the generation."""
        # Evaluate population of nets on gymnax task - vmap over key & params
        pop_rollout = jax.vmap(self.batch_rollout, in_axes=(None, 0))
        return pop_rollout(key_eval)

    @partial(jax.jit, static_argnames=("self",))
    def batch_rollout(self, key_eval):
        """Evaluate a generation of networks on RL/Supervised/etc. task."""
        # vmap over different MC fitness evaluations for single network
        batch_rollout = jax.vmap(self.single_rollout, in_axes=(0, None))
        return batch_rollout(key_eval)

    @partial(jax.jit, static_argnames=("self",))
    def single_rollout(self, key_input, deterministic=True):
        """Rollout a pendulum episode with lax.scan."""
        # Reset the environment
        key_reset, key_episode = jax.random.split(key_input)
        obs, state = self.env.reset(key_reset, self.env_params)

        def policy_step(state_input, _):
            """lax.scan compatible step transition in jax env."""
            obs, state, key, cum_reward, valid_mask = state_input
            key, key_step, key_net = jax.random.split(key, 3)
            dist = self.model.get_action_dist(obs[:3])
            action = dist.mode() if deterministic else dist.sample(seed=key_net)
            next_obs, next_state, reward, done, _ = self.env.step(
                key_step, state, action, self.env_params
            )
            new_cum_reward = cum_reward + reward * valid_mask
            new_valid_mask = valid_mask * (1 - done)
            carry = [
                next_obs,
                next_state,
                key,
                _,
                new_cum_reward,
                new_valid_mask,
            ]
            y = [obs, action, reward, next_obs, done]
            return carry, y

        # Scan over episode step loop
        carry_out, scan_out = jax.lax.scan(
            policy_step,
            [
                obs,
                state,
                key_episode,
                jnp.array([0.0]),
                jnp.array([1.0]),
            ],
            (),
            self.num_env_steps,
        )
        # Return the sum of rewards accumulated by agent in episode rollout
        obs, action, reward, next_obs, done = scan_out
        cum_return = carry_out[-2]
        state = carry_out[1]
        return obs, action, reward, next_obs, done, cum_return, state

    @property
    def input_shape(self):
        """Get the shape of the observation."""
        key = jax.random.key(0)
        obs, _ = self.env.reset(key, self.env_params)
        return obs.shape


# --- Validation Loop (JAX) ---
def validation_loop_jax(
    actor: Actor,
    config: Config,
    val_dataset: SmallBowelDataset,
    rng_key: jnp.ndarray,
):
    print("Starting JAX validation...")
    val_results_summary = {"val_reward_sum": [], "val_length": [], "val_coverage": []}

    # This will be populated by the env parameters from the dataset
    rolloutwrapper = RolloutWrapperSmallBowel(actor, num_env_steps=config.max_episode_steps)

    for i, subj in tqdm(enumerate(val_dataset), desc="Validation"):
        current_subject_best_reward = -float("inf")
        best_subject_metrics = {}
        env_params = dataset_sample_to_envparams(subj, config)
        rolloutwrapper.env_params = env_params
        for _ in range(config.get("val_rollouts_per_subject", 10)):
            rng_key, rollout_key = jax.random.split(rng_key, 3)

            obs, action, reward, next_obs, done, cum_return, state = rolloutwrapper.single_rollout(
                rollout_key, deterministic=True
            )

            if cum_return > current_subject_best_reward:
                current_subject_best_reward = cum_return
                best_subject_metrics = {
                    "reward": reward.mean(),
                    "length": state.length,
                    "coverage": cum_return,
                }

        val_results_summary["val_reward_sum"].append(best_subject_metrics["reward"])
        val_results_summary["val_length"].append(best_subject_metrics["length"])
        val_results_summary["val_coverage"].append(best_subject_metrics["coverage"])

    final_metrics = {
        f"validation/avg_{k.split('_')[-1]}": np.mean(np.array(v))
        for k, v in val_results_summary.items()
        if v
    }
    print(f"Validation Results: {final_metrics}")
    return final_metrics, rng_key


# --- GAE Calculation Function ---
def calculate_gae_jax(
    trajectories: TransitionJax,  # Shape: (num_steps, num_envs, ...)
    last_values: jnp.ndarray,  # Shape: (num_envs,) V(s_T)
    gamma: float,
    gae_lambda: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:  # advantages, targets
    """Calculates GAE and value targets."""

    def _calc_gae_step(gae_next_val_carry, transition_step):
        gae, next_val = gae_next_val_carry
        # transition_step has fields like .reward, .done, .value, all shaped (num_envs,)
        delta = (
            transition_step.reward
            + gamma * next_val * (1 - transition_step.done)
            - transition_step.value
        )
        gae = delta + gamma * gae_lambda * (1 - transition_step.done) * gae
        return (gae, transition_step.value), gae

    # Scan over the steps dimension (axis 0 of trajectories)
    _, advantages = jax.lax.scan(
        _calc_gae_step,
        (jnp.zeros_like(last_values), last_values),  # initial (gae, next_val)
        trajectories,  # PyTree of arrays with leading dim num_steps
        reverse=True,
    )
    targets = advantages + trajectories.value  # Q-value estimate using GAE
    return advantages, targets


def save_checkpoint(
    train_state,
    collected_frames_total,
    ppo_update_idx,
    best_val_metric,
    rng_key,
    prefix="checkpoint_jax_",
    keep=3,
    overwrite=False,
):
    save_data = {
        "train_state": train_state,
        "collected_frames_total": collected_frames_total,
        "num_gradient_updates_total": train_state.step,
        "current_ppo_update_idx": ppo_update_idx + 1,
        "best_val_metric": best_val_metric,
        "rng_key": rng_key,
    }
    checkpoints.save_checkpoint(
        ckpt_dir=config.checkpoint_dir,
        target=save_data,
        step=collected_frames_total,
        prefix=prefix,
        keep=keep,
        overwrite=overwrite,
    )


# --- Main Training Function (JAX) ---
def train_jax_ppo(config: Config, train_dataset: SmallBowelDataset, val_dataset: SmallBowelDataset):
    rng_key = jax.random.key(config.seed)
    rngs = nnx.Rngs(rng_key)

    if config.get("track_wandb", False):
        wandb.init(
            project=getattr(config, "wandb_project", "jax_ppo_sb"),
            config=config,
            name=getattr(config, "run_name", "ppo_jax"),
        )

    num_envs = config.num_envs
    env_instance = SmallBowel()
    action_dim = env_instance.action_dim

    model = ActorCritic(
        rngs=rngs,
        input_channels=env_instance.observation_space.shape[0],
        patch_size=env_instance.default_params.image_shape[:3],
    )

    rng_key, model_init_key = jax.random.split(rng_key)

    num_ppo_updates = config.total_timesteps // (config.frames_per_batch * num_envs)
    total_gradient_steps_approx = (
        num_ppo_updates
        * config.update_epochs
        * ((config.frames_per_batch * num_envs) // config.batch_size)
    )

    lr_scheduler_fn = optax.cosine_decay_schedule(
        init_value=config.learning_rate,
        decay_steps=total_gradient_steps_approx,
        alpha=getattr(config, "lr_eta_min", 1e-6) / config.learning_rate,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adamw(learning_rate=lr_scheduler_fn, eps=1e-5),
    )

    train_state = nnx.Optimizer(model, optimizer)

    collected_frames_total, num_gradient_updates_total = 0, 0
    current_ppo_update_idx, best_val_metric = 0, -float("inf")
    if config.get("reload_checkpoint_path"):
        try:
            restored_target = {
                "train_state": train_state,
                "collected_frames_total": 0,
                "num_gradient_updates_total": 0,
                "current_ppo_update_idx": 0,
                "best_val_metric": -float("inf"),
                "rng_key": rng_key,
            }
            ckpt_dir_to_load = config.checkpoint_dir
            loaded_data = checkpoints.restore_checkpoint(
                ckpt_dir=ckpt_dir_to_load,
                target=restored_target,
                prefix="best_model_jax_",
                step=None,
            )
            if not loaded_data:
                loaded_data = checkpoints.restore_checkpoint(
                    ckpt_dir=ckpt_dir_to_load,
                    target=restored_target,
                    prefix="checkpoint_jax_",
                    step=None,
                )
            if loaded_data:
                train_state = loaded_data["train_state"]
                collected_frames_total = int(loaded_data["collected_frames_total"])
                num_gradient_updates_total = int(loaded_data["num_gradient_updates_total"])
                current_ppo_update_idx = int(loaded_data["current_ppo_update_idx"])
                best_val_metric = float(loaded_data["best_val_metric"])
                rng_key = loaded_data["rng_key"]
                print(
                    f"Checkpoint loaded. Resuming from PPO update {current_ppo_update_idx}, {collected_frames_total} frames."
                )
            else:
                print(
                    f"No checkpoint found at {ckpt_dir_to_load} with known prefixes. Starting fresh."
                )
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting fresh.")

    pbar = tqdm(
        total=config.total_timesteps,
        desc="Training (JAX)",
        unit="frames",
        initial=collected_frames_total,
    )

    rollout_wrapper = RolloutWrapperSmallBowel(
        model=model.actor,
        num_env_steps=config.max_episode_steps,
        env_params=env_instance.default_params,
    )

    dataset_iterator = cycle(iter(train_dataset))
    for ppo_update_idx in range(current_ppo_update_idx, num_ppo_updates):
        subj = next(dataset_iterator)
        env_params = dataset_sample_to_envparams(subj, config)
        rollout_wrapper.env_params = env_params
        trajectories = rollout_wrapper.batch_rollout(rng_key)

        # Stack trajectories: list of PyTrees -> PyTree of stacked arrays
        # Each leaf will have shape (frames_per_batch, num_envs, ...)
        batch_trajectories = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *trajectories)
        # --- GAE Calculation ---
        last_values = model.critic(
            batch_trajectories["obs"][3:]
        )  # Use the critic for last value estimation
        advantages, targets = calculate_gae_jax(
            batch_trajectories, last_values, config.gamma, config.gae_lambda
        )

        def _update_epoch(update_state: nnx.Optimizer, unused):

            def _loss_fn(traj, adv, target):
                # Rerun network
                pi, value_pred = update_state.model(traj.obs)
                log_prob_new, entropy = pi.log_prob(traj.action), pi.entropy().mean()

                ratio = jnp.exp(log_prob_new - traj.log_prob)
                loss_actor = -jnp.minimum(
                    ratio * adv,
                    jnp.clip(ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon) * adv,
                ).mean()

                loss_critic = optax.huber_loss(value_pred, target).mean()
                # value_pred_clipped = traj_batch.value + (
                #     value - traj_batch.value
                # ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                # value_losses = jnp.square(value - targets)
                # value_losses_clipped = jnp.square(value_pred_clipped - targets)
                # value_loss = (
                #     0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                # )

                total_loss = (
                    loss_actor + config.vf_coef * loss_critic - config.ent_coef * entropy
                )
                kl_approx = ((ratio - 1) - jnp.log(ratio)).mean()

                return total_loss, (loss_actor, loss_critic, entropy, kl_approx)

            grads, aux = nnx.grad(_loss_fn, has_aux=True)(update_state.model)
            update_state.update(grads)
            return update_state, aux
        # --- PPO Update ---
        train_state, infos = jax.lax.scan(
            _update_epoch,
            train_state,
            (batch_trajectories, advantages, targets),
            length=config.update_epochs,
        )
        avg_losses_dict = {
            "actor": jnp.mean(infos[0]),
            "critic": jnp.mean(infos[1]),
            "entropy": jnp.mean(infos[2]),
            "kl": jnp.mean(infos[3]),
        }
        grad_norm = jnp.linalg.norm(train_state.optimizer.target.gradients)

        # --- Logging ---
        def get_mean_ep_stat(key):
            vals = batch_trajectories.info.get(key)  # Use unflattened for info
            return jnp.nanmean(vals) if vals is not None and vals.size > 0 else jnp.nan

        log_data = {
            "losses/policy_loss": avg_losses_dict["actor"],
            "losses/value_loss": avg_losses_dict["critic"],
            "losses/entropy": avg_losses_dict["entropy"],
            "losses/kl_div": avg_losses_dict["kl"],
            "losses/grad_norm": grad_norm,
            "charts/learning_rate": lr_scheduler_fn(train_state.step)
            if callable(lr_scheduler_fn)
            else lr_scheduler_fn,  # train_state.step is num gradient updates
            "charts/num_gradient_updates": train_state.step,
            "charts/num_ppo_updates": ppo_update_idx + 1,
            "train/reward_mean_rollout": batch_trajectories.reward.mean(),
            "train/ep_reward_mean": get_mean_ep_stat("returned_episode_reward"),
            "train/ep_length_mean": get_mean_ep_stat("returned_episode_length"),
            "train/ep_coverage_mean": get_mean_ep_stat("returned_episode_coverage"),
            "train/ep_wall_gradient_mean": get_mean_ep_stat("wall_gradient"),
            "charts/max_gdt_achieved": get_mean_ep_stat("max_gdt_achieved"),
            "charts/action_mean": trajectories.action.astype(jnp.float32).mean(),
        }
        pbar.set_postfix({
            k.split("/")[-1]: f"{v:.2f}"
            for k, v in log_data.items()
            if isinstance(v, (float, np.ndarray, jax.Array))
            and ("loss" in k or "reward" in k or "coverage" in k)
        })
        if config.get("track_wandb"):
            log_wandb(log_data, step=collected_frames_total)

        # --- Validation and Checkpointing ---
        if (ppo_update_idx + 1) % config["eval_interval"] == 0:
            rng_key, val_rng = jax.random.split(rng_key)
            # Pass only params for validation to avoid TrainState issues if not needed
            val_metrics, rng_key = validation_loop_jax(
                train_state.model.actor, config, val_dataset, val_rng
            )
            if config.get("track_wandb"):
                log_wandb(val_metrics, step=collected_frames_total)

            current_metric_val = val_metrics.get(config["metric_to_optimize"], -float("inf"))
            if current_metric_val > best_val_metric:
                best_val_metric = current_metric_val
                print(
                    f"  New best validation metric ({config['metric_to_optimize']}): {best_val_metric:.4f}"
                )
                save_checkpoint(
                    train_state,
                    collected_frames_total,
                    ppo_update_idx,
                    best_val_metric,
                    rng_key,
                    config,
                    prefix="best_model_jax_",
                    keep=3,
                    overwrite=True,
                )

        if (ppo_update_idx + 1) % config["save_freq"] == 0:
            save_checkpoint(
                train_state,
                collected_frames_total,
                ppo_update_idx,
                best_val_metric,
                rng_key,
                config,
                prefix="checkpoint_jax_",
                keep=3,
            )

        if collected_frames_total >= config["total_timesteps"]:
            break

    pbar.close()
    save_checkpoint(
        train_state,
        collected_frames_total,
        ppo_update_idx,
        best_val_metric,
        rng_key,
        config,
        prefix="final_model_jax_",
        keep=3,
        overwrite=True,
    )

    print(f"Final model saved at frame {collected_frames_total}.")
    if config.get("track_wandb"):
        wandb.finish()


# --- Main Execution Example ---
if __name__ == "__main__":
    config = parse_args()
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    dataset = SmallBowelDataset(config.data_dir, config)
    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset = Subset(dataset, list(range(train_size)))
    val_dataset = Subset(dataset, list(range(train_size, len(dataset))))
    train_jax_ppo(config, train_dataset, val_dataset)
