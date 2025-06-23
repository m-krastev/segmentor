import torch
import numpy as np
from pathlib import Path
from itertools import cycle
from torch.utils.data import DataLoader
from typing import Optional

from .environment import SmallBowelEnv, make_sb_env
from .config import parse_args
from .dataset import SmallBowelDataset

def get_first(x):
    return x[0]

def run_perfect_episode(
    env: SmallBowelEnv,
    save_dir: Optional[Path] = None,
    num_episodes: int = 1,
):
    """
    Runs a perfect episode using the sample_perfect_action method.
    """
    print(f"Running {num_episodes} perfect episodes...")
    all_final_rewards = []
    all_final_coverages = []
    all_final_steps = []

    for i in range(num_episodes):
        print(f"\n--- Starting Perfect Episode {i+1}/{num_episodes} ---")
        tensordict = env.reset()
        done = tensordict["done"].item()
        total_reward = 0.0
        step_count = 0

        while not done:
            # Get the perfect action from the environment
            perfect_action = env.sample_perfect_action()
            
            # Create a tensordict for the step method
            action_td = tensordict.select("actor").set("action", perfect_action.unsqueeze(0))
            
            # Perform the step
            tensordict = env.step(action_td)
            
            reward = tensordict["reward"].item()
            done = tensordict["done"].item()
            total_reward += reward
            step_count += 1

            current_pos = env.current_pos_vox
            goal_pos = env.goal
            dist_to_goal = np.linalg.norm(np.array(current_pos) - np.array(goal_pos))

            print(f"Step {step_count:03d}: Current Pos: {current_pos}, Dist to Goal: {dist_to_goal:.2f}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")

        final_coverage = tensordict["info"]["final_coverage"].item()
        final_step_count = tensordict["info"]["final_step_count"].item()
        final_total_reward = tensordict["info"]["total_reward"].item()

        print(f"--- Episode {i+1} Finished ---")
        print(f"Final Total Reward: {final_total_reward:.2f}")
        print(f"Final Coverage: {final_coverage:.3f}")
        print(f"Final Step Count: {final_step_count}")

        all_final_rewards.append(final_total_reward)
        all_final_coverages.append(final_coverage)
        all_final_steps.append(final_step_count)

        if save_dir:
            episode_save_dir = save_dir / f"episode_{i+1}"
            episode_save_dir.mkdir(parents=True, exist_ok=True)
            env.save_path(episode_save_dir)
            print(f"Path visualization saved to {episode_save_dir}")

    avg_reward = np.mean(all_final_rewards)
    avg_coverage = np.mean(all_final_coverages)
    avg_steps = np.mean(all_final_steps)

    print(f"\n--- Perfect Run Test Summary ({num_episodes} episodes) ---")
    print(f"Average Final Reward: {avg_reward:.2f}")
    print(f"Average Final Coverage: {avg_coverage:.3f}")
    print(f"Average Final Steps: {avg_steps:.2f}")

    return {
        "avg_reward": avg_reward,
        "avg_coverage": avg_coverage,
        "avg_steps": avg_steps,
        "all_rewards": all_final_rewards,
        "all_coverages": all_final_coverages,
        "all_steps": all_final_steps,
    }

if __name__ == "__main__":
    # Parse arguments from command line or use defaults
    config = parse_args()
    
    # Initialize dataset
    dataset = SmallBowelDataset(data_dir=config.data_dir, config=config)
    env = make_sb_env(dataset, config=config)

    results_dir = Path("perfect_run_results")
    results_dir.mkdir(exist_ok=True)

    run_perfect_episode(env, save_dir=results_dir, num_episodes=len(dataset))
