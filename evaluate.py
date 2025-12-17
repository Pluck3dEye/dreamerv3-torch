#!/usr/bin/env python3
"""
Standalone evaluation script for DreamerV3 trained models.
Loads a checkpoint and runs evaluation episodes with visualization.
"""

import argparse
import functools
import pathlib
import sys
import torch
import numpy as np
from collections import defaultdict
import ruamel.yaml as yaml

import tools


def load_config(config_names):
    """Load config from configs.yaml using the same method as dreamer.py."""
    configs = yaml.safe_load(
        (pathlib.Path(__file__).parent / "configs.yaml").read_text()
    )
    
    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value
    
    name_list = ["defaults", *config_names] if config_names else ["defaults"]
    defaults = {}
    for name in name_list:
        if name in configs:
            recursive_update(defaults, configs[name])
    
    # Convert to argparse namespace (same as dreamer.py)
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    
    config = parser.parse_args([])
    return config


def make_env(config):
    """Create environment based on task."""
    task = config.task
    
    if task.startswith("highway_"):
        from envs.highway import HighwayEnv
        env_name = task.split("_", 1)[1]  # e.g., "highway" from "highway_highway"
        
        env = HighwayEnv(
            name=env_name,
            size=tuple(config.size) if hasattr(config, 'size') else (64, 64),
            obs_type=getattr(config, 'highway_obs_type', 'image'),
            action_type=getattr(config, 'highway_action_type', 'discrete'),
            action_repeat=config.action_repeat,
            vehicles_count=getattr(config, 'highway_vehicles_count', 50),
            vehicles_density=getattr(config, 'highway_vehicles_density', 1.5),
            use_reward_shaping=getattr(config, 'highway_reward_shaping', True),
        )
        return env
    else:
        raise ValueError(f"Unknown task: {task}")


class DreamerAgent:
    """Wrapper for Dreamer agent for evaluation."""
    
    def __init__(self, agent, config):
        self.agent = agent
        self.config = config
        self._state = None
        
    def reset(self):
        self._state = None
        
    def __call__(self, obs):
        """Get action from observation."""
        with torch.no_grad():
            # Prepare observation
            obs_dict = {}
            for key, val in obs.items():
                if isinstance(val, np.ndarray):
                    tensor = torch.tensor(val, dtype=torch.float32)
                    if tensor.dtype == torch.float32 and val.dtype == np.uint8:
                        tensor = torch.tensor(val, dtype=torch.uint8)
                    obs_dict[key] = tensor.unsqueeze(0).to(self.config.device)
                else:
                    obs_dict[key] = torch.tensor([[val]], dtype=torch.float32).to(self.config.device)
            
            # Add required fields
            if "is_first" not in obs_dict:
                is_first = 1.0 if self._state is None else 0.0
                obs_dict["is_first"] = torch.tensor([[is_first]], dtype=torch.float32).to(self.config.device)
            
            # Call agent
            action_dict, self._state = self.agent(obs_dict, self._state, training=False)
            
            # Extract action
            action = action_dict["action"].squeeze(0).cpu().numpy()
            
            return action


class VQVAEAgentWrapper:
    """Wrapper for VQ-VAE agent for evaluation."""
    
    def __init__(self, agent, config):
        self.agent = agent
        self.config = config
        self._state = None
        
    def reset(self):
        self._state = None
        
    def __call__(self, obs):
        """Get action from observation."""
        with torch.no_grad():
            # Prepare observation
            obs_dict = {}
            for key, val in obs.items():
                if isinstance(val, np.ndarray):
                    tensor = torch.tensor(val, dtype=torch.float32)
                    if tensor.dtype == torch.float32 and val.dtype == np.uint8:
                        tensor = torch.tensor(val, dtype=torch.uint8)
                    obs_dict[key] = tensor.unsqueeze(0).to(self.config.device)
                else:
                    obs_dict[key] = torch.tensor([[val]], dtype=torch.float32).to(self.config.device)
            
            # Add required fields
            if "is_first" not in obs_dict:
                is_first = 1.0 if self._state is None else 0.0
                obs_dict["is_first"] = torch.tensor([[is_first]], dtype=torch.float32).to(self.config.device)
            
            # Call agent
            action_dict, self._state = self.agent(obs_dict, self._state, training=False)
            
            # Extract action
            action = action_dict["action"].squeeze(0).cpu().numpy()
            
            return action
            
            # Call agent
            action_dict, self._state = self.agent(obs_dict, self._state, training=False)
            
            # Extract action
            action = action_dict["action"].squeeze(0).cpu().numpy()
            
            return action


def evaluate(config, agent, env, episodes=5, render=True, w_drive=(0.5, 0.5, 1.0)):
    """
    Evaluate agent with detailed metrics per episode.

    Metrics:
        collision_rate : Fraction of episodes with any collision
        offroad_rate   : Fraction of steps off the road
        success        : Goal reached without collisions
        lateral_deviation : Avg lateral deviation from lane center
        avg_reward     : Total episode reward
        minADE         : Mean error between predicted & true positions
        driving_score  : Weighted score (collision/offroad/comfort)
    """
    results = defaultdict(list)
    minade_counts = []  # track whether minADE was available per episode

    for ep in range(episodes):
        obs, info = env.reset()
        agent.reset()
        done = False
        total_reward = 0
        steps = 0

        # Episode tracking variables
        collision_frames = 0
        offroad_step = 0
        lateral_devs = []
        route_dist_traveled = 0
        route_total_length = info.get("route_length", 1.0)
        predicted_positions = []
        true_positions = []

        print(f"\nEpisode {ep + 1}/{episodes}")

        while not done:
            # Get action from agent
            action = agent(obs)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

            # Print progress every 50 steps
            if steps % 50 == 0:
                print(f"  Step {steps}: reward={total_reward:.2f}")
                
            # --- Step-level metrics ---
            collision_frames += int(info.get("crashed", False))
            offroad_step += info.get("off_road", 0)
            ego_pos = info.get("ego_position", None)
            lane_center = info.get("lane_center", None)
            signed_lat = info.get("lateral_offset_signed", None)
            # Prefer precomputed lateral offsets; fall back to center distance if absent.
            if signed_lat is not None:
                lateral_devs.append(abs(float(signed_lat)))
            elif "lateral_offset_abs" in info:
                lateral_devs.append(float(info["lateral_offset_abs"]))
            elif ego_pos is not None and lane_center is not None:
                lateral_devs.append(np.linalg.norm(np.array(ego_pos) - np.array(lane_center)))

            route_dist_traveled += info.get("route_progress", 0)

            pred_pos = info.get("predicted_position", None)
            true_pos = info.get("true_future_position", None)
            if pred_pos is not None and true_pos is not None:
                predicted_positions.append(np.array(pred_pos))
                true_positions.append(np.array(true_pos))

        # --- Episode metrics ---
        # Collision ends the episode; per-episode flag used for overall rate (#collision episodes / total).
        collision_rate = 1.0 if collision_frames > 0 else 0.0
        offroad_rate = offroad_step / max(steps, 1)
        print(f"  offroad_step: {offroad_step}")
        
        success_mode = getattr(config, "highway_success_mode", "goal_flag")
        if success_mode == "no_collision_episode":
            success = int((collision_frames == 0) and (offroad_step == 0))
        else:
            success = int((collision_frames == 0) and info.get("goal_reached", False))
        
        lateral_deviation = np.mean(lateral_devs) if lateral_devs else 0.0
        avg_reward = total_reward

        # minADE computation
        if predicted_positions and true_positions:
            pred = np.stack(predicted_positions)
            true = np.stack(true_positions)
            minADE = np.mean(np.linalg.norm(pred - true, axis=1))
            minade_counts.append(True)
        else:
            minADE = np.nan
            minade_counts.append(False)

        # Driving score (weighted combination)
        # Comfort index may be absent; default to 0.0 to avoid None math.
        comfort_index = float(info.get("comfort_index", 0.0) or 0.0)

        # Convert to discomfort
        discomfort = -np.log(max(comfort_index, 1e-6))

        # Map back to bounded goodness
        comfort_term = 1.0 / (1.0 + discomfort)

        driving_score = w_drive[0] * (1 - collision_rate) + w_drive[1] * (1 - offroad_rate) + w_drive[2] * comfort_term

        # Save results
        results["collision_rate"].append(collision_rate)
        results["offroad_rate"].append(offroad_rate)
        results["success"].append(success)
        results["lateral_deviation"].append(lateral_deviation)
        results["avg_reward"].append(avg_reward)
        results["minADE"].append(minADE)
        results["driving_score"].append(driving_score)
        results["steps"].append(steps)

        crashed = info.get("crashed", False)
        status = "CRASHED" if crashed else "SURVIVED"
        print(f"  Done: {steps} steps, reward={total_reward:.2f}, success={success}, {status}")

    # Aggregate statistics
    summary = {}
    for k, v in results.items():
        arr = [np.nan if x is None else x for x in v]
        if np.all(np.isnan(arr)):
            summary[k] = (np.nan, np.nan)
        else:
            summary[k] = (np.nanmean(arr), np.nanstd(arr))

    summary["minADE_available_episodes"] = (int(sum(minade_counts)), len(minade_counts))

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION METRICS SUMMARY")
    print("=" * 50)
    for k, v in summary.items():
        if k == "minADE_available_episodes":
            have, total = v
            print(f"minADE coverage: {have}/{total} episodes with data")
            continue
        mean, std = v
        if np.isnan(mean):
            print(f"{k}: n/a (no data)")
        else:
            print(f"{k}: {mean:.3f} ± {std:.3f}")
    print("=" * 50)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model (Dreamer or VQ-VAE)")
    parser.add_argument("--logdir", type=str, required=True, help="Path to training logdir")
    parser.add_argument("--config", type=str, default="highway", help="Config name from configs.yaml")
    parser.add_argument("--world_model", type=str, default=None, 
                        help="World model type: 'dreamer' or 'vqvae' (auto-detected from config if not specified)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    args = parser.parse_args()
    
    logdir = pathlib.Path(args.logdir)
    print(f"Loading from: {logdir}")
    
    # Load config from yaml (same method as dreamer.py)
    config = load_config([args.config])
    config.device = args.device
    config.logdir = logdir
    
    # Determine world model type
    world_model_type = args.world_model or getattr(config, 'world_model', 'dreamer')
    print(f"World model: {world_model_type}")
    print(f"Task: {config.task}")
    print(f"Device: {config.device}")
    
    # Create environment
    env = make_env(config)
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Get action count
    acts = env.action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    
    # Create dummy logger
    class DummyLogger:
        def __init__(self):
            self.step = 0
        def scalar(self, *args, **kwargs): pass
        def image(self, *args, **kwargs): pass
        def video(self, *args, **kwargs): pass
        def write(self, *args, **kwargs): pass
    
    logger = DummyLogger()
    
    # Create agent based on world model type
    if world_model_type == 'vqvae':
        from vqvae_agent import VQVAEAgent
        agent = VQVAEAgent(
            env.observation_space,
            env.action_space,
            config,
            logger,
            None,  # dataset not needed for eval
        ).to(config.device)
        wrapper_class = VQVAEAgentWrapper
    else:
        from dreamer import Dreamer
        agent = Dreamer(
            env.observation_space,
            env.action_space,
            config,
            logger,
            None,  # dataset not needed for eval
        ).to(config.device)
        wrapper_class = DreamerAgent
    
    # Load checkpoint
    ckpt_path = logdir / "latest.pt"
    if not ckpt_path.exists():
        ckpt_path = logdir / "checkpoint.pt"
    
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=config.device, weights_only=False)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    agent.eval()
    
    print("Model loaded successfully!")
    
    # Wrap agent
    eval_agent = wrapper_class(agent, config)
    
    # Run evaluation
    results = evaluate(
        config=config,
        agent=eval_agent,
        env=env,
        episodes=args.episodes,
        render=True,
    )
    
    env.close()


if __name__ == "__main__":
    main()
