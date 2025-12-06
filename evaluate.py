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


def evaluate(config, agent, env, episodes=5, render=True):
    """Run evaluation episodes."""
    results = defaultdict(list)
    
    for ep in range(episodes):
        obs, info = env.reset()
        agent.reset()
        done = False
        total_reward = 0
        steps = 0
        
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
        
        # Episode finished
        crashed = info.get("crashed", False)
        status = "CRASHED" if crashed else "SURVIVED"
        print(f"  Done: {steps} steps, reward={total_reward:.2f}, {status}")
        
        results["steps"].append(steps)
        results["reward"].append(total_reward)
        results["crashed"].append(crashed)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Episodes: {episodes}")
    print(f"Avg Steps: {np.mean(results['steps']):.1f} ± {np.std(results['steps']):.1f}")
    print(f"Avg Reward: {np.mean(results['reward']):.2f} ± {np.std(results['reward']):.2f}")
    print(f"Crash Rate: {np.mean(results['crashed'])*100:.1f}%")
    print("="*50)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate DreamerV3 model")
    parser.add_argument("--logdir", type=str, required=True, help="Path to training logdir")
    parser.add_argument("--config", type=str, default="highway", help="Config name from configs.yaml")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    args = parser.parse_args()
    
    logdir = pathlib.Path(args.logdir)
    print(f"Loading from: {logdir}")
    
    # Load config from yaml (same method as dreamer.py)
    config = load_config([args.config])
    config.device = args.device
    config.logdir = logdir
    
    print(f"Task: {config.task}")
    print(f"Device: {config.device}")
    
    # Create environment
    env = make_env(config)
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Import Dreamer
    from dreamer import Dreamer
    
    # Get action count
    acts = env.action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    
    # Create agent (need a dummy dataset)
    train_eps = {}
    
    def make_dataset(episodes, config):
        generator = tools.sample_episodes(episodes, config.batch_length)
        dataset = tools.from_generator(generator, config.batch_size)
        return dataset
    
    # Create dummy logger
    class DummyLogger:
        def __init__(self):
            self.step = 0
        def scalar(self, *args, **kwargs): pass
        def image(self, *args, **kwargs): pass
        def video(self, *args, **kwargs): pass
        def write(self, *args, **kwargs): pass
    
    logger = DummyLogger()
    
    # Create Dreamer agent
    agent = Dreamer(
        env.observation_space,
        env.action_space,
        config,
        logger,
        None,  # dataset not needed for eval
    ).to(config.device)
    
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
    eval_agent = DreamerAgent(agent, config)
    
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
