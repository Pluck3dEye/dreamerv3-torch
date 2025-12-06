"""
Highway-Env wrappers for DreamerV3.

This module provides gymnasium-compatible wrappers for highway-env autonomous driving
environments, optimized for use with DreamerV3 world model training.

Supported environments:
- highway: Multi-lane highway driving with traffic
- intersection: Navigating through intersections
- merge: Highway merging scenarios
- roundabout: Roundabout navigation
- parking: Parking lot maneuvering
- racetrack: Circuit racing

Usage:
    from envs.highway import HighwayEnv, DEFAULT_REWARD_CONFIGS
    
    # Create environment
    env = HighwayEnv('highway', obs_type='image', action_type='discrete')
    
    # Or with custom reward config
    env = HighwayEnv('highway', reward_config={'high_speed_reward': 0.8})
"""

# Import from modular files
from .highway_rewards import DEFAULT_REWARD_CONFIGS, get_reward_config
from .highway_base import HighwayEnv, HighwayEnvKinematics, ENV_NAME_MAPPING

# Re-export for backward compatibility
__all__ = [
    'HighwayEnv',
    'HighwayEnvKinematics', 
    'DEFAULT_REWARD_CONFIGS',
    'get_reward_config',
    'ENV_NAME_MAPPING',
]
