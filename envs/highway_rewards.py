"""
Reward configurations for highway-env environments.
Each environment type has its own default reward configuration.
"""

# Default reward configuration for different environment types
DEFAULT_REWARD_CONFIGS = {
    "highway": {
        # 1) Speed – still important, but less overpowering
        "high_speed_reward": 0.4,            # from 0.6 -> 0.4
        "reward_speed_range": [23.0, 27.0],  # ~83-97 km/h (matches traffic speed)

        # 2) Safety: MUCH stronger
        "collision_reward": -5.0,            # from -2.0 -> -5.0
        "on_road_reward": 0.02,

        # 3) Lane behavior – push harder to escape blocking
        "lane_change_reward": 0.0,
        "smart_lane_change_reward": 0.25,    # from 0.15 -> 0.25
        "blocked_lane_penalty": 0.6,         # from 0.30 -> 0.6
        "overtake_reward": 0.2,              # slightly up from 0.15

        # 4) Distance / tailgating – harsher
        "min_safe_distance": 20.0,           # from 15.0 -> 20.0
        "safe_distance_penalty": 1.0,        # from 0.40 -> 1.0 (2.5x)

        # 5) Lane "look ahead" / slow vehicle detection
        "look_ahead_distance": 50.0,
        "slow_vehicle_threshold": 0.9,       # from 0.85 -> 0.9

        # 6) Progress / heading shaping
        "progress_reward_scale": 0.005,
        "heading_reward": 0.2,

        # 7) Episode-level shaping
        "success_reward": 1.0,
        "shaped_reward_weight": 0.85,

        # Optional
        "normalize_reward": False,
    },
    
    "intersection": {
        "high_speed_reward": 0.2,
        "reward_speed_range": [10, 15],
        "collision_reward": -1.0,
        "on_road_reward": 0.1,
        "arrived_reward": 1.0,           # Bonus for successfully crossing
        "safe_distance_reward": 0.2,
        "min_safe_distance": 10.0,
        "normalize_reward": True,
    },
    
    "merge": {
        "high_speed_reward": 0.3,
        "reward_speed_range": [20, 28],
        "collision_reward": -1.0,
        "on_road_reward": 0.1,
        "merging_speed_reward": 0.2,     # Reward for matching highway speed
        "safe_distance_reward": 0.15,
        "min_safe_distance": 12.0,
        "normalize_reward": True,
    },
    
    "roundabout": {
        "high_speed_reward": 0.2,
        "reward_speed_range": [8, 12],
        "collision_reward": -1.0,
        "on_road_reward": 0.15,
        "safe_distance_reward": 0.2,
        "min_safe_distance": 8.0,
        "normalize_reward": True,
    },
    
    "parking": {
        "collision_reward": -1.0,
        "on_road_reward": 0.0,
        "goal_reward": 1.0,              # Reward for reaching parking spot
        "goal_distance_reward": 0.5,     # Shaped reward based on distance to goal
        "heading_reward": 0.2,           # Reward for correct heading
        "normalize_reward": True,
    },
    
    "racetrack": {
        "high_speed_reward": 0.5,
        "reward_speed_range": [15, 25],
        "collision_reward": -1.0,
        "on_road_reward": 0.3,
        "lane_centering_reward": 0.2,    # Reward for staying centered
        "normalize_reward": True,
    },

    "twoway" :{
        "collision_reward": 0,
        "left_lane_constraint": 1,
        "left_lane_reward": 0.2,
        "high_speed_reward": 0.8,
    }
}


def get_reward_config(env_name: str, custom_config: dict = None) -> dict:
    """
    Get reward configuration for an environment.
    
    Args:
        env_name: Environment name (e.g., 'highway', 'intersection')
        custom_config: Optional custom config to merge with defaults
        
    Returns:
        Merged reward configuration dict
    """
    # Get default config for this environment type
    default_config = DEFAULT_REWARD_CONFIGS.get(env_name, DEFAULT_REWARD_CONFIGS["highway"])
    
    # Merge with custom config if provided
    config = default_config.copy()
    if custom_config:
        config.update(custom_config)
    
    return config
