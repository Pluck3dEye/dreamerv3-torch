# Highway-Env Complete Configuration Reference

This document provides comprehensive documentation for **all** configuration options in highway-env for DreamerV3 training.

## Table of Contents
1. [Environment Configuration](#environment-configuration)
2. [Observation Configuration](#observation-configuration)
3. [Action Configuration](#action-configuration)
4. [Reward Configuration](#reward-configuration)
5. [Simulation Configuration](#simulation-configuration)
6. [Rendering Configuration](#rendering-configuration)
7. [Vehicle Configuration](#vehicle-configuration)
8. [Road Configuration](#road-configuration)
9. [Custom Reward Shaping](#custom-reward-shaping)
10. [Speed Reference](#speed-reference)
11. [File Locations](#file-locations)

---

## Environment Configuration

### Core Environment Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `duration` | int | 40 | Episode duration in policy steps |
| `lanes_count` | int | 4 | Number of highway lanes |
| `vehicles_count` | int | 50 | Number of traffic vehicles |
| `controlled_vehicles` | int | 1 | Number of ego vehicles |
| `initial_lane_id` | int/None | None | Starting lane (None = random) |
| `ego_spacing` | float | 2.0 | Initial spacing from other vehicles |
| `vehicles_density` | float | 1.0 | Vehicle density multiplier |
| `offroad_terminal` | bool | False | End episode if vehicle goes offroad |

### Example Configuration

```python
env_config = {
    "duration": 40,          # 40 policy steps per episode
    "lanes_count": 4,        # 4-lane highway
    "vehicles_count": 50,    # Dense traffic
    "initial_lane_id": None, # Random starting lane
    "vehicles_density": 1.0, # Normal density
}
```

---

## Observation Configuration

### Observation Types

| Type | String Key | Description |
|------|------------|-------------|
| `KinematicObservation` | `"Kinematics"` | Numerical vehicle states (position, velocity, heading) |
| `GrayscaleObservation` | `"GrayscaleImage"` | Grayscale camera image |
| `OccupancyGridObservation` | `"OccupancyGrid"` | Bird's-eye view occupancy grid |
| `TimeToCollisionObservation` | `"TimeToCollision"` | Time-to-collision grid |
| `LidarObservation` | `"Lidar"` | Simulated LIDAR scan |
| `AttributesObservation` | `"Attributes"` | Vehicle attribute dictionary |
| `MultiAgentObservation` | `"MultiAgentObservation"` | Observations for multiple agents |

### Kinematics Observation

```python
observation_config = {
    "type": "Kinematics",
    "features": ["presence", "x", "y", "vx", "vy", "heading"],  # Observation features
    "vehicles_count": 5,     # Number of observed vehicles
    "features_range": {      # Feature normalization ranges
        "x": [-100, 100],
        "y": [-100, 100],
        "vx": [-20, 20],
        "vy": [-20, 20]
    },
    "absolute": False,       # Relative to ego vehicle
    "order": "sorted",       # Sort by distance ("sorted", "shuffled", or None)
    "normalize": True,       # Normalize features
    "clip": True,            # Clip to range
    "see_behind": False,     # Include vehicles behind
    "observe_intentions": False,  # Include turn signals
    "include_obstacles": True     # Include static obstacles
}
```

#### Available Features
- `presence`: Whether vehicle exists (0 or 1)
- `x`: Longitudinal position
- `y`: Lateral position
- `vx`: Longitudinal velocity
- `vy`: Lateral velocity
- `heading`: Vehicle heading angle
- `cos_h`: Cosine of heading
- `sin_h`: Sine of heading
- `cos_d`: Cosine of direction to ego
- `sin_d`: Sine of direction to ego
- `long_off`: Longitudinal offset
- `lat_off`: Lateral offset
- `ang_off`: Angular offset

### Grayscale Observation

```python
observation_config = {
    "type": "GrayscaleImage",
    "observation_shape": (64, 64),  # Image height x width
    "stack_size": 4,                # Frame stacking
    "weights": [0.2989, 0.5870, 0.1140],  # RGB to gray weights
    "scaling": None,                # Override env scaling
    "centering_position": None      # Override env centering
}
```

### Occupancy Grid Observation

```python
observation_config = {
    "type": "OccupancyGrid",
    "features": ["presence", "vx", "vy"],  # Features per cell
    "grid_size": [[-27.5, 27.5], [-27.5, 27.5]],  # Grid extent [x, y]
    "grid_step": [5, 5],           # Cell size [x, y]
    "features_range": {...},        # Feature normalization
    "absolute": False,              # Absolute or relative
    "align_to_vehicle_axes": False, # Align grid to vehicle
    "clip": True,                   # Clip features
    "as_image": False               # Return as image format
}
```

---

## Action Configuration

### Action Types

| Type | String Key | Description |
|------|------------|-------------|
| `DiscreteMetaAction` | `"DiscreteMetaAction"` | High-level discrete actions (lane change, speed) |
| `DiscreteAction` | `"DiscreteAction"` | Low-level discrete acceleration/steering |
| `ContinuousAction` | `"ContinuousAction"` | Continuous acceleration and steering |
| `MultiAgentAction` | `"MultiAgentAction"` | Actions for multiple agents |

### Discrete Meta Action (Default)

```python
action_config = {
    "type": "DiscreteMetaAction",
    "longitudinal": True,    # Enable speed control (FASTER, SLOWER)
    "lateral": True,         # Enable lane changes (LANE_LEFT, LANE_RIGHT)
    "target_speeds": None    # Custom target speeds (default: np.linspace(0, 30, 5))
}
```

**Actions Available:**
| Index | Action | Description |
|-------|--------|-------------|
| 0 | `LANE_LEFT` | Change to left lane |
| 1 | `IDLE` | Maintain current behavior |
| 2 | `LANE_RIGHT` | Change to right lane |
| 3 | `FASTER` | Increase target speed |
| 4 | `SLOWER` | Decrease target speed |

### Continuous Action

```python
action_config = {
    "type": "ContinuousAction",
    "acceleration_range": [-5.0, 5.0],  # m/s² (default: based on vehicle)
    "steering_range": [-0.7854, 0.7854], # radians (±45°)
    "speed_range": [0, 30],              # m/s speed limits
    "longitudinal": True,                # Enable acceleration control
    "lateral": True,                     # Enable steering control
    "dynamical": False,                  # Use dynamical model
    "clip": True                         # Clip to ranges
}
```

**Action Space:** `Box([-1, -1], [1, 1], (2,), float32)`
- `action[0]`: Acceleration (-1 = max brake, +1 = max accelerate)
- `action[1]`: Steering (-1 = full left, +1 = full right)

### Discrete Action

```python
action_config = {
    "type": "DiscreteAction",
    "acceleration_range": [-5.0, 5.0],  # m/s²
    "steering_range": [-0.7854, 0.7854], # radians
    "longitudinal": True,
    "lateral": True,
    "dynamical": False,
    "clip": True,
    "actions_per_axis": 3    # Number of discrete levels per axis (3x3=9 actions)
}
```

---

## Reward Configuration

### Native Highway-Env Rewards

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `collision_reward` | float | -1.0 | Reward on collision |
| `right_lane_reward` | float | 0.1 | Reward for being in right lane |
| `high_speed_reward` | float | 0.4 | Reward for high speed |
| `lane_change_reward` | float | 0.0 | Reward/penalty for lane change |
| `reward_speed_range` | [float, float] | [20, 30] | Speed range for speed reward [min, max] |
| `normalize_reward` | bool | True | Normalize total reward to [0, 1] |

### Our Modified Settings (Lane-Neutral)

```python
# In highway_base.py - we disable lane bias
reward_config = {
    "collision_reward": -1.0,      # Keep collision penalty
    "right_lane_reward": 0.0,      # DISABLED - was causing right lane bias
    "high_speed_reward": 0.4,      # Standard speed reward
    "lane_change_reward": 0.0,     # DISABLED - neutral about changes
    "reward_speed_range": [23, 27], # Match traffic speed
    "normalize_reward": True
}
```

### Reward Calculation

The native reward is calculated as:
```python
reward = collision_reward * crashed \
       + right_lane_reward * (lane / (lanes_count - 1)) \
       + high_speed_reward * speed_reward_fn(speed) \
       + lane_change_reward * lane_changed

# Where speed_reward_fn is:
speed_reward = (speed - min_speed) / (max_speed - min_speed)  # Scaled to [0, 1]
```

---

## Simulation Configuration

### Timing Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `simulation_frequency` | int | 15 | Physics updates per second |
| `policy_frequency` | int | 1 | Agent decisions per second |

**Relationship:**
- Steps per action = `simulation_frequency / policy_frequency`
- With defaults: 15 physics updates per agent decision

### Example

```python
simulation_config = {
    "simulation_frequency": 15,  # 15 Hz physics
    "policy_frequency": 5,       # 5 Hz agent (3 physics steps per action)
}
```

---

## Rendering Configuration

### Display Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `screen_width` | int | 600 | Render window width (pixels) |
| `screen_height` | int | 150 | Render window height (pixels) |
| `scaling` | float | 5.5 | Pixels per meter |
| `centering_position` | [float, float] | [0.3, 0.5] | Ego position in frame [x, y] (0-1) |
| `show_trajectories` | bool | False | Show predicted vehicle paths |
| `render_agent` | bool | True | Render the ego vehicle |
| `offscreen_rendering` | bool | False | Render without display |
| `real_time_rendering` | bool | False | Sync to real-time clock |
| `manual_control` | bool | False | Enable keyboard control |

### Example Configuration

```python
render_config = {
    "screen_width": 600,
    "screen_height": 150,
    "scaling": 5.5,              # 5.5 pixels per meter
    "centering_position": [0.3, 0.5],  # Ego at 30% from left
    "show_trajectories": True,   # Visualize predictions
    "render_agent": True,
    "offscreen_rendering": False,
    "real_time_rendering": True  # For visualization
}
```

---

## Vehicle Configuration

### Vehicle Types

| Type | Module Path | Description |
|------|-------------|-------------|
| `IDMVehicle` | `highway_env.vehicle.behavior.IDMVehicle` | Intelligent Driver Model (default) |
| `LinearVehicle` | `highway_env.vehicle.behavior.LinearVehicle` | Simple linear controller |
| `AggressiveVehicle` | `highway_env.vehicle.behavior.AggressiveVehicle` | Aggressive IDM variant |
| `DefensiveVehicle` | `highway_env.vehicle.behavior.DefensiveVehicle` | Defensive IDM variant |
| `ControlledVehicle` | `highway_env.vehicle.behavior.ControlledVehicle` | Base controlled vehicle |

### IDMVehicle Parameters

The Intelligent Driver Model (IDM) is the default traffic model:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `COMFORT_ACC_MAX` | 3.0 m/s² | Comfortable acceleration |
| `COMFORT_ACC_MIN` | -5.0 m/s² | Comfortable deceleration |
| `DISTANCE_WANTED` | 10.0 m | Desired gap distance |
| `TIME_WANTED` | 1.5 s | Desired time headway |
| `DELTA` | 4.0 | Acceleration exponent |
| `DELTA_RANGE` | [3.5, 4.5] | Randomization range |
| `ACC_MAX` | 6.0 m/s² | Maximum acceleration |
| `LANE_CHANGE_MIN_ACC_GAIN` | 0.2 m/s² | Min acceleration advantage for lane change |
| `LANE_CHANGE_MAX_BRAKING_IMPOSED` | 2.0 m/s² | Max braking imposed on others |
| `LANE_CHANGE_DELAY` | 1.0 s | Delay between lane changes |

### Vehicle Type Configuration

```python
env_config = {
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",  # Default
    # Or for more aggressive traffic:
    # "other_vehicles_type": "highway_env.vehicle.behavior.AggressiveVehicle",
}
```

---

## Road Configuration

### Default Road Layout

- 4 lanes (configurable via `lanes_count`)
- Lane width: 4 meters
- Straight infinite highway

### Lane Numbering
- Lane 0: Leftmost (fast lane)
- Lane N-1: Rightmost (slow lane)

---

## Custom Reward Shaping

Our DreamerV3 integration adds custom reward shaping on top of highway-env's native rewards.

### Configuration in `configs.yaml`

```yaml
highway:
  reward:
    # Speed reward
    speed_reward: 1.0           # Weight for speed reward
    target_speed: 25.0          # Optimal speed (m/s)
    speed_tolerance: 5.0        # Acceptable deviation (m/s)
    
    # Safety settings
    collision_reward: -5.0      # Strong collision penalty
    safe_distance_penalty: 1.0  # Weight for distance penalty
    min_safe_distance: 15.0     # Minimum safe following distance (m)
    
    # Smoothness
    smoothness_penalty: 0.1     # Penalty for erratic actions
    
    # Progress
    progress_reward: 0.1        # Reward per meter forward
    
    # Speed range (matches highway-env)
    reward_speed_range: [23.0, 27.0]  # [min, max] m/s
```

### Reward Components

1. **Native Highway-Env Reward**: Base reward from environment
2. **Speed Reward**: Bonus for maintaining target speed
3. **Distance Penalty**: Penalty for tailgating
4. **Collision Penalty**: Additional crash penalty
5. **Smoothness Penalty**: Penalize erratic steering/acceleration
6. **Progress Reward**: Reward forward movement

---

## Speed Reference

### Understanding `reward_speed_range`

```python
reward_speed_range: [23.0, 27.0]  # [min_speed, max_speed] in m/s
```

| Speed (m/s) | Speed (km/h) | Speed (mph) | Reward Level |
|-------------|--------------|-------------|--------------|
| < 23.0 | < 82.8 | < 51 | 0% (too slow) |
| 23.0 | 82.8 | 51 | 0% (threshold) |
| 25.0 | 90.0 | 56 | 50% (middle) |
| 27.0 | 97.2 | 60 | 100% (max) |
| > 27.0 | > 97.2 | > 60 | 100% (capped) |

### Speed Conversion Table

| m/s | km/h | mph | Typical Use |
|-----|------|-----|-------------|
| 15 | 54 | 34 | Urban highway |
| 20 | 72 | 45 | City highway |
| 23 | 83 | 51 | Our minimum |
| 25 | 90 | 56 | IDM target |
| 27 | 97 | 60 | Our maximum |
| 30 | 108 | 67 | Fast highway |
| 35 | 126 | 78 | Very fast |

### Why [23, 27] m/s?

Traffic vehicles use IDMVehicle with target speed ~25 m/s:
- Setting [23, 27] encourages matching traffic flow
- Allows slightly faster driving for overtaking
- Avoids rewarding dangerous speeding

---

## File Locations

| File | Purpose |
|------|---------|
| `envs/highway.py` | Entry point for imports |
| `envs/highway_base.py` | Main environment class with native config overrides |
| `envs/highway_rewards.py` | Custom reward shaping definitions |
| `configs.yaml` | Runtime configuration |
| `evaluate.py` | Standalone evaluation script |

---

## Complete Example Configuration

```python
# Full highway-env configuration
config = {
    # Environment
    "duration": 40,
    "lanes_count": 4,
    "vehicles_count": 50,
    "controlled_vehicles": 1,
    "initial_lane_id": None,
    "ego_spacing": 2.0,
    "vehicles_density": 1.0,
    "offroad_terminal": False,
    
    # Observation
    "observation": {
        "type": "Kinematics",
        "features": ["presence", "x", "y", "vx", "vy"],
        "vehicles_count": 5,
        "normalize": True,
        "absolute": False
    },
    
    # Action
    "action": {
        "type": "DiscreteMetaAction",
        "longitudinal": True,
        "lateral": True
    },
    
    # Simulation
    "simulation_frequency": 15,
    "policy_frequency": 1,
    
    # Vehicles
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    
    # Rewards (lane-neutral)
    "collision_reward": -1.0,
    "right_lane_reward": 0.0,
    "high_speed_reward": 0.4,
    "lane_change_reward": 0.0,
    "reward_speed_range": [23, 27],
    "normalize_reward": True,
    
    # Rendering
    "screen_width": 600,
    "screen_height": 150,
    "scaling": 5.5,
    "centering_position": [0.3, 0.5],
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": False,
    "real_time_rendering": False,
    "manual_control": False
}
```

---

## Tuning Guide

### For Safer Driving
```yaml
collision_reward: -10.0      # Stronger penalty
safe_distance_penalty: 2.0   # Maintain distance
min_safe_distance: 20.0      # Larger buffer
```

### For Faster Driving
```yaml
reward_speed_range: [25.0, 30.0]  # Higher speeds
speed_reward: 2.0                  # Emphasize speed
```

### For Dense Traffic
```yaml
vehicles_count: 80           # More vehicles
vehicles_density: 1.5        # Higher density
safe_distance_penalty: 1.5   # More careful
```

### For Long Episodes
```yaml
duration: 100                # Longer episodes
simulation_frequency: 15     # Keep physics rate
policy_frequency: 5          # More decisions
```

---

## All Parameters Quick Reference

### Environment Parameters
```python
{
    "duration": 40,                    # Episode length in steps
    "lanes_count": 4,                  # Number of lanes
    "vehicles_count": 50,              # Traffic vehicle count
    "controlled_vehicles": 1,          # Ego vehicles
    "initial_lane_id": None,           # Start lane (None=random)
    "ego_spacing": 2.0,                # Initial vehicle spacing
    "vehicles_density": 1.0,           # Traffic density multiplier
    "offroad_terminal": False,         # End on offroad
}
```

### Observation Parameters
```python
{
    "observation": {
        "type": "Kinematics",          # Observation type
        "features": [...],              # Feature list
        "vehicles_count": 5,            # Vehicles to observe
        "features_range": {...},        # Normalization ranges
        "absolute": False,              # Absolute coordinates
        "order": "sorted",              # Vehicle ordering
        "normalize": True,              # Normalize features
        "clip": True,                   # Clip to range
        "see_behind": False,            # See rear vehicles
        "observe_intentions": False,    # Include intentions
        "include_obstacles": True,      # Include obstacles
    }
}
```

### Action Parameters
```python
{
    "action": {
        "type": "DiscreteMetaAction",  # Action type
        "longitudinal": True,           # Speed control
        "lateral": True,                # Lane changes
        "target_speeds": None,          # Custom speeds
    }
}
```

### Reward Parameters
```python
{
    "collision_reward": -1.0,          # Collision penalty
    "right_lane_reward": 0.0,          # Right lane reward
    "high_speed_reward": 0.4,          # Speed reward weight
    "lane_change_reward": 0.0,         # Lane change reward
    "reward_speed_range": [23, 27],    # Speed range [min,max]
    "normalize_reward": True,          # Normalize to [0,1]
}
```

### Simulation Parameters
```python
{
    "simulation_frequency": 15,        # Physics Hz
    "policy_frequency": 1,             # Decision Hz
}
```

### Rendering Parameters
```python
{
    "screen_width": 600,               # Window width
    "screen_height": 150,              # Window height
    "scaling": 5.5,                    # Pixels per meter
    "centering_position": [0.3, 0.5],  # Ego position in frame
    "show_trajectories": False,        # Show predictions
    "render_agent": True,              # Render ego
    "offscreen_rendering": False,      # Headless mode
    "real_time_rendering": False,      # Real-time sync
    "manual_control": False,           # Keyboard control
}
```

### Vehicle Parameters
```python
{
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
}
```
