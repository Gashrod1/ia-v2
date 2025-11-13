"""
Script pour visualiser les bots en temps réel (x1) sans les entraîner.
Lance simplement une simulation et affiche le jeu dans RocketSim Visualizer.
"""

import numpy as np
import os
import torch

print("=" * 60)
print("BOT VISUALIZATION MODE - No training, just watching")
print("=" * 60)

# CRITICAL: Verify CUDA is available
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Monkey-patch torch.load to always use CPU mapping when CUDA is not available
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if not torch.cuda.is_available() and 'map_location' not in kwargs:
        kwargs['map_location'] = torch.device('cpu')
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from rewards import SaveBoostReward, TouchStrengthReward
import rlgym_sim
from rlgym_sim.utils.reward_functions import CombinedReward
from rlgym_sim.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, VelocityBallToGoalReward, \
    EventReward, FaceBallReward
from rlgym_sim.utils.obs_builders import DefaultObs
from rlgym_sim.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition
from rlgym_sim.utils.state_setters import RandomState
from rlgym_sim.utils import common_values
from rlgym_sim.utils.action_parsers import ContinuousAction
import rocketsimvis_rlgym_sim_client as rsv

# Game timing constants
TICK_SKIP = 8
GAME_TICK_RATE = 120
STEP_TIME = TICK_SKIP / GAME_TICK_RATE

# Environment configuration (same as bot.py)
team_size = 1
spawn_opponents = True

# Terminal conditions
terminal_conditions = [NoTouchTimeoutCondition(500), GoalScoredCondition()]

# State setter
state_setter = RandomState(ball_rand_speed=True, 
                           cars_rand_speed=True, 
                           cars_on_ground=False)

# Rewards (not used for visualization but needed for env)
reward_fn = CombinedReward.from_zipped(
    (EventReward(team_goal=1, concede=-1), 50),
    (VelocityBallToGoalReward(), 10.0),
    (TouchStrengthReward(), 1.5),
    (VelocityPlayerToBallReward(), 0.5),
    (SaveBoostReward(), 0.3),
    (FaceBallReward(), 0.05),
)

# Observation builder
obs_builder = DefaultObs(
    pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z]),
    ang_coef=1 / np.pi,
    lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
    ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL)

# Action parser
action_parser = ContinuousAction()

# Create environment
print("\nCreating RocketSim environment...")
env = rlgym_sim.make(tick_skip=TICK_SKIP,
                     team_size=team_size,
                     spawn_opponents=spawn_opponents,
                     terminal_conditions=terminal_conditions,
                     reward_fn=reward_fn,
                     obs_builder=obs_builder,
                     action_parser=action_parser,
                     state_setter=state_setter)

print("Environment created successfully!")

# Find latest checkpoint
print("\nSearching for latest checkpoint...")
latest_checkpoint_dir = None
checkpoint_base = os.path.join("data", "checkpoints")
try:
    if os.path.isdir(checkpoint_base):
        run_dirs = [d for d in os.listdir(checkpoint_base) if d.startswith("rlgym-ppo-run") and os.path.isdir(os.path.join(checkpoint_base, d))]
        if run_dirs:
            latest_run = max(run_dirs, key=lambda d: os.path.getmtime(os.path.join(checkpoint_base, d)))
            latest_run_dir = os.path.join(checkpoint_base, latest_run)
            
            checkpoint_subdirs = [d for d in os.listdir(latest_run_dir) if d.isdigit() and os.path.isdir(os.path.join(latest_run_dir, d))]
            if checkpoint_subdirs:
                latest_checkpoint = max(checkpoint_subdirs, key=int)
                latest_checkpoint_dir = os.path.join(latest_run_dir, latest_checkpoint)
                print(f"✓ Found checkpoint: {latest_checkpoint_dir}")
except Exception as e:
    print(f"✗ No checkpoint found: {e}")

# Load policy if checkpoint exists
policy = None
if latest_checkpoint_dir and os.path.exists(os.path.join(latest_checkpoint_dir, "policy.pt")):
    print("\nLoading policy from checkpoint...")
    try:
        from rlgym_ppo.ppo import PPOLearner
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Create a dummy PPO learner just to load the policy
        obs_space_size = obs_builder.get_obs_space(env._match)[0]
        act_space_size = action_parser.get_action_space()[0]
        
        policy_path = os.path.join(latest_checkpoint_dir, "policy.pt")
        policy_dict = torch.load(policy_path, map_location=device)
        
        # Create policy network with same architecture as training
        from rlgym_ppo.ppo import DiscreteFF
        policy = DiscreteFF(obs_space_size, act_space_size, (512, 512, 256), device)
        policy.load_state_dict(policy_dict)
        policy.eval()
        
        print(f"✓ Policy loaded successfully from {policy_path}")
        print(f"✓ Using device: {device}")
    except Exception as e:
        print(f"✗ Failed to load policy: {e}")
        print("Will use random actions instead")
        policy = None
else:
    print("\nNo checkpoint found - will use random actions")

print("\n" + "=" * 60)
print("STARTING VISUALIZATION")
print("=" * 60)
print("Make sure RocketSim Visualizer is running!")
print("Press Ctrl+C to stop")
print("=" * 60 + "\n")

# Main visualization loop
obs = env.reset()
step_count = 0
episode_count = 0

try:
    import time
    while True:
        # Get actions (either from policy or random)
        if policy is not None:
            with torch.no_grad():
                obs_tensor = torch.from_numpy(np.asarray([obs])).float().to(policy.device)
                action = policy.get_action(obs_tensor, deterministic=True)[0].cpu().numpy()
        else:
            # Random actions
            action = env.action_space.sample()
        
        # Step environment
        obs, reward, done, state = env.step(action)
        
        # Send state to visualizer
        rsv.send_state_to_rocketsimvis(state)
        
        step_count += 1
        
        # Real-time delay (x1 speed)
        time.sleep(STEP_TIME)
        
        # Reset on episode end
        if done:
            episode_count += 1
            blue_score = state.blue_score
            orange_score = state.orange_score
            print(f"Episode {episode_count} finished | Score: Blue {blue_score} - {orange_score} Orange | Steps: {step_count}")
            obs = env.reset()
            step_count = 0
            
except KeyboardInterrupt:
    print("\n\nVisualization stopped by user")
    print(f"Total episodes watched: {episode_count}")
