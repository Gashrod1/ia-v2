import numpy as np
import os
import torch

# CRITICAL: Verify CUDA is available
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("WARNING: CUDA NOT AVAILABLE - Training will be VERY slow on CPU!")

# Monkey-patch torch.load to always use CPU mapping when CUDA is not available
# This fixes the "Attempting to deserialize object on a CUDA device" error
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if not torch.cuda.is_available() and 'map_location' not in kwargs:
        kwargs['map_location'] = torch.device('cpu')
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from rlgym_sim.utils.gamestates import GameState
from rlgym_ppo.util import MetricsLogger
from rewards import InAirReward, SpeedTowardBallReward, HandbrakePenalty, FlipDisciplineReward, SaveBoostReward, TouchStrengthReward

# Game timing constants
TICK_SKIP = 8  # Number of physics ticks per step
GAME_TICK_RATE = 120  # Rocket League runs at 120 ticks per second
STEP_TIME = TICK_SKIP / GAME_TICK_RATE  # Time between steps in seconds (8/120 = 0.0667 seconds)


class ExampleLogger(MetricsLogger):
    def __init__(self):
        super().__init__()
        # Per-episode tracking (reset each episode)
        self.episode_touches = {}  # Dict par joueur
        self.episode_shots_on_goal = 0
        self.episode_saves = 0
        self.episode_duration = 0
        
        # Cumulative tracking (across all episodes)
        self.total_touches = 0
        self.total_goals = 0
        self.total_episodes = 0
        
        # Previous state tracking
        self.prev_ball_toucher = -1
        self.prev_blue_score = 0
        self.prev_orange_score = 0
        
    def _collect_metrics(self, game_state: GameState) -> list:
        # Track episode duration
        self.episode_duration += 1
        
        # Detect ball touches (when last_touch changes)
        if game_state.last_touch != self.prev_ball_toucher and game_state.last_touch != -1:
            self.total_touches += 1
            player_id = game_state.last_touch
            if player_id not in self.episode_touches:
                self.episode_touches[player_id] = 0
            self.episode_touches[player_id] += 1
            self.prev_ball_toucher = game_state.last_touch
            
        # Detect goals
        goals_scored = 0
        if game_state.blue_score > self.prev_blue_score:
            goals_scored += (game_state.blue_score - self.prev_blue_score)
            self.prev_blue_score = game_state.blue_score
            
        if game_state.orange_score > self.prev_orange_score:
            goals_scored += (game_state.orange_score - self.prev_orange_score)
            self.prev_orange_score = game_state.orange_score
        
        if goals_scored > 0:
            self.total_goals += goals_scored
        
        # Collect data to report
        return {
            'ball_height': game_state.ball.position[2],
            'ball_speed': np.linalg.norm(game_state.ball.linear_velocity),
            'num_players': len(game_state.players),
            'blue_score': game_state.blue_score,
            'orange_score': game_state.orange_score,
        }

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        self.total_episodes += 1
        
        # Calculate averages across the iteration
        avg_ball_height = np.mean([m['ball_height'] for m in collected_metrics])
        avg_ball_speed = np.mean([m['ball_speed'] for m in collected_metrics])
        
        # Calculate touches per player
        total_episode_touches = sum(self.episode_touches.values())
        num_players = collected_metrics[0]['num_players'] if collected_metrics else 1
        avg_touches_per_player = total_episode_touches / num_players if num_players > 0 else 0
        
        # Calculate average episode length (in seconds)
        avg_episode_duration = self.episode_duration / len(collected_metrics) if collected_metrics else 0
        avg_episode_duration_seconds = avg_episode_duration * STEP_TIME
        
        report = {
            # Game stats
            "Stats/Total Touches": self.total_touches,
            "Stats/Total Goals": self.total_goals,
            "Stats/Total Episodes": self.total_episodes,
            "Stats/Touches Per Episode": total_episode_touches,
            "Stats/Touches Per Player": avg_touches_per_player,
            "Stats/Episode Duration (steps)": avg_episode_duration,
            "Stats/Episode Duration (seconds)": avg_episode_duration_seconds,
            
            # Ball stats
            "Ball/Average Height": avg_ball_height,
            "Ball/Average Speed": avg_ball_speed,
            
            # Cumulative timesteps (required)
            "Cumulative Timesteps": cumulative_timesteps
        }
        
        # Reset episode stats
        self.episode_touches = {}
        self.episode_duration = 0
        
        wandb_run.log(report)


def build_rocketsim_env():
    import rlgym_sim
    from rlgym_sim.utils.reward_functions import CombinedReward
    from rlgym_sim.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, VelocityBallToGoalReward, \
        EventReward, FaceBallReward
    from rlgym_sim.utils.obs_builders import DefaultObs
    from rlgym_sim.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition
    from rlgym_sim.utils.state_setters import RandomState
    from rlgym_sim.utils import common_values
    from rlgym_sim.utils.action_parsers import ContinuousAction

    # ===== ENVIRONMENT CONFIGURATION =====
    spawn_opponents = True              # Phase 1: No opponents (focus on ball control)
    team_size = 1                        # Single agent training
    timeout_seconds = 12                 # Timeout between 10-15 seconds
    timeout_ticks = int(round(timeout_seconds * GAME_TICK_RATE / TICK_SKIP))

    action_parser = ContinuousAction()
    terminal_conditions = [NoTouchTimeoutCondition(timeout_ticks), GoalScoredCondition()]
    
    # RandomState provides better training by spawning cars and ball with random positions/velocities
    # cars_on_ground=False means cars spawn airborne 50% of the time (helps with air training)
    state_setter = RandomState(ball_rand_speed=True, 
                               cars_rand_speed=True, 
                               cars_on_ground=False)
    
    reward_fn = CombinedReward.from_zipped(
        # Format is (func, weight)
        # PHASE 2: Bot hits ball consistently, now learn to score goals
        (EventReward(team_goal=1, concede=-1), 50),  # MASSIVELY increased - scoring is the objective!
        (VelocityBallToGoalReward(), 10.0),          # DOUBLED - push ball toward goal is PRIMARY
        (TouchStrengthReward(), 1.5),                # REDUCED - don't farm weak touches
        (VelocityPlayerToBallReward(), 0.5),         # REDUCED - getting to ball is less important
        (SpeedTowardBallReward(), 0.3),              # REDUCED - approach is minor
        (SaveBoostReward(), 0.3),                    # REDUCED - boost management is secondary
        (FaceBallReward(), 0.05),                    # REDUCED - minor correction only
        (InAirReward(), 0.1),                        # REDUCED - don't forget jumping
    )

    obs_builder = DefaultObs(
        pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z]),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL)

    env = rlgym_sim.make(tick_skip=TICK_SKIP,
                         team_size=team_size,
                         spawn_opponents=spawn_opponents,
                         terminal_conditions=terminal_conditions,
                         reward_fn=reward_fn,
                         obs_builder=obs_builder,
                         action_parser=action_parser,
                         state_setter=state_setter)
    
    import rocketsimvis_rlgym_sim_client as rsv
    type(env).render = lambda self: rsv.send_state_to_rocketsimvis(self._prev_state)

    return env

if __name__ == "__main__":
    from rlgym_ppo import Learner
    
    metrics_logger = ExampleLogger()

    # ========================================================================
    # HARDWARE CONFIGURATION
    # ========================================================================
    n_proc = 64                  # Number of parallel game instances (adjust for your CPU)
    minibatch_size = 50_000      # Must divide evenly into ppo_batch_size
    device = "cuda:0"            # "cuda:0" for GPU, "cpu" for CPU-only training
    
    # Network architecture - adjust based on your hardware
    # Bigger networks learn better but require more GPU/CPU power
    policy_size = (512, 512, 256)
    critic_size = (512, 512, 256)

    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.75)))

    # Discover latest checkpoint/run folder under data/checkpoints.
    latest_checkpoint_dir = None
    checkpoint_base = os.path.join("data", "checkpoints")
    try:
        # Find directories starting with the common prefix used by rlgym-ppo runs
        if os.path.isdir(checkpoint_base):
            run_dirs = [d for d in os.listdir(checkpoint_base) if d.startswith("rlgym-ppo-run") and os.path.isdir(os.path.join(checkpoint_base, d))]
            if run_dirs:
                # Choose the most recently modified run directory (robust to different naming schemes)
                latest_run = max(run_dirs, key=lambda d: os.path.getmtime(os.path.join(checkpoint_base, d)))
                latest_run_dir = os.path.join(checkpoint_base, latest_run)
                
                # Within the run directory, find the latest numbered checkpoint subdirectory
                checkpoint_subdirs = [d for d in os.listdir(latest_run_dir) if d.isdigit() and os.path.isdir(os.path.join(latest_run_dir, d))]
                if checkpoint_subdirs:
                    # Sort by the numeric value to get the highest checkpoint number
                    latest_checkpoint = max(checkpoint_subdirs, key=int)
                    latest_checkpoint_dir = os.path.join(latest_run_dir, latest_checkpoint)
    except Exception:
        # If anything goes wrong (missing folder, permissions, etc.), leave None so learner won't try to load
        latest_checkpoint_dir = None
    
    learner = Learner(build_rocketsim_env,
                      n_proc=n_proc,
                      min_inference_size=min_inference_size,
                      metrics_logger=metrics_logger,
                      
                      # Device configuration
                      device=device,
                      
                      # ===== DATA COLLECTION SETTINGS =====
                      ts_per_iteration=50_000,              # Phase 1: 50k (increase to 100k in Phase 2)
                      exp_buffer_size=150_000,              # 3x ts_per_iteration for better learning
                      ppo_batch_size=50_000,                # Should equal ts_per_iteration
                      ppo_minibatch_size=minibatch_size,    # Adjust based on VRAM/RAM availability
                      
                      # ===== NETWORK ARCHITECTURE =====
                      policy_layer_sizes=policy_size,
                      critic_layer_sizes=critic_size,
                      
                      # ===== PPO HYPERPARAMETERS =====
                      ppo_ent_coef=0.01,                    # Entropy coefficient - encourages exploration (0.01 is golden value)
                      ppo_epochs=2,                         # 2-3 is optimal (2 for speed, 3 for quality)
                      
                      # ===== LEARNING RATES =====
                      # Phase 1: 2e-4 (high for fast early learning)
                      # Phase 2: 1e-4 (lower once bot is hitting ball)
                      # Phase 3: 0.8e-4 or lower (for advanced mechanics)
                      # Target clip fraction: ~0.08 (you have 0.1-0.2, too high!)
                      policy_lr=8e-5,                         # Lowered from 1.2e-4 to reduce clip fraction
                      critic_lr=8e-5,                         # Keep same as policy_lr
                      
                      # ===== NORMALIZATION =====
                      standardize_returns=True,
                      standardize_obs=True,
                      
                      # ===== CHECKPOINTING =====
                      save_every_ts=500_000,                # Save every 500k steps
                      checkpoint_load_folder=latest_checkpoint_dir,
                      
                      # ===== TRAINING DURATION =====
                      timestep_limit=10e15,                 # Effectively infinite - stop manually
                      
                      # ===== RENDERING =====
                      render=False,                         # Set to True to watch bot play (slows training)
                      render_delay=STEP_TIME,
                      
                      # ===== LOGGING =====
                      log_to_wandb=True)
    learner.learn()