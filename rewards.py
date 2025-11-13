import numpy as np
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData

class InAirReward(RewardFunction): # We extend the class "RewardFunction"
    # Empty default constructor (required)
    def __init__(self):
        super().__init__()

    # Called when the game resets (i.e. after a goal is scored)
    def reset(self, initial_state: GameState):
        pass # Don't do anything when the game resets

    # Get the reward for a specific player, at the current state
    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        
        # "player" is the current player we are getting the reward of
        # "state" is the current state of the game (ball, all players, etc.)
        # "previous_action" is the previous inputs of the player (throttle, steer, jump, boost, etc.) as an array
        
        if not player.on_ground:
            # We are in the air! Return full reward
            return 1
        else:
            # We are on ground, don't give any reward
            return 0
        

# Import CAR_MAX_SPEED from common game values
from rlgym_sim.utils.common_values import CAR_MAX_SPEED

class SpeedTowardBallReward(RewardFunction):
    # Default constructor
    def __init__(self):
        super().__init__()

    # Do nothing on game reset
    def reset(self, initial_state: GameState):
        pass

    # Get the reward for a specific player, at the current state
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # Velocity of our player
        player_vel = player.car_data.linear_velocity
        
        # Difference in position between our player and the ball
        # When getting the change needed to reach B from A, we can use the formula: (B - A)
        pos_diff = (state.ball.position - player.car_data.position)
        
        # Determine the distance to the ball
        # The distance is just the length of pos_diff
        dist_to_ball = np.linalg.norm(pos_diff)
        
        # We will now normalize our pos_diff vector, so that it has a length/magnitude of 1
        # This will give us the direction to the ball, instead of the difference in position
        # Normalizing a vector can be done by dividing the vector by its length
        dir_to_ball = pos_diff / dist_to_ball

        # Use a dot product to determine how much of our velocity is in this direction
        # Note that this will go negative when we are going away from the ball
        speed_toward_ball = np.dot(player_vel, dir_to_ball)
        
        if speed_toward_ball > 0:
            # We are moving toward the ball at a speed of "speed_toward_ball"
            # The maximum speed we can move toward the ball is the maximum car speed
            # We want to return a reward from 0 to 1, so we need to divide our "speed_toward_ball" by the max player speed
            reward = speed_toward_ball / CAR_MAX_SPEED
            return reward
        else:
            # We are not moving toward the ball
            # Many good behaviors require moving away from the ball, so I highly recommend you don't punish moving away
            # We'll just not give any reward
            return 0


class HandbrakePenalty(RewardFunction):
    """
    Penalizes excessive handbrake usage to encourage smoother driving.
    The bot needs to learn proper steering instead of relying on handbrake.
    """
    def __init__(self, penalty_weight=0.5):
        super().__init__()
        self.penalty_weight = penalty_weight

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # previous_action format: [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
        # handbrake is at index 7 (0-indexed)
        
        if len(previous_action) > 7:
            handbrake = previous_action[7]
            
            # If handbrake is being used, give negative reward
            if handbrake > 0.5:  # Binary action, usually 0 or 1
                return -self.penalty_weight
        
        # No penalty if not using handbrake
        return 0


class FlipDisciplineReward(RewardFunction):
    """
    Encourage strategic flip usage: 
    - Reward flips when very close to ball (attacking)
    - Allow flips when far from ball (recovering speed)
    - PENALIZE flips in mid-range (loses control during approach)
    """
    def __init__(self, close_distance=400, far_distance=2000, penalty=1.0):
        super().__init__()
        self.close_distance = close_distance  # Distance pour "attaquer" la balle
        self.far_distance = far_distance      # Distance pour "récupérer"
        self.penalty = penalty
        self.last_on_ground = True
        self.flip_detected = False
        
    def reset(self, initial_state: GameState):
        self.last_on_ground = True
        self.flip_detected = False
    
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # Détecter un flip : le joueur était au sol et saute maintenant (double saut)
        # previous_action: [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
        
        if len(previous_action) <= 5:
            return 0
        
        jump_action = previous_action[5]  # Index 5 = jump
        currently_on_ground = player.on_ground
        
        # Détecter transition sol -> air avec input jump = flip probable
        # Note: détection simplifiée - le vrai flip est double jump mais difficile à détecter
        # On pénalise les sauts en approche moyenne
        
        if not currently_on_ground and jump_action > 0.5:
            # Le joueur saute (potentiellement flip)
            pos_diff = state.ball.position - player.car_data.position
            dist_to_ball = np.linalg.norm(pos_diff)
            
            # Zone dangereuse : distance moyenne (perd contrôle)
            if self.close_distance < dist_to_ball < self.far_distance:
                # PÉNALITÉ : flip en approche = perte de contrôle
                return -self.penalty
            
            # Zones OK : très proche (attaque) ou très loin (récupération)
            # Pas de pénalité, comportement neutre
            return 0
        
        self.last_on_ground = currently_on_ground
        return 0


class SaveBoostReward(RewardFunction):
    """
    Rewards having boost to encourage collecting and not wasting it.
    Uses sqrt to make boost more valuable when you have less of it.
    """
    def __init__(self):
        super().__init__()
    
    def reset(self, initial_state: GameState):
        pass
    
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # player.boost_amount ranges from 0 to 1
        # Using sqrt makes low boost more important: 0->0.5 boost is more valuable than 0.5->1
        return np.sqrt(player.boost_amount)


class TouchStrengthReward(RewardFunction):
    """
    Scales touch reward based on how much the ball's velocity changed.
    Strong shots/touches give more reward than weak pushes.
    Prevents farming touches by constantly pushing the ball.
    """
    def __init__(self):
        super().__init__()
        self.previous_ball_vel = None
    
    def reset(self, initial_state: GameState):
        self.previous_ball_vel = initial_state.ball.linear_velocity.copy()
    
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = 0
        
        if player.ball_touched:
            # Calculate change in ball velocity
            current_ball_vel = state.ball.linear_velocity
            vel_change = np.linalg.norm(current_ball_vel - self.previous_ball_vel)
            
            # Normalize by max possible ball speed (around 6000 uu/s)
            MAX_BALL_SPEED = 6000
            reward = vel_change / MAX_BALL_SPEED
            
            # Clamp to [0, 1] range
            reward = min(reward, 1.0)
        
        # Update previous velocity for next step
        self.previous_ball_vel = state.ball.linear_velocity.copy()
        
        return reward
