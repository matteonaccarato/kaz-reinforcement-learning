import gymnasium
import math
import numpy as np
from gymnasium import spaces
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType

# actions
# 0: forward
# 1: backward
# 2: turn left
# 3: turn right
# 4: shoot


class CustomWrapper(BaseWrapper):
    """
    An example of a custom wrapper that flattens the symbolic vector state of the environment.

    Wrappers are useful to do state pre-processing (e.g. feature engineering) that does not need to be learned by the agent.
    """

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        print("observation space: ", super().observation_space(agent))
        print("flattened observation space: ", spaces.flatten_space(super().observation_space(agent)))
        return spaces.flatten_space(super().observation_space(agent))
    
    def get_angle(self, x, y):
        return (math.atan2(y, x) * 180 / math.pi - 90) % 360 - 180

    def observe(self, agent: AgentID) -> ObsType | None:
        # obs = super().observe(agent)
        # cur_agent = obs[1]
        # zombies = obs[-4:]
        # return cur_agent + zombies
    
        obs = super().observe(agent)
        cur_agent = obs[1]
        zombies = obs[-4:]

        current_angle = np.array([self.get_angle(cur_agent[4]-1, cur_agent[3])])
        zombies = [([target[0], self.get_angle(target[2], target[1])] if target[0] != 0 else [1, 0]) for target in zombies]
        zombies = np.array(zombies).flatten()

        print("current angle: ", current_angle)
        print("zombies: ", zombies)

        return np.concatenate([current_angle, zombies])


class CustomPredictFunction:
    """A random archer agent."""

    def __init__(self, env):
        self.env = env

    def angular_difference_degrees(self, a, b):
        diff = (a - b) % 360
        if diff > 180:
            diff -= 360
        return diff

    def __call__(self, observation, agent, *args, **kwargs):
        print("observation: ", observation)
        return self.env.action_space(agent).sample()
        # use as target the zombie with max 3rd element of the observation if different from 0
        # cur_agent = observation[0]
        # eligible_zombies = [zombie for zombie in observation[-4:] if zombie[0] != 0]
        # if not eligible_zombies:
        #     return self.env.action_space(agent).sample()
        
        # target = max(eligible_zombies, key=lambda x: x[2])

        # # compute needed angle to hit the target (coordinates are target[1], target[2])
        # current_angle = (math.atan2(cur_agent[4]-1, cur_agent[3]) * 180 / math.pi - 90) % 360 - 180
        # target_angle = (math.atan2(target[2] + 0.25, target[1]) * 180 / math.pi - 90) % 360 - 180

        # angle_difference = self.angular_difference_degrees(current_angle, target_angle)

        # if abs(angle_difference) > 5:  # threshold for adjustment
        #     if angle_difference > 0: return 2  # turn right
        #     else:                    return 3  # turn left

        # return 4  # shoot
