import gymnasium
from gymnasium import spaces
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType
import math

# 0 : forward
# 1 : backward
# 2 : left
# 3 : right
# 4 : shoot

class CustomWrapper(BaseWrapper):
    """
    An example of a custom wrapper that flattens the symbolic vector state of the environment.

    Wrappers are useful to do state pre-processing (e.g. feature engineering) that does not need to be learned by the agent.
    """

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return spaces.flatten_space(super().observation_space(agent))

    def observe(self, agent: AgentID) -> ObsType | None:
        obs = super().observe(agent)
        flat_obs = obs.flatten()
        """
        Each row of the matrix (this is an 5 wide vector) has a breakdown that looks something like this:
            - The first value is the absolute distance between an entity and the current agent.
            - The next four values are relative position and absolute angles of each entity relative to the current agent.
        The first two values are position values relative to the current agent.
        The final two values are the angle of the entity represented as a directional unit vector relative to the world.
        """
        curr_agent = obs[1]
        zombies = obs[-4:]
        return curr_agent, zombies

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
        curr_agent, zombies = observation
        
        # curr agent ->    _ x y angle1 angle2
        # zombies    -> dist x y angle1 angle2

        elibible_zombies = [zombie for zombie in zombies if zombie[0] > 0]
        if not elibible_zombies:
            return 4
        target_zombie = min(elibible_zombies, key=lambda x: x[0])
        # print(f"target {target_zombie[0]}")

        current_angle = (math.atan2(curr_agent[4]-1, curr_agent[3]) * 180 / math.pi - 90) % 360 - 180
        target_angle = (math.atan2(target_zombie[2], target_zombie[1]) * 180 / math.pi - 90) % 360 - 180

        if target_angle < 155 and target_angle > 0:
            target_angle -= 15
        if target_angle > -155 and target_angle < 0:
            target_angle += 15

        angle_diff = self.angular_difference_degrees(current_angle, target_angle)

        if abs(angle_diff) > 3:
            # print(f"curr_angle {current_angle} target_angle {target_angle} angle_diff: {angle_diff}")
            if angle_diff > 0:
                return 2
            else:
                return 3    
        return 4