from pathlib import Path
from typing import Callable

import gymnasium
import pettingzoo
import math
from gymnasium import spaces
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.rl_module import RLModule, MultiRLModule
from ray.rllib.core.rl_module import MultiRLModuleSpec, RLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv, ParallelPettingZooEnv
from ray.rllib.examples.rl_modules.classes.random_rlm import RandomRLModule
from ray.tune.registry import register_env
import numpy as np
import torch

import sys
import os
# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import create_environment

import matplotlib.pyplot as plt


FOLDER = "multiagent_ppo_results_full"
lr = "3e-4to1e-4to1e-6"
batch_size = 8000
sgd = 15
iterations = 1000        
class CustomWrapper(BaseWrapper):
    # This is an example of a custom wrapper that flattens the symbolic vector state of the environment
    # Wrapper are useful to inject state pre-processing or feature that does not need to be learned by the agent

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        raw_space = spaces.flatten_space(super().observation_space(agent))
        total_dim = raw_space.shape[0] + 4
        return spaces.Box(low=-1, high=1, shape=(total_dim,), dtype=np.float32)

    def angular_difference_degrees(self, a, b):
        diff = (a - b) % 2
        if diff > 1:
            diff -= 2
        return diff

    def observe(self, agent: AgentID) -> ObsType | None:
        obs = super().observe(agent)
        flat_obs = obs.flatten()

        cur_agent = obs[0]
        zombies = obs[-4:]

        current_angle = (math.atan2(cur_agent[4], cur_agent[3]) / math.pi)
        target_angles = [math.atan2(zombie[2], zombie[1]) / math.pi if zombie[0] > 0 else current_angle for zombie in zombies]
        angle_differences = [self.angular_difference_degrees(current_angle, target_angle) for target_angle in target_angles]

        return np.concatenate((flat_obs, angle_differences), dtype=np.float32)


class CustomPredictFunction(Callable):
    """ This is an example of an instantiation of the CustomPredictFunction that loads a trained RLLib algorithm from
    a checkpoint and extract the policies from it"""

    def __init__(self, env):

        # Here you should load your trained model(s) from a checkpoint in your folder
        best_checkpoint = (Path(FOLDER + "_lr" + str(lr) + "_bs" + str(batch_size) + "_sgd" + str(sgd) + "_iter" + str(iterations)) / "learner_group" / "learner" / "rl_module").resolve()
        self.modules = MultiRLModule.from_checkpoint(best_checkpoint)

    def __call__(self, observation, agent, *args, **kwargs):
        rl_module = self.modules[agent]
        fwd_ins = {"obs": torch.Tensor(observation).unsqueeze(0)}
        fwd_outputs = rl_module.forward_inference(fwd_ins)
        action_dist_class = rl_module.get_inference_action_dist_cls()
        action_dist = action_dist_class.from_logits(
            fwd_outputs["action_dist_inputs"]
        )
        action = action_dist.sample()[0].numpy()
        return action




    
def algo_config(id_env, policies, policies_to_train):
    
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(env=id_env, disable_env_checking=True)
        .env_runners(num_env_runners=4)
        .multi_agent(
            policies={x for x in policies},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
            policies_to_train=policies_to_train,
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    x: RLModuleSpec(module_class=PPOTorchRLModule, model_config={"fcnet_hiddens": [64, 64]})
                    if x in policies_to_train
                    else
                    RLModuleSpec(module_class=RandomRLModule)
                    for x in policies},
            ))
        .training(
            train_batch_size=batch_size,
            num_sgd_iter=sgd,
            lr=[
                (0, 3e-4),
                (iterations/2 * batch_size, 1e-4),
                (iterations * batch_size, 1e-6)
            ],
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
        )
        .debugging(log_level="ERROR")

    )

    return config


def training(env, checkpoint_path, max_iterations=500):
    # Translating the PettingZoo environment to an RLLib environment
    rllib_env = ParallelPettingZooEnv(pettingzoo.utils.conversions.aec_to_parallel(env))
    id_env = "knights_archers_zombies_v10"
    register_env(id_env, lambda config: rllib_env)

    # Fix seeds
    np.random.seed(42)
    torch.manual_seed(42)

    # Define the configuration for the PPO algorithm
    print("Policies: ", env.agents)
    policies = [x for x in env.agents]
    policies_to_train = policies
    config = algo_config(id_env, policies, policies_to_train)

    # Prepare live plotting
    plt.ion()
    fig, (ax_reward, ax_steps) = plt.subplots(2, 1, figsize=(4, 6))

    rewards_0 = []
    rewards_1 = []
    steps = []

    reward_line_0, = ax_reward.plot(rewards_0, label="archer_0 reward", color='blue')
    reward_line_1, = ax_reward.plot(rewards_1, label="archer_1 reward", color='red')
    ax_reward.set_xlabel("Iteration")
    ax_reward.set_ylabel("Mean Episode Reward")
    ax_reward.set_title("Live Training Rewards")
    ax_reward.legend()
    ax_reward.grid(True)

    steps_line_0, = ax_steps.plot(steps, label="archer steps", color='green')
    ax_steps.set_xlabel("Iteration")
    ax_steps.set_ylabel("Agent Steps")
    ax_steps.set_title("Live Training Steps")
    ax_steps.legend()
    ax_steps.grid(True)

    # Train the model
    algo = config.build()
    for i in range(max_iterations):
        result = algo.train()

        result.pop("config")
        if "env_runners" in result and "agent_episode_returns_mean" in result["env_runners"]:

            reward_0 = result["env_runners"]["agent_episode_returns_mean"]["archer_0"]
            reward_1 = result["env_runners"]["agent_episode_returns_mean"]["archer_1"]
            step = result["env_runners"]["agent_steps"]["archer_0"]

            rewards_0.append(reward_0)
            rewards_1.append(reward_1)
            steps.append(step)

            # Update reward plot
            reward_line_0.set_ydata(rewards_0)
            reward_line_1.set_ydata(rewards_1)
            reward_line_0.set_xdata(range(len(rewards_0)))
            reward_line_1.set_xdata(range(len(rewards_1)))
            ax_reward.relim()
            ax_reward.autoscale_view()

            # Update steps plot
            steps_line_0.set_ydata(steps)
            steps_line_0.set_xdata(range(len(steps)))
            ax_steps.relim()
            ax_steps.autoscale_view()

            plt.tight_layout()
            plt.pause(0.01)

        if i % 5 == 0:
            save_result = algo.save(checkpoint_path)
            path_to_checkpoint = save_result.checkpoint.path
            print(
                "An Algorithm checkpoint has been created inside directory: "
                f"'{path_to_checkpoint}'."
            )

    plt.ioff()
    fig.tight_layout()
    plt.savefig(f"learning_curves/{FOLDER}/training_rewards_and_steps_lr{lr}_bs{batch_size}_sgd{sgd}_iter{iterations}.png")



if __name__ == "__main__":

    num_agents = 2
    visual_observation = False

    # Create the PettingZoo environment for training
    env = create_environment(num_agents=num_agents, visual_observation=visual_observation)
    env = CustomWrapper(env)

    # Running training routine
    checkpoint_path = str(Path(FOLDER + "_lr" + str(lr) + "_bs" + str(batch_size) + "_sgd" + str(sgd) + "_iter" + str(iterations)).resolve())
    training(env, checkpoint_path, max_iterations = iterations)
