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

# result = algo.train()
# result.keys() ->
# dict_keys(['num_env_steps_sampled', 
#   'num_module_steps_sampled', 'num_module_steps_sampled_lifetime', 
#   'num_agent_steps_sampled_lifetime', 'num_env_steps_sampled_lifetime', 'num_agent_steps_sampled', 
#   'episode_duration_sec_mean', 'num_episodes', 'episode_return_mean', 'num_episodes_lifetime', 'module_episode_returns_mean', 
#   'episode_len_max', 'episode_len_mean', 'episode_len_min', 
#   'episode_return_max', 'episode_return_min', 
#   'agent_steps', 'agent_episode_returns_mean'])

# result["learners"]["archer_0"].keys() ->
# dict_keys(['mean_kl_loss', 'num_trainable_parameters', 'default_optimizer_learning_rate', 
#   'gradients_default_optimizer_global_norm', 'vf_loss_unclipped', 'vf_loss', 'total_loss', 
#   'curr_entropy_coeff', 'num_non_trainable_parameters', 'policy_loss', 
#   'diff_num_grad_updates_vs_sampler_policy', 'module_train_batch_size_mean', 
#   'weights_seq_no', 'num_module_steps_trained_lifetime', 'vf_explained_var', 
#   'curr_kl_coeff', 'entropy', 'num_module_steps_trained'])



class CustomWrapper(BaseWrapper):
    """
    An example of a custom wrapper that flattens the symbolic vector state of the environment.

    Wrappers are useful to do state pre-processing (e.g. feature engineering) that does not need to be learned by the agent.
    """

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        # current_angle ∈ [-180, 180]
        # zombies: 4 × [target[0] ∈ [0, 1], angle ∈ [-180, 180]]
        low = np.array([-180], dtype=np.float32)
        high = np.array([180], dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)
    
    def get_angle(self, x, y):
        return (math.atan2(y, x) * 180 / math.pi - 90) % 360 - 180
    
    def angular_difference_degrees(self, a, b):
        diff = (a - b) % 360
        if diff > 180:
            diff -= 360
        return diff

    def observe(self, agent: AgentID) -> ObsType | None:
        obs = super().observe(agent)
        cur_agent = obs[1]
        zombies = obs[-4:]

        current_angle = self.get_angle(cur_agent[4]-1, cur_agent[3])
        # zombies = [[target[2] + cur_agent[2], self.get_angle(target[2], target[1])] if target[0] != 0 else [1, 0] for target in zombies]
        # zombies = [[target[0], self.angular_difference_degrees(current_angle, target[1])] for target in zombies]
        # zombies = np.array(zombies).flatten()

        zombie = [self.angular_difference_degrees(current_angle, self.get_angle(zombies[0][2], zombies[0][1]))]

        return np.array(zombie)
    
    # def step(self, action):
    #     super().step(action)
    #     agent = self.agent_selection

    #     if agent == 'archer_0':
    #         obs = self.observe(agent)

    #         if agent in self.rewards and abs(obs[0]) > 20 and abs(obs[0]) < 25 and action == 4:
    #             self.rewards[agent] += 0.05



class CustomPredictFunction(Callable):
    """ This is an example of an instantiation of the CustomPredictFunction that loads a trained RLLib algorithm from
    a checkpoint and extract the policies from it"""

    def __init__(self, env):

        # Here you should load your trained model(s) from a checkpoint in your folder
        best_checkpoint = (Path("ppo_results") / "learner_group" / "learner" / "rl_module").resolve()
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
            train_batch_size=512,
            lr=1e-4,
            gamma=0.99,
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
    policies = [x for x in env.agents]
    policies_to_train = policies
    config = algo_config(id_env, policies, policies_to_train)

    # Prepare live plotting
    plt.ion()
    fig, (ax_reward, ax_steps) = plt.subplots(2, 1, figsize=(8, 10))

    rewards = []
    steps = []

    reward_line, = ax_reward.plot(rewards, label="archer_0 reward")
    ax_reward.set_xlabel("Iteration")
    ax_reward.set_ylabel("Mean Episode Reward")
    ax_reward.set_title("Live Training Rewards")
    ax_reward.legend()
    ax_reward.grid(True)

    steps_line, = ax_steps.plot(steps, label="archer_0 steps", color='orange')
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
            reward = result["env_runners"]["agent_episode_returns_mean"]["archer_0"]
            step = result["env_runners"]["agent_steps"]["archer_0"]
            print(f"Iteration {i} - archer_0 reward: {reward}, steps: {step}")
            rewards.append(reward)
            steps.append(step)

            # Update reward plot
            reward_line.set_ydata(rewards)
            reward_line.set_xdata(range(len(rewards)))
            ax_reward.relim()
            ax_reward.autoscale_view()

            # Update steps plot
            steps_line.set_ydata(steps)
            steps_line.set_xdata(range(len(steps)))
            ax_steps.relim()
            ax_steps.autoscale_view()

            plt.pause(0.01)

            if reward > 40:  # Early stopping criterion
                break

        if i % 5 == 0:
            save_result = algo.save(checkpoint_path)
            path_to_checkpoint = save_result.checkpoint.path
            print(
                "An Algorithm checkpoint has been created inside directory: "
                f"'{path_to_checkpoint}'."
            )

    plt.ioff()
    fig.tight_layout()
    plt.savefig("learning_curves/training_rewards_and_steps.png")



if __name__ == "__main__":

    num_agents = 1
    visual_observation = False

    # Create the PettingZoo environment for training
    env = create_environment(num_agents=num_agents, visual_observation=visual_observation)
    env = CustomWrapper(env)

    # Running training routine
    checkpoint_path = str(Path("ppo_results").resolve())
    training(env, checkpoint_path, max_iterations = 1000)
