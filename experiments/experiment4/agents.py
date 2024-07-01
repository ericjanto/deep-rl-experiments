import os
import gymnasium as gym
import numpy as np
from torch.optim import Adam
from typing import Dict, Iterable
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Normal

from experiments.experiment3.agents import Agent
from experiments.experiment3.networks import FCNetwork
from experiments.experiment3.replay import Transition


class DiagGaussian(torch.nn.Module):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def sample(self):
        eps = Variable(torch.randn(*self.mean.size()))
        return self.mean + self.std * eps


class DDPG(Agent):
    """ DDPG
        :attr critic (FCNetwork): fully connected critic network
        :attr critic_optim (torch.optim): PyTorch optimiser for critic network
        :attr policy (FCNetwork): fully connected actor network for policy
        :attr policy_optim (torch.optim): PyTorch optimiser for actor network
        :attr gamma (float): discount rate gamma
        """

    def __init__(
            self,
            action_space: gym.Space,
            observation_space: gym.Space,
            gamma: float,
            critic_learning_rate: float,
            policy_learning_rate: float,
            critic_hidden_size: Iterable[int],
            policy_hidden_size: Iterable[int],
            tau: float,
            **kwargs,
    ):
        """
        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        :param gamma (float): discount rate gamma
        :param critic_learning_rate (float): learning rate for critic optimisation
        :param policy_learning_rate (float): learning rate for policy optimisation
        :param critic_hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected critic
        :param policy_hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected policy
        :param tau (float): step for the update of the target networks
        """
        super().__init__(action_space, observation_space)
        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.shape[0]

        self.upper_action_bound = action_space.high[0]
        self.lower_action_bound = action_space.low[0]

        # ######################################### #
        #  BUILD YOUR NETWORKS AND OPTIMIZERS HERE  #
        # ######################################### #
        # self.actor = Actor(STATE_SIZE, policy_hidden_size, ACTION_SIZE)
        self.actor = FCNetwork(
            (STATE_SIZE, *policy_hidden_size, ACTION_SIZE), output_activation=torch.nn.Tanh
        )
        self.actor_target = FCNetwork(
            (STATE_SIZE, *policy_hidden_size, ACTION_SIZE), output_activation=torch.nn.Tanh
        )

        self.actor_target.hard_update(self.actor)
        # self.critic = Critic(STATE_SIZE + ACTION_SIZE, critic_hidden_size)
        # self.critic_target = Critic(STATE_SIZE + ACTION_SIZE, critic_hidden_size)

        self.critic = FCNetwork(
            (STATE_SIZE + ACTION_SIZE, *critic_hidden_size, 1), output_activation=None
        )
        self.critic_target = FCNetwork(
            (STATE_SIZE + ACTION_SIZE, *critic_hidden_size, 1), output_activation=None
        )
        self.critic_target.hard_update(self.critic)

        self.policy_optim = Adam(self.actor.parameters(), lr=policy_learning_rate, eps=1e-3)
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_learning_rate, eps=1e-3)


        # ############################################# #
        # WRITE ANY HYPERPARAMETERS YOU MIGHT NEED HERE #
        # ############################################# #
        self.gamma = gamma
        self.critic_learning_rate = critic_learning_rate
        self.policy_learning_rate = policy_learning_rate
        self.tau = tau

        # ################################################### #
        # DEFINE A GAUSSIAN THAT WILL BE USED FOR EXPLORATION #
        # ################################################### #
        mean = torch.zeros(ACTION_SIZE)
        std = 0.1 * torch.ones(ACTION_SIZE)
        # self.noise = DiagGaussian(mean, std)
        self.noise = Normal(mean, std)

        # ############################### #
        # WRITE ANY AGENT PARAMETERS HERE #
        # ############################### #

        self.saveables.update(
            {
                "actor": self.actor,
                "actor_target": self.actor_target,
                "critic": self.critic,
                "critic_target": self.critic_target,
                "policy_optim": self.policy_optim,
                "critic_optim": self.critic_optim,
            }
        )


    def save(self, path: str, suffix: str = "") -> str:
        """Saves saveable PyTorch models under given path

        The models will be saved in directory found under given path in file "models_{suffix}.pt"
        where suffix is given by the optional parameter (by default empty string "")

        :param path (str): path to directory where to save models
        :param suffix (str, optional): suffix given to models file
        :return (str): path to file of saved models file
        """
        torch.save(self.saveables, path)
        return path


    def restore(self, filename: str, dir_path: str = None):
        """Restores PyTorch models from models file given by path

        :param filename (str): filename containing saved models
        :param dir_path (str, optional): path to directory where models file is located
        """

        if dir_path is None:
            dir_path = os.getcwd()
        save_path = os.path.join(dir_path, filename)
        checkpoint = torch.load(save_path)
        for k, v in self.saveables.items():
            v.load_state_dict(checkpoint[k].state_dict())


    def schedule_hyperparameters(self, timestep: int, max_timesteps: int):
        """Updates the hyperparameters
        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ### PUT YOUR CODE HERE ###
        min_policy_learning_rate = 1e-4
        min_critic_learning_rate = 1e-4

        # Exponential decay factor
        decay_factor = 0.99

        # Exponentially decay the learning rates
        self.policy_learning_rate = max(self.policy_learning_rate * decay_factor ** (timestep / max_timesteps), min_policy_learning_rate)
        self.critic_learning_rate = max(self.critic_learning_rate * decay_factor ** (timestep / max_timesteps), min_critic_learning_rate)


    def act(self, obs: np.ndarray, explore: bool):
        """Returns an action (should be called at every timestep)
        When explore is False you should select the best action possible (greedy). However, during exploration,
        you should be implementing exporation using the self.noise variable that you should have declared in the __init__.
        Use schedule_hyperparameters() for any hyperparameters that you want to change over time.

        :param obs (np.ndarray): observation vector from the environment
        :param explore (bool): flag indicating whether we should explore
        :return (sample from self.action_space): action the agent should perform
        """
        ### PUT YOUR CODE HERE ###
        obs = torch.Tensor(obs).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs).squeeze().numpy()

        if explore:
            noise = self.noise.sample().item()
            action += noise

        # Clip the action to the action space bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)        

        return action


    def update(self, batch: Transition) -> Dict[str, float]:
        """Update function for DQN
        This function is called after storing a transition in the replay buffer. This happens
        every timestep. It should update your critic and actor networks, target networks with soft
        updates, and return the q_loss and the policy_loss in the form of a dictionary.

        :param batch (Transition): batch vector from replay buffer
        :return (Dict[str, float]): dictionary mapping from loss names to loss values
        """
        ### PUT YOUR CODE HERE ###
        # Critic network update:
        # L(θ) = (r + γ(1-d_t)Q(μ(s_t+1;\pho'), s_t+1;θ') - Q(a_t, s_t;θ))^2.

        # Actor network update:
        # L(θ) = 1/N * Σ^N_i -Q(s_i,μ(s_i;\pho);θ)).

        # Update critic
        # μ(s_t+1;\pho')
        next_actions = self.actor_target(batch.next_states)
        next_state_actions = torch.cat([batch.next_states, next_actions], dim=-1)

        # Q(next_actions, s_t+1;θ')
        # next_q_values = self.critic_target(batch.next_states, next_actions)
        next_q_values = self.critic_target(next_state_actions)


        # r + γ(1-d_t) * next_q_values
        target_q_values = batch.rewards + self.gamma * (1 - batch.done) * next_q_values

        q_values = self.critic(torch.cat([batch.states, batch.actions], dim=-1))

        # L(θ) = (q_values - target_q_values)^2
        q_loss = F.mse_loss(q_values, target_q_values.detach())

        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.step()

        # Update actor
        # μ(s_t;\pho)
        actions_pred = self.actor(batch.states)

        # 1/N \sum -Q(s_t, actions_pred;θ)
        p_loss = -self.critic(torch.cat([batch.states, actions_pred], dim=-1)).mean()

        self.policy_optim.zero_grad()
        p_loss.backward()
        self.policy_optim.step()

        # Soft update of target networks
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            # copy_ updates target_param in place
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)

        return {
            "q_loss": q_loss.item(),
            "p_loss": p_loss.item(),
        }