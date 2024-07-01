from abc import ABC, abstractmethod
from collections import defaultdict
import random
from typing import List, Dict, DefaultDict
from gymnasium.spaces import Space
from gymnasium.spaces.utils import flatdim


class Agent(ABC):
    """Base class for Q-Learning agent

    """

    def __init__(
        self,
        action_space: Space,
        obs_space: Space,
        gamma: float,
        epsilon: float,
        **kwargs
    ):
        """Constructor of base agent for Q-Learning

        Initializes basic variables of the Q-Learning agent
        namely the epsilon, learning rate and discount rate.

        :param action_space (int): action space of the environment
        :param obs_space (int): observation space of the environment
        :param gamma (float): discount factor (gamma)
        :param epsilon (float): epsilon for epsilon-greedy action selection

        :attr n_acts (int): number of actions
        :attr q_table (DefaultDict): table for Q-values mapping (obses[t], ACT) pairs of observations
            and actions to respective Q-values
        """

        self.action_space = action_space
        self.obs_space = obs_space
        self.n_acts = flatdim(action_space)

        self.epsilon: float = epsilon
        self.gamma: float = gamma

        self.q_table: DefaultDict = defaultdict(lambda: 0)

    def act(self, obs: int) -> int:
        """Implement the epsilon-greedy action selection here

        :param obs (int): received observation representing the current environmental state
        :return (int): index of selected action
        """
        ### PUT YOUR CODE HERE ###

        # The epsilon-greedy action selection works by selecting a random action with probability epsilon
        # and otherwise selecting the action with the highest Q-value for the current observation.
        # The Q-values for the current observation can be accessed using self.q_table[(obs, act)] for each action.

        if random.uniform(0, 1) <= self.epsilon:
            action = random.randint(0, self.n_acts - 1)
        else:
            action = max(list(range(self.n_acts)), key = lambda act: self.q_table[(obs, act)])
        return action

    @abstractmethod
    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ...

    @abstractmethod
    def learn(self):
        ...


class QLearningAgent(Agent):
    """Agent using the Q-Learning algorithm"""

    def __init__(self, alpha: float, **kwargs):
        """Constructor of QLearningAgent

        Initializes some variables of the Q-Learning agent, namely the epsilon, discount rate
        and learning rate alpha.

        :param alpha (float): learning rate alpha for Q-learning updates
        """

        super().__init__(**kwargs)
        self.alpha: float = alpha

    def learn(
        self, obs: int, action: int, reward: float, n_obs: int, done: bool
    ) -> float:
        """Updates the Q-table based on agent experience

        :param obses[t] (int): received observation representing the current environmental state
        :param action (int): index of applied action
        :param reward (float): received reward
        :param n_obs (int): received observation representing the next environmental state
        :param done (bool): flag indicating whether a terminal state has been reached
        :return (float): updated Q-value for current observation-action pair
        """
        ### PUT YOUR CODE HERE ###
        # The Q-value update for the Q-learning algorithm is given by:
        # Q(s, a) = Q(s, a) + alpha * (r + gamma * max_a(Q(s', a)) - Q(s, a))
        # where s is the current observation, a is the action, r is the reward, s' is the next observation,
        # and max_a(Q(s', a)) is the maximum Q-value for the next observation.
        if done:
            target = reward
        else:
            target = reward + self.gamma * max([self.q_table[(n_obs, act)] for act in range(self.n_acts)])

        self.q_table[(obs, action)] += self.alpha * (target - self.q_table[(obs, action)])
        return self.q_table[(obs, action)]

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        self.epsilon = 1.0 - (min(1.0, timestep / (0.20 * max_timestep))) * 0.99


class MonteCarloAgent(Agent):
    """Agent using the Monte-Carlo algorithm for training"""

    def __init__(self, **kwargs):
        """Constructor of MonteCarloAgent

        Initializes some variables of the Monte-Carlo agent, namely epsilon,
        discount rate and an empty observation-action pair dictionary.

        :attr sa_counts (Dict[(obses[t], Act), int]): dictionary to count occurrences observation-action pairs
        """
        super().__init__(**kwargs)
        self.sa_counts = {}

    def learn(
        self, obses: List[int], actions: List[int], rewards: List[float]
    ) -> Dict:
        """Updates the Q-table based on agent experience

        :param obses (List(int)): list of received observations representing environmental states
            of trajectory (in the order they were encountered)
        :param actions (List[int]): list of indices of applied actions in trajectory (in the
            order they were applied)
        :param rewards (List[float]): list of received rewards during trajectory (in the order
            they were received)
        :return (Dict): A dictionary containing the updated Q-value of all the updated state-action pairs
            indexed by the state action pair.
        """
        updated_values = {}
        ### PUT YOUR CODE HERE ###

        # TODO: this results in mean return = 0.0 for all episodes. need to
        # investigate why.

        def first_visit(obses, actions, t):
            prev = set((obses[i], actions[i]) for i in range(t))
            return (obses[t], actions[t]) not in prev

        G = 0
        for t in range(len(obses)-1, -1, -1):
            G = self.gamma * G + rewards[t]

            if first_visit(obses, actions, t):
                prev_return = self.q_table.get((obses[t], actions[t]), 0)
                self.sa_counts[(obses[t], actions[t])] = self.sa_counts.get((obses[t], actions[t]), 0) + 1
                updated_values[(obses[t], actions[t])] = prev_return + (G - prev_return) / self.sa_counts[(obses[t], actions[t])]
                self.q_table[(obses[t], actions[t])] = updated_values[(obses[t], actions[t])]

        return updated_values

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        self.epsilon = 1.0 - (min(1.0, timestep / (0.9 * max_timestep))) * 0.8
