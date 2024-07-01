from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Dict, Optional, Hashable

from experiments.constants import EX1_CONSTANTS as CONSTANTS
from experiments.experiment1.mdp import MDP, Transition, State, Action


class MDPSolver(ABC):
    """Base class for MDP solvers

    :attr mdp (MDP): MDP to solve
    :attr gamma (float): discount factor gamma to use
    :attr action_dim (int): number of actions in the MDP
    :attr state_dim (int): number of states in the MDP
    """

    def __init__(self, mdp: MDP, gamma: float):
        """Constructor of MDPSolver

        Initialises some variables from the MDP, namely the state and action dimension variables

        :param mdp (MDP): MDP to solve
        :param gamma (float): discount factor (gamma)
        """
        self.mdp: MDP = mdp
        self.gamma: float = gamma

        self.action_dim: int = len(self.mdp.actions)
        self.state_dim: int = len(self.mdp.states)

    def decode_policy(self, policy: Dict[int, np.ndarray]) -> Dict[State, Action]:
        """Generates greedy, deterministic policy dict

        Given a stochastic policy from state indeces to distribution over actions, the greedy,
        deterministic policy is generated choosing the action with highest probability

        :param policy (Dict[int, np.ndarray of float with dim (num of actions)]):
            stochastic policy assigning a distribution over actions to each state index
        :return (Dict[State, Action]): greedy, deterministic policy from states to actions
        """
        new_p = {}
        for state, state_idx in self.mdp._state_dict.items():
            new_p[state] = self.mdp.actions[np.argmax(policy[state_idx])]
        return new_p

    @abstractmethod
    def solve(self):
        """Solves the given MDP"""
        ...


class ValueIteration(MDPSolver):
    """MDP solver using the Value Iteration algorithm"""

    def _calc_value_func(self, theta: float) -> np.ndarray:
        """Calculates the value function

        Useful Variables:
        1. `self.mpd` -- Gives access to the MDP.
        2. `self.mdp.R` -- 3D NumPy array with the rewards for each transition.
            E.g. the reward of transition [3] -2-> [4] (going from state 3 to state 4 with action
            2) can be accessed with `self.R[3, 2, 4]`
        3. `self.mdp.P` -- 3D NumPy array with transition probabilities.
            *REMEMBER*: the sum of (STATE, ACTION, :) should be 1.0 (all actions lead somewhere)
            E.g. the transition probability of transition [3] -2-> [4] (going from state 3 to
            state 4 with action 2) can be accessed with `self.P[3, 2, 4]`

        :param theta (float): theta is the stop threshold for value iteration
        :return (np.ndarray of float with dim (num of states)):
            1D NumPy array with the values of each state.
            E.g. V[3] returns the computed value for state 3
        """
        V = np.zeros(self.state_dim)
        ### PUT YOUR CODE HERE ###
        
        print("Beginning value iteration for given MDP...\n")
        iteration = 0
        while True:
            print(f"Iteration: {iteration} \t Current Values: {V}")
            delta = 0
            initial_values = V
            V = np.zeros_like(V)

            for state in range(self.state_dim):
                action_values = np.zeros(self.action_dim)
                for action in range(self.action_dim):
                    next_state_probabilities = self.mdp.P[state, action, :]
                    next_states = next_state_probabilities.nonzero()[0]

                    for next_state in next_states:
                        r = self.mdp.R[state, action, next_state]
                        action_values[action] += (
                            next_state_probabilities[next_state]
                            * (r + self.gamma * initial_values[next_state])
                        )

                V[state] = np.max(action_values)
                delta = max(delta, abs(initial_values[state] - V[state]))

            if delta < theta:
                print(
                    f"\nMax difference in state value from previous iteration = {delta} which is less than threshold {theta}. Value Iteration terminating...\n"
                )
                break

            iteration += 1
        return V

    def _calc_policy(self, V: np.ndarray) -> np.ndarray:
        """Calculates the policy

        :param V (np.ndarray of float with dim (num of states)):
            A 1D NumPy array that encodes the computed value function (from _calc_value_func(...))
            It is indexed as (State) where V[State] is the value of state 'State'
        :return (np.ndarray of float with dim (num of states, num of actions):
            A 2D NumPy array that encodes the calculated policy.
            It is indexed as (STATE, ACTION) where policy[STATE, ACTION] has the probability of
            taking action 'ACTION' in state 'STATE'.
            REMEMBER: the sum of policy[STATE, :] should always be 1.0
            For deterministic policies the following holds for each state S:
            policy[S, BEST_ACTION] = 1.0
            policy[S, OTHER_ACTIONS] = 0
        """
        policy = np.zeros([self.state_dim, self.action_dim])
        ### PUT YOUR CODE HERE ###
        
        print("Calculating policy based on value function...\n")

        for state in range(self.state_dim):
            action_values = np.zeros(self.action_dim)
            for action in range(self.action_dim):
                next_state_probabilities = self.mdp.P[state, action, :]
                next_states = next_state_probabilities.nonzero()[0]

                for next_state in next_states:
                    r = self.mdp.R[state, action, next_state]
                    action_values[action] += (
                        next_state_probabilities[next_state]
                        * (r + self.gamma * V[next_state])
                    )

            best_action = np.argmax(action_values)
            policy[state][best_action] = 1.0

        print(policy)

        return policy

    def solve(self, theta: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """Solves the MDP

        Compiles the MDP and then calls the calc_value_func and
        calc_policy functions to return the best policy and the
        computed value function

        :param theta (float, optional): stop threshold, defaults to 1e-6
        :return (Tuple[np.ndarray of float with dim (num of states, num of actions),
                       np.ndarray of float with dim (num of states)):
            Tuple of calculated policy and value function
        """
        self.mdp.ensure_compiled()
        V = self._calc_value_func(theta)
        policy = self._calc_policy(V)

        return policy, V


class PolicyIteration(MDPSolver):
    """MDP solver using the Policy Iteration algorithm"""

    def _policy_eval(self, policy: np.ndarray) -> np.ndarray:
        """Computes one policy evaluation step

        :param policy (np.ndarray of float with dim (num of states, num of actions)):
            A 2D NumPy array that encodes the policy.
            It is indexed as (STATE, ACTION) where policy[STATE, ACTION] has the probability of
            taking action 'ACTION' in state 'STATE'.
            REMEMBER: the sum of policy[STATE, :] should always be 1.0
            For deterministic policies the following holds for each state S:
            policy[S, BEST_ACTION] = 1.0
            policy[S, OTHER_ACTIONS] = 0
        :return (np.ndarray of float with dim (num of states)):
            A 1D NumPy array that encodes the computed value function
            It is indexed as (State) where V[State] is the value of state 'State'
        """
        V = np.zeros(self.state_dim)
        ### PUT YOUR CODE HERE ###

        # Understanding: Policy evaluation calculates the state values given a
        # policy. The state values are the expected return from a state given a
        # policy.

        num_states, num_actions = policy.shape  # 3, 2

        print("Beginning policy evaluation for given policy and MDP...\n")
        iteration = 0

        while True:
            print(f"Iteration: {iteration} \t Current Values: {V}")
            delta = 0
            initial_values = V  # values from prev iteration
            # We reset V since we overwrite based on previous V values:
            V = np.zeros_like(V)
            for state in range(num_states):
                for action in range(num_actions):
                    next_state_probabilities = self.mdp.P[state, action, :]
                    next_states = next_state_probabilities.nonzero()[0]
                    # print(
                    #     f"""Next states for state {self.mdp._get_state_label(state)}
                    #     and action {self.mdp._get_action_label(action)}:
                    #     {list(map(self.mdp._get_state_label, next_states))}"""
                    # )

                    for next_state in next_states:
                        r = self.mdp.R[state, action, next_state]
                        V[state] += (
                            policy[state][action]
                            * next_state_probabilities[next_state]
                            * (r + self.gamma * initial_values[next_state])
                        )

                print(V[state])
                delta = max(delta, abs(initial_values[state] - V[state]))

            if delta < self.theta:
                # print(
                #     f"\nMax difference in state value from previous iteration = {delta} which is less than threshold {self.theta}. Policy Evaluation terminating...\n"
                # )
                break

            iteration += 1

        # print(f"Final state values: {V}")
        return V

    def _policy_improvement(self) -> Tuple[np.ndarray, np.ndarray]:
        """Computes policy iteration until a stable policy is reached

        Useful Variables (As with Value Iteration):
        1. `self.mpd` -- Gives access to the MDP.
        2. `self.mdp.R` -- 3D NumPy array with the rewards for each transition.
            E.g. the reward of transition [3] -2-> [4] (going from state 3 to state 4 with action
            2) can be accessed with `self.R[3, 2, 4]`
        3. `self.mdp.P` -- 3D NumPy array with transition probabilities.
            *REMEMBER*: the sum of (STATE, ACTION, :) should be 1.0 (all actions lead somewhere)
            E.g. the transition probability of transition [3] -2-> [4] (going from state 3 to
            state 4 with action 2) can be accessed with `self.P[3, 2, 4]`

        :return (Tuple[np.ndarray of float with dim (num of states, num of actions),
                       np.ndarray of float with dim (num of states)):
            Tuple of calculated policy and value function
        """
        policy = np.zeros([self.state_dim, self.action_dim])
        # policy (np.ndarray of float with dim (num of states, num of actions)):
        #     A 2D NumPy array that encodes the policy.
        #     It is indexed as (STATE, ACTION) where policy[STATE, ACTION] has the probability of
        #     taking action 'ACTION' in state 'STATE'.
        #     REMEMBER: the sum of policy[STATE, :] should always be 1.0
        #     For deterministic policies the following holds for each state S:
        #     policy[S, BEST_ACTION] = 1.0
        #     policy[S, OTHER_ACTIONS] = 0

        V = np.zeros([self.state_dim])
        ### PUT YOUR CODE HERE ###

        # Perturb the policy to be stochastic. This is needed because when
        # initialising the policy to zeros (as given above), there is a chance
        # that taking action 0 is always the best action.
        # Then, if the policy is already optimal, the value function will not
        # change from its initial value (premature convergence).
        for state in range(self.state_dim):
            policy[state] = np.random.dirichlet(np.ones(self.action_dim), size=1)

        # state_action_values holds the value of each state and action pair.
        # This is where we maximise the action over.
        # Shape is (num_states, num_actions), where
        # state_action_values[state][action] holds the value of the state and action.
        state_action_values = np.nan * policy

        valid_actions = np.zeros([self.state_dim, self.action_dim])
        for state in range(self.state_dim):
            for action in range(self.action_dim):
                if np.sum(self.mdp.P[state, action, :]) > 0:
                    valid_actions[state, action] = 1

        i = 0
        while True:
            print(f"Policy Iteration: {i}")
            policy_stable = True
            V = self._policy_eval(policy)
            for state in range(self.state_dim):
                # There is only one action with probability 1.0
                old_action = np.nanargmax(policy[state])
                for action in range(self.action_dim):
                    if valid_actions[state, action] == 1:
                        state_action_values[state][action] = 0.0

                        next_state_probabilities = self.mdp.P[state, action, :]
                        next_states = next_state_probabilities.nonzero()[0]
                        for next_state in next_states:
                            state_action_values[state][action] += (
                                next_state_probabilities[next_state]
                                * (
                                    self.mdp.R[state, action, next_state]
                                    + self.gamma * V[next_state]
                                )
                            )

                greedy_action = np.nanargmax(state_action_values[state])

                for action in range(self.action_dim):
                    policy[state][action] = 1.0 if action == greedy_action else 0.0

                # print(f"{self.mdp._get_action_label(old_action)} vs {self.mdp._get_action_label(greedy_action)}\n")
                # print(old_action)
                if old_action != greedy_action:
                    policy_stable = False

            # print(
            #     f"State action values with previous policy:\n {state_action_values}\n"
            # )
            # print(f"Greedy policy after policy improvement:\n {policy}")

            if policy_stable:
                # print("Policy stable. Terminating policy iteration...")
                break

            i += 1

        return policy, V

    def solve(self, theta: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """Solves the MDP

        This function compiles the MDP and then calls the
        policy improvement function that the student must implement
        and returns the solution

        :param theta (float, optional): stop threshold, defaults to 1e-6
        :return (Tuple[np.ndarray of float with dim (num of states, num of actions),
                       np.ndarray of float with dim (num of states)]):
            Tuple of calculated policy and value function
        """
        self.mdp.ensure_compiled()
        self.theta = theta
        return self._policy_improvement()


if __name__ == "__main__":
    mdp = MDP()
    mdp.add_transition(
        #         start action end prob reward
        Transition("rock0", "jump0", "rock0", 1, 0),
        Transition("rock0", "stay", "rock0", 1, 0),
        Transition("rock0", "jump1", "rock0", 0.1, 0),
        Transition("rock0", "jump1", "rock1", 0.9, 0),
        Transition("rock1", "jump0", "rock1", 0.1, 0),
        Transition("rock1", "jump0", "rock0", 0.9, 0),
        Transition("rock1", "jump1", "rock1", 0.1, 0),
        Transition("rock1", "jump1", "land", 0.9, 10),
        Transition("rock1", "stay", "rock1", 1, 0),
        Transition("land", "stay", "land", 1, 0),
        Transition("land", "jump0", "land", 1, 0),
        Transition("land", "jump1", "land", 1, 0),
    )

    # jump0: jump left
    # jump1: jump right
    # stay: stay in place

    # intuitively, the policy should be to jump right at rock0, jump right at
    # rock1, and any action at land (since it's a terminal state).

    # rock1 is the only state from which a reward can be achieved, so the value
    # function should be highest at rock1.

    solver = ValueIteration(mdp, CONSTANTS["gamma"])
    policy, valuefunc = solver.solve()
    print("---Value Iteration---")
    print("Policy:")
    print(solver.decode_policy(policy))
    print("Value Function")
    print(valuefunc)

    solver = PolicyIteration(mdp, CONSTANTS["gamma"])
    policy, valuefunc = solver.solve()
    print("---Policy Iteration---")
    print("Policy:")
    print(solver.decode_policy(policy))
    print("Value Function")
    print(valuefunc)
