from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Dict, Optional, Hashable

from rl2021.utils import MDP, Transition, State, Action


class MDPSolver(ABC):
    """Base class for MDP solvers

    **DO NOT CHANGE THIS CLASS**

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
        """Solves the given MDP
        """
        ...


class ValueIteration(MDPSolver):
    """MDP solver using the Value Iteration algorithm
    """

    def _calc_value_func(self, theta: float) -> np.ndarray:
        """Calculates the value function

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q1**

        **DO NOT ALTER THE MDP HERE**

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

        
        while True:
            delta = 0

            # Calculate the value for each state
            for state in range(0, self.state_dim):
                v = V[state] # save current value
                
                # Calculate the value for each action in this state (Bellman)
                cur_vs = np.zeros(self.action_dim) # array to save the values for each action in
                for action in range(0, self.action_dim):
                    cur_vs[action] = np.sum([self.mdp.P[state, action, future_state]*(self.mdp.R[state, action, future_state] + self.gamma*V[future_state]) for future_state in range(0, self.state_dim)])
                
                # Assign the maximum value as the new state value
                V[state] = max(cur_vs)

                # Calculate stopping criterion
                delta = max(delta, np.abs(v-V[state]))

            # Break the loop if converged enough
            if delta < theta: 
                break

        return V

    def _calc_policy(self, V: np.ndarray) -> np.ndarray:
        """Calculates the policy

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q1**

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

        # Get the deterministic policy for each state with matched action
        for state in range(0,self.state_dim):

            # Calculate the value for each action in this state (Bellman)
            cur_vs = np.zeros(self.action_dim) # array to save the values for each action in
            for action in range(0, self.action_dim):
                cur_vs[action] = np.sum([self.mdp.P[state, action, future_state]*(self.mdp.R[state, action, future_state] + self.gamma*V[future_state]) for future_state in range(0, self.state_dim)])

            policy[state, np.argmax(cur_vs)] = 1 # Deterministic policy to the most valuable action # TODO: Break Ties
        return policy

    def solve(self, theta: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """Solves the MDP

        Compiles the MDP and then calls the calc_value_func and
        calc_policy functions to return the best policy and the
        computed value function

        **DO NOT CHANGE THIS FUNCTION**

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
    """MDP solver using the Policy Iteration algorithm
    """

    def _policy_eval(self, policy: np.ndarray) -> np.ndarray:
        """Computes one policy evaluation step

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q1**

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
        
        while True:
            delta = 0

            # Change value for each state
            for state in range(0, self.state_dim):
                v = V[state] # save old state value

                #TODO use all actions instead of argmax
                action = np.argmax(policy[state,:]) # get action choosen by policy

                # Assign new value based on the action choosen
                V[state] = np.sum([self.mdp.P[state, action, future_state]*(self.mdp.R[state, action, future_state] + self.gamma*V[future_state]) for future_state in range(0, self.state_dim)])

                delta = max(0, np.abs(V[state]-v)) # update stopping criterion

            # Break the loop if converged enough
            if delta < solver.theta:
                break

        return np.array(V)

    def _policy_improvement(self) -> Tuple[np.ndarray, np.ndarray]:
        """Computes one policy improvement iteration

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q1**

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
        V = np.zeros([self.state_dim])
        
        while True:

            policy_stable = True

            # Improve the policy for each iteration
            for state in range(0, self.state_dim):
                old_action = np.argmax(policy[state,:]) # save the old optimal action to check convergence later

                # Update the values for each action given the state
                cur_vs = np.zeros(self.action_dim) # array to save the values for each action in
                for action in range(0, self.action_dim):
                    cur_vs[action] = np.sum([self.mdp.P[state, action, future_state]*(self.mdp.R[state, action, future_state] + self.gamma*V[future_state]) for future_state in range(0, self.state_dim)])

                # Update deterministic policy
                policy[state,:] = 0
                policy[state,np.argmax(cur_vs)] = 1

                # If the best action changed --> not stable
                if old_action != np.argmax(policy[state,:]):
                    policy_stable = False

            # Policy is converged if actions did not change
            if policy_stable:
                break
            
            # If the policy is not stable --> make policy evaluation
            else: 
                V = self._policy_eval(policy)

        return policy, V

        


        

    def solve(self, theta: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """Solves the MDP

        This function compiles the MDP and then calls the
        policy improvement function that the student must implement
        and returns the solution

        **DO NOT CHANGE THIS FUNCTION**

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
        Transition("high", "wait", "high", 1, 2),
        Transition("high", "search", "high", 0.8, 5),
        Transition("high", "search", "low", 0.2, 5),
        Transition("high", "recharge", "high", 1, 0),
        Transition("low", "recharge", "high", 1, 0),
        Transition("low", "wait", "low", 1, 2),
        Transition("low", "search", "high", 0.6, -3),
        Transition("low", "search", "low", 0.4, 5),
    )

    solver = ValueIteration(mdp, 0.9)
    policy, valuefunc = solver.solve()
    print("---Value Iteration---")
    print("Policy:")
    print(solver.decode_policy(policy))
    print("Value Function")
    print(valuefunc)

    solver = PolicyIteration(mdp, 0.9)
    policy, valuefunc = solver.solve()
    print("---Policy Iteration---")
    print("Policy:")
    print(solver.decode_policy(policy))
    print("Value Function")
    print(valuefunc)