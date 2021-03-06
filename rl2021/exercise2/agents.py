from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np
import random
from typing import List, Dict, DefaultDict
from gym.spaces import Space
from gym.spaces.utils import flatdim


class Agent(ABC):
    """Base class for Q-Learning agent

    **ONLY CHANGE THE BODY OF THE act() FUNCTION**

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
        :attr q_table (DefaultDict): table for Q-values mapping (OBS, ACT) pairs of observations
            and actions to respective Q-values
        """

        self.action_space = action_space
        self.obs_space = obs_space
        self.n_acts = flatdim(action_space)

        self.epsilon: float = epsilon
        self.gamma: float = gamma

        self.q_table: DefaultDict = defaultdict(lambda: 0)

    def act(self, obs: np.ndarray) -> int:
        """Implement the epsilon-greedy action selection here

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q2**

        :param obs (np.ndarray of float with dim (observation size)):
            received observation representing the current environmental state
        :return (int): index of selected action
        """
        
        u = np.random.uniform(low = 0.0, high = 1.0, size = 1)

        if u <= self.epsilon:
            selected_action = np.random.randint(low = 0, high = self.n_acts-1)

        else:
            selected_action = np.argmax([self.q_table[(obs, action)] for action in range(self.n_acts)])
            if selected_action.size > 1:
                selected_action = np.random.choice(selected_action)
        return selected_action

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
    """Agent using the Q-Learning algorithm

    """

    def __init__(self, alpha: float, **kwargs):
        """Constructor of QLearningAgent

        Initializes some variables of the Q-Learning agent, namely the epsilon, discount rate
        and learning rate alpha.

        :param alpha (float): learning rate alpha for Q-learning updates
        """

        super().__init__(**kwargs)
        self.alpha: float = alpha

    def learn(
        self, obs: np.ndarray, action: int, reward: float, n_obs: np.ndarray, done: bool
    ) -> float:
        """Updates the Q-table based on agent experience

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q2**

        :param obs (np.ndarray of float with dim (observation size)):
            received observation representing the current environmental state
        :param action (int): index of applied action
        :param reward (float): received reward
        :param n_obs (np.ndarray of float with dim (observation size)):
            received observation representing the next environmental state
        :param done (bool): flag indicating whether a terminal state has been reached
        :return (float): updated Q-value for current observation-action pair
        """
        # TODO: Change Done implementation
        # TODO: Change Range for max_next-value
        max_next_value = np.max([self.q_table[(n_obs, action)] for action in range(self.n_acts)])
        target_value = reward + self.gamma*(1-done)*max_next_value 
        self.q_table[(obs, action)] = self.q_table[(obs, action)] + self.alpha*(target_value - self.q_table[(obs, action)])

        return self.q_table[(obs, action)]

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q2**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """

        """
        if timestep == 0:
            self.max_alpha = self.alpha
            self.min_alpha = 0.0001
            self.alpha_difference = self.max_alpha - self.min_alpha

        self.alpha = self.max_alpha - timestep/max_timestep * self.alpha_difference"""
        self.epsilon = 1.0-(min(1.0, timestep/(0.07*max_timestep)))*0.95
        #max_deduct, decay = 0.95, 0.07
        #self.epsilon =  1.0 - (min(1.0, timestep/(decay * max_timestep))) * max_deduct



class MonteCarloAgent(Agent):
    """Agent using the Monte-Carlo algorithm for training
    """

    def __init__(self, **kwargs):
        """Constructor of MonteCarloAgent

        Initializes some variables of the Monte-Carlo agent, namely epsilon,
        discount rate and an empty observation-action pair dictionary.

        :attr sa_counts (Dict[(Obs, Act), int]): dictionary to count occurrences observation-action pairs
        """
        super().__init__(**kwargs)
        self.sa_counts = {}
        self.returns = defaultdict(lambda: [])

    def learn(
        self, obses: List[np.ndarray], actions: List[int], rewards: List[float]
    ) -> Dict:
        """Updates the Q-table based on agent experience

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q2**

        :param obses (List(np.ndarray) with numpy arrays of float with dim (observation size)):
            list of received observations representing environmental states of trajectory (in
            the order they were encountered)
        :param actions (List[int]): list of indices of applied actions in trajectory (in the
            order they were applied)
        :param rewards (List[float]): list of received rewards during trajectory (in the order
            they were received)
        :return (Dict): A dictionary containing the updated Q-value of all the updated state-action pairs
            indexed by the state action pair.
        """
        updated_values = {}
        #returns = defaultdict(lambda: [])
        obses = np.asarray(obses)
        actions = np.asarray(actions)
        
        g = 0
        for t in range(len(obses)-2, 0, -1):
            g = self.gamma * g + rewards[t+1]

            # Gets indices where current action and state appeared earlier
            state_id = np.where(obses[0:t] == obses[t])[0]
            act_id = np.where(actions[0:t] == actions[t])[0]

            # if both indices are not empty and same indices are included in array
            # Than pari s_t, a_t appears in earlier timestep
            if state_id.size != 0 and act_id.size != 0 and np.any(state_id == act_id):
                continue
            else:
                self.returns[(obses[t], actions[t])].append(g)
                #print(returns[(obses[t], actions[t])])
                updated_values[(obses[t], actions[t])] = np.mean(self.returns[(obses[t], actions[t])])
        
        self.q_table.update(updated_values)
        return updated_values

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q2**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        self.epsilon = 0.7 - (min(0.7, timestep / (0.2*max_timestep)))*0.95
        self.epsilon = min(self.epsilon, 1 - min(1, timestep/(0.95*max_timestep)))