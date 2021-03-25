from abc import ABC, abstractmethod
from collections import defaultdict
import random
import sys
from typing import List, Dict, DefaultDict
from copy import deepcopy
import numpy as np
from gym.spaces import Space, Box
from gym.spaces.utils import flatdim

from rl2021.exercise5.matrix_game import actions_to_onehot

def obs_to_tuple(obs):
    return tuple([tuple(o) for o in obs])


class MultiAgent(ABC):
    """Base class for multi-agent reinforcement learning

    **DO NOT CHANGE THIS BASE CLASS**

    """

    def __init__(
        self,
        num_agents: int,
        action_spaces: List[Space],
        observation_spaces: List[Space],
        gamma: float,
        **kwargs
    ):
        """Constructor of base agent for Q-Learning

        Initializes basic variables of MARL agents
        namely epsilon, learning rate and discount rate.

        :param num_agents (int): number of agents
        :param action_spaces (List[Space]): action spaces of the environment for each agent
        :param observation_spaces (List[Space]): observation spaces of the environment for each agent
        :param gamma (float): discount factor (gamma)

        :attr n_acts (List[int]): number of actions for each agent
        """

        self.num_agents = num_agents
        self.action_spaces = action_spaces
        self.observation_spaces = observation_spaces
        self.n_acts = [flatdim(action_space) for action_space in action_spaces]

        self.gamma: float = gamma

    @abstractmethod
    def act(self, obs: List[np.ndarray]) -> List[int]:
        """Chooses an action for all agents given observations

        :param obs (List[np.ndarray] of float with dim (observation size)):
            received observations representing the current environmental state for each agent
        :return (List[int]): index of selected action for each agent
        """
        ...

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


class IndependentQLearningAgents(MultiAgent):
    """Agent using the Independent Q-Learning algorithm

    ** YOU NEED TO IMPLEMENT THE FUNCTIONS IN THIS CLASS **
    """

    def __init__(self, learning_rate: float =0.5, epsilon: float =1.0, **kwargs):
        """Constructor of IndependentQLearningAgents

        :param learning_rate (float): learning rate for Q-learning updates
        :param epsilon (float): epsilon value for all agents


        :attr q_tables (List[DefaultDict]): tables for Q-values mapping (OBS, ACT) pairs of observations
            and actions to respective Q-values for all agents

        Initializes some variables of the Independent Q-Learning agents, namely the epsilon, discount rate
        and learning rate
        """

        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        # initialise Q-tables for all agents
        self.q_tables: List[DefaultDict] = [defaultdict(lambda: 0) for i in range(self.num_agents)]


    def act(self, obss: List[np.ndarray]) -> List[int]:
        """Implement the epsilon-greedy action selection here

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        :param obss (List[np.ndarray] of float with dim (observation size)):
            received observations representing the current environmental state for each agent
        :return (List[int]): index of selected action for each agent
        """
        actions = []

        rng = np.random.default_rng()

        for i, obs in enumerate(obss):
            # Make epsilon-greedy action selection for each agent where i is the index of the agent
            u = np.random.uniform(low = 0.0, high = 1.0, size = 1)

            if u <= self.epsilon:
                selected_action = rng.choice(range(self.n_acts[i]))

            else:
                val_id = [self.q_tables[i][(obs, action)] for action in range(self.n_acts[i])]
                selected_action = np.argmax(val_id)
                # selected_action = np.argmax([self.q_tables[i][(obs, action)] for action in range(self.n_acts[i])])
                # if selected_action.size > 1:
                #     selected_action = np.random.choice(selected_action)
            actions.append(selected_action)
            
        return actions


    def learn(
        self, obss: List[np.ndarray], actions: List[int], rewards: List[float], n_obss: List[np.ndarray], dones: List[bool]
    ) -> List[float]:
        """Updates the Q-tables based on agents' experience

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        :param obss (List[np.ndarray] of float with dim (observation size)):
            received observations representing the current environmental state for each agent
        :param action (List[int]): index of applied action of each agent
        :param rewards (List[float]): received reward for each agent
        :param n_obss (List[np.ndarray] of float with dim (observation size)):
            received observations representing the next environmental state for each agent
        :param dones (List[bool]): flag indicating whether a terminal state has been reached for each agent
        :return (List[float]): updated Q-values for current observation-action pair of each agent
        """
        updated_values = []

        for i, (obs, action, reward, n_obs, done) in enumerate(zip(obss, actions, rewards, n_obss, dones)):
            max_next_value = np.max([self.q_tables[i][(n_obs, action)] for action in range(self.n_acts[i])])
            target_value = reward + self.gamma*(1-done)*max_next_value 
            self.q_tables[i][(obs, action)] = self.q_tables[i][(obs, action)] + self.learning_rate*(target_value - self.q_tables[i][(obs, action)])
            updated_values.append(self.q_tables[i][(obs, action)])
        
        return updated_values


    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ### PUT YOUR CODE HERE ###
        if timestep == 0:
            min_alpha = 1/1000
            self.max_alpha = self.learning_rate
            self.diff = (self.max_alpha - min_alpha)
            
        self.learning_rate = self.max_alpha - (self.diff) * (timestep / max_timestep)
        
        max_deduct, decay = 0.95, 0.07
       
        self.epsilon =  1.0 - (min(1.0, timestep/(decay * max_timestep))) * max_deduct
        #raise NotImplementedError("Needed for Q5")


class JointActionLearning(MultiAgent):
    """Agents using the Joint Action Learning algorithm with Opponent Modelling

    ** YOU NEED TO IMPLEMENT THE FUNCTIONS IN THIS CLASS **
    """

    def __init__(self, learning_rate: float =0.5, epsilon: float =1.0, **kwargs):
        """Constructor of JointActionLearning

        :param learning_rate (float): learning rate for Q-learning updates
        :param epsilon (float): epsilon value for all agents

        :attr q_tables (List[DefaultDict]): tables for Q-values mapping (OBS, ACT) pairs of
            observations and joint actions to respective Q-values for all agents
        :attr models (List[DefaultDict[DefaultDict]]): each agent holding model of other agent
            mapping observation to other agent actions to count of other agent action

        Initializes some variables of the Joint Action Learning agents, namely the epsilon, discount
        rate and learning rate
        """

        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.n_acts = [flatdim(action_space) for action_space in self.action_spaces]

        # initialise Q-tables for all agents
        self.q_tables: List[DefaultDict] = [defaultdict(lambda: 0) for _ in range(self.num_agents)]

        # initialise models for each agent mapping state to other agent actions to count of other agent action
        # in state
        self.models = [defaultdict(lambda: defaultdict(lambda: 0)) for _ in range(self.num_agents)] 

        # count observations - count for each agent
        self.c_obss = [defaultdict(lambda: 0) for _ in range(self.num_agents)]


    def calc_evs(self, agent_id, obss):
        EV = [0] * self.n_acts[agent_id]
        #possible_join_actions_other_agents = self.models[agent_id][obss[agent_id]].keys()

        actions_other_agents = deepcopy(self.n_acts)
        del actions_other_agents[agent_id]

        possible_join_actions_other_agents = list(*[range(action) for action in actions_other_agents])

        for act_idx, action in enumerate(range(self.n_acts[agent_id])):                    
            
            for pja in possible_join_actions_other_agents:
                if type(pja) == int:
                    pja = np.asarray([pja])
                joint_actions = np.concatenate([pja[:agent_id], np.asarray([action]), pja[agent_id:]])
                C = self.models[agent_id][obss[agent_id]][tuple(pja)]
                N = self.c_obss[agent_id][obss[agent_id]]
                Q = self.q_tables[agent_id][(obss[agent_id], tuple(joint_actions))]

                EV[act_idx] += C/N*Q

        return EV


    def act(self, obss: List[np.ndarray]) -> List[int]:
        """Implement the epsilon-greedy action selection here

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        :param obss (List[np.ndarray] of float with dim (observation size)):
            received observations representing the current environmental state for each agent
        :return (List[int]): index of selected action for each agent
        """
        joint_action = []

        rng = np.random.default_rng()

        for agent_id in range(self.num_agents):

            u = np.random.uniform(low = 0.0, high = 1.0, size = 1)

            if u <= self.epsilon:
                joint_action.append(rng.choice(range(self.n_acts[agent_id])))

            else:
                action_q_values = [self.q_tables[agent_id][(obss[agent_id], act)] for act in range(self.n_acts[agent_id])]
                
                max_acts = np.argwhere(action_q_values == np.amax(action_q_values)).reshape(1,-1)[0]
                joint_action.append(np.random.choice(max_acts))

        return joint_action



    def learn(
        self, obss: List[np.ndarray], actions: List[int], rewards: List[float], n_obss: List[np.ndarray], dones: List[bool]
    ) -> List[float]:
        """Updates the Q-tables and models based on agents' experience

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        :param obss (List[np.ndarray] of float with dim (observation size)):
            received observations representing the current environmental state for each agent
        :param action (List[int]): index of applied action of each agent
        :param rewards (List[float]): received reward for each agent
        :param n_obss (List[np.ndarray] of float with dim (observation size)):
            received observations representing the next environmental state for each agent
        :param dones (List[bool]): flag indicating whether a terminal state has been reached for each agent
        :return (List[float]): updated Q-values for current observation-action pair of each agent
        """
        updated_values = [0]*self.num_agents
        
        for agent_id, (obs, reward, done) in enumerate(zip(obss, rewards, dones)):
            joint_actions_other_agents = actions[:agent_id] + actions[agent_id+1 :]
            self.c_obss[agent_id][obs] += 1
            self.models[agent_id][obs][joint_actions_other_agents[0]] += 1

            #max_ev = self.get_max_ev(agent_id, obs)
            max_ev = max(self.calc_evs(agent_id, obss))
            target_value = reward + self.gamma*(1-done)*max_ev
            self.q_tables[agent_id][(obs, tuple(actions))] += self.learning_rate*(target_value - self.q_tables[agent_id][(obs, tuple(actions))])

            updated_values[agent_id] = self.q_tables[agent_id][(obs, tuple(actions))]
        return updated_values


    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ### PUT YOUR CODE HERE ###
        #raise NotImplementedError("Needed for Q5")