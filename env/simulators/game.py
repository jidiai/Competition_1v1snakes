# -*- coding:utf-8  -*-
# 作者：zruizhi
# 创建时间： 2020/7/10 10:24 上午
# 描述：
from abc import ABC, abstractmethod
import numpy as np
from utils.discrete import Discrete


class Game(ABC):
    def __init__(self, n_player, is_obs_continuous, is_act_continuous, game_name, agent_nums, obs_type):
        self.n_player = n_player
        self.current_state = None
        self.all_observes = None
        self.is_obs_continuous = is_obs_continuous
        self.is_act_continuous = is_act_continuous
        self.game_name = game_name
        self.agent_nums = agent_nums
        self.obs_type = obs_type

    def get_config(self, player_id):
        raise NotImplementedError

    def get_render_data(self, current_state):
        return current_state

    def set_current_state(self, current_state):
        raise NotImplementedError

    @abstractmethod
    def is_terminal(self):
        raise NotImplementedError

    def get_next_state(self, all_action):
        raise NotImplementedError

    def get_reward(self, all_action):
        raise NotImplementedError

    @abstractmethod
    def step(self, all_action):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    def set_action_space(self):
        raise NotImplementedError

    def is_single_valid_action(self, single_action, action_space, index):
        for i in range(len(action_space)):
            if isinstance(action_space[i], Discrete):
                if single_action is None or single_action[i] is None:
                    raise Exception("The input action should be not None")
                if len(single_action[i]) != action_space[i].n:
                    raise Exception("The input action dimension should be {}, not {}".format(action_space[i].n, len(single_action[i])))
                if not sum(single_action[i]) == 1:
                    raise Exception("Illegal input action")
                elif 1 not in single_action[i]:
                    raise Exception("The input action is out of range")
            else:
                if single_action is None or single_action[i] is None:
                    raise Exception("The input action should be not None")
                if not isinstance(single_action[i], np.ndarray) and not isinstance(single_action[i], list):
                    raise Exception("For continuous action, the input should be numpy.ndarray or list")
                if isinstance(single_action[i], list):
                    single_action[i] = np.array(single_action[i])
                if single_action[i].shape != action_space[i].shape:
                    raise Exception("The input action dimension should be {}, not {}".format(action_space[i].shape, single_action[i].shape))
                if not action_space[i].contains(single_action[i]):
                    raise Exception("The input action is out of range")

