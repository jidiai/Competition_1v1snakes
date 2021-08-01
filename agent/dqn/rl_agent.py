# -*- coding:utf-8  -*-
# Time  : 2021/5/27 下午3:38
# Author: Yahui Cui

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


def make_grid_map(board_width, board_height, beans_positions:list, snakes_positions:dict):
    snakes_map = [[[0] for _ in range(board_width)] for _ in range(board_height)]
    for index, pos in snakes_positions.items():
        for p in pos:
            snakes_map[p[0]][p[1]][0] = index

    for bean in beans_positions:
        snakes_map[bean[0]][bean[1]][0] = 1

    return snakes_map


def get_surrounding(state, width, height, x, y):
    surrounding = [state[(y - 1) % height][x],  # up
                   state[(y + 1) % height][x],  # down
                   state[y][(x - 1) % width],  # left
                   state[y][(x + 1) % width]]  # right

    return surrounding


# Self position:        0:head_x; 1:head_y
# Head surroundings:    2:head_up; 3:head_down; 4:head_left; 5:head_right
# Beans positions:      (6, 7) (8, 9) (10, 11) (12, 13) (14, 15)
# Other snake positions: (16, 17) -- (other_x - self_x, other_y - self_y)
def get_observations(state, agent_trained_index, obs_dim):
    state_copy = state.copy()

    agents_index = state_copy["controlled_snake_index"]

    # if agents_index != agent_trained_index:
    #     error = "训练的智能体：{name}, 观测的智能体：{url}".format(name=agents_index, url=agent_trained_index)
    #     raise Exception(error)

    board_width = state_copy['board_width']
    board_height = state_copy['board_height']
    beans_positions = state_copy[1]
    snakes_positions = {key: state_copy[key] for key in state_copy.keys() & {2, 3}}
    snake_map = make_grid_map(board_width, board_height, beans_positions, snakes_positions)
    state = np.array(snake_map)
    state = np.squeeze(snake_map, axis=2)

    snakes_position = np.array(snakes_positions[agents_index], dtype=object)

    beans_position = np.array(beans_positions).flatten()

    observations = np.zeros((1, obs_dim)) # todo

    # self head position
    observations[0][:2] = snakes_position[0][:]

    # head surroundings
    head_x = snakes_position[0][1]
    head_y = snakes_position[0][0]
    head_surrounding = get_surrounding(state, board_width, board_height, head_x, head_y)
    observations[0][2:6] = head_surrounding[:]

    # beans positions
    observations[0][6:16] = beans_position[:]

    # other snake head positions
    snakes_other_position = np.array(snakes_positions[3], dtype=object) # todo
    observations[0][16:] = snakes_other_position[0][:]

    return observations


class Critic(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class DQN(object):
    def __init__(self, state_dim, action_dim, num_agent, hidden_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agent = num_agent

        self.hidden_size = hidden_size

        self.critic_eval = Critic(self.state_dim, self.action_dim, self.hidden_size)
        self.critic_target = Critic(self.state_dim, self.action_dim, self.hidden_size)

    def choose_action(self, observation):
        observation = torch.tensor(observation, dtype=torch.float).view(1, -1)
        action = torch.argmax(self.critic_eval(observation)).item()
        return action

    def load(self, file):
        base_path = os.path.dirname(os.path.abspath(__file__))
        file = os.path.join(base_path, file)
        self.critic_eval.load_state_dict(torch.load(file))
        self.critic_target.load_state_dict(torch.load(file))


def to_joint_action(actions, num_agent):
    joint_action = []
    for i in range(num_agent):
        action = actions
        one_hot_action = [0] * 4
        one_hot_action[action] = 1
        one_hot_action = [one_hot_action]
        joint_action.append(one_hot_action)
    return joint_action

agent = DQN(18, 4, 1, 256)
agent.load('critic_20000.pth')