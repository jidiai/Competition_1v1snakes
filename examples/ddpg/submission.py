import os
import sys

import torch
import numpy as np

from torch import nn
from typing import Union
from torch.distributions import Categorical


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(f'device: {device}')

Activation = Union[str, nn.Module]

_str_to_activation = {
    'relu': torch.nn.ReLU(),
    'tanh': nn.Tanh(),
    'identity': nn.Identity(),
    'softmax': nn.Softmax(dim=-1),
}


def mlp(sizes,
        activation: Activation = 'relu',
        output_activation: Activation = 'identity'):
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]

    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act]
    return nn.Sequential(*layers)


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
def get_observations(state, info, agents_index, obs_dim, height, width):
    state = np.array(state)
    state = np.squeeze(state, axis=2)
    observations = np.zeros((len(agents_index), obs_dim))
    snakes_position = np.array(info['snakes_position'], dtype=object)
    beans_position = np.array(info['beans_position']).flatten()
    for i, j in enumerate(agents_index):
        # self head position
        observations[i][:2] = snakes_position[j][0][:]

        # head surroundings
        head_x = snakes_position[j][0][1]
        head_y = snakes_position[j][0][0]
        head_surrounding = get_surrounding(state, width, height, head_x, head_y)
        observations[i][2:6] = head_surrounding[:]

        # beans positions
        observations[i][6:16] = beans_position[:]

        # other snake positions
        snake_heads = [snake[0] for snake in snakes_position]
        snake_heads = np.array(snake_heads[1:])
        snake_heads -= snakes_position[j][0]
        observations[i][16:] = snake_heads.flatten()[:]
    return observations


def logits2action(logits):
    logits = torch.Tensor(logits).to(device)
    actions = [Categorical(out).sample().item() for out in logits]
    return np.array(actions)


HIDDEN_SIZE = 256


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, num_agents, output_activation='tanh'):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agents = num_agents

        sizes_prev = [obs_dim, HIDDEN_SIZE]
        sizes_post = [HIDDEN_SIZE, HIDDEN_SIZE, act_dim]

        self.prev_dense = mlp(sizes_prev)
        self.post_dense = mlp(sizes_post, output_activation=output_activation)

    def forward(self, obs_batch):
        out = self.prev_dense(obs_batch)
        out = self.post_dense(out)
        return out


class LSTMNet(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 batch_first=True,
                 bidirectional=True):
        super(LSTMNet, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=batch_first,
            bidirectional=bidirectional
        )

    def forward(self, data, ):
        output, (_, _) = self.lstm(data)
        return output


class BiCNet:
    def __init__(self, obs_dim, act_dim, num_agent):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agent = num_agent
        self.device = device
        self.output_activation = 'softmax'
        self.actor = Actor(obs_dim, act_dim, num_agent, self.output_activation).to(self.device)

    def choose_action(self, obs):
        obs = torch.Tensor([obs]).to(self.device)
        logits = self.actor(obs).cpu().detach().numpy()[0]
        actions = logits2action(logits)
        return actions

    def load_model(self, filename):
        base_path = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(base_path, filename)
        self.actor.load_state_dict(torch.load(filepath, map_location=device))
        print(f'Model loaded. Path: {filepath}')


def to_joint_action(actions, num_agent):
    joint_action = []
    for i in range(num_agent):
        action = actions[i]
        one_hot_action = [0] * 4
        one_hot_action[action] = 1
        one_hot_action = [one_hot_action]
        joint_action.append(one_hot_action)
    return joint_action


agent = BiCNet(18, 4, 1)
agent.load_model('actor.pth')


def my_controller(observation_list, a, b):
    obs_dim = 18
    obs = observation_list[0]
    board_width = obs['board_width']
    board_height = obs['board_height']
    ctrl_agent_index = [obs['controlled_snake_index'] for obs in observation_list]
    state_map = obs['state_map']
    info = {"beans_position": obs[1], "snakes_position": [obs[key] for key in obs.keys() & {2, 3}]}

    observations = get_observations(state_map, info, ctrl_agent_index, obs_dim, board_height, board_width)
    actions = agent.choose_action(observations)

    return to_joint_action(actions, len(ctrl_agent_index))
