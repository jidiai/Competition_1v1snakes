import numpy as np
import torch
import torch.nn as nn

from typing import Union
from torch.distributions import Categorical

from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from agent.greedy.greedy_agent import greedy_snake
from types import SimpleNamespace as SN
import yaml
import os

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def hard_update(source, target):
    target.load_state_dict(source.state_dict())


def soft_update(source, target, tau):
    for src_param, tgt_param in zip(source.parameters(), target.parameters()):
        tgt_param.data.copy_(tgt_param.data * (1.0 - tau) + src_param.data * tau)


Activation = Union[str, nn.Module]

_str_to_activation = {
    'relu': nn.ReLU(),
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

    for i in agents_index:
        # self head position
        observations[i][:2] = snakes_position[i][0][:]

        # head surroundings
        head_x = snakes_position[i][0][1]
        head_y = snakes_position[i][0][0]
        head_surrounding = get_surrounding(state, width, height, head_x, head_y)
        observations[i][2:6] = head_surrounding[:]

        # beans positions
        observations[i][6:16] = beans_position[:]

        # other snake positions
        snake_heads = [snake[0] for snake in snakes_position]
        snake_heads = np.array(snake_heads[1:])
        snake_heads -= snakes_position[i][0]
        observations[i][16:] = snake_heads.flatten()[:]

    return observations


def get_reward(info, snake_index, reward, final_result):
    step_reward = np.zeros(len(snake_index))
    for i in snake_index:
        if final_result == 1:       # done and won
            step_reward[i] += 100
        elif final_result == 2:     # done and lose
            step_reward[i] -= 50
        elif final_result == 3:     # done and draw
            step_reward[i] -= 20
        else:                       # not done
            if reward[i] > 0:           # eat a bean
                step_reward[i] += 20
            else:                       # just move
                snakes_position = np.array(info['snakes_position'], dtype=object)
                beans_position = np.array(info['beans_position'], dtype=object)
                snake_heads = [snake[0] for snake in snakes_position]
                self_head = np.array(snake_heads[i])
                dists = [np.sqrt(np.sum(np.square(other_head - self_head))) for other_head in beans_position]
                step_reward[i] -= min(dists)
                if reward[i] < 0:
                    step_reward[i] -= 10
    return step_reward


def logits_random(act_dim, logits):
    logits = torch.Tensor(logits).to(device)
    acs = [Categorical(out).sample().item() for out in logits]
    num_agents = len(logits)
    actions = np.random.randint(act_dim, size=num_agents << 1)
    actions[:num_agents] = acs[:]
    return actions

def append_random(act_dim, action):
    action = torch.Tensor([action]).to(device)
    acs = [out for out in action]
    num_agents = len(action)
    actions = np.random.randint(act_dim, size=num_agents << 1)
    actions[:num_agents] = acs[:]
    return actions

def logits_greedy(state, info, logits, height, width):
    state = np.squeeze(np.array(state), axis=2)
    beans = info['beans_position']
    snakes = info['snakes_position']

    logits = torch.Tensor(logits).to(device)
    logits_action = np.array([Categorical(out).sample().item() for out in logits])
    greedy_action = greedy_snake(state, beans, snakes, width, height, [1])

    action_list = np.zeros(2)
    action_list[0] = logits_action[0]
    action_list[1] = greedy_action[0]

    return action_list


def get_surrounding(state, width, height, x, y):
    surrounding = [state[(y - 1) % height][x],  # up
                   state[(y + 1) % height][x],  # down
                   state[y][(x - 1) % width],  # left
                   state[y][(x + 1) % width]]  # right

    return surrounding


def save_config(args, save_path):
    file = open(os.path.join(str(save_path), 'config.yaml'), mode='w', encoding='utf-8')
    yaml.dump(vars(args), file)
    file.close()


def load_config(args, log_path):
    file = open(os.path.join(str(log_path), 'config.yaml'), "r")
    config_dict = yaml.load(file, Loader=yaml.FullLoader)
    print("@", config_dict)
    args = SN(**config_dict)
    print("@@", args)
    return args


# def set_algos():
#     with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
#         try:
#             config_dict = yaml.load(f, Loader=yaml.FullLoader)
#         except yaml.YAMLError as exc:
#             assert False, "default.yaml error: {}".format(exc)
#
#     args = SN(**config_dict)
#     return args


