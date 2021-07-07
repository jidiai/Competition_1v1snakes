import os
import argparse

from utils import *
from log_path import make_logpath
from collections import namedtuple
from dqn import DQN
from env.chooseenv import make
from tensorboardX import SummaryWriter

import numpy as np
import random
import torch


def main(args):
    env = make('snakes_1v1', conf=None)
    game_name = args.game_name
    print(f'game name: {args.game_name}')

    ctrl_agent_index = [0]
    ctrl_agent_num = len(ctrl_agent_index)

    width = env.board_width
    print(f'Game board width: {width}')
    height = env.board_height
    print(f'Game board height: {height}')
    action_dim = env.get_action_dim()
    print(f'action dimension: {action_dim}')
    obs_dim = 18
    print(f'observation dimension: {obs_dim}')

    # set seed
    torch.manual_seed(args.seed_nn)
    np.random.seed(args.seed_np)
    random.seed(args.seed_random)

    # 定义保存路径
    run_dir, log_dir = make_logpath(game_name, args.algo)
    writer = SummaryWriter(str(log_dir))

    # 保存训练参数 以便复现
    if args.train_redo:
        config_dir = os.path.join(os.path.dirname(log_dir), 'run%i' % (args.run_redo))
        load_config(args, config_dir)
        save_config(args, log_dir)
    else:
        save_config(args, log_dir)

    Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])
    model = DQN(obs_dim, action_dim, ctrl_agent_num, args)
    episode = 0

    while episode < args.max_episodes:
        state, info = env.reset()
        obs = get_observations(state, info, ctrl_agent_index, obs_dim, height, width)

        episode += 1
        step = 0
        episode_reward = np.zeros(2)

        while True:
            action = model.choose_action(obs)
            actions = append_random(action_dim, action)

            # actions = model.choose_action(obs)

            next_state, reward, done, _, info = env.step(env.encode(actions))

            reward = np.array(reward)
            episode_reward += reward

            if done:
                if np.sum(episode_reward[0]) > np.sum(episode_reward[1]):
                    step_reward = get_reward(info, ctrl_agent_index, reward, final_result=1)
                elif np.sum(episode_reward[0]) < np.sum(episode_reward[1]):
                    step_reward = get_reward(info, ctrl_agent_index, reward, final_result=2)
                else:
                    step_reward = get_reward(info, ctrl_agent_index, reward, final_result=3)
                next_obs = np.zeros((ctrl_agent_num, obs_dim))
            else:
                step_reward = get_reward(info, ctrl_agent_index, reward, final_result=0)
                next_obs = get_observations(next_state, info, ctrl_agent_index, obs_dim, height, width)

            done = np.array([done] * ctrl_agent_num)

            # store transitions
            trans = Transition(obs, actions, step_reward, np.array(next_obs), done)
            model.store_transition(trans)
            model.learn()
            obs = next_obs
            state = next_state
            step += 1

            if args.episode_length <= step or (True in done):
                print(f'[Episode {episode:05d}] score: {episode_reward[0]} reward: {step_reward[0]:.2f}')

                reward_tag = 'reward'
                loss_tag = 'loss'
                writer.add_scalars(reward_tag, global_step=episode,
                                   tag_scalar_dict={'score': episode_reward[0], 'reward': step_reward[0]})
                if model.loss:
                    writer.add_scalars(loss_tag, global_step=episode,
                                       tag_scalar_dict={'loss': model.loss})
                    print(f'\t\t\t\tloss {model.loss:.3f}')

                if episode % args.save_interval == 0:
                    model.save(run_dir, episode)

                env.reset()
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--game_name', default='snake1v1')
    parser.add_argument('--algo', default='dqn', help='dqn')

    # trainer
    parser.add_argument('--max_episodes', default=5000, type=int)
    parser.add_argument('--episode_length', default=5000, type=int)
    parser.add_argument('--save_interval', default=1000, type=int)
    parser.add_argument('--model_episode', default=0, type=int)
    parser.add_argument('--train_redo', default=False, type=bool)
    parser.add_argument('--run_redo', default=None, type=int)

    # algo
    parser.add_argument('--output_activation', default='softmax', type=str, help='tanh/softmax')
    parser.add_argument('--buffer_size', default=int(1e5), type=int)
    parser.add_argument('--tau', default=0.001, type=float)
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--lr_a', default=0.0001, type=float)
    parser.add_argument('--lr_c', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epsilon', default=0.2, type=float)
    parser.add_argument('--epsilon_speed', default=0.99998, type=float)
    parser.add_argument('--epsilon_end', default=0.05, type=float)
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--target_replace', default=100, type=int)

    # seed
    parser.add_argument('--seed_nn', default=1, type=int)
    parser.add_argument('--seed_np', default=1, type=int)
    parser.add_argument('--seed_random', default=1, type=int)

    # evaluation
    parser.add_argument('--evaluate_rate', default=50)

    args = parser.parse_args()
    main(args)
