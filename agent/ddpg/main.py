import os
import argparse
import datetime

from tensorboardX import SummaryWriter
from ddpg_agent import DDPG
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from common.utils import *
from common.log_path import make_logpath
from env.chooseenv import make
import numpy as np
import random

def main(args):

    print(f'game name: {args.game_name}')
    print(f'device: {device}')
    print(f'model episode: {args.model_episode}')
    print(f'save interval: {args.save_interval}')

    env = make('snakes_1v1', conf=None)
    game_name = args.game_name

    num_agents = env.n_player
    print(f'Total agent number: {num_agents}')
    ctrl_agent_index = [0]
    print(f'Agent control by the actor: {ctrl_agent_index}')
    ctrl_agent_num = len(ctrl_agent_index)

    width = env.board_width
    print(f'Game board width: {width}')
    height = env.board_height
    print(f'Game board height: {height}')

    act_dim = env.get_action_dim()
    print(f'action dimension: {act_dim}')
    obs_dim = 18
    print(f'observation dimension: {obs_dim}')

    # 设置seed, 以便复现
    torch.manual_seed(args.seed_nn)
    np.random.seed(args.seed_np)
    random.seed(args.seed_random)

    # 定义保存路径
    run_dir, log_dir = make_logpath(game_name)
    print('run_dir', type(run_dir), os.path.dirname(log_dir))
    writer = SummaryWriter(str(log_dir))

    # 保存训练参数 以便复现
    if args.train_redo:
        config_dir = os.path.join(os.path.dirname(log_dir), 'run%i' % (args.run_redo))
        load_config(args, config_dir)  # config_dir: str
        save_config(args, log_dir)
    else:
        save_config(args, log_dir)

    model = DDPG(obs_dim, act_dim, ctrl_agent_num, args)

    episode = 0

    while episode < args.max_episodes:

        # Receive initial observation state s1
        state, info = env.reset()
        obs = get_observations(state, info, ctrl_agent_index, obs_dim, height, width)

        episode += 1
        step = 0
        episode_reward = np.zeros(2)

        while True:

            # For each agent i, select and execute action a:t,i = a:i,θ(s_t) + Nt
            logits = model.choose_action(obs)
            actions = logits_random(act_dim, logits)
            # actions = logits_greedy(state, info, logits, height, width)

            # Receive reward [r_t,i]i=1~n and observe new state s_t+1
            next_state, reward, done, _, info = env.step(env.encode(actions))

            # reward shaping
            reward = np.array(reward)
            episode_reward += reward
            # done = len(info['snakes_position']) < num_agents or done

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
            # Store transition in R
            model.replay_buffer.push(obs, logits, step_reward, next_obs, done)

            model.update()

            obs = next_obs
            state = next_state
            step += 1

            if args.episode_length <= step or (True in done):

                print(f'[Episode {episode:05d}] score: {episode_reward[0]} reward: {step_reward[0]:.2f}')

                reward_tag = 'reward'
                loss_tag = 'loss'
                writer.add_scalars(reward_tag, global_step=episode,
                                   tag_scalar_dict={'score': episode_reward[0], 'reward': step_reward[0]})
                if model.c_loss and model.a_loss:
                    writer.add_scalars(loss_tag, global_step=episode,
                                       tag_scalar_dict={'actor': model.a_loss, 'critic': model.c_loss})

                if model.c_loss and model.a_loss:
                    print(f'\t\t\t\ta_loss {model.a_loss:.3f} c_loss {model.c_loss:.3f}')

                if episode % args.save_interval == 0:
                    model.save_model(episode)

                env.reset()
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--game_name', default='snake1v1')
    parser.add_argument("--algo", default="ddpg", help="ddpg")

    # trainer
    parser.add_argument('--max_episodes', default=5000, type=int)
    parser.add_argument('--episode_length', default=5000, type=int)
    parser.add_argument("--save_interval", default=10000, type=int)
    parser.add_argument("--model_episode", default=0, type=int)
    parser.add_argument('--train_redo', default=False, type=bool)
    parser.add_argument('--run_redo', default=None, type=int)

    # algo
    parser.add_argument('--output_activation', default="softmax", type=str, help="tanh/softmax")
    parser.add_argument('--buffer_size', default=int(1e5), type=int)
    parser.add_argument('--tau', default=0.001, type=float)
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--lr_a', default=0.0001, type=float)
    parser.add_argument('--lr_c', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epsilon', default=0, type=int)
    parser.add_argument('--epsilon_speed', default=0.99998, type=int)

    # seed
    parser.add_argument('--seed_nn', default=1, type=int)
    parser.add_argument('--seed_np', default=1, type=int)
    parser.add_argument('--seed_random', default=1, type=int)

    args = parser.parse_args()

    main(args)
