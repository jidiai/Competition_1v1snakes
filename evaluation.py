import numpy as np

from agent.dqn.rl_agent import get_observations
from agent.greedy.greedy_agent import greedy_snake, make_grid_map
from agent.dqn.rl_agent import agent as dqn_snake
from env.chooseenv import make
from tabulate import tabulate
import argparse


def print_state(state, actions, step):
    state = np.array(state)
    state = np.squeeze(state, axis=2)
    print(f'----------------- STEP:{step} -----------------')
    print(f'state:\n{state}')
    print(f'actions: {actions}\n')


def get_actions(state, algo):

    actions = np.random.randint(4, size=1)

    # dqn
    if algo == 'dqn':
        agent_trained_index = [0] # todo
        obs = get_observations(state, agent_trained_index, obs_dim=18)
        actions[:] = dqn_snake.choose_action(obs)

    # greedy
    if algo == 'greedy':
        greedy_info = get_greedy_info(state)

        actions[:] = greedy_snake(greedy_info['state'],
                                  greedy_info['beans'],
                                  greedy_info['snakes'],
                                  greedy_info['width'],
                                  greedy_info['height'],
                                  greedy_info['ctrl_agent_index'])[:]

    return actions

def get_greedy_info(observation):
    obs = observation.copy()
    ctrl_agent_index = [obs['controlled_snake_index']]
    board_width = obs['board_width']
    board_height = obs['board_height']
    beans_positions = obs[1]
    snakes_positions = {key: obs[key] for key in obs.keys() & {2, 3}}
    snake_map = make_grid_map(board_width, board_height, beans_positions, snakes_positions)
    state_map = np.squeeze(np.array(snake_map), axis=2)

    greedy_info = {'state': state_map,
                   'beans': beans_positions,
                   'snakes': snakes_positions,
                   'width': board_width,
                   'height': board_height,
                   'ctrl_agent_index': ctrl_agent_index}
    return greedy_info


    # greedy_snake(state_map, beans_positions, snakes_positions, board_width, board_height, ctrl_agent_index)

def get_join_actions(obs, algo_list):
    first_action = get_actions(obs[0], algo_list[0])
    second_action = get_actions(obs[1], algo_list[1])
    actions = np.zeros(2)
    actions[0] = first_action[:]
    actions[1] = second_action[:]
    return actions


def run_game(env, algo_list, episode, verbose=False):
    width = env.board_width
    height = env.board_height
    obs_dim = 18
    agent_index = [0, 1]
    total_reward = np.zeros(2)
    num_win = np.zeros(3)

    for i in range(1, episode + 1):
        episode_reward = np.zeros(2)
        state = env.reset()

        # obs = get_observations(state, agent_index, obs_dim, height, width)

        action_list = get_join_actions(state, algo_list)
        joint_action = env.encode(action_list)

        step = 0
        if verbose:
            print_state(state, action_list, step)

        while True:
            next_state, reward, done, _, info = env.step(joint_action)
            episode_reward += reward

            if done:
                if np.sum(episode_reward[0]) > np.sum(episode_reward[1]):
                    num_win[0] += 1
                elif np.sum(episode_reward[0]) < np.sum(episode_reward[1]):
                    num_win[1] += 1
                else:
                    num_win[2] += 1

                if not verbose:
                    print('.', end='')
                    if i % 100 == 0 or i == episode:
                        print()
                break

            state = next_state
            step += 1
            # obs = get_observations(state, info, agent_index, obs_dim, height, width)

            action_list = get_join_actions(state, algo_list)
            joint_action = env.encode(action_list)

            if verbose:
                print_state(state, action_list, step)

        total_reward += episode_reward

    # calculate results
    total_reward /= episode
    print(f'\nResult base on {episode} ', end='')
    print('episode:') if episode == 1 else print('episodes:')

    header = ['Name', algo_list[0], algo_list[1]]
    data = [['score', total_reward[0], total_reward[1]],
            ['win', num_win[0], num_win[1]]]
    print(tabulate(data, headers=header, tablefmt='pretty', floatfmt='.3f'))


if __name__ == "__main__":
    env_type = 'snakes_1v1'

    game = make(env_type, conf=None)

    parser = argparse.ArgumentParser()
    parser.add_argument("--my_ai", default="dqn", help="dqn/random/greedy")
    parser.add_argument("--opponent", default="greedy", help="dqn/random/greedy")
    parser.add_argument("--episode", default=100)
    args = parser.parse_args()

    # [greedy, dqn, random]
    agent_list = [args.my_ai, args.opponent]
    run_game(game, algo_list=agent_list, episode=args.episode, verbose=False)
