import time

import gym
import matplotlib.pyplot as plt
from tqdm import tqdm

from constants import *


class RandomAgent(object):
    def __init__(self, episodes=100):
        self.episodes = episodes
        envname = "LunarLander-v2"
        self.env = gym.make(envname).env

        # print('Environment:', envname)
        # print('action_space_:', self.env.action_space)
        # print('observation_space:', self.env.observation_space)

        self.actions = range(self.env.action_space.n)

    def solve(self):
        start_time = time.time()
        rewards = []
        states = []

        for episode in tqdm(range(self.episodes)):
            state = self.env.reset()
            done = False
            score = 0
            action = self._take_action(state)
            iterations = 0
            while not done:
                iterations += 1
                states.append(state)
                state_next, reward, done, info = self.env.step(action)
                action_next = self._take_action(state)
                state, action = state_next, action_next
                score += reward
                # if done:
                #     print('Episode {},Iterations:{} Score:{}'.format(episode + 1, iterations, round(score, 1)))
                if iterations >= MAX_ITERATIONS:
                    break
            rewards.append(score)
        plt.plot(rewards)
        plt.show()
        end_time = time.time()
        # print("---------------------------------------------")
        # print('training took:', end_time - start_time, 'seconds')
        # print("---------------------------------------------")
        return np.array(states)

    def _take_action(self, state):
        return np.random.choice(self.actions)
def r_agents(episodes=20000):
    agent = RandomAgent(episodes=episodes)
    states = agent.solve()
    get_observations_limits(states, 20)
    get_observations_limits(states, 10)
    get_observations_limits(states, 8)
    get_observations_limits(states, 5)
    get_observations_limits(states, 3)


def get_observations_limits(states, bins=10):
    def discretize(data, bins):
        split = np.array_split(np.sort(data), bins)
        cutoffs = [-float('inf')]
        cutoffs.extend([round(x[-1], 3) for x in split])
        cutoffs.append(float('inf'))
        # cutoffs = cutoffs[:-1]
        return cutoffs

    # pprint(states)
    state_description = ['x', 'y', 'x_', 'y_', 'theta', 'theta_']

    print(states.shape)
    columns = states.transpose()

    for x in range(len(state_description)):
        dat = columns[x]
        cutoff = discretize(dat, bins)
        sample_data = np.random.choice(dat, 10)
        state_map[state_description[x]] = cutoff
    print('bins', bins)
    print(state_map)
