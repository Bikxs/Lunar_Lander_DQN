import time

import gym
from tqdm import tqdm

from constants import *


class SARSAAgent(object):
    def __init__(self, episodes=20000, alpha=0.1, gamma=0.90, epsilon=1.0, epsilon_decay=.99):
        self.name = "SARSA"
        self.Q = {}
        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        self.episodes = episodes
        envname = "LunarLander-v2"
        self.env = gym.make(envname).env
        self.env.seed(1984)
        np.random.seed(1984)

        self.actions = [x for x in range(self.env.action_space.n)]
        self.states = []
        for x in range(len(state_map['x']) + 1):
            for y in range(len(state_map['y']) + 1):
                for x_ in range(len(state_map['x_']) + 1):
                    for y_ in range(len(state_map['y_']) + 1):
                        for theta in range(len(state_map['theta']) + 1):
                            for theta_ in range(len(state_map['theta_']) + 1):
                                for legL in [0, 1]:
                                    for legR in [0, 1]:
                                        self.states.append((x, y, x_, y_, theta, theta_, legL, legR))
        for state in self.states:
            for action in self.actions:
                self.Q[state, action] = 0

    def train(self):
        start_time = time.time()
        rewards = []

        for episode in tqdm(range(self.episodes)):
            if episode % 500 == 0:
                done = False
                state = discrete_state(self.env.reset())
                action = self._max_action(state)
                score = 0
                iterations = 0
                while not done:
                    iterations += 1
                    state, reward, done, info = self.env.step(action)
                    state = discrete_state(state)
                    score += reward
                    action = self._max_action(state)
                    if iterations >= MAX_ITERATIONS:
                        break
            state = self.env.reset()
            state = discrete_state(state)
            done = False
            score = 0
            action = self._take_action(state)
            iterations = 0
            while not done:
                iterations += 1
                state_next, reward, done, info = self.env.step(action)
                state_next = discrete_state(state_next)
                action_next = self._take_action(state)
                self.Q[state, action] = self.Q[state, action] + \
                                        self.alpha * (reward + self.gamma * self.Q[
                    state_next, action_next] - self.Q[
                                                          state, action])
                state, action = state_next, action_next
                score += reward
                if iterations >= MAX_ITERATIONS:
                    break
            rewards.append(score)
            self.epsilon = self.epsilon * self.epsilon_decay if self.epsilon > 0.01 else 1.0

        end_time = time.time()
        time_taken = end_time - start_time
        return rewards, time_taken

    def best_action(self, state):
        return self._max_action(state)

    def Q_table(self, state, action):
        return self.Q[state, action]

    def _max_action(self, state):
        values = np.array([self.Q[state, action] for action in self.actions])
        return np.argmax(values)

    def _take_action(self, state):
        """Specifically, an action_ should be selected uniformly at random if a random number drawn uniformly between 0 and 1 is less than ϵ.
                If the greedy action_ is selected, the action_ with lowest index should be selected in case of ties."""
        if np.random.random() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return self._max_action(state)
    def table_visitations(self):
        zeros = 0
        for key, value in self.Q.items():
            if abs(value) <= 0.001:
                zeros += 1
        return (1 - (zeros / (len(self.states) * len(self.actions)))) * 100

    def table_size(self):
        return len(self.states) * len(self.actions)
    def reset(self):
        return discrete_state(self.env.reset())

    def take_action(self, action):
        state, reward, done, info = self.env.step(action)
        return discrete_state(state), reward, done, info