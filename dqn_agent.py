import os
import random
import time
import warnings
from collections import deque

import gym
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras import Sequential, models
from tensorflow.keras.activations import relu, linear
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.optimizers import Adam
from tensorflow.python.client import device_lib

warnings.filterwarnings("ignore")
from constants import MAX_ITERATIONS

print(device_lib.list_local_devices())


class DQNAgent:
    def __init__(self, episodes=1000, alpha=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=.998):
        self.learning_rate = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        self.episodes = episodes

        self.name = "DQN"
        envname = "LunarLander-v2"
        self.file_name = f"models/{self.name} {envname} alpha_{int(self.alpha * 1000)} gamma_{int(self.gamma * 1000)} epsilon_{int(self.epsilon * 1000)}"
        self.env = gym.make(envname).env
        self.env.seed(1984)
        np.random.seed(1984)

        self.action_space = self.env.action_space.n
        self.observation_space = self.env.observation_space.shape[0]
        self.update_counter = 0
        self.target_update_counter = 0

        self.replay_memory = deque(maxlen=50000)
        self.min_replay_memory_start = 5000
        self.batch_size = 128

        models_dir = "models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        if os.path.exists(self.file_name):
            self.model = self.load_model()
        else:
            self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

    @property
    def alpha(self):
        return self.learning_rate

    def create_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.observation_space, activation=relu))
        model.add(Dense(16, activation=relu))
        model.add(Dense(16, activation=relu))
        model.add(Dense(self.action_space, activation=linear))
        model.compile(loss=mean_squared_error, optimizer=Adam(lr=self.learning_rate))

        return model

    def save_model(self):
        self.target_model.save(self.file_name)

    def load_model(self):
        return models.load_model(self.file_name)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_space)

        predicted_actions = self.target_model.predict(state)
        return np.argmax(predicted_actions[0])

    def best_action(self, state):
        predicted_actions = self.target_model.predict(state)
        return np.argmax(predicted_actions[0])

    def update_weights(self):
        def update_weights_target():
            weights = self.model.get_weights()
            target_weights = self.target_model.get_weights()
            for i in range(len(target_weights)):
                target_weights[i] = weights[i]
            self.target_model.set_weights(target_weights)

        # update weights if we have enough data in replay_memomry and the counter flag is set
        if len(self.replay_memory) < self.min_replay_memory_start or self.update_counter != 0:
            return

        mini_batch = random.sample(self.replay_memory, self.batch_size)
        states, actions, rewards, next_states, done_list = self.extract_columns(mini_batch)
        targets = rewards + self.gamma * (np.amax(self.target_model.predict_on_batch(next_states), axis=1)) * (
                1 - done_list)
        target_vec = self.target_model.predict_on_batch(states)
        indexes = np.array([i for i in range(self.batch_size)])
        target_vec[[indexes], [actions]] = targets

        self.model.fit(states, target_vec, epochs=1, verbose=0)

        if self.target_update_counter == 0:
            # self.target_model.set_weights(self.model.get_weights())
            update_weights_target()
        self.target_update_counter += 1
        self.target_update_counter = self.target_update_counter % 10

    def extract_columns(self, batch):
        states = np.array([i[0] for i in batch])
        actions = np.array([i[1] for i in batch])
        rewards = np.array([i[2] for i in batch])
        next_states = np.array([i[3] for i in batch])
        dones = np.array([i[4] for i in batch])
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        return np.squeeze(states), actions, rewards, next_states, dones

    def reshape_state(self, state):
        return np.reshape(state, [1, self.observation_space])

    def train(self):

        start_time = time.time()
        rewards = []

        for episode in range(self.episodes):
            # for episode in tqdm(range(self.episodes)):
            done = False
            to_render = False  # episode % 100 == 0
            state = self.reshape_state(self.env.reset())
            score = 0.0
            iterations = 0
            while not done:
                iterations += 1
                if to_render:
                    self.env.render()
                action = self.get_action(state)

                next_state, reward, done, info = self.env.step(action)
                next_state = self.reshape_state(next_state)

                # Remeber/put experience in memory
                self.replay_memory.append((state, action, reward, next_state, done))

                score += reward
                state = next_state

                # updated counter variable
                self.update_counter += 1
                self.update_counter = self.update_counter % (self.batch_size / 2)

                self.update_weights()

                if iterations >= MAX_ITERATIONS:
                    break
            print(
                f"Episode {episode}/{self.episodes}, Iterations={iterations}, Epsilon={round(self.epsilon, 4)}, Rewards={round(score, 1)}")
            rewards.append(score)
            # Decay/Reset epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            # keep saving a copy of the trained model
            if episode % 500 == 0 and episode > 100:
                self.save_model()
        end_time = time.time()
        time_taken = end_time - start_time
        self.save_model()
        return rewards, time_taken

    def table_visitations(self):
        return 0.

    def table_size(self):
        return 0

    def reset(self):
        return self.reshape_state(self.env.reset())

    def take_action(self, action):
        state, reward, done, info = self.env.step(action)
        return self.reshape_state(state), reward, done, info
