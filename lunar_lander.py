import os

import matplotlib.pyplot as plt

from constants import *
from dqn_agent import DQNAgent
from random_agent import RandomAgent


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


def train_test(agent: DQNAgent, variable_name, variable_value):
    def make_plot(title, folder, name, rewards):
        def dump_rewards(file_name):
            with open(file_name, 'w') as f:
                for item in rewards:
                    f.write("%s\n" % item)

        plt.figure()
        plt.title(title)
        plt.plot(rewards)
        plt.ylabel('Rewards')
        plt.xlabel('Episodes')
        filename = 'plots/' + folder
        if not os.path.exists(filename):
            os.makedirs(filename)
        plt.savefig(filename + '/' + name + '.png', bbox_inches='tight')
        dump_rewards(filename + '/' + name + '_rewards.txt')

    def test(testing_episodes=100):
        rewards = []
        for episode in range(testing_episodes):
            done = False
            state = agent.reset()
            action = agent.best_action(state)
            score = 0
            iterations = 0
            while not done:
                iterations += 1
                state, reward, done, info = agent.take_action(action)
                score += reward
                action = agent.best_action(state)
                if iterations >= MAX_ITERATIONS:
                    break
            rewards.append(score)
            print(f"Episode {episode}/{testing_episodes}, Iterations={iterations}, Rewards={round(score, 1)}")
        return rewards

    if variable_name is None:
        subtitle = agent.name + " Agent\n(defaults \u03B3=" + str(agent.gamma) + ", lr=" + str(
            round(agent.learning_rate, 5)) + ", \u025B=" + str(round(1.0, 3)) + ", \u03B4=" + str(
            round(agent.epsilon_decay, 3)) + ")"
    else:
        subtitle = agent.name + " Agent\n" + variable_name + "=" + str(variable_value)
    title = "Training " + subtitle
    print(title)
    rewards, time_taken = agent.train()
    print('\tTime Taken to Train: ', int(time_taken / 60), ' minutes and ', int(time_taken % 60), ' seconds', sep='')

    print('\tQ-Table Size: ', agent.table_size(), sep='')
    print('\tQ-Table Visitations: ', round(agent.table_visitations() * 100, 1), '%', sep='')
    make_plot(title, subtitle, "training", rewards)
    if variable_name is None:
        subtitle = agent.name + " Agent\n(defaults \u03B3=" + str(agent.gamma) + ", lr=" + str(
            round(agent.learning_rate, 5)) + ", \u025B=" + str(round(1.0, 3)) + ", \u03B4=" + str(
            round(agent.epsilon_decay, 3)) + ")"
    else:
        subtitle = agent.name + " Agent\n" + variable_name + "=" + str(variable_value)
    title = "Testing Trained " + subtitle
    print(title)
    rewards = test()
    make_plot(title, subtitle, "testing", rewards)
    print('\tAveraged Rewards:', sum(rewards) / len(rewards))
    print("----------------------------------------------")


def agents(Agent, episodes=1000):
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    print("----------------------------------------------")
    # main agent
    agent = Agent(episodes=episodes)
    train_test(agent, None, None)
    gammas = [0.8, 0.85, 0.9, 0.95, 0.99]
    learning_rates = [0.01, 0.005, 0.001]
    epsilons = [1.0]
    epsilon_decays = [0.95, .99, 0.995, .999]

    for gamma in gammas:
        agent = Agent(episodes=episodes, gamma=gamma)
        train_test(agent, "gamma,\u03B3", gamma)
    for alpha in learning_rates:
        agent = Agent(episodes=episodes, alpha=alpha)
        train_test(agent, "learning_rate,lr", alpha)
    for epsilon in epsilons:
        agent = Agent(episodes=episodes, epsilons=epsilons)
        train_test(agent, "epsilon,\u025B", epsilon)
    for epsilon_decay in epsilon_decays:
        agent = Agent(episodes=episodes, epsilon_decay=epsilon_decay)
        train_test(agent, "epsilon_decay,\u03B4", epsilon_decay)


def r_agents(episodes=20000):
    agent = RandomAgent(episodes=episodes)
    states = agent.solve()
    get_observations_limits(states, 20)
    get_observations_limits(states, 10)
    get_observations_limits(states, 8)
    get_observations_limits(states, 5)
    get_observations_limits(states, 3)


if __name__ == '__main__':
    if True:
        agents(DQNAgent, 700)
    else:
        agents(SARSAAgent, 1000)
        agents(QLearningAgent, 1000)
