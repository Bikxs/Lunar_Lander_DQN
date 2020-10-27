import os

import matplotlib.pyplot as plt

from constants import *
from dqn_agent import DQNAgent


def dump_rewards(file_name, rewards):
    with open(file_name, 'w') as f:
        for item in rewards:
            f.write("%s\n" % item)


def make_plot(title, folder, name, rewards):
    plt.figure()
    plt.title(title)
    plt.plot(rewards)
    plt.ylabel('Rewards')
    plt.xlabel('Episodes')
    filename = 'plots/' + folder
    if not os.path.exists(filename):
        os.makedirs(filename)
    plt.savefig(filename + '/' + name + '.png', bbox_inches='tight')
    dump_rewards(filename + '/' + name + '_rewards.txt', rewards)


def test(agent, testing_episodes):
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


def train_test(datum):
    variable_name, variable_value, agent = datum
    print(datum)

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
    rewards = test(agent, 100)
    make_plot(title, subtitle, "testing", rewards)
    avgr = sum(rewards) / len(rewards)
    print('\tAveraged Rewards:', avgr)
    print("----------------------------------------------")
    return title, avgr


def agents(episodes=1000):
    print("----------------------------------------------")
    # main agent
    agent = DQNAgent(episodes=episodes)
    # train_test(agent, None, None)
    gammas = [0.8, 0.85, 0.9, 0.95, 0.99]
    learning_rates = [0.01, 0.005, 0.001]
    epsilons = [1.0]
    epsilon_decays = [0.95, .99, 0.995, .999]

    agents = [(None, None, agent)]
    agents.extend([("gamma,\u03B3", gamma, DQNAgent(episodes=episodes, gamma=gamma)) for gamma in gammas])
    agents.extend([("learning_rate,lr", alpha, DQNAgent(episodes=episodes, alpha=alpha)) for alpha in learning_rates])
    agents.extend([("gamma,\u03B3", gamma, DQNAgent(episodes=episodes, gamma=gamma)) for gamma in gammas])
    agents.extend([("epsilon,\u025B", epsilon, DQNAgent(episodes=episodes, epsilon=epsilon)) for epsilon in epsilons])
    agents.extend([("epsilon_decay,\u03B4", epsilon_decay, DQNAgent(episodes=episodes, epsilon_decay=epsilon_decay)) for
                   epsilon_decay in epsilon_decays])

    return agents


results = []


def collect_result(result):
    global results
    print(results)
    results.append(result)


def para_train_test():
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    agents_data = agents(1000)
    print("Agents:", len(agents_data))
    for datum in agents_data:
        train_test(datum)

    import multiprocessing as mp
    print("Number of processors: ", mp.cpu_count())

    pool = mp.Pool(mp.cpu_count() - 1)
    pool.map_async(train_test, agents_data, callback=collect_result())
    print(results)
    pool.close()


if __name__ == '__main__':
    para_train_test()
