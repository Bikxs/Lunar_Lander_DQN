import numpy as np

MAX_ITERATIONS = 2000

state_map = {
    'theta': [-float('inf'), -0.5, -0.1, 0., 0.1, 0.5],
    'theta_': [-float('inf'), -0.4, -0.1, 0., 0.1, 0.4],
    'x': [-float('inf'), -0.4, -0.1, 0.1, 0.4, ],
    'x_': [-float('inf'), -0.3, -0.1, 0.1, 0.3],
    'y': [-float('inf') - 0.1, 0.1, 0.5, 1.0],
    'y_': [-float('inf'), -0.5, -0.1, 0.1]}


def discrete_state(observation):
    x, y, x_, y_, theta, theta_, legL, legR = observation
    right = True
    x = np.digitize(x, state_map['x'], right=right)
    y = np.digitize(y, state_map['y'], right=right)
    x_ = np.digitize(x_, state_map['x_'], right=right)
    y_ = np.digitize(y_, state_map['y_'], right=right)
    theta = np.digitize(theta, state_map['theta'], right=right)
    theta_ = np.digitize(theta_, state_map['theta_'], right=right)
    legL = int(legL)
    legR = int(legR)
    return x, y, x_, y_, theta, theta_, legL, legR
