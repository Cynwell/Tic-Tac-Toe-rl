from game_nn import TicTatToe, QNAgent
from random import random, seed

import warnings
warnings.filterwarnings('ignore')

# Create agent and random seed
o_agent = QNAgent()
x_agent = QNAgent()
random_prob = 0.5
seed()

# Win rate tracking
x_win = []
o_win = []
draw = []
statistic = [0, 0, 0] # O win, X win, draw

# Training loop
for i in range(25000):
    env = TicTatToe() # Reinitialize board
    o_state = env.state()
    o_actions = env.valid_actions()
    while True:
        o_action = o_agent.act(o_state, o_actions, random() < random_prob)
        o_reward = env.move(o_action)
        if env.check_draw():  # Game draw
            statistic[2] += 1
            break
        elif o_reward == 1:   # First player win
            o_agent.train(o_state, o_action, o_next_state, o_next_actions, 1)
            x_agent.train(x_state, x_action, x_next_state, x_next_actions, -1)
            statistic[0] += 1
            break
        elif len(o_actions) < 9:  # Update for the second agent starts in second round
            x_next_state = env.state()
            x_next_actions = env.valid_actions()
            x_agent.train(x_state, x_action, x_next_state, x_next_actions, 0)
            x_state = x_next_state
            x_actions = x_next_actions
        else:
            x_state =  env.state()
            x_actions = env.valid_actions()
        x_action = x_agent.act(x_state, x_actions, random() < random_prob)
        x_reward = env.move(x_action)
        o_next_state = env.state()
        o_next_actions = env.valid_actions()
        if x_reward == 1:  # Second player win
            o_agent.train(o_state, o_action, o_next_state, o_next_actions, -1)
            x_agent.train(x_state, x_action, x_next_state, x_next_actions, 1)
            statistic[1] += 1
            break
        else:
            o_agent.train(o_state, o_action, o_next_state, o_next_actions, 0)
            o_state = o_next_state
            o_actions = o_next_actions
    if i != 0 and i % 100 == 0:  # Gather statistics every 100 games 
        random_prob *= 0.99  # Decrease move randomness
        print(i, 'games played')
        o_win.append(statistic[0] / 100)
        x_win.append(statistic[1] / 100)
        draw.append(statistic[2] / 100)
        statistic = [0, 0, 0]

# Plot changes in win rate throughout trainig
import matplotlib.pyplot as plt
plt.plot(o_win, c='red')
plt.plot(x_win, c='blue')
plt.plot(draw, c='green')
plt.show()

# Test the resulting agent by playing against it
env = TicTatToe()
while True:
    action = int(input('Your action:'))
    reward = env.move(action)
    env.visualize()
    state = env.state()
    if reward == 1 or env.check_draw():
        break
    actions = env.valid_actions()
    action = x_agent.act(state, actions)
    reward = env.move(action)
    env.visualize()
    if reward == 1 or env.check_draw():
        break