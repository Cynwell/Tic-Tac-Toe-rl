from game_nn import TicTatToe, QNAgent
from random import random, seed


agent = QNAgent()
random_prob = 0.2
seed()

x_win = []
o_win = []
draw = []
statistic = [0, 0, 0] # O win, X win, draw
for i in range(1000):
    env = TicTatToe()
    state = env.state()
    actions = env.valid_actions()
    while True:
        r = random()
        player = env.current_player
        action = agent.act(state, actions, player, r < random_prob)
        reward = env.move(action)
        next_state =  env.state()
        next_actions = env.valid_actions()
        experience = state, action, next_state, next_actions, reward, player
        agent.train(experience)
        if reward == 1:
            statistic[player] += 1
            break
        elif env.check_draw():
            statistic[2] += 1
            break
        else:
            states = next_state
            actions = next_actions
    if i != 0 and i % 20 == 0:
        x_win.append(statistic[1] / 20)
        o_win.append(statistic[0] / 20)
        draw.append(statistic[2] / 20)
        statistic = [0, 0, 0]


import matplotlib.pyplot as plt
plt.plot(x_win, c='orange')
plt.plot(o_win, c='green')
plt.plot(draw, c='blue')
plt.show()