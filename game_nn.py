import torch
from torch import nn
import torch.nn.functional as F

from random import choice

class TicTatToe:

    def __init__(self):
        self.grid = [0 for i in range(18)] + [1 for i in range(9)]
        self.current_player = 0

    def move(self, pos):
        self.grid[pos + 18] = 0
        pos = self.current_player * 9 + pos
        self.grid[pos] = 1
        self.current_player = (self.current_player + 1) % 2
        return self.check_win()

    def check_draw(self):
        return sum(self.grid[18:]) == 0

    def check_win(self):
        sum_check = []
        for i in [0, 9]:
            # Columns
            sum_check.append(self.grid[i] + self.grid[i + 3] + self.grid[i + 6])
            sum_check.append(self.grid[i + 1] + self.grid[i + 4] + self.grid[i + 7])
            sum_check.append(self.grid[i + 2] + self.grid[i + 5] + self.grid[i + 8])
            # Rows
            sum_check.append(self.grid[i] + self.grid[i + 1] + self.grid[i + 2])
            sum_check.append(self.grid[i + 3] + self.grid[i + 4] + self.grid[i + 5])
            sum_check.append(self.grid[i + 6] + self.grid[i + 7] + self.grid[i + 8])
            # Diagonal
            sum_check.append(self.grid[i] + self.grid[i + 4] + self.grid[i + 8])
            sum_check.append(self.grid[i + 2] + self.grid[i + 4] + self.grid[i + 6])
        for s in sum_check:
            if s == 3:
                return 1
        return 0

    def state(self):
        t = torch.tensor(self.grid).float()
        return t.view(1, 27)

    def valid_actions(self):
        a = []
        for i in range(9):
            if self.grid[i + 18] == 1:
                a.append(i)
        return torch.LongTensor(a)

    def visualize(self):
        s = ''
        for i in range(9):
            if self.grid[i] == 1:
                s += 'O'
            if self.grid[9 + i] == 1:
                s += 'X'
            if self.grid[18 + i] == 1:
                s += ' '
            if i % 3 == 2:
                s += '\n'
        print(s)



class QNAgent(nn.Module):

    def __init__(self):
        super(QNAgent, self).__init__()
        self.fc1 = nn.Linear(27, 120)
        self.fc2 = nn.Linear(120, 120)
        self.fc3 = nn.Linear(120, 9)
        self.optim = torch.optim.SGD(self.parameters(), lr=0.05)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = torch.tanh(self.fc3(x))
        return y

    def train(self, experience, lr=0.6):
        state, action, next_state, next_actions, reward, player = experience
        if player == 1:
            reward = -reward
        if reward == 1 or len(next_actions) == 0:
            next_best_q = 0
        elif player == 0:
            next_q = self.forward(next_state).view(9)
            next_best_q = torch.min(next_q[next_actions])
        else:
            next_q = self.forward(next_state).view(9)
            next_best_q = torch.max(next_q[next_actions])
        current_q = self.forward(state)[0, action]
        y = torch.tensor(reward + 0.9 * next_best_q)
        loss = (current_q - y.detach()) ** 2
        loss.backward()
        self.optim.step()
        self.optim.zero_grad()

    def act(self, state, actions, player, random=False):
        if random:
            actions = actions.tolist()
            return choice(actions)
        elif player == 0:
            a = self.forward(state).view(9)
            a = a[actions] # select valid Q values
            a = torch.argmax(a) # Select the index of best Q values
            return int(actions[a])
        else:
            a = self.forward(state).view(9)
            a = a[actions] # select valid Q values
            a = torch.argmin(a) # Select the index of best Q values
            return int(actions[a])


