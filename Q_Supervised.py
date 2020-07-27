import pickle as pk
import torch
from game_nn import TicTatToe, QSAgent

def s2tensor(s):
    s = eval(s)
    t = [0 for i in range(27)]
    for i, j in enumerate(s):
        t[j * 9 + i] = 1
    t = t[9:] + t[:9]
    return torch.tensor(t).float()

def q2tensor(q):
    t = []
    for i in range(9):
        if i in q:
            t.append(q[i])
        else:
            t.append(0.)
    return torch.tensor(t).float()

X = []
Y = []
f = open('table', 'rb')
q_table = pk.load(f)
for s, q in q_table.items():
    s = s2tensor(s)
    q = q2tensor(q)
    X.append(s)
    Y.append(q)

agent = QSAgent()
X = torch.stack(X, dim=0)
Y = torch.stack(Y, dim=0)
optim = torch.optim.SGD(agent.parameters(), lr=0.005)

no_batch = len(q_table) // 100
no_epoch = 200
loss_record = []
for j in range(no_epoch):
    total_loss = 0
    for i in range(no_batch):
        if i != no_batch:
            x = X[100 * i: 100*(i + 1)]
            y = Y[100 * i: 100*(i + 1)]
        else:
            x = X[100 * i:]
            y = Y[100 * i:]
        loss = torch.sum((agent(x) - y) ** 2)
        loss.backward()
        optim.step()
        optim.zero_grad()
        total_loss += float(loss)
    loss_record.append(total_loss)
    print(j, ' ', total_loss)


env = TicTatToe()
while True:
    action = int(input('Your action:'))
    reward = env.move(action)
    env.visualize()
    state = env.state()
    if reward == 1 or env.check_draw():
        break
    actions = env.valid_actions()
    action = agent.act(state, actions, 1)
    reward = env.move(action)
    env.visualize()
    if reward == 1 or env.check_draw():
        break
    


