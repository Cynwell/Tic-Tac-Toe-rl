# Tic-Tac-Toe-RL
## Teach computer how to play Tic Tac Toe via Q-Learning (Pytorch)
- Q_Table : Tabular Q-learning implemented in Jupyter notebook. Positive Q values suggest good move for first player while negative Q values suggest good move for second player. Next state is the state immediately after a move is peformed.
- Q_Supervised : A single neural network to fit the Q-table values learned from Q_table.ipynb.
- Q_Network : Q-Learning with neural network from the ground with two agent compete against each other. Next state the state after the opponent has performed a move.
- game_tab : Environment class for tabular learning
- game_nn : Environment and agent classes for neural network learning and supervised learning
- table : Serialized byte file for storing the Q-table
## Q-value update rules for this project
- Q-table / Q-network update immediately after each move
- Q(s<sub>t</sub> , a<sub>t</sub>) = reward + 0.9 * Q(s<sub>t+1</sub> , a<sub>best</sub>)