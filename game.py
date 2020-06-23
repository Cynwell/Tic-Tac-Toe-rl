import numpy as np


# 1D implementation
class TicTacToe:
    def __init__(self, board=None, board_size=3, players=None, first_player=None):
        self.board_size = board_size
        if board is None:
            self.board = np.zeros((self.board_size ** 2), dtype=int)
        else:
            # Make sure the board is of 1D array
            self.board = np.array(board)

        self.EMPTY = 0
        if players is None:
            self.players = {self.EMPTY: '?', 1: 'X', 2: 'O'}
        else:
            self.players = players

        self.playerCount = len(self.players) - 1
        self.history = list()
        if first_player is None:
            self.player = 1
        else:
            self.player = first_player

    def visualize(self):
        print(f'Round {len(self.history) + 1}:', end='')
        for i, c in enumerate(self.board):
            if i % self.board_size == 0:
                print()
            print(self.players[c], end='\t')
        print()

    # The pos is 0-based
    def placeMove(self, pos):
        if self.isAvailable(pos):
            self.board[pos] = self.player
            self.history.append([pos, self.player])
            self.player = self.getNextPlayer()
        return self.checkGameEnds()

    def isAvailable(self, pos):
        if 0 <= pos < self.board_size ** 2:
            return self.board[pos] == self.EMPTY
        return False

    def getCurrentPlayer(self):
        return self.player

    def getNextPlayer(self):
        if self.player == self.playerCount:
            self.player = 1
        else:
            self.player += 1
        return self.player

    def checkGameEnds(self):
        # -1 means no one wins and all spaces has used up
        # 0 means the game hasn't ended yet
        # If a player won, it'll return the corresponding integer that represents the player
        reshaped_board = self.board.reshape(self.board_size, self.board_size)
        for row in reshaped_board:
            if len(set(row)) == 1 and row[0] != 0:
                return row[0]

        for column in reshaped_board.transpose():
            if len(set(column)) == 1 and column[0] != 0:
                return column[0]

        diagonal = [reshaped_board[i][i] for i in range(self.board_size)]
        if len(set(diagonal)) == 1 and diagonal[0] != 0:
            return diagonal[0]

        inverted_diagonal = [np.flip(reshaped_board, 0)[i][i] for i in range(self.board_size)]
        if len(set(inverted_diagonal)) == 1 and inverted_diagonal[0] != 0:
            return inverted_diagonal[0]

        if self.EMPTY in set(reshaped_board.flatten().tolist()):
            return 0
        else:
            return -1

    def explain(self):
        state = self.checkGameEnds()
        if state == -1:
            print('No one wins and all spaces has used up.')
        elif state == 0:
            print('The game hasn\'t ended yet.')
        else:
            print(f'Player {state} ({self.players[state]}) won.')


def main():
    board_size = 3
    game = TicTacToe(board_size=board_size)
    while game.checkGameEnds() == 0:
        game.visualize()
        player = game.getCurrentPlayer()
        pos = int(input(f'Player {player}\'s move (0 to {board_size ** 2 - 1}): '))
        game.placeMove(pos)
    game.visualize()
    game.explain()


if __name__ == '__main__':
    main()
