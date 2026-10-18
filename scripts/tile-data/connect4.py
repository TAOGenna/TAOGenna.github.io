"""Replay the repo's MCTS self-play (same seed as its make_animation) and dump
moves + per-column root visit counts as JSON for the bento tile.

Run: python scripts/tile-data/connect4.py   (needs numpy + tqdm)
"""
import contextlib, copy, io, json, os, random, sys

import numpy as np

REPO = os.path.expanduser("~/Desktop/kenyi/projects/Connect4-MonteCarlo")
sys.path.insert(0, REPO)
from monte_carlo_connect4 import MCTS, Board, Node  # noqa: E402

random.seed(7)
np.random.seed(7)
BUDGET = 200  # same as the repo's animation

board, turn, moves = Board(), 1, []
while not board.IsTerminal() and board.Winner() == 0:
    root = Node(state=copy.deepcopy(board))
    with contextlib.redirect_stdout(io.StringIO()):  # MCTS prints UCB arrays
        best = MCTS(BUDGET, root)
    visits = [0] * 7
    for ch in root.children:
        d = np.argwhere(ch.state.board != board.board)[0]
        visits[int(d[1])] = int(ch.N)
    r, c = np.argwhere(best.state.board != board.board)[0]
    board.board[r, c] = turn
    board.next[c] += 1
    moves.append({"r": int(r), "c": int(c), "p": turn, "visits": visits})
    turn = -turn

# winning line (first 4-in-a-row found)
b, win = board.board, None
for r in range(6):
    for c in range(7):
        for dr, dc in ((0, 1), (1, 0), (1, 1), (-1, 1)):
            cells = [(r + i * dr, c + i * dc) for i in range(4)]
            if all(0 <= y < 6 and 0 <= x < 7 for y, x in cells) and b[r, c] != 0 and all(b[y, x] == b[r, c] for y, x in cells):
                win = win or cells
out = os.path.join(os.path.dirname(__file__), "../../src/data/tiles/connect4.json")
json.dump({"moves": moves, "win": win}, open(out, "w"), separators=(",", ":"))
print(len(moves), "moves, win:", win, "->", os.path.normpath(out))
