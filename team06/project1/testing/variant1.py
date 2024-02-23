# This is necessary to find the main code
import sys
sys.path.insert(0, '../../../Bomberman')
sys.path.insert(1, '../../project1')

# Import necessary stuff
import random
from game import Game

# TODO This is your code!
sys.path.insert(1, '../../../team06')
from clydeML import Clyde

wins = 0
num_tries = 500
initial_seed = 300
winning_seeds = []
for i in range(num_tries):
    random.seed(initial_seed + i)
    g = Game.fromfile('../map.txt')

    # TODO Add your character
    g.add_character(Clyde("me", # name
                                "C",  # avatar
                                0, 0  # position
    ))
    if g.go(1):
        wins += 1
        winning_seeds.append(initial_seed + i)

print(f"Guy won {wins} times out of {num_tries} iterations, winning seeds were: {winning_seeds}")

