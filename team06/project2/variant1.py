# This is necessary to find the main code
import sys
import random
sys.path.insert(0, '../../Bomberman')
sys.path.insert(1, '..')

# Import necessary stuff
from game import Game

# TODO This is your code!
sys.path.insert(1, '../team06')
from clydeML import ClydeML

initial_seed = 200
num_tries = 1
wins = 0
winning_seeds = []
for i in range(num_tries):
    random.seed(initial_seed+i)
    # Create the game
    g = Game.fromfile('map.txt')

    # TODO Add your character
    g.add_character(ClydeML("me", # name
                                "C",  # avatar
                                0, 0,  # position
                                filename="variant1-3.json"
    ))
    if g.go(1):
        wins += 1
        winning_seeds.append(initial_seed + i)
    # input()

print(f"Guy won {wins} times out of {num_tries} iterations, winning seeds were: {winning_seeds}")


