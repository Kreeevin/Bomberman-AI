# This is necessary to find the main code
import sys
sys.path.insert(0, '../../Bomberman')
sys.path.insert(1, '..')

# Import necessary stuff
import random
from game import Game
from monsters.selfpreserving_monster import SelfPreservingMonster

# TODO This is your code!
sys.path.insert(1, '../team06')
from reset import ResetChar
from abusing_bugs import BuggyCharacter
from interactivecharacter import InteractiveCharacter

# Create the game
# 277 monster cheats, guy dies
# 123 guy wins
# 124 guy wins
# 125 guy loses
# 126 guy wins
# 127 guy wins
# 128 guy wins

# random.seed(129) # TODO Change this if you want different random choices
# g = Game.fromfile('map.txt')
# g.add_monster(SelfPreservingMonster("aggressive", # name
#                                     "A",          # avatar
#                                     3, 13,        # position
#                                     2             # detection range
# ))

# # TODO Add your character
# g.add_character(ResetChar("me", # name
#                               "C",  # avatar
#                               0, 0  # position
# ))

wins = 0
num_tries = 20
initial_seed = 300
winning_seeds = []
for i in range(num_tries):
    random.seed(initial_seed + i)
    g = Game.fromfile('map.txt')
    g.add_monster(SelfPreservingMonster("aggressive", # name
                                        "A",          # avatar
                                        3, 13,        # position
                                        2             # detection range
    ))

    # TODO Add your character
    g.add_character(ResetChar("me", # name
                                "C",  # avatar
                                0, 0  # position
    ))
    if g.go(1):
        wins += 1
        winning_seeds.append(initial_seed + i)

print(f"Guy won {wins} times out of {num_tries} iterations, winning seeds were: {winning_seeds}")

# Run!
# g.go(1)
