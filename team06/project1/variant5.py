# This is necessary to find the main code
import sys
sys.path.insert(0, '../../Bomberman')
sys.path.insert(1, '..')

# Import necessary stuff
import random
from game import Game
from monsters.stupid_monster import StupidMonster
from monsters.selfpreserving_monster import SelfPreservingMonster

# TODO This is your code!
sys.path.insert(1, '../team06')
from testcharacter import TestCharacter
from reset import ResetChar
from abusing_bugs import BuggyCharacter

# Create the game
# 123 - wins
# 124 - loses
# 125 - loses
# 126 - win
# 127 - win
# 128 - lose

random.seed(128) # TODO Change this if you want different random choices
g = Game.fromfile('map.txt')
g.add_monster(StupidMonster("stupid", # name
                            "S",      # avatar
                            3, 5,     # position
))
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

# Run!
g.go(1)
