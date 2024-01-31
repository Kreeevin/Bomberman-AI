# This is necessary to find the main code
import sys
sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back
from world import World

class TestCharacter(CharacterEntity):

    def do(self, wrld):
        # Commands
        dx, dy = 0, 0
        bomb = False
        # Handle input
        ### this is us

        # Execute commands
        self.move(dx, dy)
        if bomb:
            self.place_bomb()

# helper methods
    def manhattan_dist(a: tuple[int, int], b: tuple[int, int]):
        return a[0] + b[0] + a[1] + b[1]

    def euclidean_dist(a: tuple[int, int], b: tuple[int, int]):
        pass


# write a*
    def a_star(self, wrld: World):
        
        
        pass

# write depth [X] minimax, evaluate goodness using a*
# [MAYBE] write markov decision processes