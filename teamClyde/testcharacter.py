# This is necessary to find the main code
import sys
sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back
from world import World
from priority_queue import PriorityQueue

class TestCharacter(CharacterEntity):
    
    def do(self, wrld):
        # Commands
        dx, dy = 0,0
        bomb = False
        # Handle input
        ### this is us
        if wrld.exitcell is not None:
            path = self.a_star(wrld, (self.x, self.y), wrld.exitcell)
            if len(path) > 0:
                nextNode = path[1]
                dx, dy = (nextNode[0] - self.x, nextNode[1] - self.y)
        else:
            print("No Exit Found")
        # Execute commands
        self.move(dx, dy)
        if bomb:
            self.place_bomb()

# helper methods
    def manhattan_dist(self, a: tuple[int, int], b: tuple[int, int]):
        return a[0] + b[0] + a[1] + b[1]

    def euclidean_dist(self, a: tuple[int, int], b: tuple[int, int]):
        return ((a[0]+b[0])**2 + (a[1]+b[1])**2)**(1/2)
        
    def is_cell_occupied(self, wrld: World, c: tuple[int, int]):
        return (wrld.wall_at(c[0], c[1]) or 
                wrld.monsters_at(c[0], c[1]) or 
                wrld.explosion_at(c[0], c[1]))

    def is_cell_walkable(self, wrld: World, c: tuple[int, int],):
        # init variables

        width = wrld.width()
        height = wrld.height()
        x, y = c

        if x >= width or y >= height or x < 0 or y < 0:    # if cell is out of bounds 
            return False                                                                    
        
        if self.is_cell_occupied(wrld, c):
            return False

        if wrld.bomb_at(x, y) and not (x == self.x and y == self.y): # bomb we did not just place down
            return False
        
        return True


    def neighbors_of_8(self, wrld,  pos: tuple[int, int]):
        # init neighbor array
        neighbors = []

        #loop through neighbors
        for y_offset in [-1,0,1]: 
            for x_offset in [-1,0,1]:

                point = (pos[0] + x_offset, pos[1] + y_offset) # calculate the point to check

                if self.is_cell_walkable(wrld, point) and point != pos: 
                    neighbors.append(point) #append to return list if walkable

        return neighbors
 
# write a*
    def a_star(self, wrld: World, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
        """
        Calculates the Optimal path using the A* algorithm.
        Publishes the list of cells that were added to the original map.
        :param wrld [World] The world data.
        :param start [int]           The starting grid location to pathfind from.
        :param goal [int]           The target grid location to pathfind to.
        :return        [list[tuple(int, int)]] The Optimal Path from start to goal.
        """
        print("Executing A* from (%d,%d) to (%d,%d)" % (start[0], start[1], goal[0], goal[1]))

        # Check if start and goal are walkable
        if(not self.is_cell_walkable(wrld, start)):
            print('start blocked')
            return []
        elif(not self.is_cell_walkable(wrld, goal)):
            print('goal blocked')
            return []

        #Priority queue for the algorithm
        q = PriorityQueue()

        # dictionary of all the explored points keyed by their coordinates tuple
        explored={} 
        q.put((start,None,0),self.euclidean_dist(start,goal))

        while not q.empty():
            element = q.get()
            cords = element[0]
            g = element[2] #cost so far at this element
            explored[cords] = element

            if cords == goal:
                # Once we've hit the goal, reconstruct the path and then return it
                return self.reconstructPath(explored,start,goal)
            
            neighbors=self.neighbors_of_8(wrld, cords)
            
            for i in range(len(neighbors)):
                neighbor=neighbors[i]
                if explored.get(neighbor) is None or explored.get(neighbor)[2] > g + 1:
                    f = g + 1 + self.euclidean_dist(neighbor,goal)
                    q.put((neighbor,cords,g+1),f)
        
        # this only happens if no exit can be fond, queue runs out
        print('Could not reach goal')
        
        return []

        
    def reconstructPath(self, explored: dict, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:   
        """
        A helper function to reconstruct the path from the explored dictionary
        :param explored [dict] The dictionary of explored nodes
        :param start [tuple(int, int)] The starting point
        :param goal [tuple(int, int)] The goal point
        :return        [list[tuple(int, int)]] The Optimal Path from start to goal.
        """
        
        cords = goal
        path = []
       
        # Loops backwards through the explored dictionary to reconstruct the path
        while cords != start:
            
            element = explored[cords]
            path = [cords] + path
            cords = element[1]
            
            if cords == None:
                # This should never happen given the way the algorithm is implemented
                print('Could not reconstruct path')
                return []

        return path

# write depth [X] minimax, evaluate goodness using a* distance
    def minimax():
        
        pass

# [MAYBE] write markov decision processes (he hasn't finished teaching this so maybe not)
   