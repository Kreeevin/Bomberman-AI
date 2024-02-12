# This is necessary to find the main code
import sys
sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back
from world import World
from priority_queue import PriorityQueue
import math
import numpy as np
from events import Event
from types import MethodType
from sensed_world import SensedWorld
from monsters.stupid_monster import StupidMonster
from monsters.selfpreserving_monster import SelfPreservingMonster
from entity import ExplosionEntity


EIGHT_MOVEMENT = [(-1,-1), (0, -1), (1, -1),
                  (-1, 0),           (1, 0),
                  (-1, 1),  (0, 1),  (1, 1)]

DEBUG = True

decay = 1

def debug(str):
    if DEBUG:
        print(str)

class BuggyCharacter(CharacterEntity):

    def __init__(self, name, avatar, x, y):
        super().__init__(name, avatar, x, y)
        self.turncount = 0
        self.from_world = SensedWorld.from_world

    def do(self, world):
        # Commands
        dx, dy = 0,0
        bomb = False
        
        self.rocketman(world)

        # Execute commands
        self.move(dx, dy)

        debug(f"New player postion: {self.nextpos()}")
        
        if bomb:
            self.place_bomb()
        self.turncount += 1

    def rocketman(self, world):
        if self.turncount == 0:
            SensedWorld.from_world = MethodType(lambda cls, world:world, SensedWorld)

        if self.turncount == 1:
            world.monsters = {0:[]}

        if self.turncount > 1:
            path = self.a_star(world, (self.x, self.y), world.exitcell, ignoreWalls=True)
            if len(path) > 0:
                nextNode = path[0]
                print(f"Next node is {nextNode}")
                dx, dy = (nextNode[0] - self.x, nextNode[1] - self.y)

            if world.wall_at(self.x+dx, self.y+dy):
                world.grid[self.x+dx][self.y+dy] = False

            self.explosions[self.index(self.x,self.y)] = ExplosionEntity(self.x+dx, self.y+dy, self.expl_duration, self)

    def earthbender(self, world):
        if self.turncount == 0:
            SensedWorld.from_world = MethodType(lambda cls, world:world, SensedWorld)

        if self.turncount == 1:
            for x in range(world.width()):
                for y in range(world.height()):
                    world.grid[x][y] = False

            for mList in world.monsters.values():
                # Secondary loop because of weird format of dictionaries (multiple monsters at same index?)
                for monster in mList:
                    for (dx,dy) in EIGHT_MOVEMENT:
                        world.grid[monster.x+dx][monster.y+dy] = True

        if self.turncount > 1:
            path = self.a_star(world, (self.x, self.y), world.exitcell, ignoreWalls=False)
            if len(path) > 0:
                nextNode = path[0]
                print(f"Next node is {nextNode}")
                dx, dy = (nextNode[0] - self.x, nextNode[1] - self.y)

    def timelord(self, world):
        if self.turncount == 0:
            SensedWorld.from_world = MethodType(lambda cls, world:world, SensedWorld)

        if self.turncount >= 1:
            for mList in world.monsters.values():
                # Delete monster's movement
                for monster in mList:
                    monster.move(0,0)
            path = self.a_star(world, (self.x, self.y), world.exitcell, ignoreWalls=False)
            if len(path) > 0:
                nextNode = path[0]
                print(f"Next node is {nextNode}")
                dx, dy = (nextNode[0] - self.x, nextNode[1] - self.y)



    def a_star(self, wrld: World, start: tuple[int, int], goal: tuple[int, int], ignoreWalls:bool = True) -> list[tuple[int, int]]:
        """
        Calculates the Optimal path using the A* algorithm.
        Publishes the list of cells that were added to the original map.
        :param wrld [World] The world data.
        :param start [int]           The starting grid location to pathfind from.
        :param goal [int]           The target grid location to pathfind to.
        :return        [list[tuple(int, int)]] The Optimal Path from start to goal.
        """
        # print("Executing A* from (%d,%d) to (%d,%d)" % (start[0], start[1], goal[0], goal[1]))

        # Check if start and goal are walkable
        if(not self.isCellWalkable(wrld, start)):
            print('start blocked')
            return []
        elif(not self.isCellWalkable(wrld, goal)):
            print('goal blocked')
            return []

        #Priority queue for the algorithm
        q = PriorityQueue()

        # dictionary of all the explored points keyed by their coordinates tuple
        explored={} 
        q.put((start,None,0),self.euclideanDist(start,goal))

        while not q.empty():
            element = q.get()
            cords = element[0]
            g = element[2] #cost so far at this element
            explored[cords] = element

            if cords == goal:
                # Once we've hit the goal, reconstruct the path and then return it
                return self.reconstructPath(explored,start,goal)
            
            neighbors=self.neighbors_of_8(wrld, cords, ignoreWalls)
            
            for i in range(len(neighbors)):
                neighbor=neighbors[i]
                if explored.get(neighbor) is None or explored.get(neighbor)[2] > g + 1:
                    f = g + 1 + self.euclidean_dist(neighbor,goal)
                    q.put((neighbor,cords,g+1),f)
        
        # this only happens if no exit can be fond, queue runs out
        debug('Could not reach goal')
        
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
        # debug(f"A* found path of length {len(path)}")
        return path
    
    def euclideanDist(self, a: tuple[int, int], b: tuple[int, int]):
        return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**(1/2)

    
    def isCellWalkable(self, world: World, pos: tuple[int, int]):
        # init variables

        width = world.width()
        height = world.height()
        x, y = pos

        if x >= width or y >= height or x < 0 or y < 0:    # if cell is out of bounds 
            return False                                                                    
        
        if world.wall_at(x, y):
            return False
        
        return True



    
    def neighbors_of_8(self, wrld, pos: tuple[int, int], ignoreWalls: bool = False):
        # init neighbor array
        neighbors = []

        #loop through neighbors
        for y_offset in [-1,0,1]: 
            for x_offset in [-1,0,1]:

                point = (pos[0] + x_offset, pos[1] + y_offset) # calculate the point to check

                if self.isCellWalkable(wrld, point) and point != pos or ignoreWalls: 
                    neighbors.append(point) #append to return list if walkable

        return neighbors