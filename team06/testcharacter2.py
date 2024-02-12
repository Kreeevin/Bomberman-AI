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

DEBUG = True
EIGHT_MOVEMENT = [(-1,-1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]

def debug(str):
    if DEBUG:
        print(str)

class TestCharacter2(CharacterEntity):
    
    def do(self, wrld):
        # Commands
        dx, dy = 0,0
        bomb = False
        self.wavefront = self.make_wavefront(wrld, wrld.exitcell)
        # Handle input
        ### this is us
        if wrld.exitcell is not None:
            #   WORKING CODE FOR SCENARIO 1 - JUST FOLLOWS A*
            # path = self.a_star(wrld, (self.x, self.y), wrld.exitcell)
            # if len(path) > 0:
            #     nextNode = path[0]
            #     print(f"Next node is {nextNode}")
            #     dx, dy = (nextNode[0] - self.x, nextNode[1] - self.y)
            #     if wrld.wall_at(self.x+dx, self.y+dy):
            #         print("Tried to walk into a wall")

            action = self.minimax(wrld, 4)
            dx, dy = action

            # Just in case something is buggy
            self.maybe_place_bomb = False

            debug(f"Player chose to move ({dx},{dy}). Tried to place bomb: {bomb}")
            if wrld.wall_at(self.x+dx, self.y+dy):
                debug("Player tried to walk into a wall")
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
        return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**(1/2)
        
    def is_cell_occupied(self, wrld: World, c: tuple[int, int]):
        return wrld.wall_at(c[0], c[1])

    def is_cell_walkable(self, wrld: World, c: tuple[int, int], me: CharacterEntity = None):
        if me is None:
            me = self
        # init variables

        width = wrld.width()
        height = wrld.height()
        x, y = c

        if x >= width or y >= height or x < 0 or y < 0:    # if cell is out of bounds 
            return False                                                                    
        
        if self.is_cell_occupied(wrld, c):
            return False

        # if wrld.bomb_at(x, y) and not (x == me.x and y == me.y): # bomb we did not just place down
        #     return False
        
        return True

    def neighbors_of_4(self, wrld,  pos: tuple[int, int]):
        # init neighbor array
        neighbors = []
        if self.is_cell_walkable(wrld, (pos[0]-1, pos[1])):
            neighbors.append((pos[0]-1, pos[1]))
        if self.is_cell_walkable(wrld, (pos[0]+1, pos[1])):
            neighbors.append((pos[0]+1, pos[1]))
        if self.is_cell_walkable(wrld, (pos[0], pos[1]-1)):
            neighbors.append((pos[0], pos[1]-1))
        if self.is_cell_walkable(wrld, (pos[0], pos[1]+1)):
            neighbors.append((pos[0], pos[1]+1))
        return neighbors

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
        # print("Executing A* from (%d,%d) to (%d,%d)" % (start[0], start[1], goal[0], goal[1]))

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
    
    def make_wavefront(self, wrld, exitcell):
        if exitcell is None or exitcell == []:
            return {exitcell: 0}

        q = PriorityQueue()

        # dictionary of all the explored points keyed by their coordinates tuple
        explored={} 
        q.put((exitcell,0),0)

        while not q.empty():
            element = q.get()
            cords = element[0]
            g = element[1] #cost so far at this element
            explored[cords] = g

            neighbors=self.neighbors_of_8(wrld, cords)
            
            for neighbor in (neighbors):
                if explored.get(neighbor) is None:
                    q.put((neighbor,g+1),g+1)

        return explored
        
# write depth [X] minimax, evaluate goodness using wavefront distance
# Writing alpha-beta pruning after this part shouldn't be too too bad
    
    def minimax(self, world: World, depth: int) -> tuple[int, int]:
        action, reward = self.maxValue(world, depth, (float("-inf"), float("inf")))
        debug(f"Chose action {action}, which should result in a reward of {reward}")
        return action

    def evaluateState(self, world: World, newEvents: list[Event]) -> int:
        
        eventReward = 0
        for event in newEvents:
            if event.tpe == Event.BOMB_HIT_CHARACTER:
                eventReward += -1000
            if event.tpe == Event.CHARACTER_KILLED_BY_MONSTER:
                eventReward += -1000
            if event.tpe == Event.CHARACTER_FOUND_EXIT:
                eventReward += 690
            if event.tpe == Event.BOMB_HIT_WALL:
                eventReward += 10
            if event.tpe == Event.BOMB_HIT_MONSTER:
                eventReward += 50

        me = world.me(self)

        if me is None:
            # We either won or died, evaluate accordingly
            return eventReward

        # Compare position to wavefront for distance evaluation to exit
        try:
            distToExit = self.wavefront[(me.x, me.y)]
        except KeyError:
            distToExit = self.euclidean_dist((me.x, me.y), world.exitcell)
        
        # find dist to closest monster
        monsterPenalty = 0

        for mList in world.monsters.values():
            # Secondary loop because of weird format of dictionaries (multiple monsters at same index?)
            for m in mList:
                monsterDistToExit = self.wavefront[(m.x,m.y)]
                if distToExit - monsterDistToExit < 0:
                    # Monster is closer to exit than us 
                    dist = len(self.a_star(world, (me.x, me.y), (m.x, m.y)))
                    
                    if dist <= 5:
                        monsterPenalty += 20*(5-dist)
                
        return eventReward - monsterPenalty - distToExit

    def maxValue(self, world: World, depth: int, alphabeta: tuple) -> tuple[int,int]:

        if world.me(self) is None:
            return ((0,0), float("-inf"))

        if depth == 0:
            return self.evaluateState(world, None)

        prevBest = None


        # Loop through all player actions
        for (dx,dy) in EIGHT_MOVEMENT:
            # if dx == 0 and dy == 0:
            #     bomb = True
            # else:
            #     bomb = False
                
            if alphabeta[0] > alphabeta[1]:
                return prevBest
            
            # Grab current instance of player character
            me = world.me(self)

            # Perform action
            me.move(dx, dy) 

            #if next movement puts char inside wall, that state is ignored
            if not self.is_cell_walkable(world, me.nextpos()):
                continue
                
            # Either make recursive call or evaluate
            val = ((dx,dy), self.maxValue(world, depth-1, alphabeta)[1])

            # Save best outcome
            if prevBest is None or val[1] > prevBest[1]:
                # Update alpha-beta value
                alphabeta = (alphabeta[0], val[1])
                prevBest = val
        
        return prevBest

    def minValue(self, world: World, depth: int, alphabeta: tuple) -> tuple[int,int]:

        # This isn't really a min function as much as it is letting the monsters decide 
        # what they would do because for project 1 they are random
        
        # For each monster
        for mList in world.monsters.values():
            # Secondary loop because of weird format of dictionaries (multiple monsters at same index?)
            for m in mList:
                m.do(world)

        # if alphabeta[0] > alphabeta[1]:
        #     return prevBest
        

        newWorld, newEvents = world.next()

        #TODO: how to evaluate world state for each min?
        

        evaluation = self.maxValue(world, depth-1, alphabeta)
        # alphabeta = (val[0], alphabeta[1])

        return evaluation