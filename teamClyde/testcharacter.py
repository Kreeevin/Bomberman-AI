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

DEBUG = False
EIGHT_MOVEMENT = [(-1,-1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]

def debug(str):
    if DEBUG:
        print(str)

class TestCharacter(CharacterEntity):


    
    def do(self, wrld):
        # Commands
        dx, dy = 0,0
        bomb = False
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

            dx, dy = self.minimax(True, wrld, 3)
            debug(f"Player chose to move ({dx},{dy})")
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

# write depth [X] minimax, evaluate goodness using a* distance
# Writing alpha-beta pruning after this part shouldn't be too too bad
    
    def minimax(self, max: bool, world: World, depth: int) -> tuple[int, int]:
        if max:
            return self.maxValue(world, depth)[0]
        else:
            return self.minValue(world, depth)[0]

    def evaluateState(self, world: World, newEvents: list[Event]) -> int:
        
        eventReward = 0
        for event in newEvents:
            if event.tpe == Event.BOMB_HIT_CHARACTER:
                eventReward -= float("-inf")
            if event.tpe == Event.CHARACTER_KILLED_BY_MONSTER:
                eventReward -= float("-inf")
            if event.tpe == Event.CHARACTER_FOUND_EXIT:
                eventReward += 100
            if event.tpe == Event.BOMB_HIT_WALL:
                eventReward += 10
            if event.tpe == Event.BOMB_HIT_MONSTER:
                eventReward += 50

        me = world.me(self)

        if me is None:
            # We either won or died, evaluate accordingly
            return eventReward

        distToExit = len(self.a_star(world, (me.x, me.y), world.exitcell))
        
        # find dist to closest monster
        closestDist = None
        for mList in world.monsters.values():
            # Secondary loop because of weird format of dictionaries (multiple monsters at same index?)
            for m in mList:
                dist = self.euclidean_dist((me.x, me.y), (m.x, m.y))
                if closestDist is None or dist <= closestDist:
                    closestDist = dist
                    
        if closestDist is None:
            monsterPenalty = 0
        else:
            if closestDist > 4:
                monsterPenalty = 0
            else:
                monsterPenalty = 5 + (5 - closestDist)**2
                # return distToExit**(abs(5 - closestDist))

        num_available_moves = 0

        for (dx,dy) in EIGHT_MOVEMENT:
            if self.is_cell_walkable(world, (me.x+dx, me.y+dy)):
                num_available_moves += 1

        movementReward = num_available_moves/2

        debug(f"Monster Penalty: {monsterPenalty}, Distance to Exit: {distToExit}")
        return eventReward + movementReward - monsterPenalty - distToExit

    def maxValue(self, world: World, depth: int) -> tuple[int,int]:

        if world.me(self) is None:
            return ((0,0), float("-inf"))
            
        prevBest = None
        # Loop through all player actions
        for a in EIGHT_MOVEMENT:
            # Grab current instance of player character
            me = world.me(self)
            # Perform action
            me.move(a[0], a[1]) 
            # Update world
            newWorld, newEvents = world.next()
            # Either make recursive call or evaluate
            if depth != 0:  
                val = (a, self.minValue(newWorld, depth-1)[1])
            else:
                val = (a, self.evaluateState(newWorld, newEvents))
            # Save best outcome
            if prevBest is None or val[1] > prevBest[1]:
                prevBest = val
        
        return prevBest

    def minValue(self, world: World, depth: int) -> tuple[int,int]:

        # Check if there are any monsters
        if len(world.monsters.values()) == 0:
            # If no monsters, just return world value
            return ((0,0), self.evaluateState(world, []))
                
        prevWorst = None
        # For each monster
        for mList in world.monsters.values():
            # Secondary loop because of weird format of dictionaries (multiple monsters at same index?)
            for m in mList:
                # Try all actions
                for a in EIGHT_MOVEMENT:
                    m.move(a[0], a[1])
                    newWorld, newEvents = world.next()
                    if depth != 0:  
                        val = (a, self.maxValue(newWorld, depth-1))
                    else:
                        val = (a, self.evaluateState(newWorld, newEvents))
                        if prevWorst is None or val[1] < prevWorst[1]:
                            prevWorst = val
        
        if prevWorst is None:
            debug("This shouldn't happen!")
            return ((0,0), self.evaluateState(world, []))

        return prevWorst


# TODO: Alpha-Beta Pruning (Do as group)
# TODO: Improved evaluate that rewards having escape routes
# TODO: Add metrics for num loops, time, etc.