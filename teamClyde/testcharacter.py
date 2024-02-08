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

class TestCharacter(CharacterEntity):
    
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

            action = self.minimax(wrld, 3)
            dx, dy = action[0]
            bomb = action[1]

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
        
    def simulateBombExplosion(self, world):
        # Makes a new wavefront of the world assuming all current bombs have exploded.
        for b in world.bombs.values():
            world.add_blast(b)

        newWorld, newEvents = world.next()

        if Event.BOMB_HIT_WALL in newEvents:
            debug("Bomb simulation worked")

        return self.make_wavefront(newWorld, newWorld.exitcell)

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
                eventReward += float("-inf")
            if event.tpe == Event.CHARACTER_KILLED_BY_MONSTER:
                eventReward += float("-inf")
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

        # Compare position to wavefront for distance evaluation to exit
        try:
            distToExit = 0.75*self.wavefront[(me.x, me.y)]
        except KeyError:
            distToExit = self.euclidean_dist((me.x, me.y), world.exitcell)
        
        # find dist to closest monster
        closestDist = None
        for mList in world.monsters.values():
            # Secondary loop because of weird format of dictionaries (multiple monsters at same index?)
            for m in mList:
                dist = self.euclidean_dist((me.x, me.y), (m.x, m.y))
                if closestDist is None or dist <= closestDist:
                    closestDist = dist

        bombDistToExit = 0 
        for b in world.bombs.values():
            # Offer a large reward if a bomb would make a shorter path to the exit
            resultingWorldState = self.simulateBombExplosion(world)
            bombDistToExit += 10*(distToExit - 0.75*resultingWorldState[(me.x, me.y)])
        
        if closestDist is None:
            monsterPenalty = 0
        else:
            if closestDist > 2.5:
                monsterPenalty = 0
            else:
                monsterPenalty = 5 + (5 - closestDist)**2

        num_available_moves = 0

        for (dx,dy) in EIGHT_MOVEMENT:
            if self.is_cell_walkable(world, (me.x+dx, me.y+dy)):
                num_available_moves += 1

        movementReward = num_available_moves/2
        
        # debug(f"If I used a bomb, the new path would reward {bombDistToExit}")

        return eventReward + movementReward - monsterPenalty - distToExit + bombDistToExit

    def maxValue(self, world: World, depth: int, alphabeta: tuple) -> tuple[int,int]:

        if world.me(self) is None:
            return (((0,0), False), float("-inf"))

        prevBest = (((0,0), False), float("-inf"))

        actions = EIGHT_MOVEMENT.copy()
        actions.append((0,0))

        # Loop through all player actions
        for (dx,dy) in actions:
            if dx == 0 and dy == 0:
                bomb = True
            else:
                bomb = False
                
            if alphabeta[0] > alphabeta[1]:
                return prevBest
            
            # Grab current instance of player character
            me = world.me(self)

            me.move(dx, dy) 

            #if next movement puts char inside wall, that state is ignored
            if not self.is_cell_walkable(world, me.nextpos()):
                continue
                
            # Perform action
            if bomb:
                me.place_bomb()
            else:
                me.maybe_place_bomb = False
            # Update world
            newWorld, newEvents = world.next()
            # Either make recursive call or evaluate
            if depth != 0 and len(newWorld.monsters.values()) != 0:
                val = (((dx,dy),bomb), self.minValue(newWorld, depth, alphabeta)[1])
            else:
                val = (((dx,dy),bomb), self.evaluateState(newWorld, newEvents))
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
        

        evaluation = self.maxValue(newWorld, depth-1, alphabeta)
        # alphabeta = (val[0], alphabeta[1])

        return evaluation


# TODO: Alpha-Beta Pruning (Do as group)

# functions:

def maxValueAB(state: World, alphabeta: tuple): # returns a utility value
    if terminalTest(state): 
        return # utility(state)
        pass 

    v = float("-inf")
    for a in actions(state):
        v = max(v, minValueAB(result(state, actions), alphabeta[0], alphabeta[1]))

        if v >= alphabeta[1]: return v
        alphabeta[0] = max(alphabeta[0], v)
        pass

    return v

    pass

def minValueAB(state: World, alphabeta: tuple): # returns a utility value
    if terminalTest(state): 
        return # utility(state)
        pass 

    v = float("inf")
    for a in actions(state):
        v = min(v, maxValueAB(result(state, actions), alphabeta[0], alphabeta[1]))

        if v <= alphabeta[0]: return v
        alphabeta[1] = max(alphabeta[1], v)
        pass

    return v

    pass

# terminal_test determines if a state ends the gane or not
#does character die? no? great you pass

def expectimax_AB_pruning(state: World) -> tuple [int, int]: # output is action, check datatype
    v = maxValueAB(state, float("-inf"), float("inf"))
    #return the action in Actions(state) with value v
    pass

# TODO: Improved evaluate that rewards having escape routes

# TODO: Add metrics for num loops, time, etc.