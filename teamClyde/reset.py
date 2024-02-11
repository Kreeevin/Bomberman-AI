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

from monsters.stupid_monster import StupidMonster
from monsters.selfpreserving_monster import SelfPreservingMonster


EIGHT_MOVEMENT = [(-1,-1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]

DEBUG = True

def debug(str):
    if DEBUG:
        print(str)

class ResetChar(CharacterEntity):
    
    def do(self, world):
        # Commands
        dx, dy = 0,0
        bomb = False
        self.wavefront = self.make_wavefront(world, world.exitcell)

        me = world.me(self)
        me.move(0, 0)
        action, utility = self.expectimax(world, 5)

        debug(f"Chose action {action}, which has a utility of {utility}")

        if action is not None:
            (dx, dy) = action

        # Execute commands
        self.move(dx, dy)

        debug(f"New player postion: {self.nextpos()}")

        if bomb:
            self.place_bomb()


    def expectimax(self, world, depth) -> tuple:

        newWorld, newEvents = world.next()
        
        if self.isTerminal(newWorld, depth):
            return None, self.evaluateState(newWorld, newEvents)
        
        # update
        return self.maxNode(newWorld, depth)


    def isTerminal(self, world, depth):
        isTerminalState = world.me(self) is None
        
        return depth <= 0 or isTerminalState
    
    
    def maxNode(self, world, depth):
        bestMove = None
        me = world.me(self)
        validMoves = self.validMoves(world, me)
        debug(f"Player Position: {me.x, me.y}, Valid Moves: {validMoves}")
        for (dx, dy) in validMoves:
            me.move(dx, dy)
            utility = self.chanceNode(world, depth)
            
            if bestMove is None or utility > bestMove[1]:
                bestMove = (dx,dy), utility

        debug(f"Player chose action {bestMove[0]}, which has utility {bestMove[1]}")
        return bestMove
    
    # Returns chance utility
    def chanceNode(self, world, depth):
        # For each monster
        utility = 0
        numMonsters = 0
        for [monster] in world.monsters.values():
            
            numMonsters += 1
            
            # Check what type of monster            
            isRandom = False
            
            if type(monster) == SelfPreservingMonster:
                if monster.must_change_direction(world):
                    isRandom = True
            elif type(monster) == StupidMonster:
                isRandom = True
            
            # if self preserving, check if next step is random
            # if not random, perform next step
            # if random monster or random step, perform chance node behavior
            if isRandom:
                # chance node behavior   
                # if not random, perform next step
                validMoves = self.validMoves(world, monster)
                for (dx, dy) in validMoves:
                    monster.move(dx, dy)

                    _, partialUtility = self.expectimax(world, depth-1)

                    utility += partialUtility / len(validMoves)

            else:     
                monster.do(world)

                utility += self.expectimax(world, depth-1)[1]

            # debug(f"Monster behavior is Random: {isRandom}, Behavior found has resulting utility: {utility}")

        return utility / numMonsters


    def validMoves(self, world, entity):
        validMoves = []
        for dx,dy in EIGHT_MOVEMENT:
            if self.isCellWalkable(world, (entity.x + dx, entity.y + dy)):
                validMoves.append((dx, dy))

        return validMoves

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

    def evaluateState(self, world: World, newEvents: list[Event]) -> int:
        # Use a* distance for distance to monster, use the walls to your advantage
        # Account for if monster is along path to exit
        # Run a* for yourself to the exit and monster to the exit, compare lengths
        
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
                if distToExit - monsterDistToExit < 5:
                    # Monster is closer to exit than us (or farther by less than 3 moves)
                    dist = len(self.a_star(world, (me.x, me.y), (m.x, m.y)))
                    
                    if dist <= 5:
                        monsterPenalty += 20*(5-dist)
                
        return eventReward - monsterPenalty - distToExit
    
    
    def neighbors_of_8(self, wrld, pos: tuple[int, int]):
        # init neighbor array
        neighbors = []

        #loop through neighbors
        for y_offset in [-1,0,1]: 
            for x_offset in [-1,0,1]:

                point = (pos[0] + x_offset, pos[1] + y_offset) # calculate the point to check

                if self.isCellWalkable(wrld, point) and point != pos: 
                    neighbors.append(point) #append to return list if walkable

        return neighbors
 

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
    

    def euclidean_dist(self, a: tuple[int, int], b: tuple[int, int]):
        return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**(1/2)