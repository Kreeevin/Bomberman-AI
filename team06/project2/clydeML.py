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
import random
import json

# TODO: Normalize features between 0 and 1 - allows for comparing weights later
# TODO: Separate random vs intelligent monster instead of type of monster

# TODO: Introduce curiousity so sometimes choose the best action


featureNames = ["distToExit", "numWallsOnPath", "distToMonster", "typeClosestMonster", "numMonsters", 
                "canPlaceBomb", "inBombPath", "timeUntilBombExplodes", "nextToExplosion", "bombHitWall", 
                "numMovesAvailable", "bombHitMonster", "bombHitChar", "charKilledByMonster", "charWins"]

EIGHT_MOVEMENT = [(-1,-1), (0, -1), (1, -1),
                  (-1, 0),           (1, 0),
                  (-1, 1),  (0, 1),  (1, 1)]


decay = 1

DEBUG = True
def debug(str):
    if DEBUG:
        print(str)

class Clyde(CharacterEntity):

    def __init__(self, name, avatar, x, y, learningFactor = 0.5, futureDecay = 0.9):
        super().__init__(name, avatar, x, y)
        self.turncount = 0
        self.learningFactor = learningFactor
        self.futureDecay = futureDecay
        self.doneLearning = False
        self.prevWeights = self.readWeights()

    def do(self, world):
        # Commands
        dx, dy = 0,0
        bomb = False

        # Neccessary so that entities don't repeat their previous move if timestep is
        # incremented before their next move is redefined
        for mList in world.monsters.values():
            for monster in mList:
                monster.move(0,0)
        me = world.me(self)
        me.move(0, 0)
        
        # if not self.doneLearning:
        action, weights = self.qLearning(world, self.prevWeights)

        self.prevWeights = weights
        
        # else:

        #     policySearchThreshold = 20
        #     if self.turncount > policySearchThreshold:
        #         # At what point do we want to run this instead of just letting QLearning decide our next move
        #         optimizedWeights = self.policySearch(weights, world)
                
        #         self.prevWeights = optimizedWeights
        #         self.saveWeights(optimizedWeights)

        # Execute commands
        
        self.performAction(world, action)

        debug(f"New player postion: {self.nextpos()}\t Was bomb placement attempted: {bomb}")

        theoryland, theoreticalEvents = world.next()
        theoreticalEvents = list(map(lambda e:e.tpe, theoreticalEvents))
        print(f"Theoretical Events: {theoreticalEvents}")
        
        
        if (Event.BOMB_HIT_CHARACTER in theoreticalEvents or 
            Event.CHARACTER_KILLED_BY_MONSTER in theoreticalEvents or 
            Event.CHARACTER_FOUND_EXIT in theoreticalEvents or 
            theoryland.explosion_at(*self.nextpos())):
            self.saveWeights(self.prevWeights)
            
        

    def bestMove(self, world, weights):

        ### AFTER TRAINING, JUST A MAX NODE TO CHOOSE MOVE

        me = world.me(self)
        if me is None:
            return None
        actions = self.validMoves(world, me) # (0,0) places bomb
        actions.append((0,0))
        bestMove = None

        for dx, dy in actions:
            self.performAction(world, (dx, dy))
            
            newWorld, _ = world.next()
            newFeatures, _ = self.featuresOfState(newWorld)
            newUtility = self.evaluateState(newFeatures, weights)
            
            if bestMove is None or newUtility > bestMove[1]:
                bestMove = ((dx,dy), newUtility)
        
        if bestMove is None:
            print("Player has no valid move")

        return bestMove
            
    def performAction(self, world : SensedWorld, action):
        me = world.me(self)
        dx, dy = action
        bomb = False
        if (dx, dy) == (0,0):
            if not self.canPlaceBomb(world, me):
                return False
            else:
                bomb = True

        self.move(dx, dy)
        if bomb:
            self.place_bomb()
            
        return True
    
    def canPlaceBomb(self, world, me):
        # Borrow for loop from world class to check if bomb can be placed
        for k,b in world.bombs.items():
            if b.owner == me:
                return False
        return True
           
    def qLearning(self, world, weights):
        # Get all legal actions
        actions = self.validMoves(world, world.me(self)) # (0, 0) is place bomb
        actions.append((0,0))
        # Choose a random move
        action = random.choice(actions)
        # Perform random move
        self.performAction(world, action)
        # Increment time step
        sensedWorld, _ = world.next()
        # Get features of new world
        newFeatures, newReward = self.featuresOfState(sensedWorld)
        print(f"Features: {newFeatures}")
        # Check for features + rewards
        # sensedWorld.scores[-1] - world.scores[-1]
        newWeights = self.updateWeights(sensedWorld, newReward, newFeatures, weights)
        # yippee!
        return action, newWeights
        
    def stopTraining(self, world, weights) -> bool:
        
        tolerance = .01 # TODO tune this real good
        done = True
        
        for idx in range(len(weights)):
            # if any weights have changed more than the tolerance, keep training
            if abs(weights[idx] - self.prevWeights[idx]) > tolerance:
                done = False

        return done
                
    
    def evaluateState(self, features, weights):
        currentUtility = 0

        for idx in range(len(weights)):

            # debug(f"weight @ index {idx} = {weights[idx]}")
            # debug(f"features @ index {idx} = {features[idx]}")

            currentUtility += weights[idx]*features[idx]
        
        return currentUtility
    
    def updateWeights(self, newWorld, reward, features, weights):
        # Get utility of current state after current action
        currentStateVal = self.evaluateState(features, weights)
        # Get utility of the best move next turn, 
        nextTurnBestMove = self.bestMove(newWorld, weights)

        if nextTurnBestMove is None: 
            nextTurnBestMoveUtility = 0
        else: 
            nextTurnBestMoveUtility = nextTurnBestMove[1]

        delta = reward + self.futureDecay*nextTurnBestMoveUtility - currentStateVal

        print(f"Delta: {delta}, Reward: {reward}, nextTurnBestMove: {nextTurnBestMoveUtility}, CurrentStateVal: {currentStateVal}")
        print(f"Weights: {weights}")
        # Update weights according to learning factor& delta
        for idx in range(len(weights)):
            weights[idx] += self.learningFactor*delta*features[idx]

        return weights

    def policySearch(self, weights, world) -> list[float]:
        pass

    def a_star(self, wrld: World, start: tuple[int, int], goal: tuple[int, int], ignoreWalls:bool = False) -> list[tuple[int, int]]:
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
                    f = g + 1 + self.euclideanDist(neighbor,goal)
                    q.put((neighbor,cords,g+1),f)
        
        # this only happens if no exit can be fond, queue runs out
        print('Could not reach goal')
        
        return []

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

    # calculate list of features
    def featuresOfState(self, world: World) -> tuple[list[int], int]:
        
        
        
        distToExit              = 0
        numWallsOnPath          = 0
        
        distToMonster           = 0 # TODO: fuck around and find out - 0? inf? -1? 
        typeClosestMonster      = 0
        numMonsters             = 0

        inBombPath              = 0
        timeUntilBombExplodes   = 0
        nextToExplosion         = 0
        canPlaceBomb            = 1

        numMovesAvailable       = 0

        bombHitChar             = 0
        charKilledByMonster     = 0 
        charWins                = 0
        bombHitWall             = 0
        bombHitMonster          = 0
        

        me = world.me(self)

        if me is not None:

            aStarPath = self.a_star(world, (me.x,me.y), world.exitcell, ignoreWalls=True)
            for p in aStarPath:
                if world.wall_at(*p):
                    numWallsOnPath += 1

            distToExit = len(aStarPath)
            
            for mList in world.monsters.values():
                    # Secondary loop because of weird format of dictionaries (multiple monsters at same index?)
                    for monster in mList:
                        
                        dist = len(self.a_star(world, (me.x, me.y), (monster.x, monster.y), ignoreWalls=False))
                            
                        if distToMonster == 0 or dist < distToMonster:
                            distToMonster = dist
                        
                            if type(monster) == SelfPreservingMonster:
                                if monster.rnge > 1:
                                    typeClosestMonster = 3
                                else:
                                    typeClosestMonster = 2
                            elif type(monster) == StupidMonster:
                                typeClosestMonster = 1
            
            # explosion        
            # Bool for if the player is able to place a bomb
            dangerBombs = []

            if not self.canPlaceBomb(world, me):
                canPlaceBomb = 0
            
            # in path
            for bomb in world.bombs.values():
                # if guy is same x coord as bomb and diff in y coord is <= range
                if me.x == bomb.x and abs(me.y - bomb.y) <= world.expl_range:
                    dangerBombs.append(bomb)
                    inBombPath = 1
                # if guy is same y coord as bomb and diff in x coord is <= range
                if me.y == bomb.y and abs(me.x - bomb.x) <= world.expl_range:
                    dangerBombs.append(bomb)
                    inBombPath = 1
                        
            # time til boom
            for danger in dangerBombs:
                if inBombPath: #is this needed?
                    timeUntilBombExplodes += danger.timer

            # next to explosion
            # check all valid move positions
            validMoves = self.validMoves(world, me)
            for v in validMoves:
                #if explosion:
                if world.explosion_at((me.x + v[0]), (me.y + v[1])):
                    nextToExplosion = 1
                    break

            numMovesAvailable = len(self.validMoves(world, me))

        # events: all binary
        reward = 0
        for event in world.events:
            if event.tpe == Event.BOMB_HIT_CHARACTER:
                bombHitChar = 1
                reward -= 5000
            if event.tpe == Event.CHARACTER_KILLED_BY_MONSTER:
                charKilledByMonster = 1
                reward -= 1000
            if event.tpe == Event.CHARACTER_FOUND_EXIT:
                charWins = 1
                reward += 1000
            if event.tpe == Event.BOMB_HIT_WALL:
                bombHitWall = 1
                reward += 50
            if event.tpe == Event.BOMB_HIT_MONSTER:
                bombHitMonster = 1
                reward += 250

        return ([1/(1+distToExit), numWallsOnPath, distToMonster, typeClosestMonster, numMonsters, 
                canPlaceBomb, inBombPath, timeUntilBombExplodes, nextToExplosion, bombHitWall, numMovesAvailable,
                bombHitMonster, bombHitChar, charKilledByMonster, charWins], reward)

    # Returns number of valid moves for a given entity

    def validMoves(self, world, entity):
        validMoves = []
        for dx,dy in EIGHT_MOVEMENT:
            if self.isCellWalkable(world, (entity.x + dx, entity.y + dy)):
                validMoves.append((dx, dy))

        return validMoves

    # Helpers for reading from / saving weights to a file

    def saveWeights(self, weights: list, fileName = "weights.json"):
        featureDict = {}
        with open(fileName, "w+") as file:
            for index, weight in enumerate(weights):
                featureDict[featureNames[index]] = weight
            json.dump(featureDict, file)
            print("Weights saved to file successfully")
    
    def readWeights(self, fileName = "weights.json") -> list[float]:
        weights = []
        try:
            with open(fileName) as file:
                file_weights = json.load(file)
                for key in featureNames:
                    weights.append(file_weights[key])
                print(f"Weights successfully read: {weights}")
            return weights
        except:
            return [1]*len(featureNames)

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
    
    def euclideanDist(self, a: tuple[int, int], b: tuple[int, int]):
        return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**(1/2)

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