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

featureNames = ["distToExit", "numWallsOnPath", "distToMonster", "typeClosestMonster", "numMonsters", 
                "canPlaceBomb", "inBombPath", "timeUntilBombExplodes", "nextToExplosion", "bombHitWall", "numMovesAvailable",
                "bombHitMonster", "bombHitChar", "charKilledByMonster", "charWins"]

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
        
        _, theoreticalEvents = world.next()
        if (Event.BOMB_HIT_CHARACTER in theoreticalEvents or 
            Event.CHARACTER_KILLED_BY_MONSTER in theoreticalEvents or 
            Event.CHARACTER_FOUND_EXIT in theoreticalEvents):
            self.saveWeights(self.prevWeights)

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
        

    def executeBestMove(self, world, weights):

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
        # Check for features + rewards
        newWeights = self.updateWeights(sensedWorld, newReward, newFeatures, weights)
        # yippee!
        return action, newWeights
        
    def stopTraining(self, world, weights) -> bool:
        
        tolerance = .01 # TODO tune this real good
        done = True
        
        for idx in range(len(weights)):
            # if any weights have changed more than the tolerance, keep training
            if abs(weights[idx] - self.prevWeights[idx]) > self.tolerance:
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
        nextTurnBestMove = self.executeBestMove(newWorld, weights)
        if nextTurnBestMove is None: nextTurnBestMoveUtility = 0
        else: nextTurnBestMoveUtility = nextTurnBestMove[1]

        delta = reward + self.futureDecay*nextTurnBestMoveUtility - currentStateVal
        # Update weights according to learning factor& delta
        for idx in range(len(weights)):
            weights[idx] += self.learningFactor*delta*features[idx]

        return weights

    def policySearch(self, weights, world) -> list[float]:
        pass

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
            
            for mList in world.monsters.values():
                    # Secondary loop because of weird format of dictionaries (multiple monsters at same index?)
                    for monster in mList:
                        
                        dist = len(self.a_star(world, (me.x, me.y), (monster.x, monster.y)))
                            
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
                reward -= 1000
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

        return ([distToExit, numWallsOnPath, distToMonster, typeClosestMonster, numMonsters, 
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
        with open(fileName) as file:
            for index, weight in enumerate(weights):
                featureDict[featureNames[index]] = weight
            json.dump(featureDict, file)
    
    def readWeights(self, fileName = "weights.json"):
        weights = []
        try:
            with open(fileName) as file:
                file_weights = json.load(file)
                for key in featureNames:
                    weights.append(file_weights[key])
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