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
from clyde import Clyde

# TODO: Normalize features between 0 and 1 - allows for comparing weights later
# TODO: Separate random vs intelligent monster instead of type of monster

# TODO: Introduce curiousity so sometimes choose the best action


featureNames = ["distToExit", # Already normalized, 1/(1+dist)
                "proportionWallsOnPath", # Divide current number by length of path
                "dirGoalNegX", # bool 0, 1 for direction
                "dirGoalPosX", # bool 0, 1  for direction
                "dirGoalNegY", # bool 0, 1 for direction
                "dirGoalPosY", # bool 0, 1  for direction
                "distToRandomMonster", # 1 / (1+dist)^2
                "distToAggressiveMonster", # 1/(1+dist)^2
                "numMonsters", # int -- numMonsters / initial number of monsters
                "dirMonsterNegX", # bool 0, 1 for direction
                "dirMonsterPosX", # bool 0, 1  for direction
                "dirMonsterNegY", # bool 0, 1 for direction
                "dirMonsterPosY", # bool 0, 1  for direction
                "canPlaceBomb",  # bool
                "inBombPath",  # bool
                "dirBombNegX", # bool 0, 1 for direction
                "dirBombPosX", # bool 0, 1  for direction
                "dirBombNegY", # bool 0, 1 for direction
                "dirBombPosY", # bool 0, 1  for direction
                "timeUntilBombExplodes", # time til explosion / total explosion time
                "nextToExplosion",  # bool
                "numMovesAvailable", # Just divide by 8 (don't count bomb as a move)
                "bombHitWall", "bombHitMonster", "bombHitChar", "charKilledByMonster", "charWins"] #Events -- all bools


EIGHT_MOVEMENT = [(-1,-1), (0, -1), (1, -1),
                  (-1, 0),           (1, 0),
                  (-1, 1),  (0, 1),  (1, 1)]


decay = 1

DEBUG = True
def debug(str):
    if DEBUG:
        print(str)

class ClydeML(CharacterEntity):

    def __init__(self, name, avatar, x, y, learningFactor = 0.5, futureDecay = 0.5):
        super().__init__(name, avatar, x, y)
        self.turncount = 0
        self.learningFactor = learningFactor
        self.futureDecay = futureDecay
        self.doneLearning = False
        self.sum_features = [0] * len(featureNames)
        self.avg_features = [0] * len(featureNames)
        self.num_samples = 0
        self.prevWeights = self.readWeights()
        self.freePathToExit = False

    def do(self, world):
        # Commands

        if self.turncount == 0:
            # Initialize helpful class variables like self.initial_monster_count or wavefront
            self.initialize_helper_variables(world)
            self.x, self.y = self.random_non_wall(world)

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
        
        self.performAction(world, action)

        debug(f"New player postion: {self.nextpos()}\t Was bomb placement attempted: {self.maybe_place_bomb}")

        theoryland, theoreticalEvents = world.next()
        theoreticalEvents = list(map(lambda e:e.tpe, theoreticalEvents))
        debug(f"Theoretical Events: {theoreticalEvents}")
        
        if (Event.BOMB_HIT_CHARACTER in theoreticalEvents or 
            Event.CHARACTER_KILLED_BY_MONSTER in theoreticalEvents or 
            Event.CHARACTER_FOUND_EXIT in theoreticalEvents or 
            theoryland.explosion_at(*self.nextpos())):
            self.saveWeights(self.prevWeights)
        
        self.turncount += 1
            
        

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
            newFeatures, reward = self.featuresOfState(newWorld)
            newUtility = self.evaluateStateUtility(newFeatures, weights) + reward
            
            if bestMove is None or newUtility > bestMove[1]:
                bestMove = ((dx,dy), newUtility)
        
        if bestMove is None:
            print("Player has no valid move")

        return bestMove
    
    def curiousMove(self, world, weights):

        ### DURING TRAINING, JUST A MAX NODE TO CHOOSE MOVE THAT IS FARTHEST FROM AVERAGE
        me = world.me(self)
        if me is None:
            return None
        actions = self.validMoves(world, me) # (0,0) places bomb
        actions.append((0,0))
        curiosityMove = None

        for dx, dy in actions:
            self.performAction(world, (dx, dy))
            
            newWorld, _ = world.next()
            newFeatures, _ = self.featuresOfState(newWorld)
            newDistToAvgState = self.feature_diff(newFeatures)
            
            if curiosityMove is None or newDistToAvgState > curiosityMove[1]:
                curiosityMove = ((dx,dy), newDistToAvgState)
        
        if curiosityMove is None:
            print("Player has no valid move")

        print(f"Choosing move {curiosityMove[0]}, which is {curiosityMove[1]} away from the average move")

        return curiosityMove
            
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

        # Choose a curious move
        # action, _ = self.curiousMove(world, weights)


        if random.random() < (self.turncount+self.trainingDuration)/1000:
            action, _ = self.bestMove(world, weights)
            debug("PICKS BEST MOVE")
        else:
            non_death_actions = []
            for (dx,dy) in actions:
                me = world.me(self)
                if (dx, dy) == (0,0) and not self.canPlaceBomb(world, me):
                    continue
                if world.explosion_at(self.x+dx, self.y+dy):
                    continue
                inPath, bomb = self.in_bomb_path(world, me, (dx, dy))
                if inPath and bomb is not None and bomb.timer < 2:
                    continue
                else: non_death_actions.append((dx, dy))
            if non_death_actions == []:
                non_death_actions.append((0,0))
            action = random.choice(non_death_actions)
            debug("PICKS RANDOM MOVE")
        # Perform random move
        self.performAction(world, action)
        # Increment time step
        sensedWorld, _ = world.next()
        # Get features of new world
        newFeatures, reward = self.featuresOfState(sensedWorld)
        
        # SKILL ISSUE: EXPECTIMAX DOESN'T LIKE WALLS

        # if newFeatures[24] or newFeatures[25] or newFeatures[26]:
        #     # If the curious move will cause the player to die,
        #     # Find the optimal move according to expectimax instead
        #     debug("RUNNING EXPECTIMAX TO TRY AND AVOID DYING")
        #     newAction, _ = self.maxNode(world, 3)
        #     if newAction is not None:
        #         # Perform random move
        #         self.performAction(world, newAction)
        #         # Increment time step
        #         sensedWorld, _ = world.next()
        #         # Get features of new world
        #         newFeatures, reward = self.featuresOfState(sensedWorld)

        #         action = newAction

        self.avg_features = self.rolling_average(newFeatures)
        # debug(f"Features: {newFeatures}")
        self.debug_features(newFeatures)
        # Check for features + rewards
        newWeights = self.updateWeights(sensedWorld, reward, newFeatures, weights)
        # yippee!
        return action, newWeights
    
    def rolling_average(self, features: tuple[float]) -> tuple[float]:
        avg_features = []
        self.num_samples += 1
        for i in range(len(self.sum_features)):
            self.sum_features[i] += features[i]
            avg_features.append(self.sum_features[i]/self.num_samples)
        # debug(f"Average features with {self.num_samples} samples:")
        # debug(f"{avg_features}")
        return tuple(avg_features)

    def feature_diff(self, features: tuple[float]) -> float:
        sum = 0
        binary_feature_indexes = [2,3,4,5,9,10,11,12,13,14,15,16,17,18,20,22,23,24,25,26]
        for i in range(len(featureNames)):
            if i in binary_feature_indexes:
                continue
            sum += (self.avg_features[i] - features[i])**2
        return math.sqrt(sum)

    def stopTraining(self, world, weights) -> bool:
        
        tolerance = .01 # TODO tune this real good
        done = True
        
        for idx in range(len(weights)):
            # if any weights have changed more than the tolerance, keep training
            if abs(weights[idx] - self.prevWeights[idx]) > tolerance:
                done = False

        return done
                
    
    def evaluateStateUtility(self, features, weights):
        currentUtility = 0

        for idx in range(len(weights)):

            # debug(f"weight @ index {idx} = {weights[idx]}")
            # debug(f"features @ index {idx} = {features[idx]}")

            currentUtility += weights[idx]*features[idx]
        
        return currentUtility
    
    def updateWeights(self, newWorld, reward, features, weights):
        # Get utility of current state after current action
        currentStateVal = self.evaluateStateUtility(features, weights)
        # Get utility of the best move next turn, 
        nextTurnBestMove = self.bestMove(newWorld, weights)

        if nextTurnBestMove is None: 
            nextTurnBestMoveUtility = 0
        else: 
            nextTurnBestMoveUtility = nextTurnBestMove[1]
            
        delta = reward + self.futureDecay*nextTurnBestMoveUtility - currentStateVal

        debug(f"\nDelta: {delta}, Reward: {reward}, nextTurnBestMove: {nextTurnBestMoveUtility}, CurrentStateVal: {currentStateVal}")
        debug(f"Weights: {weights}\n")
        # Update weights according to learning factor& delta
        for idx in range(len(weights)):
            weights[idx] += self.learningFactor*delta*features[idx]
            # Bind weights from -1000 to 1000 
            # if math.fabs(weights[idx]) > 1000:
            #     weights[idx] *= 1000/math.fabs(weights[idx])

        return weights

    def policySearch(self, weights, world) -> list[float]:
        # Decide later if this is actually needed or just a nice to have
        pass

    def in_bomb_path(self, world, entity, action=(0,0)):
        for bomb in world.bombs.values():
                # if guy is same x coord as bomb and diff in y coord is <= range
                if entity.x+action[0] == bomb.x and abs(entity.y+action[1] - bomb.y) <= world.expl_range:
                    return True, bomb
                # if guy is same y coord as bomb and diff in x coord is <= range
                if entity.y+action[1] == bomb.y and abs(entity.x+action[0] - bomb.x) <= world.expl_range:
                    return True, bomb
                    
        return False, None

    # calculate list of features
    def featuresOfState(self, world: World) -> tuple[list[int], int]:
        
        rewards = 0
        
        normalizedDistToExit    = 0 
        proportionWallsOnPath   = 0 

        dirGoalNegX             = 0
        dirGoalPosX             = 0
        dirGoalNegY             = 0
        dirGoalPosY             = 0
        
        distToRandomMonster     = 0 
        distToAggressiveMonster = 0
        numMonsters             = 0

        dirMonsterNegX          = 0
        dirMonsterPosX          = 0
        dirMonsterNegY          = 0
        dirMonsterPosY          = 0

        inBombPath              = 0 
        timeUntilBombExplodes   = 0 
        nextToExplosion         = 0 
        canPlaceBomb            = 1 

        dirBombNegX             = 0
        dirBombPosX             = 0
        dirBombNegY             = 0
        dirBombPosY             = 0

        numMovesAvailable       = 0 

        bombHitChar             = 0 
        charKilledByMonster     = 0 
        charWins                = 0 
        bombHitWall             = 0 
        bombHitMonster          = 0 
        

        me = world.me(self)

        if me is not None:

            aStarPath = self.a_star(world, (me.x,me.y), world.exitcell, ignoreWalls=True)
            numWallsOnPath = 0
            self.tiles = {}
            for p in aStarPath:
                self.set_cell_color(*p, Fore.CYAN+Back.RED)
                if world.wall_at(*p):
                    numWallsOnPath += 1

            if len(aStarPath) >= 2:
                
                if me.nextpos() == aStarPath[0]:
                    debug(f"bro followed a step of a* :D")
                    # Reward moving in the right direction
                    rewards += 5
                
            proportionWallsOnPath = numWallsOnPath/len(aStarPath)

            normalizedDistToExit = 1/(1+len(aStarPath))

            exitcell_x = world.exitcell[0]
            exitcell_y = world.exitcell[1]

            if exitcell_x - me.x > 0: dirGoalPosX = 1
            if exitcell_x - me.x < 0: dirGoalNegX = 1
            if exitcell_y - me.y > 0: dirGoalPosY = 1
            if exitcell_y - me.y < 0: dirGoalNegY = 1

            distClosest = 0
            for mList in world.monsters.values():
                # Secondary loop because of weird format of dictionaries (multiple monsters at same index?)
                for monster in mList:
                    numMonsters += 1
                    dist = len(self.a_star(world, (me.x, me.y), (monster.x, monster.y), ignoreWalls=False))
                    if type(monster) == SelfPreservingMonster:
                        
                        if distToAggressiveMonster == 0 or dist < distToAggressiveMonster:
                            distToAggressiveMonster = dist

                        if distClosest == 0 or dist < distClosest:

                            if monster.x - me.x > 0: dirMonsterPosX = 1
                            else: dirMonsterPosX = 0
                            if monster.x - me.x < 0: dirMonsterNegX = 1
                            else: dirMonsterNegX = 0
                            if monster.y - me.y > 0: dirMonsterPosX = 1
                            else: dirMonsterPosX = 0
                            if monster.y - me.y < 0: dirMonsterNegX = 1
                            else: dirMonsterNegX = 0
                        
                    elif type(monster) == StupidMonster:
                        if distToRandomMonster == 0 or dist < distToRandomMonster:
                                distToRandomMonster = dist
                        
                        if distClosest == 0 or dist < distClosest:
                            if monster.x - me.x > 0: dirMonsterPosX = 1
                            else: dirMonsterPosX = 0
                            if monster.x - me.x < 0: dirMonsterNegX = 1
                            else: dirMonsterNegX = 0
                            if monster.y - me.y > 0: dirMonsterPosX = 1
                            else: dirMonsterPosX = 0
                            if monster.y - me.y < 0: dirMonsterNegX = 1
                            else: dirMonsterNegX = 0

            # Normalize Monster Features
            if self.initial_monster_count != 0:
                numMonsters /= self.initial_monster_count
            distToAggressiveMonster = 1 - (1/(1+distToAggressiveMonster)) # inverse so reward increases as distance increases
            distToRandomMonster = 1 - (1/(1+distToRandomMonster))

            
            # Bool for if the player is able to place a bomb
            dangerBombs = []

            if not self.canPlaceBomb(world, me):
                canPlaceBomb = 0
                
            
            # in path
            closestBomb = None
            
            for bomb in world.bombs.values():
                # if guy is same x coord as bomb and diff in y coord is <= range
                if me.x == bomb.x and abs(me.y - bomb.y) <= world.expl_range:
                    dangerBombs.append(bomb)
                    inBombPath = 1
                # if guy is same y coord as bomb and diff in x coord is <= range
                if me.y == bomb.y and abs(me.x - bomb.x) <= world.expl_range:
                    dangerBombs.append(bomb)
                    inBombPath = 1
                
                distToBomb = abs(me.x - bomb.x) + abs(me.y - bomb.y)
                closestDist = -1
                
                if (distToBomb < closestDist) or closestDist == -1: 
                    closestDist = distToBomb
                    closestBomb = (bomb.x, bomb.y)
            if closestBomb is not None:
                if closestBomb[0] - me.x > 0: dirBombPosX = 1
                if closestBomb[0] - me.x < 0: dirBombNegX = 1
                if closestBomb[1] - me.y > 0: dirBombPosY = 1
                if closestBomb[1] - me.y < 0: dirBombNegY = 1
                        
            # time til boom
            for danger in dangerBombs:
                if inBombPath: #is this needed?
                    timeUntilBombExplodes = 1 - (danger.timer / world.bomb_time) # inverse makes it so theres more weight when less time

            # next to explosion
            # check all valid move positions
            validMoves = self.validMoves(world, me)
            for v in validMoves:
                #if explosion:
                if world.explosion_at((me.x + v[0]), (me.y + v[1])):
                    nextToExplosion = 1
                    break

            # Normalize number of available moves by dividing by the max number
            numMovesAvailable = len(self.validMoves(world, me)) / 8

            if numWallsOnPath == 0 and not self.freePathToExit:
                print("FREE PATH TO EXIT - PLEASE FOR FUCKS SAKE PLEASE TAKE IT")
                rewards += 250
                self.freePathToExit = True

        # events: all binary
        for event in world.events:
            if event.tpe == Event.BOMB_HIT_CHARACTER:
                bombHitChar = 1
                rewards -= 1000
            if event.tpe == Event.CHARACTER_KILLED_BY_MONSTER:
                charKilledByMonster = 1
                rewards -= 1000
            if event.tpe == Event.CHARACTER_FOUND_EXIT:
                charWins = 1
                rewards += 1000
            if event.tpe == Event.BOMB_HIT_WALL:
                bombHitWall = 1
                rewards += 20
            if event.tpe == Event.BOMB_HIT_MONSTER:
                bombHitMonster = 1
                rewards += 100

        
        
        # Cost of Living
        rewards -= 1

        # rewards = rewards * (np.math.dist((world., me.y), world.exitcell) - np.math.dist((me.x, me.y), world.exitcell)) #TODO: let me cook....

        features = [normalizedDistToExit, proportionWallsOnPath, dirGoalNegX, dirGoalPosX, dirGoalNegY, dirGoalPosY, 
                distToRandomMonster, distToAggressiveMonster, numMonsters, dirMonsterNegX, dirMonsterPosX, dirMonsterNegY, dirMonsterPosY,
                canPlaceBomb, inBombPath, dirBombNegX, dirBombPosX, dirBombNegY, dirBombPosY, timeUntilBombExplodes, nextToExplosion, numMovesAvailable, bombHitWall,
                bombHitMonster, bombHitChar, charKilledByMonster, charWins]
        
        return features, rewards

    # Returns number of valid moves for a given entity

    def validMoves(self, world, entity):
        validMoves = []
        for dx,dy in EIGHT_MOVEMENT:
            if self.isCellWalkable(world, (entity.x + dx, entity.y + dy)):
                validMoves.append((dx, dy))

        return validMoves

    def a_star(self, world: World, start: tuple[int, int], goal: tuple[int, int], ignoreWalls:bool = False) -> list[tuple[int, int]]:
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
        if(not self.isCellWalkable(world, start)):
            print('start blocked')
            return []
        elif(not self.isCellWalkable(world, goal)):
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
            
            neighbors=self.neighbors_of_8(world, cords, ignoreWalls)
            
            for i in range(len(neighbors)):
                neighbor=neighbors[i]
                if explored.get(neighbor) is None or explored.get(neighbor)[2] > g + 1:
                    costOfNode = 50 if world.wall_at(*neighbor) else 1
                    f = g + costOfNode + self.euclideanDist(neighbor,goal)
                    q.put((neighbor,cords,g + costOfNode),f)
        
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

                if self.isCellWalkable(wrld, point, ignoreWalls) and point != pos: 
                    neighbors.append(point) #append to return list if walkable

        return neighbors

    def isCellWalkable(self, world: World, pos: tuple[int, int], ignoreWalls = False):
        # init variables

        width = world.width()
        height = world.height()
        x, y = pos

        if x >= width or y >= height or x < 0 or y < 0:    # if cell is out of bounds 
            return False                                                                    
        
        if world.wall_at(x, y) and not ignoreWalls:
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
    
    def initialize_helper_variables(self, world):
        num_monsters = 0

        for mList in world.monsters.values():
                    # Secondary loop because of weird format of dictionaries (multiple monsters at same index?)
                    for monster in mList:
                        num_monsters += 1
        
        self.initial_monster_count = num_monsters
        self.initial_dist_to_exit = len(self.a_star(world, (self.x , self.y), world.exitcell, True))
        
    def random_non_wall(self, world):
        x = int(random.random()*world.width())
        y = int(random.random()*world.height())
        if world.wall_at(x,y) or world.exit_at(x,y):
            return self.random_non_wall(world)
        return (x,y)

    def debug_features(self, features):
        debug("Features:")
        debug(json.dumps({featureNames[i] : features[i] for i in range(len(features))}, indent=2, sort_keys=True))
        debug("Average Features")
        debug(json.dumps({featureNames[i] : self.avg_features[i] for i in range(len(self.avg_features))}, indent=2, sort_keys=True))
  

    # Helpers for reading from / saving weights to a file

    def saveWeights(self, weights: list, fileName = "weights.json"):
        featureDict = {}
        with open(fileName, "w+") as file:
            for index, weight in enumerate(weights):
                featureDict[featureNames[index]] = weight
            featureDict["trainingDuration"] = self.trainingDuration + 1 # save how long we have been training
            featureDict["sum_features"] = self.sum_features
            featureDict["num_samples"] = self.num_samples
            json.dump(featureDict, file)
            print("Weights saved to file successfully")
    

    def readWeights(self, fileName = "weights.json") -> list[float]:
        weights = []
        try:
            with open(fileName) as file:
                file_weights = json.load(file)
                for key in featureNames:
                    weights.append(file_weights[key])
                self.trainingDuration = file_weights["trainingDuration"] # get how long we have been training
                self.sum_features = file_weights["sum_features"]
                self.num_samples = file_weights["num_samples"]
                print(f"Weights successfully read: {weights}")
            return weights
        except:
            self.trainingDuration = 0
            return [1]*len(featureNames)