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
                "numTurnsLeft", # proportion of turns left in game
                "freePathToExit", # bool
                "dirGoalNegX", # me.x - exitcell_x / width
                "dirGoalPosX", # me.x - exitcell_x / width
                "dirGoalNegY", # me.y - exitcell_y / height
                "dirGoalPosY", # me.y - exitcell_y / height
                "distToRandomMonster", # 1 / (1+dist)^2
                "distToAggressiveMonster", # 1/(1+dist)^2
                "numMonsters", # int -- numMonsters / initial number of monsters
                "dirRandomNegX", # me.x - monster_x / width
                "dirRandomPosX", # me.x - monster_x / width
                "dirRandomNegY", # me.y - monster_y / height
                "dirRandomPosY", # me.y - monster_y / height
                "dirAggressiveNegX", # me.x - monster_x / width
                "dirAggressivePosX", # me.x - monster_x / width
                "dirAggressiveNegY", # me.y - monster_y / height
                "dirAggressivePosY", # me.y - monster_y / height
                "wallInBombPath", # 0.25 per direction with bomb
                "bombHitWall", "bombHitMonster", "bombHitChar", "charKilledByMonster", "charWins"] #Events -- all bools


EIGHT_MOVEMENT = [(-1,-1), (0, -1), (1, -1),
                  (-1, 0),           (1, 0),
                  (-1, 1),  (0, 1),  (1, 1)]


decay = 1

DEBUG = False
def debug(str):
    if DEBUG:
        print(str)

class ClydeML(Clyde):

    def __init__(self, name, avatar, x, y, futureDecay = 0.9, filename = "weights.json"):
        super().__init__(name, avatar, x, y)
        self.turncount = 0
        self.futureDecay = futureDecay
        self.filename = filename
        self.doneLearning = False
        self.sum_features = [0] * len(featureNames)
        self.avg_features = [0] * len(featureNames)
        self.num_samples = 0
        self.prevWeights = self.readWeights()
        self.learningFactor = 1 - 0.01*self.trainingDuration/(0.01*self.trainingDuration+1)
        self.freePathToExit = False
        self.maxGameLength = 1000
        self.numberOfEpisodes = 300

    def do(self, world):
        # Commands

        if self.turncount == 0:
            # Initialize helpful class variables like self.initial_monster_count or wavefront
            self.initialize_helper_variables(world)
            if self.trainingDuration < 150:
                # Only start in a random spot for the first 150 episodes
                self.x, self.y = self.random_non_wall(world)

        # Neccessary so that entities don't repeat their previous move if timestep is
        # incremented before their next move is redefined
        me = world.me(self)
        me.move(0, 0)

        closestMonster = None
        for mList in world.monsters.values():
            for monster in mList:
                monster.move(0,0)
                dist = len(self.a_star(world, (me.x,me.y), monster.nextpos(), ignoreWalls=True))
                if closestMonster is None or dist < closestMonster:
                    if dist != 0:
                        closestMonster = dist
        
        
        if self.turncount >= self.maxGameLength and False:
            self.place_bomb()
            self.move(0,0)
        elif self.freePathToExit or (closestMonster is not None and closestMonster <= 5):
            print(f"\t\t\tEXPECTIMAXING: Closest Monster {closestMonster} cells away!")
            # Should this case be a pure A* follow
            # If free path to exit just fuckin go for it bestie
            self.wavefront = self.make_wavefront(world, world.exitcell)
            action, _ = self.expectimax(world, self.depth)
            sensedWorld, _ = world.next()

            # Update weights like we do in qLearning
            newFeatures, reward = self.featuresOfState(sensedWorld)
            newWeights = self.updateWeights(sensedWorld, reward, newFeatures, self.prevWeights)
            self.prevWeights = newWeights
            if action is not None:
                self.performAction(world, action, self)
        else:
            # if not self.doneLearning:
            action, weights = self.qLearning(world, self.prevWeights)

            self.prevWeights = weights
            
            self.performAction(world, action, self)
        

        debug(f"New player postion: {self.nextpos()}\t Was bomb placement attempted: {self.maybe_place_bomb}")
        
        self.turncount += 1
            
        

    def bestMove(self, world, weights):

        ### AFTER TRAINING, JUST A MAX NODE TO CHOOSE MOVE

        me = world.me(self)
        if me is None:
            return None
        actions = self.valid_non_death_moves(world, me) # (0,0) places bomb
        actions.reverse()

        bestMove = None

        for dx, dy in actions:
            self.performAction(world, (dx, dy))
            
            newWorld, _ = world.next()
            # print(f"theres a character in the new world at the new action point: {newWorld.characters_at(self.x, self.y)}")
            newFeatures, reward = self.featuresOfState(newWorld)
            newUtility = self.evaluateStateUtility(newFeatures, weights)
            # print(f"action: ({dx},{dy}) reward: {reward}, utility: {newUtility}")
            
            if bestMove is None or newUtility+reward > bestMove[1]+bestMove[2]:
                bestMove = ((dx,dy), newUtility, reward)
        
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

        # print(f"Choosing move {curiosityMove[0]}, which is {curiosityMove[1]} away from the average move")

        return curiosityMove
            
    def performAction(self, world : SensedWorld, action, entity = None):
        if entity is None:
            me = world.me(self)
        else:
            me = entity
        dx, dy = action
        bomb = False
        if (dx, dy) == (0,0):
            if not self.canPlaceBomb(world, me):
                return False
            else:
                bomb = True

        me.move(dx, dy)
        me.maybe_place_bomb = bomb
            
        return True
    
    def canPlaceBomb(self, world, me):
        # Borrow for loop from world class to check if bomb can be placed
        for k,b in world.bombs.items():
            if b.owner == me:
                return False
        return True
           
    def qLearning(self, world, weights):
        # Get all legal actions
        actions = self.valid_non_death_moves(world, world.me(self)) # (0, 0) is place bomb

        # Choose a curious move
        # action, _ = self.curiousMove(world, weights)
        epsilon = 1-(0.025*self.trainingDuration)/(0.025*self.trainingDuration+1)
        if random.random() > epsilon:
            action, _, _ = self.bestMove(world, weights)
            print(f"PICKS BEST MOVE:\tPROBABILITY IS {epsilon}\tLearning Factor: {self.learningFactor}")
        else:
            action = random.choice(actions)
            print(f"PICKS RANDOM MOVE:\tPROBABILITY IS {epsilon}\tLearning Factor: {self.learningFactor}")
        # Perform random move
        self.performAction(world, action)
        # Increment time step
        sensedWorld, _ = world.next()
        # Get features of new world
        newFeatures, reward = self.featuresOfState(sensedWorld)
        
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

        print(f"Delta: {delta}, Reward: {reward}, nextTurnBestMove: {nextTurnBestMoveUtility}, CurrentStateVal: {currentStateVal}")
        debug(f"Weights: {weights}\n")
        # Update weights according to learning factor& delta
        for idx in range(len(weights)):
            weights[idx] += self.learningFactor*delta*features[idx]
            # Bind weights from -1000 to 1000 
            if math.fabs(weights[idx]) > 1000:
                weights[idx] *= 1000/math.fabs(weights[idx])

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
        numTurnsLeft            = 0 

        dirGoalNegX             = 0
        dirGoalPosX             = 0
        dirGoalNegY             = 0
        dirGoalPosY             = 0
        
        distToRandomMonster     = 0 
        distToAggressiveMonster = 0
        numMonsters             = 0

        dirRandomNegX          = 1
        dirRandomPosX          = 1
        dirRandomNegY          = 1
        dirRandomPosY          = 1

        dirAggressiveNegX          = 1
        dirAggressivePosX          = 1
        dirAggressiveNegY          = 1
        dirAggressivePosY          = 1

        inBombPath              = 0 
        timeUntilBombExplodes   = 0 
        nextToExplosion         = 0 
        canPlaceBomb            = 1

        wallInBombPath          = 0

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
                
                if me.dx == me.x - aStarPath[0][0]:
                    debug(f"bro followed a step of a* in the x-direction")
                    # Reward moving in the right direction
                    rewards += 15 if self.freePathToExit else 5
                elif -me.dx == me.x - aStarPath[0][0]:
                    debug(f"bro followed a step opposite a* in the x-direction")
                    # Punish moving in the wrong direction when theres a free path
                    rewards -= 15 if self.freePathToExit else 0


                if me.dy == me.y - aStarPath[0][1]:
                    debug(f"bro followed a step of a* in the y-direction")
                    # Reward moving in the right direction
                    rewards += 15 if self.freePathToExit else 5
                elif -me.dy == me.y - aStarPath[0][1]:
                    debug(f"bro followed a step opposite a* in the y-direction")
                    # Punish moving in the wrong direction when theres a free path
                    rewards -= 15 if self.freePathToExit else 0
                
            rewards -= 2*len(aStarPath)
            

            normalizedDistToExit = 1/(1+len(aStarPath))

            exitcell_x = world.exitcell[0]
            exitcell_y = world.exitcell[1]


            dirGoalPosX = max(0, exitcell_x - me.x)/world.width()
            dirGoalNegX = max(0, me.x - exitcell_x)/world.width()
            dirGoalPosY = max(0, exitcell_y - me.y)/world.height()
            dirGoalNegY = max(0, me.y - exitcell_y)/world.height()

            distClosest = None
            for mList in world.monsters.values():
                # Secondary loop because of weird format of dictionaries (multiple monsters at same index?)
                for monster in mList:
                    numMonsters += 1
                    dist = len(self.a_star(world, (me.x, me.y), (monster.x, monster.y), ignoreWalls=False))
                    debug(f"Dist to monster = {dist}, Monster Type: {type(monster)}")
                    if monster.name == "selfpreserving" or monster.name == "aggressive":
                        
                        if distToAggressiveMonster == 0 or dist < distToAggressiveMonster:
                            distToAggressiveMonster = dist

                        if distClosest is None or dist < distClosest:
                            distClosest = dist
                            dirAggressivePosX = max(0, monster.x - me.x)/world.width()
                            dirAggressiveNegX = max(0, me.x - monster.x)/world.width()
                            dirAggressivePosY = max(0, monster.y - me.y)/world.height()
                            dirAggressiveNegY = max(0, me.y - monster.y)/world.height()
                        
                    elif monster.name == "stupid":
                        if distToRandomMonster == 0 or dist < distToRandomMonster:
                            distToRandomMonster = dist
                        
                        if distClosest is None or dist < distClosest:
                            distClosest = dist
                            dirRandomPosX = max(0, monster.x - me.x)/world.width()
                            dirRandomNegX = max(0, me.x - monster.x)/world.width()
                            dirRandomPosY = max(0, monster.y - me.y)/world.height()
                            dirRandomNegY = max(0, me.y - monster.y)/world.height()

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
                
                if bomb.timer == world.bomb_time-1:
                    # Punish per bomb placement so its a bit more strategic with bomb placements
                    rewards -= 5
                
                # Use a funny little binary guy to use 4 booleans with 1 variable
                wallInDirection = 0b0000
                                #   SNWE
                for i in range(world.expl_range):
                    if bomb.x+i <  world.width() and world.wall_at(bomb.x+i, bomb.y) and wallInDirection&0b1:
                        wallInBombPath += 1/4
                        wallInDirection |= 0b1
                    if bomb.x-i >= 0 and world.wall_at(bomb.x-i, bomb.y) and wallInDirection&0b10:
                        wallInBombPath += 1/4
                        wallInDirection |= 0b10
                    if bomb.y+i <  world.height() and world.wall_at(bomb.x, bomb.y+i) and wallInDirection&0b100:
                        wallInBombPath += 1/4
                        wallInDirection |= 0b100
                    if bomb.y-i >= 0 and world.wall_at(bomb.x, bomb.y-i) and wallInDirection&0b1000:
                        wallInBombPath += 1/4
                        wallInDirection |= 0b1000
                
                distToBomb = abs(me.x - bomb.x) + abs(me.y - bomb.y)
                closestDist = -1
                
                if (distToBomb < closestDist) or closestDist == -1: 
                    closestDist = distToBomb
                    closestBomb = (bomb.x, bomb.y)

            # if closestBomb is not None:
            #     if closestBomb[0] - me.x > 0: dirBombPosX = 1
            #     if closestBomb[0] - me.x < 0: dirBombNegX = 1
            #     if closestBomb[1] - me.y > 0: dirBombPosY = 1
            #     if closestBomb[1] - me.y < 0: dirBombNegY = 1
                        
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

            if numWallsOnPath == 0 and len(aStarPath) > 0 and not self.freePathToExit:
                print("FREE PATH TO EXIT - PLEASE FOR FUCKS SAKE PLEASE TAKE IT")
                print(f"A-Star Path is: {aStarPath}, length {len(aStarPath)}")
                rewards += 250
                self.freePathToExit = True

        numTurnsLeft = 1 - self.turncount/self.maxGameLength
        if abs(self.turncount - self.maxGameLength) < 2:
            # Huge penalty for max length game
            rewards += -2500 
            
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
        rewards -= 5

        debug(f"Rewards: {rewards}")

        features = [normalizedDistToExit, numTurnsLeft, self.freePathToExit, 
                    dirGoalNegX, dirGoalPosX, dirGoalNegY, dirGoalPosY, 
                    distToRandomMonster, distToAggressiveMonster, numMonsters, 
                    dirRandomNegX, dirRandomPosX, dirRandomNegY, dirRandomPosY, 
                    dirAggressiveNegX, dirAggressivePosX, dirAggressiveNegY, dirAggressivePosY, 
                    wallInBombPath, 
                    bombHitWall, bombHitMonster, bombHitChar, charKilledByMonster, charWins]
        
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
        debug('Could not reach goal')
        
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
  
    def valid_non_death_moves(self, world, entity):
        actions = self.validMoves(world, entity)
        actions.append((0,0))
        non_death_actions = []
        for (dx,dy) in actions:
            me = world.me(self)
            if (dx, dy) == (0,0) and not self.canPlaceBomb(world, me):
                continue
            if world.explosion_at(self.x+dx, self.y+dy):
                continue
            if world.monsters_at(self.x+dx, self.y+dy):
                continue
            inPath, bomb = self.in_bomb_path(world, me, (dx, dy))
            if inPath and bomb is not None and bomb.timer < 2:
                continue
            else: non_death_actions.append((dx, dy))
        if non_death_actions == []:
            non_death_actions.append((0,0))
        return non_death_actions
        
    # Helpers for reading from / saving weights to a file

    def saveWeights(self, weights: list, fileName = None):
        if fileName is None:
            fileName = self.filename
        featureDict = {}
        with open(fileName, "w+") as file:
            for index, weight in enumerate(weights):
                featureDict[featureNames[index]] = weight
            featureDict["trainingDuration"] = self.trainingDuration + 1 # save how long we have been training
            # featureDict["sum_features"] = self.sum_features
            # featureDict["num_samples"] = self.num_samples
            json.dump(featureDict, file)
            print("Weights saved to file successfully")
    

    def readWeights(self, fileName = None) -> list[float]:
        if fileName is None:
            fileName = self.filename
        weights = []
        try:
            with open(fileName) as file:
                file_weights = json.load(file)
                for key in featureNames:
                    weights.append(file_weights[key])
                self.trainingDuration = file_weights["trainingDuration"] # get how long we have been training
                # self.sum_features = file_weights["sum_features"]
                # self.num_samples = file_weights["num_samples"]
                print(f"Weights successfully read: {weights}")
            return weights
        except:
            self.trainingDuration = 0
            return [1]*len(featureNames)
        
    def done(self, wrld):
        print("GAME OVER: Saving the weights!")
        self.saveWeights(self.prevWeights)