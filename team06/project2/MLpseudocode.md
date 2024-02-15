what we want:
- appx q-learning --> policy search

- feature-based representation of states [LIST OF INTS]
    - dist to exit
        - integer
        - run A* btwn character and exit
    - number of walls between character and exit
        - integer
        - run wall-ignoring A* and count number of walls on path
    - dist to closest monster
        - integer
        - run A*
        - steal code from project 1 evaluateState?
    - type of closest monster (0, 1, 2, 3)
        - integer
        - 0: no monster
        - 1: random/stupid monster
        - 2: selfpreserving mosnter
        - 3: aggressive mosnter
        - if equally close (or within threshold?) go with higher int value
    - number of monsters
        - integer
        - pull from world/sensed world?
        - if from sensed world, pull ater explosions
    - in path of bomb?
        - binary
        - just check x+y vals
    - dist to explosion
        - binary
        - true if theres an explosion within 1 square (dont walk into it)
    - num available moves
        - integer
        - How many open spaces next to guy
    - EVENTS
        - all binary
        - many seperate events, treat as individual features 
    
        # calculate list of features
        featuresOfState(self, world: World) -> list[features_int]

qLearning(self, featureList, previousWeights, curiosityFactor=0.1) -> tuple[action,list[weights_float]]

policySearch(self, qLearnWeights) -> list[weights_float]

- save weights to a file

do(self, world):
    features = self.featureOfState(world)
    action, weights = self.qLearning(features, world, self.prevWeights)


    if terminalState(features):
        # At what point do we want to run this instead of just letting QLearning decide our next move
        optimizedWeights = self.policySearch(weights, world)
        
        # Write optimized weights to a file for easier use
        with open(filepath) as file:
            file.write(optimizedWeights)

    # action is ((dx, dy), bomb)
    self.move(*action[0])
    if action[1]:
        self.place_bomb()
    
    

