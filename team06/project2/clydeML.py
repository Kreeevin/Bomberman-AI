
# class init shit

# helper functions

def validMoves(self, world, entity):
    validMoves = []
    for dx,dy in EIGHT_MOVEMENT:
        if self.isCellWalkable(world, (entity.x + dx, entity.y + dy)):
            validMoves.append((dx, dy))

    return validMoves

# calculate list of features
def featuresOfState(self, world: World) -> list[int]:
    me = world.me(self)
    
    distToExit          = 0
    numWallsOnPath      = 0
    
    distToMonster       = float("inf")
    typeClosestMonster  = 0
    numMonsters         = 0
    
    for mList in world.monsters.values():
            # Secondary loop because of weird format of dictionaries (multiple monsters at same index?)
            for moonster in mList:
                
                dist = len(self.a_star(world, (me.x, me.y), (monster.x, monster.y)))
                    
                if dist < distToMonster:
                    distToMonster = dist
                
                    if type(monster) == SelfPreservingMonster:
                        if monster.rnge > 1:
                            typeClosestMonster = 3
                        else:
                            typeClosestMonster = 2
                    elif type(monster) == StupidMonster:
                        typeClosestMonster = 1
    
    inExplosionPath     = False
    nextToExplosion     = False

    #explosion feature code here
    
    numMovesAvailable   = 0

    numMovesAvailable = len(self.validMoves(world, me))


    # events: all binary
    bombHitWall         = False 
    bombHitMonster      = False 
    bombHitChar         = False 
    charKilledByMonster = False 
    charWins            = False 
    
    for event in newEvents:
        if event.tpe == Event.BOMB_HIT_CHARACTER:
            bombHitChar = True
        if event.tpe == Event.CHARACTER_KILLED_BY_MONSTER:
            charKilledByMonster = True
        if event.tpe == Event.CHARACTER_FOUND_EXIT:
            charWins = True
        if event.tpe == Event.BOMB_HIT_WALL:
            bombHitWall = True
        if event.tpe == Event.BOMB_HIT_MONSTER:
            bombHitMonster = True

    return [distToExit, numWallsOnPath, distToMonster, typeClosestMonster, numMonsters, inExplosionPath, nextToExplosion, numMovesAvailable, bombHitWall, bombHitMonster, bombHitChar, charKilledByMonster, charWins]

def saveWeights(self, weights: list, fileName = "weights.txt"):
    with open(fileName) as file:

        for index, weight in enumerate(weights):
            file.write(f"{features[index]}:{weight}")

def weightsToDict(self, weights: list[float]):
    features = ["DistToExit", "NumWalls", "DistToMonster", "TypeOfMonster", "NumMonster", "InBombPath?", "DistToExplosion", "NumMoves", 
    "EventBombHitCharacter", "EventCharacterKilledByMonster", "EventCharacterFoundExit", "EventBombHitWall", "EventBombHitMonster"]
    
    