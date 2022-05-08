''' 
DETAILS OF EACH METHOD CAN BE READ IN MY THESIS: 

Diffusion In Fuzzy Lattice Systems: Exploring the Anomalous Regime,
Connecting the Steady-State, and Fat-Tailed Distributions
'''
def PickUpAndDrop(period,L,randomness):
    import numpy as np

    disorder = randomness/100
    L = L // period * period

    obstaclesTemp = np.zeros(L)
    for i in range(int(L // period)):
         obstaclesTemp[int(i * period)] = 1

    obstacles = np.zeros((L, L))
    for i in range(int(L // period)):
        obstacles[:, int(i * period)] = np.copy(obstaclesTemp)

    idx = np.argwhere(obstacles==1)

    numToMove = disorder*np.sum(obstacles)

    for i in range(int(numToMove)):
        arg = np.random.randint(0,len(idx))
        obstacles[idx[arg][0],idx[arg][1]] = 0
        idx = np.delete(idx,arg,axis=0)
        pos = np.random.randint(0,L,size=2)
        while obstacles[pos[0],pos[1]] == 1:
            pos = np.random.randint(0,L,size=2)
        obstacles[pos[0],pos[1]] = 1

    return obstacles, L


def Crystallize(L,phi):
    import numpy as np
    import time as t

    tStart = t.time()
    obstacles = np.zeros(L**2)
    obstacles[0:int(L**2*phi)] = 1
    obstacles = np.random.permutation(obstacles)
    obstacles = obstacles.reshape((L,L))

    lShift = (np.arange(L) + 1) % L
    rShift = (np.arange(L) - 1) % L

    obsD = obstacles[rShift]  # Shift all the obstacles to y+1
    obsU = obstacles[lShift]  # Shift all the obstacles to y-1
    obsR = obstacles[:, rShift]  # Shift all the obstacles to x+1
    obsL = obstacles[:, lShift]  # Shift all the obstacles to x-1

    numNearbyObstacles = obsD+obsU+obsR+obsL
    numObsNearObs = numNearbyObstacles*obstacles

    while np.max(numObsNearObs)!=np.min(numNearbyObstacles):
        movableIdx = np.argwhere(numObsNearObs==np.max(numObsNearObs))
        Pos = movableIdx[np.random.randint(0,movableIdx.shape[0])]
        availableIdx = np.argwhere((numNearbyObstacles+5*obstacles)==np.min(numNearbyObstacles+5*obstacles))
        newPos = availableIdx[np.random.randint(0,availableIdx.shape[0])]

        obstacles[Pos[0],Pos[1]] = 0
        obstacles[newPos[0],newPos[1]] = 1


        obsD = obstacles[rShift]  # Shift all the obstacles to y+1
        obsU = obstacles[lShift]  # Shift all the obstacles to y-1
        obsR = obstacles[:, rShift]  # Shift all the obstacles to x+1
        obsL = obstacles[:, lShift]  # Shift all the obstacles to x-1

        numNearbyObstacles = obsD + obsU + obsR + obsL
        numObsNearObs = numNearbyObstacles * obstacles
        tEnd = t.time()
    print('Obstacle Generation took',(tEnd-tStart)/60,'minutes')

    return obstacles


def NewWells(period,L,k):
    import numpy as np

    def potential(k,r):
        return np.exp(-r**2/(2*k**2))

    L = L // period * period

    obstaclesTemp = np.zeros(L)
    for i in range(int(L // period)):
        obstaclesTemp[int(i * period)] = 1

    obstacles = np.zeros((L, L))
    wellOrigins = np.zeros((L,L))
    for i in range(int(L // period)):
        wellOrigins[:, int(i * period)] = np.copy(obstaclesTemp)

    idx = np.argwhere(wellOrigins == 1)
    idx = np.random.permutation(idx)
    moveDists = []
    for pos in idx:
        distances = np.zeros((L, L))
        idxArray = np.argwhere(distances == 0)
        idxArrayFlat = np.copy(idxArray)
        idxArray = idxArray.reshape(L, L, 2)
        distances = np.sqrt(np.sum((idxArray - np.array([pos[0],pos[1]])) ** 2, axis=2))

        probMove = potential(k,distances)
        probMove -= obstacles
        probMove[probMove<0] = 0
        probMove[pos[0],pos[1]] = 1
        probMove = probMove.flatten()
        probMove = probMove/np.sum(probMove)
        newPos = idxArrayFlat[np.random.choice(np.arange(len(idxArrayFlat)),p=probMove)]
        moveDists.append(np.sqrt(np.sum(newPos-pos)**2))
        obstacles[pos[0],pos[1]] = 0
        obstacles[newPos[0],newPos[1]] = 1
    lamb = np.sqrt(2)/np.sqrt(np.pi)*k

    return obstacles , L

def PotentialMove(period,L,k):
    import numpy as np

    def potential(k,r):
        return np.exp(-r**2/(2*k**2))

    L = L // period * period

    obstaclesTemp = np.zeros(L)
    for i in range(int(L // period)):
        obstaclesTemp[int(i * period)] = 1

    obstacles = np.zeros((L, L))
    for i in range(int(L // period)):
        obstacles[:, int(i * period)] = np.copy(obstaclesTemp)

    idx = np.argwhere(obstacles == 1)
    idx = np.random.permutation(idx)
    moveDists = []
    for pos in idx:
        distances = np.zeros((L, L))
        idxArray = np.argwhere(distances == 0)
        idxArrayFlat = np.copy(idxArray)
        idxArray = idxArray.reshape(L, L, 2)
        distances = np.sqrt(np.sum((idxArray - np.array([pos[0],pos[1]])) ** 2, axis=2))

        probMove = potential(k,distances)
        probMove -= obstacles
        probMove[probMove<0] = 0
        probMove[pos[0],pos[1]] = 1
        probMove = probMove.flatten()
        probMove = probMove/np.sum(probMove)
        newPos = idxArrayFlat[np.random.choice(np.arange(len(idxArrayFlat)),p=probMove)]
        moveDists.append(np.sqrt(np.sum(newPos-pos)**2))
        obstacles[pos[0],pos[1]] = 0
        obstacles[newPos[0],newPos[1]] = 1
    lamb = np.sqrt(2)/np.sqrt(np.pi)*k

    return obstacles , L

def AddRandom(period,L,disorder):

    import numpy as np
    if disorder>1:
        disorder=disorder/100

    L = L // period * period

    obstaclesTemp = np.zeros(L)
    for i in range(int(L // period)):
        obstaclesTemp[int(i * period)] = 1

    obstacles = np.zeros((L, L))
    for i in range(int(L // period)):
        obstacles[:, int(i * period)] = np.copy(obstaclesTemp)

    numMissing = ( np.sum(obstacles) * disorder / (1-disorder) )
    # numMissing = ( (L/period)**2 * disorder)/(1-(disorder))

    for i in range(int(numMissing)):
        pos = np.random.randint(0,L,size=2)
        while obstacles[pos[0],pos[1]] == 1:
            pos = np.random.randint(0,L,size=2)

        obstacles[pos[0],pos[1]] = 1

    return obstacles

def NoCluster(L,Q,phi):
    import numpy as np
    import time as t

    tStart = t.time()

    numObs = int(L**2 * phi)
    obstacles = np.zeros((L,L))
    posList = np.copy(np.argwhere(obstacles==0))
    probs = np.ones(L**2)
    probs = probs/np.sum(probs)
    nearObsIdxList = []
    obsIdxList = []
    for n in range(numObs):

        idx = np.random.choice(L**2,p=probs)
        pos = posList[idx]
        obsIdxList.append(idx)
        #print('Placing obstacle at : ',pos)
        obstacles[pos[0],pos[1]] = 1

        probs[idx] = 0

        lPos = np.array( [pos[0] , (pos[1]-1)%L] )
        rPos = np.array( [pos[0], (pos[1] + 1)%L] )
        uPos = np.array( [(pos[0] - 1)%L, pos[1]] )
        dPos = np.array( [(pos[0] + 1)%L, pos[1]] )
        ulPos = (pos-1)%L
        drPos = (pos+1)%L
        urPos = np.array( [(pos[0] - 1)%L , (pos[1] + 1)%L] )
        dlPos = np.array( [(pos[0] + 1)%L , (pos[1] - 1)%L] )
        affectedPos = np.array([lPos,rPos,uPos,dPos,ulPos,drPos,urPos,dlPos])
        for pos in affectedPos:
            nearObsIdxList.append(np.argwhere(np.sum(posList==pos,axis=1)==2)[0][0])
        nearObsIdxList = np.unique(nearObsIdxList)
        nearObsIdxList = list(nearObsIdxList)
        probs = np.ones(L**2)
        probs[nearObsIdxList] = Q
        probs[obsIdxList] = 0
        if np.sum(probs) == 0:
            print('All probabilities = 0, cannot reach desired concentration, The final concentration was : ', np.sum(obstacles/L**2))
            break
        probs = probs/np.sum(probs)
    print('It took : ',(t.time()-tStart)/60,' Minutes for this set of obstacles to generate')

    return obstacles

def Tiling(L,p,N):
    import numpy as np

    L = (L // p) * p

    obstacles = np.zeros((L,L))

    tileList = np.zeros(p**2)
    tileList[0:N] = 1

    startingPos = np.arange(0,L,p)

    for i in startingPos:
        for j in startingPos:
            obsTemp = np.reshape(np.random.permutation(tileList),(p,p))
            obstacles[i:i+p,j:j+p] = np.copy(obsTemp)

    return obstacles

def WellClusterHybrid(period,L,k,Q):
    import numpy as np
    import time as t
    tStart = t.time()
    def potential(k,r):
        return np.exp(-r**2/(2*k**2))

    L = L // period * period

    lShift = (np.arange(L) + 1) % L
    rShift = (np.arange(L) - 1) % L

    obstaclesTemp = np.zeros(L)
    for i in range(int(L // period)):
        obstaclesTemp[int(i * period)] = 1

    obstacles = np.zeros((L, L))
    wellOrigins = np.zeros((L,L))
    for i in range(int(L // period)):
        wellOrigins[:, int(i * period)] = np.copy(obstaclesTemp)

    idx = np.argwhere(wellOrigins == 1)
    idx = np.random.permutation(idx)

    for pos in idx:
        distances = np.zeros((L, L))
        idxArray = np.argwhere(distances == 0)
        idxArrayFlat = np.copy(idxArray)
        idxArray = idxArray.reshape(L, L, 2)
        distances = np.sqrt(np.sum((idxArray - np.array([pos[0],pos[1]])) ** 2, axis=2))

        probMove = potential(k,distances)
        probMove -= obstacles
        probMove[probMove<0] = 0

        impactedPos = np.zeros((L,L))
        obsD = obstacles[rShift]  # Shift all the obstacles to y+1
        obsU = obstacles[lShift]  # Shift all the obstacles to y-1
        obsR = obstacles[:, rShift]  # Shift all the obstacles to x+1
        obsL = obstacles[:, lShift]  # Shift all the obstacles to x-1

        obsUR = np.copy(obsU[:, rShift])
        obsUL = np.copy(obsU[:, lShift])
        obsDR = np.copy(obsD[:, rShift])
        obsDL = np.copy(obsD[:, lShift])

        impactedPos -= (obsD+obsU+obsR+obsL+obsUR+obsUL+obsDR+obsDL)
        probMove[impactedPos<0] = Q*probMove[impactedPos<0]

        probMove[pos[0],pos[1]] = 1
        probMove = probMove.flatten()
        probMove = probMove/np.sum(probMove)
        newPos = idxArrayFlat[np.random.choice(np.arange(len(idxArrayFlat)),p=probMove)]
        obstacles[pos[0],pos[1]] = 0
        obstacles[newPos[0],newPos[1]] = 1
    print('The configuration took: ',(t.time()-tStart)/60,' minutes to make')
    return obstacles

def Random(phi,L):
    import numpy as np

    obstacles = np.zeros(L**2)
    obstacles[:int((L**2)*phi)] = 1
    obstacles = np.random.permutation(obstacles)
    obstacles = obstacles.reshape((L,L))

    return obstacles

def Periodic(period, L):
    import numpy as np
    L = L//period * period

    obstacles = np.zeros((L,L))
    obstaclesTemp = np.zeros(L)

    for i in range(int(L//period)):
        obstaclesTemp[int(i * period)] = 1

    for i in range(int(L // period)):
        obstacles[:, int(i * period)] = np.copy(obstaclesTemp)

    return obstacles

def Random_Variable_No_Overlap(phi,L,s):
    ### CANNOT PLACE OBSTACLES RIGHT BESIDE ANOTHER EXISTING OBSTACLE (I.E. MUST PLACE AN UNOBSTRUCTED SxS OBSTACLE)

    import numpy as np

    numPlacements = int(round(phi * L**2 / s**2))

    obstacles = np.zeros((L,L))
    # obstaclesTemp = np.ones((s,s))
    obsFiller = np.zeros((L,L))
    complete = True

    for i in range(numPlacements):

        choosablePos = np.argwhere(obsFiller==0)
        numOpenings = int(len(choosablePos))

        if numOpenings==0:
            complete = False
            break

        rand = np.random.randint(numOpenings)
        idx = choosablePos[rand]

        if idx[0]>(L-s) or idx[1]>(L-s):
            # print('Made it')
            for j in range(s):
                for k in range(s):
                    obstacles[(idx[0]+j)%L , (idx[1]+k)%L] = 1


        else:
            obstacles[idx[0]:idx[0]+s , idx[1]:idx[1]+s] = 1

        for j in range(2 * s - 1):
            for k in range(2 * s - 1):
                obsFiller[(idx[0] - (s - 1) + j) % L, (idx[1] - (s - 1) + k) % L] = 1

    return obstacles, complete

def Random_Variable_Overlap(phi,L,s):
    ####  CAN PLACE OBSTACLE WHEREVER, YOU WILL NOT ALWAYS PLACE DOWN AN SxS OBSTACLE
    import numpy as np

    obstacles = np.zeros((L,L))
    numObs = phi * L**2

    while np.sum(obstacles)<numObs:
        idxList = np.argwhere(obstacles==0)
        rand = np.random.randint(len(idxList))
        idx = idxList[rand]

        if idx[0]>(L-s) or idx[1]>(L-s):
            # print('Made it')
            for j in range(s):
                for k in range(s):
                    obstacles[(idx[0]+j)%L , (idx[1]+k)%L] = 1
        else:
            obstacles[idx[0]:idx[0]+s , idx[1]:idx[1]+s] = 1

    return obstacles

def CoreShell(phiR , period , LTot , LP):
    import numpy as np

    obstacles = Random(phiR,LTot)
    obsP = Periodic(period,LP)
    LP = obsP.shape[0]

    startP = int(LTot/2 - LP/2 - 1)
    endP = int(startP + LP)

    obstacles[startP:endP,startP:endP] = obsP

    return obstacles

def RectangleRandom(phi , Lx , Ly):
    import numpy as np

    obstacles = np.zeros(Lx * Ly)
    obstacles[:int((Lx*Ly)*phi)] = 1
    obstacles = np.random.permutation(obstacles)
    obstacles = obstacles.reshape((Ly,Lx))

    return obstacles

def obsGrowth(obs,s):
    import numpy as np
    L = obs.shape[0]
    lShift = (np.arange(L) + 1) % L
    rShift = (np.arange(L) - 1) % L
    for i in range(s-1):
        obs += obs[rShift]
    for j in range(s-1):
        obs += obs[:,rShift]
    obs[obs>1] = 1
    return obs