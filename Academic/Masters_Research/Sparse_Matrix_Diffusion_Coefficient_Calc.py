def SparseExactD(obs,start_coords,L):
	import numpy as np
	from LoopFiller import lakeFiller
	import scipy.sparse as ss
	import scipy.sparse.linalg as ssl
	import time as t

	tStart = t.time()
	lShift = (np.arange(L) + 1) % L
	rShift = (np.arange(L) - 1) % L
	epsilon = 1e-6
	obstacles = np.copy(obs)
	obs = lakeFiller(obs,start_coords)

	counter = 0
	indices = np.zeros_like(obs)
	for i in range(L):
		for j in range(L):
			if obs[i,j] == 0:
				indices[i,j] = counter
				counter += 1
	indices -= obs
	indices = indices.astype(int)

	indicesD = indices[rShift].ravel()
	indicesU = indices[lShift].ravel()
	indicesR = indices[:,rShift].ravel()
	indicesL = indices[:,lShift].ravel()
	indices = indices.ravel()

	N = int(np.max(indices)+1)


	Ai = []
	IAi = [0]
	JAi = []
	Rowi = []

	Aeps = []
	IAeps = [0]
	JAeps = []
	Roweps = []
	for idx in np.arange(N-1):
		i = np.argwhere(indices==idx)[0][0]
		counter = 0
		counterEps = 0
		selfi = 0
		selfEps = 0

		if indicesD[i]>=0:
			JAi.append(indicesD[i])
			#Rowi.append(idx)
			Ai.append(0.25)
			counter += 1
		else:
			selfi += 0.25 
			
		if indicesU[i]>=0:
			JAi.append(indicesU[i])
			#Rowi.append(idx)
			Ai.append(0.25)
			counter += 1
		else:
			selfi += 0.25
			
		if indicesL[i]>=0:
			JAi.append(indicesL[i])
			#Rowi.append(idx)
			JAeps.append(indicesL[i])
			#Roweps.append(idx)
			Ai.append(0.25)
			Aeps.append(-0.25)
			counter += 1
			counterEps += 1
		else:
			selfi += 0.25
			selfEps += 0.25
			
		if indicesR[i]>=0:
			JAi.append(indicesR[i])
			#Rowi.append(idx)
			JAeps.append(indicesR[i])
			#Roweps.append(idx)
			Ai.append(0.25)
			Aeps.append(0.25)
			counter += 1
			counterEps += 1
		else:
			selfi += 0.25
			selfEps -= 0.25

		Ai.append(selfi-1)
		#Rowi.append(idx)
		JAi.append(idx)

		if selfEps!=0:
			Aeps.append(selfEps)
			#Roweps.append(idx)
			JAeps.append(idx)
			counterEps += 1
			
		IAi.append(IAi[idx]+counter+1)
		IAeps.append(IAeps[idx]+counterEps)
	for i in range(N):
		#Rowi.append(N-1)
		JAi.append(i)
		Ai.append(1)
	IAi.append(IAi[N-1]+N)
	IAeps.append(IAeps[N-1])
	Ai = np.array(Ai)
	Aeps = np.array(Aeps)
	JAi = np.array(JAi)
	JAeps = np.array(JAeps)
	IAi = np.array(IAi)
	IAeps = np.array(IAeps)
	#Roweps = np.array(Roweps)
	#Rowi = np.array(Rowi)

	nVeci = np.ones(N)
	nVeci /= N

	'''
	b = np.zeros_like(nVeci)

	for i in range(len(IAeps)-1):
		for j in np.arange(IAeps[i],IAeps[i+1]):
			b[i] += Aeps[j]*nVeci[JAeps[j]]

	b = -b
	'''
	try:
		AiSparse = ss.csr_matrix((Ai,JAi,IAi),shape=(N,N))
		AepsSparse = ss.csr_matrix((Aeps,JAeps,IAeps),shape=(N,N))
	except ValueError:
		np.save('obstaclesPreFill.npy',obstacles)
		np.save('obstaclesBroke.npy',obs)

	b = -(AepsSparse*nVeci)

	#AiSparse2 = ss.csr_matrix((Ai,(Rowi,JAi)),shape=(N,N))
	nVecEps = ssl.spsolve(AiSparse,b)

	obsD = obs[rShift] # Shift all the obstacles to y+1
	obsU = obs[lShift] # Shift all the obstacles to y-1
	obsR = obs[:,rShift]  #  Shift all the obstacles to x+1
	obsL = obs[:,lShift]  #  Shift all the obstacles to x-1


	rProbs = np.ones_like(obs)/4
	lProbs = np.copy(rProbs)
	uProbs = np.copy(rProbs)
	dProbs = np.copy(rProbs)

	rProbsEps = np.zeros_like(obs)+epsilon/4
	lProbsEps = np.zeros_like(obs)-epsilon/4


	vVeciTemp = rProbs*(1-obsL)*(1-obs) - lProbs*(1-obsR)*(1-obs)

	vVecEpsTemp = rProbsEps*(1-obsL)*(1-obs) - lProbsEps*(1-obsR)*(1-obs)
	vVecEpsTemp /= epsilon

	vVeci = []
	vVecEps = []
	for i in range(L):
		for j in range(L):
			if obs[i,j]==0:
				vVeci.append(vVeciTemp[i,j])
				vVecEps.append(vVecEpsTemp[i,j])

	vVeci = np.array(vVeci)
	vVecEps = np.array(vVecEps)


	D = 2*(np.dot(vVecEps,nVeci) + np.dot(vVeci,nVecEps))
	tEnd = t.time()
	# print('It took this long to find D',(tEnd-tStart)/60)

	return D
