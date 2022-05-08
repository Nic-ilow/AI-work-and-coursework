import numpy as np
import matplotlib.pyplot as plt
import time as t

def nnChecker(pos):
	global voters, votersWithDisagreeingNeighbours, Ndisagree, l
	x,y,z = pos
	posState = voters[x,y,z]
	
	xr = (x+1)%l
	xl = (x-1)%l
	yr = (y+1)%l
	yl = (y-1)%l
	zr = (z+1)%l
	zl = (z-1)%l

	if posState != voters[xr,y,z]:
		if [(x,y,z),(xr,y,z)] not in votersWithDisagreeingNeighbours:
			votersWithDisagreeingNeighbours.append([(x,y,z),(xr,y,z)])
			votersWithDisagreeingNeighbours.append([(xr,y,z),(x,y,z)])
			Ndisagree +=2
	else:
		if [(xr,y,z),(x,y,z)] in votersWithDisagreeingNeighbours:
			votersWithDisagreeingNeighbours.remove([(xr,y,z),(x,y,z)])
			votersWithDisagreeingNeighbours.remove([(x,y,z),(xr,y,z)])
			Ndisagree -=2
	
	if posState != voters[xl,y,z]:
		if [(x,y,z),(xl,y,z)] not in votersWithDisagreeingNeighbours:
			votersWithDisagreeingNeighbours.append([(x,y,z),(xl,y,z)])
			votersWithDisagreeingNeighbours.append([(xl,y,z),(x,y,z)])
			Ndisagree +=2
	else:
		if [(xl,y,z),(x,y,z)] in votersWithDisagreeingNeighbours:
			votersWithDisagreeingNeighbours.remove([(xl,y,z),(x,y,z)])
			votersWithDisagreeingNeighbours.remove([(x,y,z),(xl,y,z)])
			Ndisagree -=2
	
	if posState != voters[x,yr,z]:
		if [(x,y,z),(x,yr,z)] not in votersWithDisagreeingNeighbours:
			votersWithDisagreeingNeighbours.append([(x,y,z),(x,yr,z)])
			votersWithDisagreeingNeighbours.append([(x,yr,z),(x,y,z)])
			Ndisagree +=2
	else:
		if [(x,yr,z),(x,y,z)] in votersWithDisagreeingNeighbours:
			votersWithDisagreeingNeighbours.remove([(x,yr,z),(x,y,z)])
			votersWithDisagreeingNeighbours.remove([(x,y,z),(x,yr,z)])
			Ndisagree -=2
	
	if posState != voters[x,yl,z]:
		if [(x,y,z),(x,yl,z)] not in votersWithDisagreeingNeighbours:
			votersWithDisagreeingNeighbours.append([(x,y,z),(x,yl,z)])
			votersWithDisagreeingNeighbours.append([(x,yl,z),(x,y,z)])
			Ndisagree +=2
	else:
		if [(x,yl,z),(x,y,z)] in votersWithDisagreeingNeighbours:
			votersWithDisagreeingNeighbours.remove([(x,yl,z),(x,y,z)])
			votersWithDisagreeingNeighbours.remove([(x,y,z),(x,yl,z)])
			Ndisagree -=2
	
	if posState != voters[x,y,zr]:
		if [(x,y,z),(x,y,zr)] not in votersWithDisagreeingNeighbours:
			votersWithDisagreeingNeighbours.append([(x,y,z),(x,y,zr)])
			votersWithDisagreeingNeighbours.append([(x,y,zr),(x,y,z)])
			Ndisagree +=2
	else:
		if [(x,y,zr),(x,y,z)] in votersWithDisagreeingNeighbours:
			votersWithDisagreeingNeighbours.remove([(x,y,zr),(x,y,z)])
			votersWithDisagreeingNeighbours.remove([(x,y,z),(x,y,zr)])
			Ndisagree -=2
	
	if posState != voters[x,y,zl]:
		if [(x,y,z),(x,y,zl)] not in votersWithDisagreeingNeighbours:
			votersWithDisagreeingNeighbours.append([(x,y,z),(x,y,zl)])
			votersWithDisagreeingNeighbours.append([(x,y,zl),(x,y,z)])
			Ndisagree +=2
	else:
		if [(x,y,zl),(x,y,z)] in votersWithDisagreeingNeighbours:
			votersWithDisagreeingNeighbours.remove([(x,y,zl),(x,y,z)])
			votersWithDisagreeingNeighbours.remove([(x,y,z),(x,y,zl)])
			Ndisagree -=2

tStart = t.time()

l = 6
V = l**3
simulations = 25


qlist = [0.2,0.3,0.5,0.8]
typeBlock = ['permute','block']
counterListPermuteMaster = []
counterListBlockMaster = []



trueP = []
for m in range(simulations):
	counterListBlock = []
	counterListPermute = []

	for q in qlist:
	

		for distribution in typeBlock:

			votersinit = np.ones(V)
			votersinit[0:int(q*V)] = -1
			p = np.sum(votersinit==1)/V

			if distribution=='permute':
				votersinit = np.random.permutation(votersinit)

			votersinit = np.reshape(votersinit,(l,l,l))
			counter = 0
			for n in range(simulations):
				voters = np.copy(votersinit)
				votersWithDisagreeingNeighbours = []
				Ndisagree = 0
				tElapsed = 0

				for i in range(l):
					for j in range(l):
						for k in range(l):
							nnChecker((i,j,k))

				while Ndisagree>(.1*V):
					idx = np.random.choice(Ndisagree)
					dt = 1/Ndisagree
					pair = votersWithDisagreeingNeighbours[idx]

					x,y,z = pair[0]
					x1,y1,z1 = pair[1]

					voters[x,y,z] = voters[x1,y1,z1]

					nnChecker((x,y,z))
					tElapsed+=dt
				if np.sum(voters)>0:
					counter+=1
			
			if distribution=='permute':
				counterListPermute.append(counter/simulations)
			else:
				counterListBlock.append(counter/simulations)
				if m==1:
					trueP.append(p)
	counterListPermuteMaster.append(np.array(counterListPermute))
	counterListBlockMaster.append(np.array(counterListBlock))


counterListBlockMaster = np.array(counterListBlockMaster)
counterListPermuteMaster = np.array(counterListPermuteMaster)
BlockMeans = []
PermuteMeans = []
BlockErrors = []
PermuteErrors = []
for i in range(len(qlist)):
	BlockMeans.append(np.mean(counterListBlockMaster[:,i]))
	PermuteMeans.append(np.mean(counterListPermuteMaster[:,i]))
	BlockErrors.append(np.std(counterListBlockMaster[:,i])/np.sqrt(simulations))
	PermuteErrors.append(np.std(counterListPermuteMaster[:,i])/np.sqrt(simulations))

tEnd = t.time()
print('This code took: ',(tEnd-tStart)/60.0,' minutes to run')
trueP.append(1)
trueP = [0] + trueP
BlockMeans.append(1)
PermuteMeans.append(1)
BlockMeans = [0]+BlockMeans
PermuteMeans = [0]+PermuteMeans
BlockErrors.append(0)
BlockErrors = [0]+BlockErrors
PermuteErrors = [0]+PermuteErrors
PermuteErrors.append(0)

straightLine = np.linspace(0,1,100)
plt.errorbar(trueP,BlockMeans,yerr=BlockErrors,label='Block initial condition',color='b',ecolor='b',fmt='.',capsize=3)
plt.errorbar(trueP,PermuteMeans,yerr=PermuteErrors,label='Random Distributed initial condition',color='orange',ecolor='orange',fmt='.',capsize=3)
plt.plot(straightLine,straightLine,label='Exact value',color='k')
plt.xlabel('Expected probability')
plt.ylabel('Simulated probability')
plt.legend()
plt.show()