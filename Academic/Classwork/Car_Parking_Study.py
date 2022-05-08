import numpy as np
import matplotlib.pyplot as plt
import random as rand
import time as t


tStart = t.time()

def park(parkRange):
	global free, occupied, tElapsed, maxPos, freeSpace, lengths

	parkAt = np.random.uniform(0,parkRange-1) #  car starts at this number and goes 1 to the right, hence parkRange-1
	
	tElapsed += maxPos/freeSpace			#  t += L/L_a
	occupied += 1

	newFreeLeft = parkAt					#  Length of 0 to parkAt is just parkAt
	newFreeRight = parkRange-(parkAt+1)		#  Length of after parkAt is parkRange-(parkAt+1)

	if (newFreeLeft)<1: # If we can't fit a unit length car, list as occupied
		occupied += newFreeLeft
	else:								  #  If we can, add it to the free region
		free.append(newFreeLeft)

	if (newFreeRight)<1: # If we can't fit a unit length car, list as occupied
		occupied += newFreeRight
	else:
		free.append(newFreeRight)


maxPosList = (1e4,1e5)		#  The system lengths we want to consider
pjamMaster = []
asymptoticMaster = []
for maxPos in maxPosList:	#  For system lengths in system legnth list
	pjam = []
	simulations = 20		#  num simulations to average over
	asymptoticMaxPos = []

	for i in range(simulations):	#  run for all simulations
		free = [maxPos]				#  everything is free initially
		freeProbabilities = np.array([1])	#  only one space so 1 probability
		occupied = 0
		tElapsed = 0
		tCut = 1
		freeSpace = maxPos-1
		N=0
		M=0
		asymptoticOccupation = []


		while len(free)>0:
			idx = np.random.choice(len(free),p=freeProbabilities) #  Choose a random index with the probability distribution

			parkRange = free.pop(idx)			#  this space is going to be parked in, so remove it from free
			park(parkRange)						#  Parking function, parks a car, updates freeSpace,free,occupied,t

			temp = np.array(free)-1				#  make an array of l-1's
			freeSpace = np.sum(temp)   			#  sum it all
			freeProbabilities = temp/freeSpace	#  get the probability distribution
			if maxPos==maxPosList[0]:			#  Weird if logic to make sure that the aymptotic arrays are all the same length, and hence a 2darray easy to work with
				if M<23:
					if tElapsed>=tCut:
						asymptoticOccupation.append(N/maxPos)
						tCut*=2
						M+=1

			else:
				if M<28:
					if tElapsed>=tCut:
						asymptoticOccupation.append(N/maxPos)
						tCut*=2
						M+=1



			N+=1
			
		pjam.append(N/maxPos)		# pjam is the number of cars at the end when theres no more spots available
		asymptoticMaxPos.append(np.array(asymptoticOccupation))

		tEnd = t.time()
		print('Computation takes this long: ',tEnd-tStart)

	pjamMaster.append(np.array(pjam))
	asymptoticMaster.append(np.array(asymptoticMaxPos))



pjamMaster = np.array(pjamMaster)



###  GETTING AVERAGES AND ERRORS
asymptoticMeans = []
asymptoticError = []
for i in range(len(asymptoticMaster[0][0])):
	asymptoticMeans.append(np.mean(asymptoticMaster[0][:,i]))
	asymptoticError.append(np.std(asymptoticMaster[0][:,i])/np.sqrt(simulations))

asymptoticMeans2 = []
asymptoticError2 = []
for i in range(len(asymptoticMaster[1][0])):
	asymptoticMeans2.append(np.mean(asymptoticMaster[1][:,i]))
	asymptoticError2.append(np.std(asymptoticMaster[1][:,i])/np.sqrt(simulations))

pjamMeans = []
pjamErrors = []
for i in range(len(pjamMaster)):
	pjamMeans.append(np.mean(pjamMaster[i]))
	pjamErrors.append(np.std(pjamMaster[i])/np.sqrt(simulations))

tSet1 = np.power(2,np.arange(len(asymptoticMeans)))
tSet2  = np.power(2,np.arange(len(asymptoticMeans2)))

print('Error on pjam for L=',maxPosList[0],' is ',pjamErrors[0])
print('Error on pjam for L=',maxPosList[1],' is ',pjamErrors[1])

###  PLOTTING

plt.figure()
ax = plt.gca()
ax.set_xscale('log')
plt.errorbar(tSet1,asymptoticMeans,yerr=asymptoticError,fmt='.',ecolor='r',color='orange',capsize=3,label='asymptotic Approach to pjam')
plt.fill_between(tSet1,pjamMeans[0]-pjamErrors[0],pjamMeans[0]+pjamErrors[0],alpha=0.5)
plt.plot(tSet1,[pjamMeans[0]]*len(tSet1),ls='dashed',color='k',label='pjam at no more free space')
plt.xlabel('t')
plt.ylabel('p')
plt.title('L= %.2f'%(maxPosList[0]))
plt.legend()

plt.figure()
ax = plt.gca()
ax.set_xscale('log')
plt.errorbar(tSet2,asymptoticMeans2,yerr=asymptoticError2,fmt='.',ecolor='r',color='orange',capsize=3,label='asymptotic Approach to pjam')
plt.fill_between(tSet2,pjamMeans[1]-pjamErrors[1],pjamMeans[1]+pjamErrors[1],alpha=0.5)
plt.plot(tSet2,[pjamMeans[1]]*len(tSet2),ls='dashed',color='k',label='pjam at no more free space')
plt.xlabel('t')
plt.ylabel('p')
plt.title('L= %.2f'%(maxPosList[1]))
plt.legend()

plt.figure()
ax = plt.gca()
ax.set_xscale('log')
ax.set_yscale('log')
plt.errorbar(tSet2,pjamMeans[1]-asymptoticMeans2,yerr=asymptoticError2+pjamErrors[1],fmt='.',ecolor='r',label='simulated with summed errors,',capsize=3)
plt.plot(tSet2,1/tSet2,label='1/t',ls='dashed',color='k')
plt.xlabel('t')
plt.ylabel('Residual of pjam')
plt.title('L= %.2f'%(maxPosList[1]))

plt.legend()


plt.show()