import numpy as np
import matplotlib.pyplot as plt
import time as t

def m(t):	#  Analytic magnetization function
	return 1/3*(1+np.exp(-t)+np.exp(-2*t))

tStart = t.time()

N = 10000	#  number of spins

tMax = 1e2	
tList = []
simulations = 100	#  number of simulations

magMaster = []
eMaster = []

for n in range(simulations):

	tElapsed = 0
	magnetization = []
	energy = []
	tCut = 1/N

	spins = np.ones(N)							#  All spins initially up
	bonds = np.random.uniform(-1,1,size=N-1)	#  N-1 bonds, uniformly distributed, -1,1 so p(J)=p(-J)

	while tElapsed<tMax:
		gamma = N 									#  N spins can be flipped, and thats our only rate
		dt = -1/gamma*np.log(np.random.uniform())	#  SSA dt
		idx = np.random.choice(N)					#  Choose a spin to look at
		tempSpins = np.copy(spins)					#  Make a copy of the spins
		tempSpins[idx] = tempSpins[idx]*(-1)		#  Flip the spin we chose
		if (-np.sum(tempSpins[:(N-1)]*bonds*tempSpins[1:N]))>(-np.sum(spins[:(N-1)]*bonds*spins[1:N])):	#  Compare the energies of flipped arrangement vs non-flipped
			spins[idx] = spins[idx]*(-1)																#  If our energy increases, confirm the spin flip

		tElapsed += dt 				#  Update time
		
		if (tElapsed-tCut)>0:		#  When we pass our cutoff time, store energy and magnetization
			if n==0:
				tList.append(tCut)
			tCut*=1.44
			energy.append(-np.sum(spins[:(N-1)]*bonds*spins[1:N]))
			magnetization.append(np.sum(spins)/N)

	magMaster.append(np.array(magnetization))			#  Save the magnetizations and energies from this simulations
	eMaster.append(np.array(energy))

magMaster = np.array(magMaster)
eMaster = np.array(eMaster)

magErrors = []
magMeans = []
eErrors = []
eMeans = []

#  Averaging and Finding Error
for i in range(len(magMaster[0])):
	magMeans.append(np.mean(magMaster[:,i]))
	magErrors.append(np.std(magMaster[:,i])/np.sqrt(simulations))
	eMeans.append(np.mean(eMaster[:,i]))
	eErrors.append(np.std(eMaster[:,i])/np.sqrt(simulations))

tEnd = t.time()
print('This code took : ',((tEnd-tStart)/60),' minutes to run')

#  Plotting

shapeList = ['s','o','p','P','x']
for i in (np.arange(5)+2):
	plt.figure(1)
	plt.scatter(tList,magMaster[i],label=('random I.C. #%.d'%(i)),marker=shapeList[i-2])
ax = plt.gca()
plt.figure(1)
plt.plot(tList,[1/3]*len(tList),label='1/3',color='k',ls='dashed')
ax.set_xscale('log')
plt.legend()
plt.xlabel('t (s)')
plt.ylabel('m(t)')

plt.figure(2)
tAnalyticList = np.arange(1/N,tList[-1],1/N)
plt.errorbar(tList,magMeans,yerr=magErrors,ecolor='r',capsize=3,fmt='.',label='<m(t)> over different IC')
plt.plot(tAnalyticList,m(tAnalyticList),ls='dashed',color='k',label='1/3*(1+exp(-t)+exp(-2t)')
ax = plt.gca()
ax.set_xscale('log')
plt.xlabel('t (s)')
plt.ylabel('m(t)')
plt.legend()

plt.figure(3)
plt.errorbar(tList,eMeans,yerr=eErrors,ecolor='r',capsize=3,fmt='.',label='<E> with errorbars in Energy')
ax = plt.gca()
ax.set_xscale('log')
plt.xlabel('t (s)')
plt.ylabel('Energy')
plt.legend()
plt.show()