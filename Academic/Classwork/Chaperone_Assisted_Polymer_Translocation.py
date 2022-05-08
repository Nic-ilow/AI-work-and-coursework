import numpy as np
import time as t
import matplotlib.pyplot as plt
import scipy.special as scis


def hop():
	global x,pore,tElapsed,dt,stepsTaken
	if np.random.uniform()<0.5: #  Hop left
		if x[pore+1]==0:
			pore+=1				#  Pore moving Right is the same as polymer going left
			#stepsTaken += 1
	
	else:						#  Hop right
		pore-=1					#  Pore moving Left is the same as polymer going right

	tElapsed += dt				#  Update time


def bind():
	global chapLocs,tElapsed,dt,pore,x,numChaps
	chaperoneLoc = np.random.choice(len(x[pore:]))+pore #  Choose a random spot beyond the pore
	if x[chaperoneLoc] == 0:							#  If it's 0, make it 1, e.g. add chaperone, save index
		x[chaperoneLoc] = 1								
		chapLocs.append(chaperoneLoc)
		numChaps += 1					
	tElapsed += dt										#  Update time
	

def unbind():
	global chapeLocs,tElapsed,dt,x,numChaps
	chaperoneLoc = np.random.choice(len(chapLocs)) 		#  Choose a random known chaperone location
	chapLocs.pop(chaperoneLoc)							#  It's no longer attached to the polymer
	x[chaperoneLoc] = 0									#  Update the polymer
	numChaps -= 1
	tElapsed += dt										#  Update time

khop = 2.0
kon = 1
koff = kon
#koff = 1
L = 1000
initialPore = int(L/2)
tMax = 1e5
simulations = 20
tStart = t.time()
konList = np.logspace(-1,-5,5,base=10)
C = 3**(1/3)*scis.gamma(2/3)/scis.gamma(1/3)

velocityMaster = []
velocityErrors = []
expectedVelocityMaster = []

for kon in konList:
	koff = kon
	velocities = []
	for i in range(simulations):
		stepsTaken = 0
		x = np.zeros(L)
		pore = initialPore
		numChaps = 0
		chapLocs = []
		tElapsed = 0

		while tElapsed<tMax:
			
			if pore==0:	
				#print('Polymer Translocated')
				break
			
			gammaTot = khop + (L-pore-numChaps)*kon + numChaps*koff
			dt = -1/gammaTot*np.log(np.random.uniform())

			nextEvent = np.random.uniform()*gammaTot

			if nextEvent<khop:
				hop()
			elif nextEvent<(khop + (L-pore)*kon):
				bind()
			else:
				unbind()

		velocities.append(initialPore/tElapsed)
		tEnd = t.time()
		print('The code has been running for: ',(tEnd-tStart)/60,' minutes')

	expectedVelocity = C*(kon**(1/3))
	velocityMaster.append(np.mean(velocities))
	velocityErrors.append(np.std(velocities)/np.sqrt(simulations))
	expectedVelocityMaster.append(expectedVelocity)

#print('The average velocity is: ',np.mean(velocities))
#print('The expected velocity is: ',expectedVelocity)
expectedVelocityMaster = np.array(expectedVelocityMaster)
velocityMaster = np.array(velocityMaster)
velocityErrors = np.array(velocityErrors)

plt.figure()
ax = plt.gca()
plt.errorbar(konList,abs(expectedVelocityMaster-velocityMaster),yerr=velocityErrors,fmt='.',ecolor='r')
ax.set_xscale('log')
plt.xlabel('kon = koff')
plt.ylabel('Residual of Velocity')

plt.figure()
ax = plt.gca()
plt.errorbar(konList,velocityMaster,yerr=velocityErrors,label='Simulated Velocity',fmt='.',ecolor='r')
plt.scatter(konList,expectedVelocityMaster,label='Exact Velocity',color='orange')
ax.set_xscale('log')
plt.xlabel('kon = koff')
plt.ylabel('Velocity')
plt.legend()

plt.show()