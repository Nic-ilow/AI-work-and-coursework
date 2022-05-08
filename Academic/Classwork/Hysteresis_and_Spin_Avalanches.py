import numpy as np
import matplotlib.pyplot as plt
import time as t

def Energy(spins,fields,external):
	return -np.sum(spins[:-1]*spins[1:]) -np.sum(fields*spins) -external*np.sum(spins) # Energy of system

def update():
	global L,spins
	Lnew = spins[:-2]+spins[2:]+fields[1:-1]+extField 			#  Local field after external field has changed
	flippedLocal = np.argwhere((np.sign(Lnew)*np.sign(L))<0)	#  Where has the local field crossed 0 (e.g. what spins are potentially flippable)
	flippable = []

	for idxPrime in flippedLocal:								#  Check all of the indexes in flippedLocal				
		idx = idxPrime[0]
		tempSpins = np.copy(spins)								#  Copy the spins array
		tempSpins[idx+1] *= -1									#  idx+1 because L_0 corresponds to the local field effect at s_1 , flip that spin

		if Energy(tempSpins,fields,extField)<Energy(spins,fields,extField):	#  If flipping the spin decreased the energy
			flippable.append(idx)											#  Append that index to a flippable list

	while len(flippable)>0:													#  While the list of flippable spins isn't exhausted

		for i in range(len(flippable)):										#  Flip all of the spins in the list
			idx = flippable[i]
			spins[idx+1] *= (-1)
		L = np.copy(Lnew)													#  Make L the local field effects before spin flipping
		Lnew = spins[:-2]+spins[2:]+fields[1:-1]+extField					#  Calculate the new L with all of the new spin flips		

		flippedLocal = np.argwhere((np.sign(Lnew)*np.sign(L))<0)			#  Compare where the local field effects have crossed 0 again
		flippable = []
		for idxPrime in flippedLocal:										#  Find which of those spins we are allowed to flip based on energy decreasing spin dynamics
			idx = idxPrime[0]
			tempSpins = np.copy(spins)
			tempSpins[idx+1] *= -1
			if Energy(tempSpins,fields,extField)<Energy(spins,fields,extField):
				flippable.append(idx)

##############################################################
##############################################################  Part A
##############################################################

N = 1000											#  Number of spins in chain
returnedList = []
for k in range(2):
	for j in range(2):
		extField = 3
		spins = np.ones(N)

		if k == 0:									#  Uniformly distribute the fields
			fields = np.random.uniform(-2,2,N)
		else:										#  Normally distribute the fields
			fields = np.random.normal(loc=0,scale=1,size=N)

		L = spins[:-2]+spins[2:]+fields[1:-1]+extField	#  Calculate the local field effect

		if j == 0:										#  Immediate descent
			extField = -3		

			update()									#  Update the spins

			extField = 3								#  Immediate increase

			update()									#  Reupdate the spins

			if all(spins==np.ones(N))==True:			#  If spins are the same as I.C. append True
				returnedList.append(True)

		else:											#  Gradual descent
			for i in range(60000):

				L = spins[:-2]+spins[2:]+fields[1:-1]+extField	#  Calculate the local field effect
				if i<30000:										#  First half of loops decrease the field
					extField -= .0002
				else:											#  Second half of loops increase the field
					extField += .0002
				
				update()										#  Update the spins every loop

			if all(spins==np.ones(N))==True:					#  If spins are the same as I.C. append True
				returnedList.append(True)

print(returnedList)												#  Print the list of whether each I.C. / gradual or Immediate returned , all True means verified
if sum(returnedList)==4:
	print('All combinations of Normally and Uniformly distributed fields, with gradual or instantaneous descent of the magnetic field return back to the initial condition')

		

######################################################
######################################################  Part B
######################################################

N = 1000														#  Number of spins in chain
simulations = 100												#  Number of chains to average over
magMaster = []
fieldArray = []

for n in range(simulations):

	spins = np.ones(N)											#  All spins start up
	fields = np.random.uniform(-2,2,N)							#  Uniform distribution of fields, between [-2,2] because Delta=2
	extField = 3												#  Initial field is 3
	magArray = []
	tStart = t.time()

	for i in range(60000):										#  60000 steps of .0002 will get us from 3 -> -3 -> 3 for field values

		L = spins[:-2]+spins[2:]+fields[1:-1]+extField			#  Calculate the local fields
		if i<30000:												#  First half of steps decrease field
			extField -= .0002
		else:													#  Last half of steps increase field
			extField += .0002

		update()												#  Upgdate spins

		magArray.append(np.sum(spins)/N)						#  Append the net magnetization
		if n == 0:												#  Get the list of field values through appending
			fieldArray.append(extField)
	magMaster.append(np.array(magArray))
	tEnd = t.time()
	print('Loop #%.d done in %.2f minutes'%(n,(tEnd-tStart)/60))

magMaster = np.array(magMaster)
fieldArray = np.array(fieldArray)

magErrors = []
magMeans = []

#  Averaging and Finding Error
for i in range(len(magMaster[0])):
	magMeans.append(np.mean(magMaster[:,i]))
	magErrors.append(np.std(magMaster[:,i])/np.sqrt(simulations))


upperFields = np.arange(-2,0,.01)
plt.plot(fieldArray,magMeans,label='averaged m(h) over 100 chains')
plt.plot(upperFields,1+3/2*upperFields-1/8*upperFields**3,ls='dashed',color='k',label='m(h) = 1+3h/2-h^3/8 for -2<h<0')
plt.xlabel('external field (h)')
plt.ylabel('m(h)')
plt.legend()
plt.show()
