"""
	allfluxSFD.py	(from ifluxSFD.py on Sep 4 2016)
	from 1 to 0 density
	(get mobilities from gradient of average density)
	nb. add multiple run option and analysis
"""
# python 2 modifications
from __future__ import division, print_function
#input = raw_input

import random as ran
import math as m
import numpy as np
import scipy.special as scis
import argparse		# for command line inputs
import time			# for default seed
import pickle		# for dumping lists and such to files

################
# initialize

parser=argparse.ArgumentParser(description='Command line inputs:') # needs to import argparse at top
parser.add_argument('-R','--R',default=0.01,help='koff/kon ratio')
parser.add_argument('-D0','--D0',default=2.7E5,help='diffusion as measured')	# Szyk measured diff
parser.add_argument('-dx','--dx',default=7.0,help='particle diam')		# nm (size of alphatTat1)
parser.add_argument('-koff','--koff',default=10.0,help='off rate (per second)')
parser.add_argument('-tmax','--tmax',default=9999999.0,help='maximum time in seconds')
parser.add_argument('-nrun','--nrun',default=0,help='number of this run')
parser.add_argument('-name','--name',default='MT',help='datafile name base in single quotes')
parser.add_argument('-seed','--seed',default=0,help='seed for random number')
parser.add_argument('-width','--width',default=16,help='width system in units of dx') 
args=parser.parse_args()

R=float(args.R)
D0=float(args.D0)
dx=float(args.dx)
koff=float(args.koff)
tmax=float(args.tmax)
nrun=int(args.nrun)
name=str(args.name)
seed=int(args.seed)+int(time.time()*1000000)   # so we essentially use the time, but we allow external offset
width=int(args.width)

ran.seed(seed)
fractionfree=R/(1.0+R)  # large R gives fractionfree=1
khop=2.0*D0/(dx*dx)/fractionfree  # 2e6, correction factor is to use bare D for hopping
#should have 2.0 factor in khop Sept 14 2016
kon = koff/R  # (per second) 0.0 gives SD (singular limit)  (0.001 gives 10/s)  R=koff/kon

# density 1 at 0
# density 0 at width (which isn't a site)

print("R:",R," D0:",D0," dx:",dx," kon:",kon," koff:",koff," khop:", khop)
print("tmax:",tmax," nrun:",nrun," name:",name," seed:",seed," width:",width)

#
x=np.zeros(width,int)  # 0 if empty, 1 if particle
Nbound=Nfree=0	# number of particles
free=[]	# positions of free particles
bound=[]# positions of bound particles
leftflux=[] # averages
rightflux=[]


###### ROUTINES ###########
def create(pos): # insert a particle at pos, randomly bound or unbound as appropriate
	global Nfree,Nbound
	x[pos]=1
	if ran.random()< fractionfree:
		free.append(pos)	# indices of free particles
		Nfree +=1
	else:
		bound.append(pos)
		Nbound +=1

def hop(this): 	# try to hop free particle of entry this
	global thisleft,thisright,Nfree  # keep track of flux
	pos=free[this]

	if ran.random()<0.5:	# never leave to left
		if pos>0 and x[pos-1]==0: 	# not blocking to left
			x[pos] = 0
			x[pos-1]=1
			free[this]=pos-1
	else:
		if pos==width-1:
			thisright +=1		# leave to right
			x[pos]= 0
			free.pop(this)
			Nfree-=1
		elif x[pos+1]==0:	# not blocking to right
			x[pos] =0
			x[pos+1]=1
			free[this]=pos+1

def output():	#
	global thisstart
	fadd	  = open('allFLUX_{0}_{1}.dat'.format(name,scale),'a')
	fadd.write("{0:.7f}\t{1:.7f}\t{2:d}\t{3:d}\t{4:.3f}\t{5:.3f}\n".format(thisleft,thisright,int(scale),width,nexttime,R))
	fadd.close()
	print(ttime,nexttime,scale,thisleft,thisright,Nbound,Nfree,time.time()-thisstart)

	np.save('DENS_{0}_{1}_{2}'.format(name,scale,nrun),avp)	# average density

#######################################

### CREATE INITIAL PARTICLES ###

# add initial particles
# at density 1.0 at left end (pos=0)
# at density 0.0 at right end (pos=width)-- just past end of array
for pos in np.arange(0,width):	# from 0 to width-1
	if ran.random() < 1.0-pos/(width*1.0):
		create(pos)	# nb. list is not generally in order)

TFACTOR = 2.0
### START EVOLUTION 
ttime =0
scale=0
nexttime=1.0  # for next average

while ttime<tmax:

	thisleft=thisright=thistot=0
	thistmax=nexttime
	avp=np.zeros(width+2) # running average of density
	thisstart=time.time()

	while thistmax >0.0:
		
		if x[0] ==0:  	# add to left at infinite rate
			x[0]=1
			free.append(0)
			Nfree +=1
			thisleft += 1
		
		totrate = Nfree*(kon+khop)+Nbound*koff #
		dt  = -1.0/totrate*m.log(1.0-ran.random())   # kmc dt
		
		# saving density every step slows things down by a factor of 2
		avp[width] +=dt  	# normalization
		avp[0:width] += x*dt

		thistmax -= dt
		ttime += dt
		nextx = ran.random()*totrate	# so nextx is where in totrate

		if nextx < Nfree*kon: # bind
			pos=free.pop(ran.randrange(Nfree)) #index of this
			bound.append(pos) # pop from free, add to bound
			Nfree  -= 1
			Nbound += 1
		elif nextx < Nfree*(kon+khop): # hop
			hop(ran.randrange(Nfree))  # hop with index of free
		else:   # unbind
			pos=bound.pop(ran.randrange(Nbound))
			free.append(pos) # pop from bound, add to free
			Nbound -= 1
			Nfree  += 1

## ADD SOME AVERAGE AND OUTPUT FOR DENSITY
	print("*",thisleft,thisright,nexttime,thistmax,ttime)
	thisleft /= (1.0*nexttime)
	thisright /= (1.0*nexttime)
	leftflux.append(thisleft)
	rightflux.append(thisright)

	avp[width+1] = (thisleft+thisright)/2.0
	print("**",thisleft,thisright,nexttime,thistmax,ttime,avp[width+1])


	output()
#	print(ttime,nexttime,scale,thisleft,thisright,Nbound,Nfree)

	scale += 1
	nexttime *= TFACTOR
###################### END OF FILE ##############
