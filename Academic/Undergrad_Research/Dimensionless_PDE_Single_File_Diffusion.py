import math as m
import numpy as np
import matplotlib.pyplot as plt
import argparse
import scipy.special as scis
import scipy.integrate as scii
import scipy.optimize as scio
import pickle
import time


parser = argparse.ArgumentParser(description='Command line inputs:')
parser.add_argument('-p0hat','--p0hat',default=1e-4)
parser.add_argument('-dxbar','--dxbar',default=6.0)
parser.add_argument('-xbarmax','--xbarmax',default=1250.0)
parser.add_argument('-dtbar','--dtbar',default=.1)
parser.add_argument('-tbarmax','--tbarmax',default=1.0e4)
args = parser.parse_args()

p0hat = float(args.p0hat)
dxbar = float(args.dxbar)
xbarmax = float(args.xbarmax)
dtbar = float(args.dtbar)
tbarmax = float(args.tbarmax)
colorList = ['b','g','r','c','m','k','y']


xbar = np.arange(0,xbarmax,dxbar)
tbar = np.arange(0,tbarmax,dtbar)

width = np.size(xbar)       

### pTot and aTot plotting array
dtbar = min(0.5/p0hat , 0.10*dxbar**2)
tArray = np.logspace(-200,0,num=200,base=1.1)*tbarmax
p2 = np.logspace(-2,2,5,base=3)
#p2 = np.array([.0625,64])
pArray = np.zeros((np.size(p2) , width))        
aArray = np.zeros_like(pArray)
pTotArray = np.zeros( ( np.size(p2) , np.size(tArray) ) )
aTotArray = np.zeros_like(pTotArray)
### Initializing counters and a timer 
tstart = time.time()
counter=0
### Array Pre-allocation
pReg = np.zeros(width+1)
pReg1 = np.zeros_like(pReg)
aReg = np.zeros(width)
pTotReg = np.zeros(np.size(tArray))
aTotReg = np.zeros_like(pTotReg)
plottingTimes = np.array([2.0,20.0,200.0])*dtbar

for p0hat in p2:
        dtbar = min(0.5/p0hat , 0.10*dxbar**2)
        if dtbar==0.5/p0hat:
                print('p0Hat Limit')
        else:
                print('dxbar Limit')
        epsilon = dtbar/2
        p = np.zeros(width+1)
        p_1 = np.zeros_like(p)
       
        telapsed = 0
        print('p0hat =',p0hat, '\n dtbar = ',dtbar, '\n dxbar = ',dxbar) 
        acetyl = np.zeros(width)
        pTot = np.zeros(np.size(tArray))
        aTot = np.zeros_like(pTot)
        ### Boundary Condition
        p[0] = p0hat
        p[width-1] = 0
        counter2 = 0
 
        while telapsed<=tbarmax: # Iterating the system of tmax amount of seconds
                if counter==0:
                        pReg[0] = 1.0
                        pReg1[1:width] = pReg[1:width] + dtbar/(dxbar**2)*(pReg[0:width-1]-2*pReg[1:width]+pReg[2:width+1])
                        aReg = aReg[0:width] + (1-aReg[0:width] )*dtbar*pReg[0:width]
                        pReg,pReg1 = pReg1,pReg
                        pReg[0] = 1.0 
                        pReg[width] = pReg[width-1]
                while (telapsed - tArray[counter2])>=0:
                        pTot[counter2] = sum((p[1:width])) * dxbar
                        aTot[counter2] = sum(acetyl[1:width]) * dxbar
                        if counter==0:
                                pTotReg[counter2] = sum((pReg[1:width]))*dxbar
                                aTotReg[counter2] = sum((aReg[1:width]))*dxbar
                        if counter2<np.size(tArray)-1:
                                counter2+=1

                pscaler = 1+p+p**2     
                p_1[1:width] = p[1:width] +   dtbar *  ( ( (p[0:width-1] - p[1:width]) / ((pscaler[0:width-1]+pscaler[1:width])/2) + (p[2:width+1] - p[1:width]) / ((pscaler[2:width+1]+pscaler[1:width])/2 )) / (dxbar**2))  
                acetyl[0:width] = acetyl[0:width] + (1 - acetyl[0:width] ) * dtbar * p[0:width]/p0hat ### NO SFD
                 
                telapsed += dtbar                     

                p,p_1 = p_1,p # Updating concentration array
                p[0] = p0hat # resetting the boundary condition
                p[width]=p[width-1] # Closed Tube Boundary
                if p0hat in [p2[0] , p2[np.size(p2)-1]]: 
                        if any(abs(telapsed-plottingTimes)<=epsilon):
                                plt.figure(8)
                                plt.plot(xbar[1:100],p[1:100],c=colorList[counter],label=('p0hat %.2e at t=%.2e'%(p0hat,telapsed)))
                                plt.figure(9)
                                plt.plot(xbar[1:100],acetyl[1:100],c=colorList[counter],label=('p0hat %.2e at t=%.2e'%(p0hat,telapsed)))

        pArray[counter,:] = p[0:width]
        aArray[counter,:] = acetyl[0:width]
        pTotArray[counter,:] = pTot
        aTotArray[counter,:] = aTot
        counter+=1
        print((time.time()-tstart)/60)

linearCutoff = np.argmax(tArray>1e2)
pTotPoly = np.zeros([2,len(pTotArray)])
aTotPoly = np.zeros_like(pTotPoly)
counter = 0
for i , value in enumerate(p2): 
        plt.figure(1)
        plt.plot(xbar,pArray[i,0:width]/value,label=('p0hat = %.2e'%value))
        plt.xlabel('Dimensionless length')
        plt.ylabel('phat/phat0')

        plt.figure(2)
        plt.plot(xbar,aArray[i,:],label=('p0hat = %.2e'%value))
        plt.xlabel('Dimensionless length')
        plt.ylabel('ahat')
        
        plt.figure(3)
        plt.loglog(tArray , pTotArray[i,:],label=('p0hat = %.2e'%value))
        plt.xlabel('Dimensionless Time')
        plt.ylabel('N(t)')
        #pTotPoly[:,i] = np.polyfit(np.log(tArray[linearCutoff:]),np.log(pTotArray[i,linearCutoff:]),1)
        #plt.plot(np.log(tArray[linearCutoff:]),np.log(tArray[linearCutoff:])*pTotPoly[0,i]+pTotPoly[1,i],label=('Slope = %.2f'%pTotPoly[0,i])) 
        
        plt.figure(4)
        plt.loglog(tArray , aTotArray[i,:],label=('p0hat = %.2e'%value))
        plt.xlabel('Dimensionless Time')
        plt.ylabel('A(t)')
        #aTotPoly[:,i] = np.polyfit(np.log(tArray[linearCutoff:]),np.log(aTotArray[i,linearCutoff:]),1)
        #plt.plot(np.log(tArray[linearCutoff:]),np.log(tArray[linearCutoff:])*aTotPoly[0,i]+aTotPoly[1,i],label=('Slope = %.2f'%aTotPoly[0,i])) 
        
        plt.figure(5) 
        plt.loglog(tArray ,pTotArray[i,:]/value,label=('p0hat = %.2e'%value))
        plt.xlabel('Dimensionless Time')
        plt.ylabel('N(t)/p0hat')

        plt.figure(6)
        plt.plot(xbar[1:width],pArray[i,1:width],label=('p0hat = %.2e'%value))
        plt.xlabel('Dimensionless Length')
        plt.ylabel('phat')

        counter+=1

plt.figure(1)
plt.plot(xbar,pReg[0:width],ls='dashed',label=('Simple Diffusion Fit'))
plt.legend()

plt.figure(2)
plt.plot(xbar,aReg,ls='dashed',label=('Simple Diffusion Fit'))
plt.legend()
plt.figure(3)
#plt.scatter(tArray , pTotReg,s=2,label=('No Single File Effects'))
#ax3.set_xscale('log')
#ax3.set_yscale('log')
plt.legend()
plt.figure(4)
#plt.scatter(tArray, aTotReg,s=2,label=('No Single File Effects'))
#ax4.set_xscale('log')
#ax4.set_yscale('log')
plt.legend()
plt.figure(5)
#ax5.set_xscale('log')
#ax5.set_yscale('log')
#plt.scatter(np.log(tArray),np.log(pTotReg),s=2,label=('No Single File Effects p0hat=1'))
plt.legend()
plt.show()
