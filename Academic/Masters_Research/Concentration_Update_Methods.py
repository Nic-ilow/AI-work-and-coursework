def testProgress(C,obstacles,rProbs,lProbs,upProbs,downProbs,stayProbs,rProbsShift,lProbsShift,upProbsShift,downProbsShift,lShift,rShift,obsLeft,obsRight,obsDown,obsUp):
	import numpy as np
	cUpdate = np.zeros_like(C)

	yDownShift = C[rShift] # Shift all the concentrations to y+1
	yUpShift = C[lShift] # Shift all the concentrations to y-1
	xrShift = C[:,rShift]  #  Shift all the concentrations to x+1
	xlShift = C[:,lShift]  #  Shift all the concentrations to x-1

	###  Update the concentration array by using the equation of the Sebastian Casault Thesis
	cUpdate += (xrShift*(1-obstacles)*rProbsShift+C*obsLeft*rProbs) \
			+(xlShift*(1-obstacles)*lProbsShift+C*obsRight*lProbs) \
			+(yUpShift*(1-obstacles)*upProbsShift+C*obsDown*upProbs) \
			+(yDownShift*(1-obstacles)*downProbsShift +C*obsUp*downProbs)\
			+(C*stayProbs)
	return cUpdate

def testProgressRectangle(C,obstacles,rProbs,lProbs,upProbs,downProbs,stayProbs,rProbsShift,lProbsShift,upProbsShift,downProbsShift,lShift,rShift,uShift,dShift,obsLeft,obsRight,obsDown,obsUp):
	import numpy as np
	cUpdate = np.zeros_like(C)

	yDownShift = C[dShift] # Shift all the concentrations to y+1
	yUpShift = C[uShift] # Shift all the concentrations to y-1
	xrShift = C[:,rShift]  #  Shift all the concentrations to x+1
	xlShift = C[:,lShift]  #  Shift all the concentrations to x-1

	###  Update the concentration array by using the equation of the Sebastian Casault Thesis
	cUpdate += (xrShift*(1-obstacles)*rProbsShift+C*obsLeft*rProbs) \
			+(xlShift*(1-obstacles)*lProbsShift+C*obsRight*lProbs) \
			+(yUpShift*(1-obstacles)*upProbsShift+C*obsDown*upProbs) \
			+(yDownShift*(1-obstacles)*downProbsShift +C*obsUp*downProbs)\
			+(C*stayProbs)
	return cUpdate

def testProgress2(C,obstacles,rProbs,lProbs,upProbs,downProbs,stayProbs,rProbsShift,lProbsShift,upProbsShift,downProbsShift,lShift,rShift,obsLeft,obsRight,obsDown,obsUp):
	import numpy as np
	cUpdate = np.zeros_like(C)

	yDownShift = C[rShift] # Shift all the concentrations to y+1
	yUpShift = C[lShift] # Shift all the concentrations to y-1
	xrShift = C[:,rShift]  #  Shift all the concentrations to x+1
	xlShift = C[:,lShift]  #  Shift all the concentrations to x-1

	###  Update the concentration array by using the equation of the Sebastian Casault Thesis
	cUpdate += (np.multiply(np.multiply(xrShift,(1-obstacles)),rProbsShift)+np.multiply(np.multiply(C,obsLeft),rProbs)) \
			+(np.multiply(np.multiply(xlShift,(1-obstacles)),lProbsShift)+np.multiply(np.multiply(C,obsRight),lProbs)) \
			+(np.multiply(np.multiply(yUpShift,(1-obstacles)),upProbsShift)+np.multiply(np.multiply(C,obsDown),upProbs)) \
			+(np.multiply(np.multiply(yDownShift,(1-obstacles)),downProbsShift)+np.multiply(np.multiply(C,obsUp),downProbs))\
			+(np.multiply(C,stayProbs))
	return cUpdate

def latticeSwell(a,a0,aMax,tElapsed,tExposure,tRelax):
	import numpy as np
	exposed = (tElapsed-tExposure)>=0
	exposed = exposed.astype(int)
	
	a = a0 + (aMax-a0) * exposed * ( 1-np.exp(-(tElapsed-tExposure)/tRelax) )

	return a

def halfSlideProgress(C,obstacles,obsL,obsR,obsU,obsD,obsUR,obsUL,obsDR,obsDL,rProbs,lProbs,dProbs,uProbs,sProbs,urProbs,ulProbs,drProbs,dlProbs,rProbsShift,lProbsShift,dProbsShift,uProbsShift,rShift,lShift):
	import numpy as np
	cUpdate = np.zeros_like(C)

	yDownShift = C[rShift]
	yUpShift = C[lShift]
	xrShift = C[:,rShift]
	xlShift = C[:,lShift]
	
	uLshift = (C[lShift,:])[:,lShift]
	uRshift = (C[lShift,:])[:,rShift]
	dRshift = (C[rShift,:])[:,rShift]
	dLshift = (C[rShift,:])[:,lShift]
	
	###  SAFE
	cUpdate += (xrShift*(1-obstacles)*rProbsShift+C*obsL*rProbs) \
			+(xlShift*(1-obstacles)*lProbsShift+C*obsR*lProbs) \
			+(yUpShift*(1-obstacles)*uProbsShift+C*obsD*uProbs) \
			+(yDownShift*(1-obstacles)*dProbsShift +C*obsU*dProbs)\
			+(C*sProbs)
	
	###  SAFE
	cUpdate += C*obsL*obsD*urProbs + C*obsR*obsD*ulProbs + C*obsL*obsU*drProbs + C*obsR*obsU*dlProbs  ###  All diagonal cases where concentration stays
	
	###  SAFE
	
	cUpdate += uRshift*(1-obstacles)*(1-obsR/2-obsU/2)*urProbs \
			+ uLshift*(1-obstacles)*(1-obsL/2-obsU/2)*ulProbs \
			+ dRshift*(1-obstacles)*(1-obsD/2-obsR/2)*drProbs \
			+ dLshift*(1-obstacles)*(1-obsD/2-obsL/2)*dlProbs 				#  All of concentration that will flow diagonally
	
	###  Wall (2 obstacles in a row) mechanism
	
	cUpdate += yUpShift*(1-obstacles) * ( obsUL * obsL * urProbs + obsUR * obsR * ulProbs ) \
			+ yDownShift*(1-obstacles) * ( obsDL * obsL * drProbs + obsDR * obsR * dlProbs ) \
			+ xrShift*(1-obstacles) * ( obsDR * obsD * urProbs + obsUR * obsU * drProbs ) \
			+ xlShift*(1-obstacles) * ( obsUL * obsU * dlProbs + obsDL * obsD * ulProbs) 

	cUpdate += yUpShift*(1-obstacles) * ( urProbs*obsUL/2*abs(obsUL-obsL) + ulProbs*obsUR/2*abs(obsUR-obsR) ) \
			+ yDownShift*(1-obstacles) * ( dlProbs*obsDR/2*abs(obsDR-obsR) + drProbs*obsDL/2*abs(obsDL-obsL) ) \
			+ xrShift*(1-obstacles) * ( urProbs*obsDR/2*abs(obsDR-obsD) + drProbs*obsUR/2*abs(obsUR-obsU) ) \
			+ xlShift*(1-obstacles) * ( ulProbs*obsDL/2*abs(obsDL-obsD) + dlProbs*obsUL/2*abs(obsUL-obsU) )

	cUpdate += ( xlShift/2*(obsD*(1-obstacles)*(1-obsDL))+yUpShift/2*(obsR*(1-obstacles)*(1-obsUR)) ) * ulProbs \
			+  ( xrShift/2*(obsD*(1-obstacles)*(1-obsDR)) + yUpShift/2*(obsL*(1-obstacles)*(1-obsUL)) ) * urProbs \
			+  ( xrShift/2*(obsU*(1-obstacles)*(1-obsUR)) + yDownShift/2*(obsL*(1-obstacles)*(1-obsDL)) ) * drProbs \
			+  ( xlShift/2*(obsU*(1-obstacles)*(1-obsUL)) + yDownShift/2*(obsR*(1-obstacles)*(1-obsDR)) ) * dlProbs

	
	return cUpdate


def noSlide(C,obstacles,obsL,obsR,obsU,obsD,obsUR,obsUL,obsDR,obsDL,rProbs,lProbs,dProbs,uProbs,sProbs,urProbs,ulProbs,drProbs,dlProbs,rProbsShift,lProbsShift,dProbsShift,uProbsShift,rShift,lShift):
	import numpy as np
	cUpdate = np.zeros_like(C)

	yDownShift = C[rShift]
	yUpShift = C[lShift]
	xrShift = C[:, rShift]
	xlShift = C[:, lShift]

	uLshift = (C[lShift, :])[:, lShift]
	uRshift = (C[lShift, :])[:, rShift]
	dRshift = (C[rShift, :])[:, rShift]
	dLshift = (C[rShift, :])[:, lShift]

	###  Update moving left right up and down, and seeing if you run into obstacles
	cUpdate += (xrShift*(1-obstacles)*rProbsShift+C*obsL*rProbs) \
			+(xlShift*(1-obstacles)*lProbsShift+C*obsR*lProbs) \
			+(yUpShift*(1-obstacles)*uProbsShift+C*obsD*uProbs) \
			+(yDownShift*(1-obstacles)*dProbsShift +C*obsU*dProbs)\
			+(C*sProbs)

	###  SAFE

	cUpdate += uRshift * (1 - obstacles) * (1 - obsR / 2 - obsU / 2) * urProbs \
			   + uLshift * (1 - obstacles) * (1 - obsL / 2 - obsU / 2) * ulProbs \
			   + dRshift * (1 - obstacles) * (1 - obsD / 2 - obsR / 2) * drProbs \
			   + dLshift * (1 - obstacles) * (1 - obsD / 2 - obsL / 2) * dlProbs  # All of concentration that will flow diagonally

	###  Reject diagonal movement based on if theres an obstacles on one of the primary axes

	cUpdate += C*(obsL/2+obsD/2)*urProbs \
			+ C*(obsL/2+obsU/2)*drProbs \
			+ C*(obsR/2+obsD/2)*ulProbs \
			+ C*(obsR/2+obsU/2)*dlProbs

	cUpdate += (xlShift / 2 * (obsD * (1 - obstacles) * (1 - obsDL)) + yUpShift / 2 * (
				obsR * (1 - obstacles) * (1 - obsUR))) * ulProbs \
			   + (xrShift / 2 * (obsD * (1 - obstacles) * (1 - obsDR)) + yUpShift / 2 * (
				obsL * (1 - obstacles) * (1 - obsUL))) * urProbs \
			   + (xrShift / 2 * (obsU * (1 - obstacles) * (1 - obsUR)) + yDownShift / 2 * (
				obsL * (1 - obstacles) * (1 - obsDL))) * drProbs \
			   + (xlShift / 2 * (obsU * (1 - obstacles) * (1 - obsUL)) + yDownShift / 2 * (
				obsR * (1 - obstacles) * (1 - obsDR))) * dlProbs

	return cUpdate

def alwaysSlide(C,obstacles,obsL,obsR,obsU,obsD,obsUR,obsUL,obsDR,obsDL,rProbs,lProbs,dProbs,uProbs,sProbs,urProbs,ulProbs,drProbs,dlProbs,rProbsShift,lProbsShift,dProbsShift,uProbsShift,rShift,lShift):
	import numpy as np
	cUpdate = np.zeros_like(C)

	yDownShift = C[rShift]
	yUpShift = C[lShift]
	xrShift = C[:, rShift]
	xlShift = C[:, lShift]

	uLshift = (C[lShift, :])[:, lShift]
	uRshift = (C[lShift, :])[:, rShift]
	dRshift = (C[rShift, :])[:, rShift]
	dLshift = (C[rShift, :])[:, lShift]

	###  Update moving left right up and down, and seeing if you run into obstacles
	cUpdate += (xrShift*(1-obstacles)*rProbsShift+C*obsL*rProbs) \
			+(xlShift*(1-obstacles)*lProbsShift+C*obsR*lProbs) \
			+(yUpShift*(1-obstacles)*uProbsShift+C*obsD*uProbs) \
			+(yDownShift*(1-obstacles)*dProbsShift +C*obsU*dProbs)\
			+(C*sProbs)

	###  All diagonal cases where concentration stays
	cUpdate += C * obsL * obsD * urProbs + C * obsR * obsD * ulProbs + C * obsL * obsU * drProbs + C * obsR * obsU * dlProbs

	###  All of the Sliding
	cUpdate += yUpShift*(1-obstacles)*(obsUL*urProbs + obsUR*ulProbs)  \
			+ yDownShift*(1-obstacles)*(obsDL*drProbs + obsDR*dlProbs) \
			+ xrShift*(1-obstacles)*(obsUR*drProbs + obsDR*urProbs) \
			+ xlShift*(1-obstacles)*(obsUL*dlProbs + obsDL*ulProbs)

	cUpdate += uRshift*(1-obstacles)*(1-obsR)*(1-obsU)*urProbs \
			+ uLshift*(1-obstacles)*(1-obsL)*(1-obsU)*ulProbs \
			+ dRshift*(1-obstacles)*(1-obsD)*(1-obsR)*drProbs \
			+ dLshift*(1-obstacles)*(1-obsD)*(1-obsL)*dlProbs 				#  All of concentration that will flow diagonally

	cUpdate += (xlShift / 2 * (obsD * (1 - obstacles) * (1 - obsDL)) + yUpShift / 2 * (
				obsR * (1 - obstacles) * (1 - obsUR))) * ulProbs \
			   + (xrShift / 2 * (obsD * (1 - obstacles) * (1 - obsDR)) + yUpShift / 2 * (
				obsL * (1 - obstacles) * (1 - obsUL))) * urProbs \
			   + (xrShift / 2 * (obsU * (1 - obstacles) * (1 - obsUR)) + yDownShift / 2 * (
				obsL * (1 - obstacles) * (1 - obsDL))) * drProbs \
			   + (xlShift / 2 * (obsU * (1 - obstacles) * (1 - obsUL)) + yDownShift / 2 * (
				obsR * (1 - obstacles) * (1 - obsDR))) * dlProbs

	return cUpdate

def pureDiag(C,obstacles,obsL,obsR,obsU,obsD,obsUR,obsUL,obsDR,obsDL,rProbs,lProbs,dProbs,uProbs,sProbs,urProbs,ulProbs,drProbs,dlProbs,rProbsShift,lProbsShift,dProbsShift,uProbsShift,rShift,lShift):
	import numpy as np
	cUpdate = np.zeros_like(C)

	yDownShift = C[rShift]
	yUpShift = C[lShift]
	xrShift = C[:, rShift]
	xlShift = C[:, lShift]

	uLshift = (C[lShift, :])[:, lShift]
	uRshift = (C[lShift, :])[:, rShift]
	dRshift = (C[rShift, :])[:, rShift]
	dLshift = (C[rShift, :])[:, lShift]

	###  Update moving left right up and down, and seeing if you run into obstacles
	cUpdate += (xrShift*(1-obstacles)*rProbsShift+C*obsL*rProbs) \
			+(xlShift*(1-obstacles)*lProbsShift+C*obsR*lProbs) \
			+(yUpShift*(1-obstacles)*uProbsShift+C*obsD*uProbs) \
			+(yDownShift*(1-obstacles)*dProbsShift +C*obsU*dProbs)\
			+(C*sProbs)

	cUpdate += (uRshift * (1-obstacles) + C * obsDL) * urProbs \
			+ (uLshift * (1-obstacles) + C * obsDR) * ulProbs \
			+ (dRshift * (1-obstacles) + C * obsUL) * drProbs \
			+ (dLshift * (1-obstacles) + C * obsUR) * dlProbs 				#  All of concentration that will flow diagonally

	return cUpdate

'''
def cudaTest(C, obstacles, obsL, obsR, obsU, obsD, obsUR, obsUL, obsDR, obsDL, rProbs, lProbs, dProbs, uProbs,
				 sProbs, urProbs, ulProbs, drProbs, dlProbs, rProbsShift, lProbsShift, dProbsShift, uProbsShift, rShift,
				 lShift,distances,k,tau):
	import numpy as np
	import pycuda.gpuarray as gpuarray

	#cUpdate = np.zeros_like(C)
	#cUpdate = gpuarray.to_gpu(np.zeros_like(C).astype(np.float32))

	yDownShift = gpuarray.to_gpu(C[rShift].astype(np.float32))
	yUpShift = gpuarray.to_gpu(C[lShift].astype(np.float32))
	xrShift = gpuarray.to_gpu(C[:, rShift].astype(np.float32))
	xlShift = gpuarray.to_gpu(C[:, lShift].astype(np.float32))

	uLshift = gpuarray.to_gpu((C[lShift, :])[:, lShift].astype(np.float32))
	uRshift = gpuarray.to_gpu((C[lShift, :])[:, rShift].astype(np.float32))
	dRshift = gpuarray.to_gpu((C[rShift, :])[:, rShift].astype(np.float32))
	dLshift = gpuarray.to_gpu((C[rShift, :])[:, lShift].astype(np.float32))

	C = gpuarray.to_gpu(C.astype(np.float32))


	###  SAFE
	cUpdate = ((xrShift * (1 - obstacles) * rProbsShift + C * obsL * rProbs).get()
			   + (xlShift * (1 - obstacles) * lProbsShift + C * obsR * lProbs).get() \
			   + (yUpShift * (1 - obstacles) * uProbsShift + C * obsD * uProbs).get() \
			   + (yDownShift * (1 - obstacles) * dProbsShift + C * obsU * dProbs).get() \
			   + (C * sProbs).get())

	###  SAFE
	cUpdate += (C * obsL * obsD * urProbs).get() + (C * obsR * obsD * ulProbs).get() + (C * obsL * obsU * drProbs).get() + (C * obsR * obsU * dlProbs).get()  ###  All diagonal cases where concentration stays

	###  SAFE

	cUpdate += (uRshift * (1 - obstacles) * (1 - obsR / 2 - obsU / 2) * urProbs).get() \
			   + (uLshift * (1 - obstacles) * (1 - obsL / 2 - obsU / 2) * ulProbs).get() \
			   + (dRshift * (1 - obstacles) * (1 - obsD / 2 - obsR / 2) * drProbs).get() \
			   + (dLshift * (1 - obstacles) * (1 - obsD / 2 - obsL / 2) * dlProbs).get() # All of concentration that will flow diagonally

	###  Wall (2 obstacles in a row) mechanism

	cUpdate += (yUpShift * (1 - obstacles) * (obsUL * obsL * urProbs + obsUR * obsR * ulProbs)).get() \
			   + (yDownShift * (1 - obstacles) * (obsDL * obsL * drProbs + obsDR * obsR * dlProbs)).get() \
			   + (xrShift * (1 - obstacles) * (obsDR * obsD * urProbs + obsUR * obsU * drProbs)).get() \
			   + (xlShift * (1 - obstacles) * (obsUL * obsU * dlProbs + obsDL * obsD * ulProbs)).get()

	cUpdate += (yUpShift * (1 - obstacles) * (
				urProbs * obsUL / 2 * abs(obsUL - obsL) + ulProbs * obsUR / 2 * abs(obsUR - obsR))).get() \
			   + (yDownShift * (1 - obstacles) * (
						   dlProbs * obsDR / 2 * abs(obsDR - obsR) + drProbs * obsDL / 2 * abs(obsDL - obsL))).get() \
			   + (xrShift * (1 - obstacles) * (
						   urProbs * obsDR / 2 * abs(obsDR - obsD) + drProbs * obsUR / 2 * abs(obsUR - obsU))).get() \
			   + (xlShift * (1 - obstacles) * (
						   ulProbs * obsDL / 2 * abs(obsDL - obsD) + dlProbs * obsUL / 2 * abs(obsUL - obsU))).get()

	cUpdate += (((xlShift / 2 * (obsD * (1 - obstacles) * (1 - obsDL)) + yUpShift / 2 * (
				obsR * (1 - obstacles) * (1 - obsUR))) * ulProbs).get() \
			   + ((xrShift / 2 * (obsD * (1 - obstacles) * (1 - obsDR)) + yUpShift / 2 * (
				obsL * (1 - obstacles) * (1 - obsUL))) * urProbs).get() \
			   + (xrShift / 2 * (obsU * (1 - obstacles) * (1 - obsUR)) + yDownShift / 2 * (
				obsL * (1 - obstacles) * (1 - obsDL))) * drProbs).get() \
			   + (xlShift / 2 * (obsU * (1 - obstacles) * (1 - obsUL)) + yDownShift / 2 * (
				obsR * (1 - obstacles) * (1 - obsDR))) * dlProbs).get()

	r2 = np.sum(((distances*cUpdate)/((k+1)*tau)).get())

	return cUpdate.get(),r2
'''


def cudaTest16(C, obstacles, obsL, obsR, obsU, obsD, rProbs, lProbs, dProbs, uProbs, sProbs,
					rProbsShift, lProbsShift, dProbsShift, uProbsShift, rShift,
				 	lShift,distances,k,tau):
	import numpy as np
	import pycuda.gpuarray as gpuarray


	yDownShift = gpuarray.to_gpu(np.copy(C[rShift]).astype(np.float32))
	yUpShift = gpuarray.to_gpu(np.copy(C[lShift]).astype(np.float32))
	xrShift = gpuarray.to_gpu(np.copy(C[:, rShift]).astype(np.float32))
	xlShift = gpuarray.to_gpu(np.copy(C[:, lShift]).astype(np.float32))

	C = gpuarray.to_gpu(C.astype(np.float32))
	###  Update the concentration array by using the equation of the Sebastian Casault Thesis
	cUpdate = ((xrShift*(1-obstacles)/6) + (C*obsL/6)
			+(xlShift*(1-obstacles)/6) + (C*obsR/6)
			+(yUpShift*(1-obstacles)/6) + (C*obsD/6)
			+(yDownShift*(1-obstacles)/6) + (C*obsU/6)
			+(C/3))

	r2 = np.sum(((distances*cUpdate)/((k+1)*tau)).get())

	return cUpdate.get(),r2
