import math
import bisect
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy import linalg
from scipy.stats import f
from time import time

def kernelWeights(u):
	return np.exp(-2*u);

def makeDirections(x, y, angleStep):
	numberOfElements = int(math.floor(np.pi/angleStep))
	directions       = np.zeros([x.size,numberOfElements])

	# Rotate x,y and store into directions
	for i in range(numberOfElements):
		directions[:,i] = x*math.cos(i*angleStep) + y*math.sin(i*angleStep)
	return directions

# def main():
if __name__ == '__main__':

	print '== BEGIN MAIN =='
	tBegin = time()

	# Load files from MATLAB (Temporary)
	# mat   = scipy.io.loadmat('hongwei_withBW.mat')
	# mat  = scipy.io.loadmat('emdb1019_down.mat')
	# mat  = scipy.io.loadmat('testLinux1019.mat')
	mat  = scipy.io.loadmat('scheresRIBOSOME_mildLPF.mat')
	# mat  = scipy.io.loadmat('ctf_synth.mat')

	data = mat["y"]
	mask = mat["BW"]
	n 	 = data.shape[0]

	# User defined parameters
	width0    = 2
	widthMax  = 8
	alpha     = 1-1e-3
	angleStep = np.pi/4

	# Precompute F statistic at chosen alpha level
	stepSize           = 1e-2;
	maxDegreeOfFreedom = (2*widthMax+1)**2
	fAlphaPreComp      = np.zeros([1, maxDegreeOfFreedom+1]);
	numComb            = 1 + 2*(np.pi/angleStep)

	tStart = time()
	for dof in range(10,maxDegreeOfFreedom+1):
		# Create F CDF with a given degree of freedom
		fCDF = f.cdf(np.linspace(0,100,100/stepSize), numComb-1, dof-numComb)

		# Find first point where CDF > alpha
		fAlphaPreComp[0,dof] = stepSize*bisect.bisect(fCDF,alpha)
	s = time() - tStart		
	print "fAlphaPreComp :: Time elapsed: %.2f seconds" % s

	# Calculate F statistic for all widths
	F      = np.zeros([n,n,widthMax-width0+1]);
	Falpha = np.zeros_like(F)

	for width in range(width0,widthMax+1):
		tStart = time()

		# Create directions (x,y) in 2D
		windowSize = (2*width+1)
		[x,y]      = np.mgrid[-1:1:complex(0,windowSize),-1:1:complex(0,windowSize)]
		[x,y]      = x.flatten(), y.flatten()
		directions = makeDirections(x,y,angleStep)

		# Compute kernel and degrees of freedom
		kernel     = kernelWeights(x**2 + y**2).flatten()
		kernelSum  = kernel.sum()
		kernelSq   = np.sqrt(kernel)
		dof        = windowSize**2

		# Calculate number of combinations of vectors
		numComb = 1 + 2*directions.shape[1]
		numVals = kernel.size

		# Form array to invert via SVD
		A = np.zeros([numVals, numComb])
		A[:,0] = np.ones_like(kernel)
		for i in range(directions.shape[1]):
			A[:,2*i+1] = np.sin(np.pi*directions[:,i])
			A[:,2*i+2] = np.cos(np.pi*directions[:,i])

		# Compute SVD
		[U, s, V] = linalg.svd(np.dot(np.diag(kernelSq),A))
		Sinv      = linalg.diagsvd(1/s,numVals,numComb)
		H = np.dot(
			np.dot(V.transpose(), np.dot(Sinv.transpose(), U.transpose())),
			np.diag(kernelSq))

		# Calcuate corresponding Falpha value
		Falpha[...,width-width0] = fAlphaPreComp[0,dof]*np.ones([n,n])

		# Extract data in from single width
		tmpIdx  = 0
		dataLevel = np.zeros([mask.sum(), dof])
		for i in range(n):
			for j in range(n):
				if mask[i,j] == 1:
					dataLevel[tmpIdx,:] = data[i-width:i+width+1, j-width:j+width+1].flatten()
					tmpIdx += 1

		FLevel = np.zeros([dataLevel.shape[0],1])

		for i in range(dataLevel.shape[0]):

			# Extract data for single point
			dataWindow = dataLevel[i,:]

			# Local weighted constant fit
			constCoef = np.dot(kernel,dataWindow) / kernelSum
			constFit  = constCoef*np.ones_like(dataWindow)

			## Local weighted sine/cosine fit
			scalingFactor = 1.0/np.max(abs(dataWindow))
			dataScaled    = dataWindow * scalingFactor					
			sincosCoef    = np.dot(H,dataScaled)
			sincosCoef    = sincosCoef / scalingFactor
			sincosFit     = np.dot(A,sincosCoef)

			# Calculate weighted residual sum of squares
			RSSconst  = (kernel*(dataWindow -  constFit)**2).sum()
			RSSsincos = (kernel*(dataWindow - sincosFit)**2).sum()

			# Calculate F statistic
			FLevel[i] = ( (RSSconst-RSSsincos)/(sincosCoef.size-constCoef.size)
							 / (RSSsincos / (dof-sincosCoef.size)) )

		# Undo reshaping and calculate Fstatistic "image"
		tmpIdx = 0;
		tmpF   = np.zeros([n,n])
		for i in range(n):
			for j in range(n):
				if mask[i,j] == 1:
					tmpF[i,j] = FLevel[tmpIdx]
					tmpIdx += 1
		F[..., width-width0] = tmpF	

		m, s = divmod(time() - tStart, 60)
		print "Width %d :: Time elapsed: %d minutes and %.2f seconds" % (width, m, s)
	
	# Calculate Resolution
	R   = F/Falpha
	val = np.amax(R,axis=2)
	res = np.argmax(R,axis=2)
	res[val<=1] = widthMax;
	
	m, s = divmod(time() - tBegin, 60)
	print "TOTAL :: Time elapsed: %d minutes and %.2f seconds" % (m, s)

	# Plot
	f1 = plt.figure()
	f2 = plt.figure()
	ax1 = f1.add_subplot(111)
	ax1.imshow(data, cmap=plt.cm.gray, interpolation="nearest")
	ax2 = f2.add_subplot(111)
	ax2.imshow(res, cmap=plt.cm.jet, interpolation="nearest")
	plt.show()

	scipy.io.savemat('output.mat', {'res':res})
	# scipy.io.savemat('output.mat', {'pyF':F,'pyFAlphaPreComp':fAlphaPreComp,'pyR':R,'pyRes':res})




# if __name__ == '__main__':
	# main()

