import math
import bisect
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy import linalg
from scipy.stats import f

def kernelWeights(u):
	return np.exp(-3*(u**2));

# def main():
if __name__ == '__main__':

	# Load files from MATLAB (Temporary)
	mat  = scipy.io.loadmat('testLinux.mat')
	data = mat["y"]
	mask = mat["BW"]

	n = data.shape[0]

	# User defined parameters
	width0   = 2
	widthMax = 6
	alpha    = 1-1e-3;

	# Precompute F statistic at chosen alpha level
	stepSize = 1e-2;
	maxDegreeOfFreedom = (2*widthMax+1)**2
	fAlphaPreComp = np.zeros([1, maxDegreeOfFreedom+1]);

	for dof in range(10,maxDegreeOfFreedom+1):
		# Create F CDF with a given degree of freedom
		fCDF = f.cdf(np.linspace(0,100,100/stepSize), 8, dof-9)

		# Find first point where CDF > alpha
		fAlphaPreComp[0,dof] = stepSize*bisect.bisect(fCDF,alpha)

	# dataList  = []
	# fStatList = []

	# Calculate F statistic for all widths
	F      = np.zeros([n,n,widthMax-width0+1]);
	Falpha = np.zeros_like(F)

	for width in range(width0,widthMax+1):

		# Create directions (x,y) in 2D
		windowSize = (2*width+1)
		[x,y]      = np.mgrid[-1:1:complex(0,windowSize),-1:1:complex(0,windowSize)]
		[x,y]      = x.flatten(), y.flatten()
		directions = np.concatenate( (x[...,np.newaxis], y[...,np.newaxis]), axis=1 )

		# Compute kernel and degrees of freedom
		kernel     = kernelWeights(np.sqrt(x**2 + y**2)).flatten()
		dof        = windowSize**2

		## Pre-computed SVD for this width
		# Calculate number of combinations of vectors
		numComb = 1 + 2*directions.shape[1]
		numVals = kernel.size

		# Form array to process with SVD
		A = np.zeros([numVals, numComb])
		A[:,0] = np.ones_like(kernel)
		for i in range(directions.shape[1]):
			A[:,2*i+1] = np.sin(np.pi*directions[:,i])
			A[:,2*i+2] = np.cos(np.pi*directions[:,i])

		# Compute SVD
		[U, s, V] = linalg.svd(np.dot(np.diag(kernel),A))
		Sinv      = linalg.diagsvd(1/s,numVals,numComb)

		# Calcuate corresponding Falpha value
		Falpha[...,width-width0] = fAlphaPreComp[0,dof]*np.ones([n,n])

		for i in range(n):
			for j in range(n):
				if mask[i,j] == 1:
					# Extract data in local window
					dataWindow = data[i-width:i+width+1, j-width:j+width+1].flatten()

					# Local weighted constant fit
					constCoef = np.dot(kernel,dataWindow) / kernel.sum()
					constFit  = constCoef*np.ones_like(dataWindow)

					## Local weighted sine/cosine fit
					scalingFactor = 1.0/np.max(abs(dataWindow))
					dataScaled    = dataWindow * scalingFactor					
					sincosCoef    = np.dot(V.transpose(), np.dot(Sinv.transpose(), 
									np.dot(U.transpose(),kernel*dataScaled)))
					sincosCoef    = sincosCoef / scalingFactor
					sincosFit     = np.dot(A,sincosCoef)

					# Calculate weighted residual sum of squares
					RSSconst  = (kernel*(dataWindow -  constFit)**2).sum()
					RSSsincos = (kernel*(dataWindow - sincosFit)**2).sum()

					# Calculate F statistic
					F[i,j,width-width0] = ( (RSSconst-RSSsincos)/(sincosCoef.size-constCoef.size)
									 / (RSSsincos / (dof-sincosCoef.size)) )

	# Calculate Resolution
	R   = F/Falpha
	val = np.amax(R,axis=2)
	res = np.argmax(R,axis=2)
	res[val<=1] = widthMax;


	# Plot
	f1 = plt.figure()
	f2 = plt.figure()
	ax1 = f1.add_subplot(111)
	ax1.imshow(data, cmap=plt.cm.gray, interpolation="nearest")
	ax2 = f2.add_subplot(111)
	ax2.imshow(res, cmap=plt.cm.jet, interpolation="nearest")
	plt.show()











# if __name__ == '__main__':
	# main()

