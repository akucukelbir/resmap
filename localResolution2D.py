import math
import bisect
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy import linalg
from scipy.stats import f

def kernelWeights(u):
	return np.exp(-3*(u**2));

def fit2DSinCosSVD(dataWindow,kernel,directions):

		# Scale data to avoid precision problems
		scalingFactor = 1.0/np.max(abs(dataWindow))
		dataScaled    = dataWindow * scalingFactor

		# Calculate number of combinations of vectors
		numComb = 1 + 2*directions.shape[1]
		numVals = kernel.size

		# Form array to process with SVD
		A = np.zeros([numVals, numComb])
		A[:,0] = np.ones_like(dataScaled)
		for i in range(directions.shape[1]):
			A[:,2*i+1] = np.sin(np.pi*directions[:,i])
			A[:,2*i+2] = np.cos(np.pi*directions[:,i])

		[U, s, V] = linalg.svd(np.dot(np.diag(kernel),A))
		Sinv      = linalg.diagsvd(1/s,numVals,numComb)

		sincosCoef = np.dot(V.transpose(), np.dot(Sinv.transpose(), np.dot(U.transpose(),kernel*dataScaled)))
		sincosCoef = sincosCoef / scalingFactor

		sincosFit = np.dot(A,sincosCoef)

		return (sincosCoef, sincosFit)

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
	fAlphaPreComp = np.zeros([1, maxDegreeOfFreedom]);

	for dof in range(10,maxDegreeOfFreedom):
		# Create F CDF with a given degree of freedom
		fCDF = f.cdf(np.linspace(0,100,100/stepSize), 8, dof-9)

		# Find first point where CDF > alpha
		fAlphaPreComp[0,dof] = stepSize*bisect.bisect(fCDF,alpha)

	# dataList  = []
	# fStatList = []

	F = np.zeros([n,n,widthMax+1]);

	for width in range(width0,widthMax+1):

		# Create directions, calculate kernel and degrees of freedom
		windowSize = (2*width+1)
		[x,y]      = np.mgrid[-1:1:complex(0,windowSize),-1:1:complex(0,windowSize)]
		[x,y]      = x.flatten(), y.flatten()
		directions = np.concatenate( (x[...,np.newaxis], y[...,np.newaxis]), axis=1 )
		kernel     = kernelWeights(np.sqrt(x**2 + y**2))
		dof        = windowSize**2

		for i in range(n):
			for j in range(n):
				if mask[i,j] == 1:
					# Extract data in local window
					dataWindow = data[i-width:i+width+1, j-width:j+width+1].flatten()

					# Local weighted constant fit
					constCoef = np.dot(kernel,dataWindow) / kernel.sum()
					constFit  = constCoef*np.ones_like(dataWindow)

					# Local weighted sine/cosine fit
					[sincosCoef, sincosFit] = fit2DSinCosSVD(dataWindow,kernel,directions)

					# Calculate weighted residual sum of squares
					RSSconst  = (kernel*(dataWindow -  constFit)**2).sum()
					RSSsincos = (kernel*(dataWindow - sincosFit)**2).sum()

					# Calculate F statistic
					F[i,j,width] = ( (RSSconst-RSSsincos)/(sincosCoef.size-constCoef.size)
									 / (RSSsincos / (dof-sincosCoef.size)) )













# if __name__ == '__main__':
	# main()

