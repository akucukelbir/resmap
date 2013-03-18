import math
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from time import time
from scipy.optimize import minimize_scalar
from scipy.ndimage import filters, morphology

# Modules found in python files in root folder
from localreshelpers import *
from blocks import *
from mrc import *

# def main():
if __name__ == '__main__':

	print '== BEGIN MAIN =='
	tBegin = time()

	# User defined parameters
	k      = 1.77	# voxel size (in Angstroms)
	M      = 4.5	# query resolution (in Angstroms)
	pValue = 1e-3	# generally between (0, 0.05]

	minRes = 2.5*k
	if M < minRes:
		print "Please choose a query resolution M > 2.5*k = %.2f" % minRes
		raise Exception('whoa')

	# Compute window size and form steerable bases
	r       = np.ceil(0.5*M/k)  	# number of pixels around center
	s       = (2.0*r*k)/M      		# scaling factor to account for overshoot due to k
	l       = np.pi*np.sqrt(2.0/5)	# lambda
	winSize = 2*r+1

	# Load files from MATLAB (Temporary)
	# mat  = scipy.io.loadmat('volScheres2275.mat')
	# data = mat["y"]
	data = mrc_image('EMD-2275.mrc')
	data.read()
	data = data.image_data

	n 	 = data.shape[0]
	N 	 = int(n)

	# Create spherical mask
	[x,y,z] = np.mgrid[ -n/2:n/2:complex(0,n),
						-n/2:n/2:complex(0,n),
						-n/2:n/2:complex(0,n) ]
	R       = np.sqrt(x**2 + y**2 + z**2)
	Rinside = R < n/2					 	

	# Compute mask separating the particle from background
	dataBlurred = filters.gaussian_filter(data, float(n/1e2))
	dataMask    = dataBlurred > np.max(dataBlurred)*1e-1
	mask        = np.bitwise_and(dataMask, R < n/2 - 2*winSize)
	del dataMask
	del R

	# Define range of x, y, z for steerable bases
	[x,y,z] = np.mgrid[	-s*l:s*l:complex(0,winSize),
						-s*l:s*l:complex(0,winSize),
						-s*l:s*l:complex(0,winSize) ]
	dirs    = make3DsteerableDirections(x, y, z)

	# Define Gaussian kernel
	kernel     = np.exp(-1*(x**2 + y**2 + z**2)).flatten()
	kernelSqrt = np.sqrt(kernel)  
	kernelSum  = kernel.sum()
	W          = np.diag(kernel)

	# Calculate shape of matrix of bases
	numberOfPoints = kernel.size;
	numberOfBases  = dirs.shape[3] + 1;

	# Form matrix of Hermite polynomials 
	A = np.zeros([numberOfPoints, numberOfBases])
	A[:,0] = np.ones_like(kernel)
	for i in range(0,6):
		tmp = dirs[:,:,:,i]
		tmp = tmp.flatten()
		A[:,i+1] = 4*(tmp**2) - 2;
	for i in range(6,16):
		tmp = dirs[:,:,:,i]
		tmp = tmp.flatten()
		A[:,i+1] = tmp**3 - 2.254*tmp    

	# Form matrix of just the constant term
	Ac = np.zeros([numberOfPoints, 1])
	Ac[:,0] = np.ones_like(kernel)

	# Invert weighted A matrix via SVD	
	H = np.dot(A, np.dot(np.linalg.pinv(np.dot(np.diag(kernelSqrt),A)), np.diag(kernelSqrt)))

	# Invert weighted Ac matrix via SVD
	Ack = np.dot(np.diag(kernelSqrt),Ac)	
	Hc  = np.dot(Ac, np.dot(Ack.T/(np.linalg.norm(Ack)**2), np.diag(kernelSqrt)))

	# Create LAMBDA matrices that correspond to WRSS = Y^T*LAMBDA*Y
	LAMBDA     = W-np.dot(W,H);
	LAMBDAc    = W-np.dot(W,Hc);
	LAMBDAdiff = np.array(LAMBDAc-LAMBDA, dtype='float32');

	## Estimate variance

	print 'Estimating variance from non-overlapping blocks in background...',
	tStart = time()

	# Calculate mask of background voxels within Rinside sphere but outside of particle mask
	maskBG = Rinside-mask

	# Use numpy stride tricks to compute "view" into NON-overlapping
	# blocks of 2*r+1. Does not allocate any extra memory
	maskBGview = rolling_window(Rinside-mask, 
					window=(winSize, winSize, winSize), 
					asteps=(winSize, winSize, winSize))

	dataBGview = rolling_window(data, 
					window=(winSize, winSize, winSize), 
					asteps=(winSize, winSize, winSize))

	# Find blocks within maskBG that are all 1s (i.e. only contain background voxels)
	blockMaskBG = np.squeeze(np.apply_over_axes(np.all, maskBGview, [3,4,5]))

	# Get the i, j, k indices where the blocks are all 1s
	blockMaskBGindex = np.array(np.where(blockMaskBG))

	dataBGcube = np.zeros(numberOfPoints,			dtype='float32')
	WRSSBG     = np.zeros(blockMaskBGindex.shape[1],dtype='float32')

	# For each block, use 3D steerable model to estimate sigma^2
	for idx in range(blockMaskBGindex.shape[1]):
		i, j, k     = blockMaskBGindex[:,idx]
		dataBGcube  = dataBGview[i,j,k,...].flatten()
		WRSSBG[idx] = np.vdot(dataBGcube, np.dot(LAMBDA, dataBGcube))
	WRSSBG = WRSSBG/np.trace(LAMBDA);

	# Estimate variance as mean of sigma^2's in each block
	variance = np.mean(WRSSBG);

	print 'done.'
	m, s = divmod(time() - tStart, 60)
	print "  :: Time elapsed: %d minutes and %.2f seconds" % (m, s)
	del maskBGview 
	del dataBGview

	## Compute Likelihood Ratio Statistic

	# Calculate weighted residual sum of squares difference
	print 'Calculating Likelihood Ratio Statistic...'
	tStart   = time()

	# Use numpy stride tricks to compute "view" into OVERLAPPING
	# blocks of 2*r+1. Does not allocate any extra memory
	dataView = rolling_window(data, window=(winSize, winSize, winSize))

	# Calculate i, j, k indices where particle mask is 1
	indexVec = np.array(np.where(mask))

	# Adjust i, j, k indices to take into account 'r'-wide padding 
	# due to taking overlapping blocks using 'rolling_window' function
	indexVecView = indexVec - int(r)	
	
	dataCube = np.zeros(numberOfPoints,		dtype='float32')
	WRSSdiff = np.zeros(indexVec.shape[1],	dtype='float32')
	
	maxIdx = indexVecView.shape[1]
	progressBarIdx = int(maxIdx/100)
	# Iterate over all points where particle mask is 1
	for idx in range(indexVecView.shape[1]):
		i, j, k       = indexVecView[:,idx]
		dataCube      = dataView[i,j,k,...].flatten()
		WRSSdiff[idx] = np.vdot(dataCube, np.dot(LAMBDAdiff, dataCube))
		if idx % progressBarIdx == 0:
			update_progress(idx/float(maxIdx))
	LRSvec = WRSSdiff/variance
	print 'done.'
	m, s = divmod(time() - tStart, 60)
	print "  :: Time elapsed: %d minutes and %.2f seconds" % (m, s)

	# Undo reshaping to get LRS in a 3D volume
	print 'Reshaping results into original 3D volume...',
	tStart = time()
	LRS    = np.zeros([n,n,n], dtype='float32')
	for idx in range(indexVec.shape[1]):
		i, j, k = indexVec[:,idx]
		LRS[i,j,k] = LRSvec[idx]
	print 'done.'
	m, s = divmod(time() - tStart, 60)
	print "  :: Time elapsed: %d minutes and %.2f seconds" % (m, s)

	## Numerically Compute Weighted X^2 Statistic

	# Calculate Eigenvalues of LAMBDAdiff
	(tmp1,tmp2) = np.linalg.eig(LAMBDAdiff)
	LAMBDAeig = np.abs(tmp1)
	del tmp1
	del tmp2

	# Remove small values and truncate for numerical stability
	LAMBDAeig = np.extract(LAMBDAeig>np.max(LAMBDAeig)*1e-2,LAMBDAeig)

	# Uncorrected Threshold
	alpha = 1-pValue
	print 'Calculating Uncorrected Threshold...',
	tStart = time()
	minResults = minimize_scalar(evaluateRuben, args=(alpha,LAMBDAeig))
	thrUncorr  = minResults.x
	print 'done.'
	m, s = divmod(time() - tStart, 60)
	print "  :: Time elapsed: %d minutes and %.2f seconds" % (m, s)

	# FDR Threshold
	print 'Calculating Benjamini-Yekutieli 2001 FDR Threshold...',
	tStart = time()
	LRSvecSorted = np.sort(LRSvec)
	kmax         = np.sum(LRSvecSorted > thrUncorr)
	kpoint       = np.size(LRSvecSorted) - kmax
	maskSum      = np.sum(mask,dtype='float32')
	maskSumConst = np.sum(1.0/np.array(range(1,maskSum)))
	for k in range(1,kmax,int(np.ceil(kmax/5e1))):
		result = rubenPython(LAMBDAeig,LRSvecSorted[kpoint+k])
		# tmp = 1-(pValue*(1.0/(maskSum+1-(kmax-k))))
		tmp = 1.0-(pValue*((kmax-k)/(maskSum*maskSumConst)))
		# print 'result[2]: %e, tmp: %e' %(result[2],tmp)
		if result[2] > tmp:
			thrFDR = LRSvecSorted[kpoint+k]
			break
	print 'done.'
	m, s = divmod(time() - tStart, 60)
	print "  :: Time elapsed: %d minutes and %.2f seconds" % (m, s)

	# # Bonferroni threshold
	# alphaBon = 1-((pValue)/maskSum)
	# print 'Calculating Bonferroni Threshold...',
	# tStart = time()
	# minResults = minimize_scalar(evaluateRuben, args=(alphaBon,LAMBDAeig),tol=1e-6)
	# thrBonferroni  = minResults.x
	# print 'done.'
	# m, s = divmod(time() - tStart, 60)
	# print ":: Time elapsed: %d minutes and %.2f seconds" % (m, s)

	# Calculate resolution
	resUnco = LRS > thrUncorr
	resFDR  = LRS > thrFDR
	# resBonf = LRS > thrBonferroni

	# scipy.io.savemat('output.mat', {'resUncorrPY':resUnco, 'resFDRPY':resFDR} )
	outputMRC = mrc_image('EMD-2275.mrc')
	outputMRC.read()
	outputMRC.change_filename('EMD-2275_res.mrc')
	outputMRC.write(np.array(resFDR,dtype='float32'))

	m, s = divmod(time() - tBegin, 60)
	print "TOTAL :: Time elapsed: %d minutes and %.2f seconds" % (m, s)

	# Plot
	f1 = plt.figure()
	f2 = plt.figure()
	ax1 = f1.add_subplot(111)
	ax1.imshow(data[:,:,int(n/2)], cmap=plt.cm.gray, interpolation="nearest")
	ax2 = f2.add_subplot(111)
	ax2.imshow(resFDR[:,:,int(n/2)], cmap=plt.cm.jet, interpolation="nearest")
	plt.show()


# if __name__ == '__main__':
	# main()

