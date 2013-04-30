import os, sys
from time import time

# Modules for coomputation and visualization
import numpy as np
# import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.ndimage import filters

# Modules found in python files in root folder
from sphericalProfile import *
from localreshelpers import *
from blocks import *
from mrc import *

def lr3D(**kwargs):

	print '== BEGIN localResolution3D ==',
	preWhiten = False
	tBegin = time()

	inputFileName = kwargs.get('inputFileName','')
	data          = kwargs.get('data', 0)
	vxSize        = kwargs.get('vxSize', 0)
	pValue        = kwargs.get('pValue', 0.05)

	Mbegin = kwargs.get('Mbegin', 0.0 ) 
	Mmax   = kwargs.get('Mmax',   0.0 )
	Mstep  = kwargs.get('Mstep',  1.0 )  

	dataMask = kwargs.get('dataMask',0)

	if Mbegin == 0.0:
		Mbegin = round((2.5*vxSize)/0.5)*0.5 # round to the nearest 0.5

	if Mmax == 0.0:
		Mmax = round((4*vxSize)/0.5)*0.5 # round to the nearest 0.5



	(fname,ext)   = os.path.splitext(inputFileName)

	M = Mbegin

	# Load files from MRC file
	print '\n\n= Loading Volume'
	tStart    = time()
	# data = mrc_image(inputFileName)
	# data.read()
	# data = data.image_data
	n 	 = data.shape[0]
	N 	 = int(n)
	m, s      = divmod(time() - tStart, 60)
	print "  :: Time elapsed: %d minutes and %.2f seconds" % (m, s)

	print '\n= Computing Mask Separating Particle from Background'
	tStart    = time()
	# We assume the particle is at the center of the volume
	# Create spherical mask
	[x,y,z] = np.mgrid[ -n/2:n/2:complex(0,n),
						-n/2:n/2:complex(0,n),
						-n/2:n/2:complex(0,n) ]
	R       = np.array(np.sqrt(x**2 + y**2 + z**2),dtype='float32')
	Rinside = R < n/2
	del (x,y,z)	

	# Compute mask separating the particle from background
	dataBlurred  = filters.gaussian_filter(data, float(n*0.02))	# kernel size 2% of n
	dataMask     = dataBlurred > np.max(dataBlurred)*1e-1
	mask         = np.bitwise_and(dataMask, R < n/2 - 9)
	maskORIG     = mask
	# maskBG       = np.bitwise_and(np.logical_not(dataMask), R < n/2 - 2)
	# maskBGsmooth = filters.gaussian_filter(maskBG, float(n*0.05))	# kernel size 5% of n
	oldSumOfMask = np.sum(mask)
	del dataMask	

	m, s      = divmod(time() - tStart, 60)
	print "  :: Time elapsed: %d minutes and %.2f seconds" % (m, s)

	if preWhiten:
		print '\n= Calculating Spherically Averaged Power Spectrum of Volume'
		tStart    = time()
		dataF     = np.fft.fftshift(np.fft.fftn(data))
		dataFabs  = np.array(np.abs(dataF), dtype='float32')
		dataFabs  = dataFabs-np.min(dataFabs)
		dataFabs  = dataFabs/np.max(dataFabs)
		dataSpect = sphericalAverage(dataFabs) + 1e-6
		m, s      = divmod(time() - tStart, 60)
		print "  :: Time elapsed: %d minutes and %.2f seconds" % (m, s)

		print '\n= Pre-whitening Volume'
		tStart    = time()
		xpoly = np.array(range(1,dataSpect.size + 1))
		ypoly = np.log(dataSpect)
		wpoly = np.array(np.bitwise_and(xpoly>n/4, xpoly<n/2-1), dtype='float32')
		pcoef = np.polynomial.polynomial.polyfit(xpoly, ypoly, 1, w=wpoly)
		peval = np.polynomial.polynomial.polyval(xpoly,pcoef)

		R[R>n/2] = n/2
		R[R<n/4] = n/4
		Reval    = np.polynomial.polynomial.polyval(R,-1*pcoef)
		pWfilter = np.exp(Reval)

		dataPW = np.real(np.fft.ifftn(np.fft.ifftshift(pWfilter*dataF)))

		dataPWF     = np.fft.fftshift(np.fft.fftn(dataPW))
		dataPWFabs  = np.array(np.abs(dataPWF), dtype='float32')
		dataPWFabs  = dataPWFabs-np.min(dataPWFabs)
		dataPWFabs  = dataPWFabs/np.max(dataPWFabs)
		dataPWSpect = sphericalAverage(dataPWFabs) + 1e-6
		
		m, s      = divmod(time() - tStart, 60)
		print "  :: Time elapsed: %d minutes and %.2f seconds" % (m, s)

		# plt.plot(xpoly,dataSpect**2,'b', 
		# 		 xpoly,np.exp(peval)**2,'r',
		# 		 xpoly,dataPWSpect**2,'m')
		# plt.yscale('log')
		# plt.show()

		# f, (ax1, ax2) = plt.subplots(1, 2)
		# ax1.imshow(data[int(n/2),:,:], cmap=plt.cm.gray, interpolation="nearest")
		# ax2.imshow(dataPW[int(n/2),:,:], cmap=plt.cm.gray, interpolation="nearest")
		# plt.show()

		data = dataPW

	resTOTAL = np.zeros_like(data)

	moreToProcess = True
	while moreToProcess:

		print '\n\n= Calculating Local Resolution for %.2f Angstroms\n' % M
		# Compute window size and form steerable bases
		r       = np.ceil(0.5*M/vxSize)			# number of pixels around center
		a       = (2*np.pi/M) * np.sqrt(2.0/5)	# scaling factor so that peak occurs at 1/M Angstroms
		winSize = 2*r+1		
		print "winSize = %.2f" % winSize

		# f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
		# ax1.imshow(data[int(3*n/9),:,:], cmap=plt.cm.gray, interpolation="nearest")
		# ax2.imshow(data[int(4*n/9),:,:], cmap=plt.cm.gray, interpolation="nearest")
		# ax3.imshow(data[int(5*n/9),:,:], cmap=plt.cm.gray, interpolation="nearest")
		# ax4.imshow(data[int(6*n/9),:,:], cmap=plt.cm.gray, interpolation="nearest")

		# f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
		# ax1.imshow(mask[int(3*n/9),:,:], cmap=plt.cm.jet, interpolation="nearest")
		# ax2.imshow(mask[int(4*n/9),:,:], cmap=plt.cm.jet, interpolation="nearest")
		# ax3.imshow(mask[int(5*n/9),:,:], cmap=plt.cm.jet, interpolation="nearest")
		# ax4.imshow(mask[int(6*n/9),:,:], cmap=plt.cm.jet, interpolation="nearest")
		# plt.show()

		# Define range of x, y, z for steerable bases
		[x,y,z] = a*np.mgrid[	-r*vxSize:r*vxSize:complex(0,winSize),
								-r*vxSize:r*vxSize:complex(0,winSize),
								-r*vxSize:r*vxSize:complex(0,winSize) ]		
		dirs    = make3DsteerableDirections(x, y, z)

		# Define Gaussian kernel
		kernel     = np.exp(-1/2*(x**2 + y**2 + z**2)).flatten()
		kernelSqrt = np.sqrt(kernel)  
		kernelSum  = kernel.sum()
		W          = np.diag(kernel)
		del (x,y,z)

		# Calculate shape of matrix of bases
		numberOfPoints = kernel.size
		numberOfBases  = dirs.shape[3] + 1

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

		# debug
		Ad = A[:,1:17]
		Hd = np.dot(Ad, np.dot(np.linalg.pinv(np.dot(np.diag(kernelSqrt),Ad)), np.diag(kernelSqrt)))
		LAMBDAd = W-np.dot(W,Hd)

		# Invert weighted A matrix via SVD	
		H = np.dot(A, np.dot(np.linalg.pinv(np.dot(np.diag(kernelSqrt),A)), np.diag(kernelSqrt)))

		# Invert weighted Ac matrix via SVD
		Ack = np.dot(np.diag(kernelSqrt),Ac)	
		Hc  = np.dot(Ac, np.dot(Ack.T/(np.linalg.norm(Ack)**2), np.diag(kernelSqrt)))

		# Create LAMBDA matrices that correspond to WRSS = Y^T*LAMBDA*Y
		LAMBDA     = W-np.dot(W,H)
		LAMBDAc    = W-np.dot(W,Hc)
		LAMBDAdiff = np.array(LAMBDAc-LAMBDA, dtype='float32')

		## Estimate variance
	# if M == Mbegin:
		print 'Estimating variance from non-overlapping blocks in background...',
		tStart = time()

		if M == Mbegin:
			# Calculate mask of background voxels within Rinside sphere but outside of particle mask
			maskBG = Rinside-mask

		# Use numpy stride tricks to compute "view" into NON-overlapping
		# blocks of 2*r+1. Does not allocate any extra memory
		maskBGview = rolling_window(maskBG, 								# WHOA!
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
			dataBGcube  = dataBGcube - np.mean(dataBGcube)
			WRSSBG[idx] = np.vdot(dataBGcube, np.dot(LAMBDAd, dataBGcube))
		WRSSBG = WRSSBG/np.trace(LAMBDAd)

		# Estimate variance as mean of sigma^2's in each block
		variance = np.mean(WRSSBG)

		print 'done.'
		m, s = divmod(time() - tStart, 60)
		print "  :: Time elapsed: %d minutes and %.2f seconds" % (m, s)
		del maskBGview 
		del dataBGview
		print "variance = %.6f" % variance

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
				sys.stdout.flush()
		LRSvec = WRSSdiff/(variance+1e-10)
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
		LAMBDAeig = np.extract(LAMBDAeig>np.max(LAMBDAeig)*1e-1,LAMBDAeig)

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
		for k in range(1,kmax,int(np.ceil(kmax/5e2))):
			result = rubenPython(LAMBDAeig,LRSvecSorted[kpoint+k])
			# tmp = 1-(pValue*(1.0/(maskSum+1-(kmax-k))))
			tmp = 1.0-(pValue*((kmax-k)/(maskSum*maskSumConst)))
			thrFDR = LRSvecSorted[kpoint+k]
			# print 'result[2]: %e, tmp: %e' %(result[2],tmp)
			if result[2] > tmp:
				# print 'HIT'
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
		res      = LRS > thrFDR
		resTOTAL = resTOTAL + M*res

		# Update the mask to voxels that failed this level's likelihood test
		mask = np.array(mask - res, dtype='bool')
		newSumOfMask = np.sum(mask)
		print "oldSumOfMask = %d" % oldSumOfMask
		print "newSumOfMask = %d" % newSumOfMask
		if oldSumOfMask-newSumOfMask < n and newSumOfMask < (n**2):
			print 'We have probably covered all the voxels of interest.'
			moreToProcess = False
		oldSumOfMask = newSumOfMask

		if M >= Mmax:
			print 'We have reached Mmax = %.2f.' % Mmax
			moreToProcess = False		

		# Update query resolution
		M += Mstep


	# Set all voxels that were outside of the mask or that failed all resolution tests to M+Mstep
	zeroVoxels = (resTOTAL==0)
	resTOTAL[zeroVoxels] = M+Mstep

	# Write results out as MRC volume
	outputMRC = mrc_image(inputFileName)
	outputMRC.read()
	outputMRC.change_filename(fname+"_res"+ext)
	outputMRC.write(np.array(resTOTAL,dtype='float32'))

	m, s = divmod(time() - tBegin, 60)
	print "\nTOTAL :: Time elapsed: %d minutes and %.2f seconds" % (m, s)

	# Plots
	resTOTALma  = np.ma.masked_where(resTOTAL == M+Mstep, resTOTAL)

	print "\nMEDIAN RESOLUTION in MASK = %.2f" % np.ma.median(resTOTALma)
	print "\nRESULT WRITTEN to MRC file: " + fname + "_res" + ext

	# f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
	# ax1.imshow(data[int(3*n/9),:,:], cmap=plt.cm.gray, interpolation="nearest")
	# ax2.imshow(data[int(4*n/9),:,:], cmap=plt.cm.gray, interpolation="nearest")
	# ax3.imshow(data[int(5*n/9),:,:], cmap=plt.cm.gray, interpolation="nearest")
	# ax4.imshow(data[int(6*n/9),:,:], cmap=plt.cm.gray, interpolation="nearest")
 
	# f2, ((ax21, ax22), (ax23, ax24)) = plt.subplots(2, 2)
	# ax21.imshow(data[int(3*n/9),:,:], cmap=plt.cm.gray, interpolation="nearest")
	# ax22.imshow(data[int(4*n/9),:,:], cmap=plt.cm.gray, interpolation="nearest")
	# ax23.imshow(data[int(5*n/9),:,:], cmap=plt.cm.gray, interpolation="nearest")
	# ax24.imshow(data[int(6*n/9),:,:], cmap=plt.cm.gray, interpolation="nearest")
	# cax = ax21.imshow(resTOTALma[int(3*n/9),:,:], cmap=plt.cm.jet, interpolation="nearest", alpha=0.25)
	# ax22.imshow(resTOTALma[int(4*n/9),:,:], cmap=plt.cm.jet, interpolation="nearest", alpha=0.25)
	# ax23.imshow(resTOTALma[int(5*n/9),:,:], cmap=plt.cm.jet, interpolation="nearest", alpha=0.25)
	# ax24.imshow(resTOTALma[int(6*n/9),:,:], cmap=plt.cm.jet, interpolation="nearest", alpha=0.25)
	# cbar = f2.colorbar(cax)
	# plt.show()

	try: 
		input = raw_input
	except NameError: 
		pass

	# raw_input("Press any key or close windows to EXIT")


# if __name__ == '__main__':
	# main()

