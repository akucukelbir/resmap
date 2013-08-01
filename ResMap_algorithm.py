'''
ResMap_algorithm: module containing main local resolution 3D algorithm. (Alp Kucukelbir, 2013)
             
Description of functions:
    ResMap_algorithm: Compute the local resolution map of a density map

Requirements:
    numpy
    scipy
    matplotlib
    ResMap submodules

'''
import os, sys
from time import time
from collections import OrderedDict

import numpy as np
from scipy import ndimage
from scipy.optimize import minimize_scalar
from scipy.ndimage import filters

import matplotlib.pyplot as plt

from ResMap_helpers import *
from ResMap_blocks import *
from ResMap_fileIO import *
from ResMap_toChimera import *
from ResMap_spectrumTools import *
from ResMap_sphericalProfile import sphericalAverage

def ResMap_algorithm(**kwargs):
	'''
	ResMap_algorithm implements the procedure described in the following article:

	A. Kucukelbir, F.J. Sigworth, and H.D. Tagare, The Local Resolution of Cryo-EM Density Maps, In Review, 2013.

	The procedure will (coarsely speaking) do the following things:	
		1. Grab the volume from the the MRC data structure (class)
		2. Calculate a mask that separates the particle from the background
		3. Pre-whiten the volume, if necessary
		4. Form the required matrices for the local sinusoid-like model of the smallest scale possible
		5. Estimate the variance from non-overlapping blocks in the background
		6. Calculate the likelihood ratio statistic
		7. Compute the FDR adjusted threshold
		8. Compare the statistic to the threshold and assign a resolution to voxels that pass
		9. Repeat until max resolution is reached or (heuristically) most points in mask are assigned
		10. Write result out to a MRC volume

	Required Parameters 
	----------
	inputFileName: string variable pointing to density map to analyze
	      	 data: density map loaded as a numpy array
	       vxSize: the voxel spacing for the density map (in Angstrom)
	       pValue: the desired significance level (usually 0.05)

	Optional Parameters 
	----------
	  minRes: starting resolution 
	  maxRes: maximum resolution to consider 
	 stepRes: step size for resolution queries 
	dataMask: mask loaded as a numpy array

	Assumptions 
	-------
	ResMap assumes that the density map being analyzed has not been filtered in any way, and that 
	some reasonable pre-whitening has been applied, such that the noise spectrum of the map is 
	relatively white, at least towards the Nyquist end of the spectrum.

	ResMap now provides some built-in pre-whitening tools.

	Returns 
	-------
	Writes out a new MRC volume in the same folder as the input MRC volume with '_resmap' appended to
	the file name. Values are in Angstrom and represent the local resolution assigned to each voxel.

	'''

	print '== BEGIN Resolution Map Calculation ==',
	tBegin = time()

	epsilon = 1e-20

	debugMode = False
	
	## Process inputs to function
	print '\n\n= Reading Input Parameters'
	tStart = time()

	inputFileName   = kwargs.get('inputFileName','')
	dataMRC         = kwargs.get('data',     0)
	vxSize          = kwargs.get('vxSize',   1.0 )
	pValue          = kwargs.get('pValue',   0.05)
	minRes          = kwargs.get('minRes',   0.0 ) 
	maxRes          = kwargs.get('maxRes',   0.0 )
	stepRes         = kwargs.get('stepRes',  1.0 ) 
	dataMask        = kwargs.get('dataMask', 0)
	variance        = kwargs.get('variance', 0.0)
	graphicalOutput = bool(kwargs.get('graphicalOutput', False))
	chimeraLaunch   = bool(kwargs.get('chimeraLaunch', False))
 
	# Extract volume from MRC class
	data   = dataMRC.matrix
	data   = data-np.mean(data)

	m, s   = divmod(time() - tStart, 60)
	print "  :: Time elapsed: %d minutes and %.2f seconds" % (m, s)

	print '\n\n= Testing Whether the Input Volume has been Low-pass Filtered\n'
	tStart = time()

	dataPowerSpectrum = calculatePowerSpectrum(data)	
	LPFtest           = isPowerSpectrumLPF(dataPowerSpectrum)

	if LPFtest['outcome']:
		print ( "=======================================================================\n"
				"|                                                                     |\n"
				"|            The volume appears to be low-pass filtered.              |\n"
				"|                                                                     |\n"				
				"|         This is not ideal, but ResMap will attempt to run.          |\n"
				"|                                                                     |\n"				
				"|        The input volume will be down-sampled within ResMap.         |\n"
				"|                                                                     |\n"				
				"=======================================================================\n")

		# Calculate the ratio by which the volume should be down-sampled
		# such that the LPF cutoff becomes the new Nyquist
		LPFfactor = round((LPFtest['factor'])/0.01)*0.01	# round to the nearest 0.01

		# Down- sample the volume using cubic splines
		data   = ndimage.interpolation.zoom(data, LPFfactor, mode='reflect')
		vxSize = float(vxSize)/LPFfactor
	else:
		print "  The volume does not appear to be low-pass filtered. Great!\n"
		LPFfactor = 0

	# Grab the volume size (assumed to be a cube)
	n = data.shape[0]

	# Calculate min res
	if minRes <= (2.2*vxSize):
		minRes = round((2.2*vxSize)/0.1)*0.1 # round to the nearest 0.1
	currentRes = minRes 

	# Calculate max res
	if maxRes == 0.0:
		maxRes = round((4.0*vxSize)/0.5)*0.5 # round to the nearest 0.5

	m, s   = divmod(time() - tStart, 60)
	print "  :: Time elapsed: %d minutes and %.2f seconds" % (m, s)

	print '\n\n= ResMap will now run with the following parameters'
	print '  inputMap:\t%s' 	% inputFileName
	print '  vxSize:\t%.2f' 		% vxSize
	print '  pValue:\t%.2f'			% pValue
	print '  minRes:\t%.2f' 		% minRes
	print '  maxRes:\t%.2f'   		% maxRes
	print '  stepRes:\t%.2f'   		% stepRes
	print '  variance:\t%.4f'   	% variance
	print '  LPFfactor:\t%.2f'   	% LPFfactor

	print '\n= Computing Mask Separating Particle from Background'
	tStart    = time()
	
	# We assume the particle is at the center of the volume
	# Create spherical mask
	[x,y,z] = np.mgrid[ -n/2:n/2:complex(0,n),
						-n/2:n/2:complex(0,n),
						-n/2:n/2:complex(0,n) ]
	R       = np.array(np.sqrt(x**2 + y**2 + z**2), dtype='float32')
	Rinside = R < n/2 - 1

	# Check to see if mask volume was already provided
	if isinstance(dataMask,MRC_Data) == False:
		# Compute mask separating the particle from background
		dataBlurred  = filters.gaussian_filter(data, float(n*0.02))	# kernel size 2% of n
		dataMask     = dataBlurred > np.max(dataBlurred)*5e-2
		del dataBlurred
	else:
		dataMask = np.array(dataMask.matrix, dtype='bool')

	mask         = np.bitwise_and(dataMask,  R < n/2 - 9)	# backoff 9 voxels from edge (make adaptive later)
	oldSumOfMask = np.sum(mask)

	m, s      = divmod(time() - tStart, 60)
	print "  :: Time elapsed: %d minutes and %.2f seconds" % (m, s)

	# PRE-WHITENING
	dataOrig = data
	if variance == 0.0:
		estimateVarianceFromBackground = True

		print '\n= Computing Soft Mask Separating Particle from Background'
		tStart      = time()

		# Dilate the mask a bit so that we don't seep into the particle when we blur it later
		boxElement  = np.ones([5, 5, 5])
		dilatedMask = ndimage.morphology.binary_dilation(dataMask, structure=boxElement, iterations=3)
		dilatedMask = np.logical_and(dilatedMask, Rinside)

		# Blur the mask
		softBGmask  = filters.gaussian_filter(np.array(np.logical_not(dilatedMask),dtype='float32'),float(n*0.02))

		# Get the background
		dataBG      = np.multiply(data,softBGmask)

		m, s        = divmod(time() - tStart, 60)
		print "  :: Time elapsed: %d minutes and %.2f seconds" % (m, s)		

		print '\n= Calculating Spherically Averaged Power Spectra'
		tStart    = time()

		# Calculate spectrum of input volume
		dataF     = np.fft.fftshift(np.fft.fftn(data))
		dataFabs  = np.array(np.abs(dataF), dtype='float32')
		dataFabs  = dataFabs-np.min(dataFabs)
		dataFabs  = dataFabs/np.max(dataFabs)
		dataSpect = sphericalAverage(dataFabs) + epsilon

		# Calculate spectrum of background volume
		dataBGF     = np.fft.fftshift(np.fft.fftn(dataBG))
		dataBGFabs  = np.array(np.abs(dataBGF), dtype='float32')
		dataBGFabs  = dataBGFabs-np.min(dataBGFabs)
		dataBGFabs  = dataBGFabs/np.max(dataBGFabs)
		dataBGSpect = sphericalAverage(dataBGFabs) + epsilon

		m, s      = divmod(time() - tStart, 60)
		print "  :: Time elapsed: %d minutes and %.2f seconds" % (m, s)

		# Usually b-factor sharpening has gone awry beyond about 10A
		# therefore we take a first shot of ramping the spectrum up or down beyond 10A
		oldElbowAngstrom = 0
		newElbowAngstrom = max(10,2.1*vxSize)

		# Sometimes the ramping is too much, so allow the user to adjust it down
		oldRampWeight = 0.0
		newRampWeight = 1.0

		# While the user changes the elbow in the Pre-Whitening Interface, repeat the pre-whitening.
		# 	this loop will stop when the user does NOT change the elbow in the interface.
		#
		#	it is a bit of a hack, but it works completely within matplotlib which is a relief
		#
		while newElbowAngstrom != oldElbowAngstrom or oldRampWeight != newRampWeight:

			preWhiteningResult = preWhitenVolume(x,y,z,				
									elbowAngstrom = newElbowAngstrom,
									dataBGSpect   = dataBGSpect,
									dataF         = dataF,
									softBGmask    = softBGmask,
									vxSize        = vxSize,
									rampWeight    = newRampWeight)

			dataPW = preWhiteningResult['dataPW']
			
			oldElbowAngstrom = newElbowAngstrom
			oldRampWeight    = newRampWeight

			newElbowAngstrom, newRampWeight = displayPreWhitening(
								elbowAngstrom = oldElbowAngstrom,
								rampWeight    = oldRampWeight,
								dataSpect     = dataSpect,
								dataBGSpect   = dataBGSpect,
								peval         = preWhiteningResult['peval'],
								dataPWSpect   = preWhiteningResult['dataPWSpect'],
								dataPWBGSpect = preWhiteningResult['dataPWBGSpect'],
								xpoly         = preWhiteningResult['xpoly'],
								vxSize 		  = vxSize,
								dataSlice     = data[int(n/2),:,:], 
								dataPWSlice   = dataPW[int(n/2),:,:]
								)

			del preWhiteningResult

		data = dataPW
	else:
		estimateVarianceFromBackground = False
		print ( "=======================================================================\n"
				"|                                                                     |\n"
				"|    You have chosen to run ResMap with your own variance estimate.   |\n"
				"|                                                                     |\n"
				"|                                                                     |\n"				
				"|                  This is strongly NOT recommended.                  |\n"
				"|                                                                     |\n"
				"|                                                                     |\n"								
				"|                 The results may not make any sense                  |\n"
				"|                 and the algorithm may simply fail.                  |\n"
				"|                                                                     |\n"	
				"|                                                                     |\n"	
				"|       Please consider using ResMap's own estimation option.         |\n"	
				"|                                                                     |\n"																	
				"=======================================================================\n")

		crudeEstimate = np.var(data[np.logical_and(Rinside,np.logical_not(mask))])

		print "Crude Estimate of Noise Variance = %.6f" % crudeEstimate

		if abs( int(np.log10(variance)) - int(np.log10(crudeEstimate)) ) > 1:
			print "\n"
			print ( "=======================================================================\n"
					"|                                                                     |\n"
					"|     It seems like your variance estimate is more than 1 order       |\n"
					"|           of magnitude off... beware of the results.                |\n"		
					"|                                                                     |\n"																	
					"=======================================================================\n")

	del (x,y,z)	

	# Initialize the ResMap result volume
	resTOTAL = np.zeros_like(data)

	# Initialize empty dictionary for histogram plotting
	resHisto = OrderedDict()

	# Continue testing larger and larger scales as long as there is "moreToProcess" (see below)
	moreToProcess = True
	while moreToProcess:
		print '\n\n= Calculating Local Resolution for %.2f Angstroms\n' % currentRes

		# Compute window size and form steerable bases
		r       = np.ceil(0.5*currentRes/vxSize)			# number of pixels around center
		a       = (2*np.pi/currentRes) * np.sqrt(2.0/5)		# scaling factor so that peak occurs at 1/currentRes Angstroms
		winSize = 2*r+1		
		print "winSize = %.2f" % winSize

		if debugMode:
			f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
			ax1.imshow(data[int(3*n/9),:,:], cmap=plt.cm.gray, interpolation="nearest")
			ax2.imshow(data[int(4*n/9),:,:], cmap=plt.cm.gray, interpolation="nearest")
			ax3.imshow(data[int(5*n/9),:,:], cmap=plt.cm.gray, interpolation="nearest")
			ax4.imshow(data[int(6*n/9),:,:], cmap=plt.cm.gray, interpolation="nearest")

			f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
			ax1.imshow(mask[int(3*n/9),:,:], cmap=plt.cm.jet, interpolation="nearest")
			ax2.imshow(mask[int(4*n/9),:,:], cmap=plt.cm.jet, interpolation="nearest")
			ax3.imshow(mask[int(5*n/9),:,:], cmap=plt.cm.jet, interpolation="nearest")
			ax4.imshow(mask[int(6*n/9),:,:], cmap=plt.cm.jet, interpolation="nearest")
			plt.show()

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

		# Form matrix of all but the constant term
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
		if estimateVarianceFromBackground==True:
			print 'Estimating variance from non-overlapping blocks in background...',
			tStart = time()

			if currentRes == minRes:
				# Calculate mask of background voxels within Rinside sphere but outside of particle mask
				maskBG = Rinside-mask

			# Use numpy stride tricks to compute "view" into NON-overlapping
			# blocks of 2*r+1. Does not allocate any extra memory
			maskBGview = rolling_window(maskBG, 								
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
			if progressBarIdx > 0 and idx % progressBarIdx == 0:
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
		if kmax == 0:
			thrFDR = sys.float_info.max
		else:
			kpoint       = np.size(LRSvecSorted) - kmax
			maskSum      = np.sum(mask,dtype='float32')
			maskSumConst = np.sum(1.0/np.array(range(1,maskSum)))
			for k in range(1, kmax, int(np.ceil(kmax/min(5e2,kmax)))):	# only compute ruben for about kmax/5e2 points
				result = rubenPython(LAMBDAeig,LRSvecSorted[kpoint+k])
				tmp    = 1.0-(pValue*((kmax-k)/(maskSum*maskSumConst)))
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
		resTOTAL = resTOTAL + currentRes*res

		# Update the mask to voxels that failed this level's likelihood test
		mask         = np.array(mask - res, dtype='bool')
		newSumOfMask = np.sum(mask)

		print "\nNumber of voxels assigned in this iteration = %d" % (oldSumOfMask-newSumOfMask)

		# Update value in histogram
		resHisto[str(currentRes)] = (oldSumOfMask-newSumOfMask)

		# Heuristic of telling whether we are likely done
		if oldSumOfMask-newSumOfMask < n and newSumOfMask < (n**2):
			print 'We have probably covered all voxels of interest.'
			moreToProcess = False
		oldSumOfMask = newSumOfMask

		if currentRes >= maxRes:
			print 'We have reached MaxRes = %.2f.' % maxRes
			moreToProcess = False		

		# Update current resolution
		currentRes += stepRes


	# Set all voxels that were outside of the mask or that failed all resolution tests to 100 A
	zeroVoxels = (resTOTAL==0)
	resTOTAL[zeroVoxels] = 100
	resTOTALma  = np.ma.masked_where(resTOTAL == 100, resTOTAL)	

	old_n = dataMRC.data_size[0]
	old_coordinates = np.mgrid[	1:n:complex(0,old_n),
								1:n:complex(0,old_n),
								1:n:complex(0,old_n) ]		

	# Up-sample the resulting resolution map if necessary
	if LPFfactor > 0:
		resTOTAL = ndimage.map_coordinates(resTOTAL, old_coordinates, order=1, mode='nearest')
		resTOTAL[resTOTAL < minRes] = minRes
		resTOTAL[resTOTAL > 100] = 100

	# Write results out as MRC volume
	(fname,ext)    = os.path.splitext(inputFileName)
	dataMRC.matrix = np.array(resTOTAL,dtype='float32')
	write_mrc2000_grid_data(dataMRC, fname+"_resmap"+ext)

	m, s = divmod(time() - tBegin, 60)
	print "\nTOTAL :: Time elapsed: %d minutes and %.2f seconds" % (m, s)

	print "\n  MEAN RESOLUTION in MASK = %.2f" % np.ma.mean(resTOTALma)
	print "MEDIAN RESOLUTION in MASK = %.2f" % np.ma.median(resTOTALma)

	print "\nRESULT WRITTEN to MRC file: " + fname + "_resmap" + ext
	
	chimeraScriptFileName = createChimeraScript(inputFileName, minRes, maxRes, int(resTOTAL.shape[0]), animated=True)

	print "\nCHIMERA SCRIPT WRITTEN to: " + chimeraScriptFileName

	if chimeraLaunch == True:
		print "\nATTEMPTING TO LAUNCH CHIMERA... "
		locations = ["",
					 "/Applications/Chimera.app/Contents/MacOS/",
					 "/usr/local/bin/",
					 "C:\\Program Files\\Chimera\\bin\\",
					 "C:\\Program Files\\Chimera 1.6\\bin\\",
					 "C:\\Program Files\\Chimera 1.7\\bin\\",
					 "C:\\Program Files\\Chimera 1.8\\bin\\",
					 "C:\\Program Files (x86)\\Chimera\\bin\\",
					 "C:\\Program Files (x86)\\Chimera 1.6\\bin\\",
					 "C:\\Program Files (x86)\\Chimera 1.7\\bin\\",
					 "C:\\Program Files (x86)\\Chimera 1.8\\bin\\"]
		try:
			try_alternatives("chimera", locations, ["--send", chimeraScriptFileName])
		except OSError:
			print "\n\n\n!!! ResMap is having trouble finding and/or launching UCSF Chimera. Please manually load the script into Chimera. !!!\n\n\n"

	if graphicalOutput == True:
		# Plots
		f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
		f.suptitle('\nSlices Through Input Volume', fontsize=14, color='#104E8B', fontweight='bold')
		ax1.imshow(dataOrig[int(3*n/9),:,:], cmap=plt.cm.gray, interpolation="nearest")
		ax2.imshow(dataOrig[int(4*n/9),:,:], cmap=plt.cm.gray, interpolation="nearest")
		ax3.imshow(dataOrig[int(5*n/9),:,:], cmap=plt.cm.gray, interpolation="nearest")
		ax4.imshow(dataOrig[int(6*n/9),:,:], cmap=plt.cm.gray, interpolation="nearest")

		# ax1.set_title('Slice ' + str(int(3*n/9)), fontsize=10, color='#104E8B')
		# ax2.set_title('Slice ' + str(int(4*n/9)), fontsize=10, color='#104E8B')
		# ax3.set_title('Slice ' + str(int(5*n/9)), fontsize=10, color='#104E8B')
		# ax4.set_title('Slice ' + str(int(6*n/9)), fontsize=10, color='#104E8B')
	 
		f2, ((ax21, ax22), (ax23, ax24)) = plt.subplots(2, 2)
		f2.suptitle('\nSlices Through ResMap Results', fontsize=14, color='#104E8B', fontweight='bold')
		# ax21.imshow(data[int(3*n/9),:,:], cmap=plt.cm.gray, interpolation="nearest")
		# ax22.imshow(data[int(4*n/9),:,:], cmap=plt.cm.gray, interpolation="nearest")
		# ax23.imshow(data[int(5*n/9),:,:], cmap=plt.cm.gray, interpolation="nearest")
		# ax24.imshow(data[int(6*n/9),:,:], cmap=plt.cm.gray, interpolation="nearest")
		im = ax21.imshow(resTOTALma[int(3*n/9),:,:], cmap=plt.cm.jet, interpolation="nearest")#, alpha=0.25)
		ax22.imshow(resTOTALma[int(4*n/9),:,:], cmap=plt.cm.jet, interpolation="nearest")#, alpha=0.25)
		ax23.imshow(resTOTALma[int(5*n/9),:,:], cmap=plt.cm.jet, interpolation="nearest")#, alpha=0.25)
		ax24.imshow(resTOTALma[int(6*n/9),:,:], cmap=plt.cm.jet, interpolation="nearest")#, alpha=0.25)

		# ax21.set_title('Slice ' + str(int(3*n/9)), fontsize=10, color='#104E8B')
		# ax22.set_title('Slice ' + str(int(4*n/9)), fontsize=10, color='#104E8B')
		# ax23.set_title('Slice ' + str(int(5*n/9)), fontsize=10, color='#104E8B')
		# ax24.set_title('Slice ' + str(int(6*n/9)), fontsize=10, color='#104E8B')
		
		cax = f2.add_axes([0.9, 0.1, 0.03, 0.8])
		f2.colorbar(im, cax=cax)

		# Histogram
		f3   = plt.figure()
		f3.suptitle('\nHistogram of ResMap Results', fontsize=14, color='#104E8B', fontweight='bold')
		axf3 = f3.add_subplot(111)
		
		axf3.bar(range(len(resHisto)), resHisto.values(), align='center')
		axf3.set_xlabel('Resolution (Angstroms)')
		axf3.set_xticks(range(len(resHisto)))
		axf3.set_xticklabels(resHisto.keys())
		axf3.set_ylabel('Number of Voxels')

		plt.show()
