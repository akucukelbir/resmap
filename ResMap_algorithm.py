'''
ResMap_algorithm: module containing main local resolution 3D algorithm.
								 (Alp Kucukelbir, 2013)

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

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

from ResMap_helpers import *
from ResMap_blocks import *
from ResMap_fileIO import *
from ResMap_toChimera import *
from ResMap_spectrumTools import *
from ResMap_sphericalProfile import sphericalAverage


def ResMap_algorithm(**kwargs):
	'''
	ResMap_algorithm

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

	np.seterr(divide='raise',invalid='raise')

	epsilon = 1e-20

	debugMode = False

	## Process inputs to function
	print '\n\n= Reading Input Parameters'
	tStart = time()

	inputFileName    = kwargs.get('inputFileName',  None)
	inputFileName1   = kwargs.get('inputFileName1', None)
	inputFileName2   = kwargs.get('inputFileName2', None)
	dataMRC          = kwargs.get('data',  0)
	dataMRC1         = kwargs.get('data1', 0)
	dataMRC2         = kwargs.get('data2', 0)

	vxSize           = kwargs.get('vxSize',   0.0 )
	pValue           = kwargs.get('pValue',   0.05)
	minRes           = kwargs.get('minRes',   0.0 )
	maxRes           = kwargs.get('maxRes',   0.0 )
	stepRes          = kwargs.get('stepRes',  1.0 )
	dataMask         = kwargs.get('dataMask', 0)

	graphicalOutput  = bool(kwargs.get('graphicalOutput', False))
	chimeraLaunch    = bool(kwargs.get('chimeraLaunch', False))
	noiseDiagnostics = bool(kwargs.get('noiseDiagnostics', False))
	
	scipionPrewhitenParams = kwargs.get('scipionPrewhitenParams', {})

	# Check for voxel size (really shouldn't ever happen)
	if vxSize == 0.0:
		print "There is a serious problem with the voxel size. Aborting."
		exit(1)
	vxSizeOrig = vxSize

	# Extract volume(s) from input MRC file(s)
	splitVolume = False
	if inputFileName:
		data  = dataMRC.matrix
		data  = data-np.mean(data)
		dataOrig = np.copy(data)
		orig_n = data.shape[0]
	elif inputFileName1 and inputFileName2:
		splitVolume = True
		data     = 0.5 * (dataMRC1.matrix + dataMRC2.matrix)
		dataDiff = 0.5 * (dataMRC1.matrix - dataMRC2.matrix)
		dataOrig = np.copy(data)
		orig_n = data.shape[0]
	else:
		print "There is a serious problem with loading files. Aborting."
		exit(1)

	m, s = divmod(time() - tStart, 60)
	print "  :: Time elapsed: %d minutes and %.2f seconds" % (m, s)


	#################################
	#
	#   Test if volume is LPF
	#
	#################################
	subVolLPF = 160
	if splitVolume == False:
		LPFtestResult = testLPF(
														data        = data,
														splitVolume = splitVolume,
														vxSize      = vxSize,
														minRes      = minRes,
														maxRes      = maxRes,
														subVolLPF   = subVolLPF
														)
	else:
		LPFtestResult = testLPF(
														data        = data,
														dataDiff    = dataDiff,
														splitVolume = splitVolume,
														vxSize      = vxSize,
														minRes      = minRes,
														maxRes      = maxRes,
														subVolLPF   = subVolLPF
														)

	LPFfactor         = LPFtestResult['LPFfactor']
	vxSize            = LPFtestResult['vxSize']
	minRes            = LPFtestResult['minRes']
	maxRes            = LPFtestResult['maxRes']
	currentRes        = LPFtestResult['currentRes']
	data              = LPFtestResult['data']
	dataF             = LPFtestResult['dataF']
	dataPowerSpectrum = LPFtestResult['dataPowerSpectrum']
	if splitVolume == True:
		dataDiff = LPFtestResult['dataDiff']


	print '\n\n= ResMap will now run with the following parameters'
	if splitVolume == False:
		print '  inputMap:\t%s' 		% inputFileName
	else:
		print '  inputMap1:\t%s' 		% inputFileName1
		print '  inputMap2:\t%s' 		% inputFileName2
	print '  vxSize:\t%.2f' 		% vxSizeOrig
	print '  pValue:\t%.2f'			% pValue
	print '  minRes:\t%.2f' 		% minRes
	print '  maxRes:\t%.2f'   		% maxRes
	print '  stepRes:\t%.2f'   		% stepRes
	print '  LPFfactor:\t%.2f'   	% LPFfactor



	#################################
	#
	#   Compute Mask
	#
	#################################
	computeMaskResult = computeMask(
																	data = data,
																	dataMask = dataMask,
																	LPFfactor = LPFfactor,
																	splitVolume = splitVolume
																	)

	mask         = computeMaskResult['mask']
	dataMask     = computeMaskResult['dataMask']
	oldSumOfMask = computeMaskResult['oldSumOfMask']
	Rinside      = computeMaskResult['Rinside']


	#################################
	#
	#   PRE-WHITENING
	#
	#################################
	if splitVolume == False:
		preWhiteningLoopResult = preWhiteningLoop(
														data              = data,
														dataF             = dataF,
														dataPowerSpectrum = dataPowerSpectrum,
														vxSize            = vxSize,
														subVolLPF         = subVolLPF,
														dataMask          = dataMask,
														splitVolume       = splitVolume,
														Rinside           = Rinside,
														LPFfactor         = LPFfactor,
														scipionPrewhitenParams = scipionPrewhitenParams
														)
	else:
		preWhiteningLoopResult = preWhiteningLoop(
														data              = data,
														dataF             = dataF,
														dataPowerSpectrum = dataPowerSpectrum,
														dataDiff          = dataDiff,
														vxSize            = vxSize,
														subVolLPF         = subVolLPF,
														dataMask          = dataMask,
														splitVolume       = splitVolume,
														Rinside           = Rinside,
														LPFfactor         = LPFfactor,
														scipionPrewhitenParams = scipionPrewhitenParams
														)

	# The "force-stop" param will serve to launch a wizard to estimate 
	# the prewhitening params and then launch completely in batch mode 
	if scipionPrewhitenParams.get('force-stop', False):
		return preWhiteningLoopResult		
	
	data = preWhiteningLoopResult['data']
	
	if splitVolume == True:
		dataDiff = preWhiteningLoopResult['dataDiff']
		
	print 'newElbowAngstrom', preWhiteningLoopResult['newElbowAngstrom']
	print 'newRampWeight', preWhiteningLoopResult['newRampWeight']

	#################################
	#
	#   Compute ResMap
	#
	#################################
	if splitVolume == False:
		computeResMapResult = computeResMap(
															data         = data,
															Rinside      = Rinside,
															mask         = mask,
															splitVolume  = splitVolume,
															currentRes   = currentRes,
															vxSize       = vxSize,
															LPFfactor    = LPFfactor,
															debugMode    = debugMode,
															pValue       = pValue,
															oldSumOfMask = oldSumOfMask,
															stepRes     = stepRes,
															maxRes     = maxRes,
															orig_n     = orig_n
															)
	else:
		computeResMapResult = computeResMap(
															data         = data,
															Rinside      = Rinside,
															mask         = mask,
															splitVolume  = splitVolume,
															currentRes   = currentRes,
															vxSize       = vxSize,
															dataDiff     = dataDiff,
															LPFfactor    = LPFfactor,
															debugMode    = debugMode,
															pValue       = pValue,
															oldSumOfMask = oldSumOfMask,
															stepRes     = stepRes,
															maxRes     = maxRes,
															orig_n     = orig_n
															)

	resTOTAL   = computeResMapResult['resTOTAL']
	resTOTALma = computeResMapResult['resTOTALma']
	resHisto   = computeResMapResult['resHisto']


	m, s = divmod(time() - tBegin, 60)
	print "\nTOTAL :: Time elapsed: %d minutes and %.2f seconds" % (m, s)


	# Write results out as MRC volume
	if splitVolume == True:
		(fname,ext)    = os.path.splitext(inputFileName1)
		dataMRC1.matrix = np.array(resTOTAL,dtype='float32')
		write_mrc2000_grid_data(dataMRC1, fname+"_resmap"+ext)
	else:
		(fname,ext)    = os.path.splitext(inputFileName)
		dataMRC.matrix = np.array(resTOTAL,dtype='float32')
		write_mrc2000_grid_data(dataMRC, fname+"_resmap"+ext)


	print "\nRESULT WRITTEN to MRC file: " + fname + "_resmap" + ext

	if splitVolume == True:
		chimeraScriptFileName = createChimeraScript(inputFileName1, minRes, maxRes, int(resTOTAL.shape[0]), animated=True)
	else:
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


	#################################
	#
	#   Call 2D visualization
	#
	#################################
	if graphicalOutput == True:
		visualize2Doutput(
											dataOrig   = dataOrig,
											minRes     = minRes,
											maxRes     = maxRes,
											resTOTALma = resTOTALma,
											resHisto   = resHisto
                  		)

	computeResMapResult['minRes'] = minRes
	computeResMapResult['maxRes'] = maxRes
	computeResMapResult['orig_n'] = orig_n
	computeResMapResult['n'] = resTOTALma.shape[0]

	return computeResMapResult


























def computeResMap(**kwargs):

	data        = kwargs.get('data', None)
	Rinside     = kwargs.get('Rinside', None)
	mask        = kwargs.get('mask', None)
	splitVolume = kwargs.get('splitVolume', False)

	currentRes  = kwargs.get('currentRes', None)
	vxSize      = kwargs.get('vxSize', None)

	dataDiff    = kwargs.get('dataDiff', None)

	LPFfactor   = kwargs.get('LPFfactor', None)

	debugMode   = kwargs.get('debugMode', False)

	pValue   = kwargs.get('pValue', None)

	oldSumOfMask   = kwargs.get('oldSumOfMask', None)

	stepRes   = kwargs.get('stepRes', 0.0)
	maxRes   = kwargs.get('maxRes', 0.0)
	minRes   = kwargs.get('minRes', 0.0)

	orig_n   = kwargs.get('orig_n', 0.0)


	print("\n=======================================================================\n"
			"|                                                                     |\n"
			"|                     ResMap Computation BEGINS                       |\n"
			"|                                                                     |\n"
			"=======================================================================")

	n = data.shape[0]

	# Initialize the ResMap result volume
	resTOTAL = np.zeros_like(data)

	# Initialize empty dictionary for histogram plotting
	resHisto = OrderedDict()

	# Define regions for noise estimation in singleVolume and splitVolume mode
	if splitVolume == False:
		# Calculate mask of background voxels within Rinside sphere but outside of particle mask
		maskBG = Rinside-mask
	else:
		maskParticle = mask

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

		# Form the G2 (cosine-line terms)
		for i in range(0,6):
			tmp = dirs[:,:,:,i]
			tmp = tmp.flatten()
			A[:,i+1] = 4*(tmp**2) - 2;

		# Form the H2 (sine-like terms)
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

		# Invert weighted Ac matrix analytically
		Ack = np.dot(np.diag(kernelSqrt),Ac)
		Hc  = np.dot(Ac, np.dot(Ack.T/(np.linalg.norm(Ack)**2), np.diag(kernelSqrt)))

		# Create LAMBDA matrices that correspond to WRSS = Y^T*LAMBDA*Y
		LAMBDA     = W-np.dot(W,H)
		LAMBDAc    = W-np.dot(W,Hc)
		LAMBDAdiff = np.array(LAMBDAc-LAMBDA, dtype='float32')
		del LAMBDA, LAMBDAc

		if splitVolume == False:
			print 'Estimating variance from non-overlapping blocks in background...',
			tStart = time()

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
			del maskBGview, dataBGview, blockMaskBG
			print "variance = %.6f" % variance

		else:
			print 'Estimating variance from non-overlapping blocks from difference map...',
			tStart = time()

			# Use numpy stride tricks to compute "view" into NON-overlapping
			# blocks of 2*r+1. Does not allocate any extra memory
			maskParticleview = rolling_window(maskParticle,
									window=(winSize, winSize, winSize),
									asteps=(winSize, winSize, winSize))

			dataDiffview = rolling_window(dataDiff,
									window=(winSize, winSize, winSize),
									asteps=(winSize, winSize, winSize))

			# Find blocks within maskParticleview that are all 1s
			blockMaskParticle = np.squeeze(np.apply_over_axes(np.all, maskParticleview, [3,4,5]))

			# Get the i, j, k indices where the blocks are all 1s
			blockMaskParticleindex = np.array(np.where(blockMaskParticle))

			dataDiffcube = np.zeros(numberOfPoints,			         dtype='float32')
			WRSSDiff     = np.zeros(blockMaskParticleindex.shape[1], dtype='float32')

			# For each block, use 3D steerable model to estimate sigma^2
			for idx in range(blockMaskParticleindex.shape[1]):
				i, j, k       = blockMaskParticleindex[:,idx]
				dataDiffcube  = dataDiffview[i,j,k,...].flatten()
				dataDiffcube  = dataDiffcube - np.mean(dataDiffcube)
				WRSSDiff[idx] = np.vdot(dataDiffcube, np.dot(LAMBDAd, dataDiffcube))
			WRSSDiff = WRSSDiff/np.trace(LAMBDAd)

			# Estimate variance as mean of sigma^2's in each block
			variance = np.mean(WRSSDiff)

			print 'done.'
			m, s = divmod(time() - tStart, 60)
			print "  :: Time elapsed: %d minutes and %.2f seconds" % (m, s)
			del maskParticleview, dataDiffview, blockMaskParticle
			print "variance = %.6f" % variance

		## Compute Likelihood Ratio Statistic

		# Calculate weighted residual sum of squares difference
		print 'Calculating Likelihood Ratio Statistic...'
		tStart = time()

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
	resTOTALma  = np.ma.masked_where(resTOTAL > currentRes, resTOTAL, copy=True)

	# Print results
	print "\n  MEAN RESOLUTION in MASK = %.2f" % np.ma.mean(resTOTALma)
	print "MEDIAN RESOLUTION in MASK = %.2f" % np.ma.median(resTOTALma)


	old_coordinates = np.mgrid[	0:n-1:complex(0,orig_n),
								0:n-1:complex(0,orig_n),
								0:n-1:complex(0,orig_n) ]

	# Up-sample the resulting resolution map if necessary
	if LPFfactor > 0:
		resTOTAL = ndimage.map_coordinates(resTOTAL, old_coordinates, order=1, mode='nearest')
		resTOTAL[resTOTAL <= minRes] = minRes
		resTOTAL[resTOTAL > currentRes] = 100

		resTOTALma = np.ma.masked_where(resTOTAL > currentRes, resTOTAL, copy=True)

	return {'resTOTAL':resTOTAL, 'resTOTALma':resTOTALma,
			'resHisto':resHisto, 'currentRes': currentRes}





























def preWhiteningLoop(**kwargs):

	# Get inputs
	data              = kwargs.get('data', None)
	dataF             = kwargs.get('dataF', None)
	dataPowerSpectrum = kwargs.get('dataPowerSpectrum', None)
	dataDiff          = kwargs.get('dataDiff', None)
	vxSize            = kwargs.get('vxSize', 0.0)
	subVolLPF         = kwargs.get('subVolLPF', 0.0)
	dataMask          = kwargs.get('dataMask', None)
	splitVolume       = kwargs.get('splitVolume', False)
	Rinside           = kwargs.get('Rinside', None)
	LPFfactor         = kwargs.get('LPFfactor', None)

	n = data.shape[0]


	# Allow to pass previous params for pre-whitening and avoid to 
	# launch the gui from scratch
	scipionPrewhitenParams = kwargs.get('scipionPrewhitenParams', {})

	# We take a first shot at ramping the spectrum up/down beyond 10A
	oldElbowAngstrom = 0
	#newElbowAngstrom = max(10,2.1*vxSize)
	newElbowAngstrom = scipionPrewhitenParams.get('newElbowAngstrom', max(10,2.1*vxSize))

	# Sometimes the ramping is too much, so we allow the user to adjust it
	oldRampWeight = 0.0
	#newRampWeight = 1.0
	newRampWeight = scipionPrewhitenParams.get('newRampWeight', 1.0)
	
	if n > subVolLPF:

		print("\n=======================================================================\n"
				"|                                                                     |\n"
				"|                 ResMap Pre-Whitening (beta) Tool                    |\n"
				"|                                                                     |\n"
				"|                    The volume is quite large.                       |\n"
				"|                                                                     |\n"
				"|          ResMap will run its pre-whitening on the largest           |\n"
				"|     cube it can fit within the particle and in the background.      |\n"
				"|                                                                     |\n"
				"|          In split volume mode, ResMap will only fit a cube          |\n"
				"|  inside the particle and use the difference map as the background.  |\n"
				"|                                                                     |\n"
				"=======================================================================\n")

		print '\n= Computing The Largest Cube Within the Particle'
		tStart = time()

		dataMaskDistance = ndimage.morphology.distance_transform_cdt(dataMask, metric='taxicab')

		m, s = divmod(time() - tStart, 60)
		print "  :: Time elapsed: %d minutes and %.2f seconds" % (m, s)

		# Only need to compute largest cube in background if in single volume mode
		if splitVolume == False:
			print '\n= Computing The Largest Cube in the Background'
			tStart = time()

			dataOutside         = np.logical_and(np.logical_not(dataMask),Rinside)
			dataOutsideDistance = ndimage.morphology.distance_transform_cdt(dataOutside, metric='taxicab')

			m, s = divmod(time() - tStart, 60)
			print "  :: Time elapsed: %d minutes and %.2f seconds" % (m, s)

		print '\n= Extracting Cubes and Calculating Spherically Averaged Power Spectra'
		tStart = time()

		# Calculate the largest box size that can be fit
		if splitVolume == True:
			# Biggest cube that fits just inside the particle
			widthBox = np.max(dataMaskDistance)
		else:
			# Biggest cube that fits both inside and oustide the particle
			widthBox = np.min((np.max(dataMaskDistance), np.max(dataOutsideDistance)))
		halfWidthBox = np.floor(widthBox/2)
		cubeSize     = 2*halfWidthBox

		# Extract a cube from inside the particle
		insideBox    = np.unravel_index(np.argmax(dataMaskDistance),(n,n,n))

		# HACK: Make sure indices are cubeSize away from the edges
		insideBox = np.maximum(insideBox,[cubeSize, cubeSize, cubeSize])
		insideBox = np.minimum(insideBox,[n-cubeSize, n-cubeSize, n-cubeSize])
		# insideBox += cubeSize*np.less(insideBox, cubeSize)
		# insideBox -= cubeSize*np.greater(insideBox, n-cubeSize)

		cubeInside   = data[insideBox[0]-halfWidthBox:insideBox[0]+halfWidthBox,
							insideBox[1]-halfWidthBox:insideBox[1]+halfWidthBox,
							insideBox[2]-halfWidthBox:insideBox[2]+halfWidthBox ];

		if splitVolume == True:
			# Extract the same cube from the difference map
			cubeOutside  = dataDiff[insideBox[0]-halfWidthBox:insideBox[0]+halfWidthBox,
									insideBox[1]-halfWidthBox:insideBox[1]+halfWidthBox,
									insideBox[2]-halfWidthBox:insideBox[2]+halfWidthBox ];
		else:
			# Extract a cube from outside the particle
			outsideBox   = np.unravel_index(np.argmax(dataOutsideDistance),(n,n,n))
			cubeOutside  = data[outsideBox[0]-halfWidthBox:outsideBox[0]+halfWidthBox,
								outsideBox[1]-halfWidthBox:outsideBox[1]+halfWidthBox,
								outsideBox[2]-halfWidthBox:outsideBox[2]+halfWidthBox ];

		# Create a hamming window
		hammingWindow1D = signal.hamming(cubeSize)
		hammingWindow2D = array_outer_product(hammingWindow1D,hammingWindow1D)
		hammingWindow3D = array_outer_product(hammingWindow2D,hammingWindow2D)

		# Multiply both cubes by the hamming window
		cubeInside      = np.multiply(cubeInside, hammingWindow3D)
		cubeOutside     = np.multiply(cubeOutside,hammingWindow3D)

		# Calculate spectrum of inside volume
		(dataF,   dataSpect)   = calculatePowerSpectrum(cubeInside)

		# Calculate spectrum of outside volume
		(dataBGF, dataBGSpect) = calculatePowerSpectrum(cubeOutside)

		del hammingWindow1D, hammingWindow2D, hammingWindow3D

		m, s = divmod(time() - tStart, 60)
		print "  :: Time elapsed: %d minutes and %.2f seconds" % (m, s)

		if scipionPrewhitenParams.get('force-stop', False):
			return locals()
		
		#   While: the user changes the elbow in the Pre-Whitening Interface, repeat: the pre-whitening.
		# 	This loop will stop when the user does NOT change the elbow in the interface.
		#	It is a bit of a hack, but it works completely within matplotlib (which is a relief)
		while newElbowAngstrom != oldElbowAngstrom or oldRampWeight != newRampWeight:

			preWhiteningResult = preWhitenCube( n = cubeSize,
									vxSize        = vxSize,
									elbowAngstrom = newElbowAngstrom,
									rampWeight    = newRampWeight,
									dataF         = dataF,
									dataBGF       = dataBGF,
									dataBGSpect   = dataBGSpect)

			cubeInsidePW = preWhiteningResult['dataPW']

			oldElbowAngstrom = newElbowAngstrom
			oldRampWeight    = newRampWeight
			
			
			if scipionPrewhitenParams.get('display', True):
				newElbowAngstrom, newRampWeight = displayPreWhitening(
									elbowAngstrom = oldElbowAngstrom,
									rampWeight    = oldRampWeight,
									dataSpect     = dataSpect,
									dataBGSpect   = dataBGSpect,
									peval         = preWhiteningResult['peval'],
									dataPWSpect   = preWhiteningResult['dataPWSpect'],
									dataPWBGSpect = preWhiteningResult['dataPWBGSpect'],
									vxSize 		  = vxSize,
									dataSlice     = cubeInside[int(cubeSize/2),:,:],
									dataPWSlice   = cubeInsidePW[int(cubeSize/2),:,:]
									)


		print '\n= Pre-whitening the Full Volume (this might take a bit of time...)'
		tStart = time()

		# Apply the pre-whitening filter on the full-sized map
		(dataF, dataPowerSpectrum) = calculatePowerSpectrum(data)
		if splitVolume == True:
			(dataDiffF, dataPowerSpectrumDoff) = calculatePowerSpectrum(dataDiff)

		pwFilterFinal = createPreWhiteningFilterFinal(	n = n,
											cubeSize      = cubeSize,
											spectrum      = dataPowerSpectrum,
											pcoef         = preWhiteningResult['pcoef'],
											elbowAngstrom = newElbowAngstrom,
											rampWeight    = newRampWeight,
											vxSize        = vxSize)

		dataPW     = np.real(fftpack.ifftn(fftpack.ifftshift(np.multiply(pwFilterFinal['pWfilter'],dataF))))
		if splitVolume == True:
			dataDiffPW = np.real(fftpack.ifftn(fftpack.ifftshift(np.multiply(pwFilterFinal['pWfilter'],dataDiffF))))

		m, s = divmod(time() - tStart, 60)
		print "  :: Time elapsed: %d minutes and %.2f seconds" % (m, s)

		if scipionPrewhitenParams.get('display', True):
			# Pre-whitening Results Plots
			f, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(18, 9))
			f.suptitle('Pre-Whitening Results', fontsize=14, color='#104E8B', fontweight='bold')
	
			vminData, vmaxData = np.min(data), np.max(data)
			ax1.imshow(data[(3*n/9),:,:], vmin=vminData, vmax=vmaxData, cmap=plt.cm.gray, interpolation="nearest")
			ax2.imshow(data[(4*n/9),:,:], vmin=vminData, vmax=vmaxData, cmap=plt.cm.gray, interpolation="nearest")
			ax3.imshow(data[(5*n/9),:,:], vmin=vminData, vmax=vmaxData, cmap=plt.cm.gray, interpolation="nearest")
			ax4.imshow(data[(6*n/9),:,:], vmin=vminData, vmax=vmaxData, cmap=plt.cm.gray, interpolation="nearest")
	
			vminDataPW, vmaxDataPW = np.min(dataPW), np.max(dataPW)
			ax5.imshow(dataPW[(3*n/9),:,:], vmin=vminDataPW, vmax=vmaxDataPW, cmap=plt.cm.gray, interpolation="nearest")
			ax6.imshow(dataPW[(4*n/9),:,:], vmin=vminDataPW, vmax=vmaxDataPW, cmap=plt.cm.gray, interpolation="nearest")
			ax7.imshow(dataPW[(5*n/9),:,:], vmin=vminDataPW, vmax=vmaxDataPW, cmap=plt.cm.gray, interpolation="nearest")
			ax8.imshow(dataPW[(6*n/9),:,:], vmin=vminDataPW, vmax=vmaxDataPW, cmap=plt.cm.gray, interpolation="nearest")
	
			ax1.set_title('Slice ' + str(int(3*n/9)), fontsize=10, color='#104E8B')
			ax2.set_title('Slice ' + str(int(4*n/9)), fontsize=10, color='#104E8B')
			ax3.set_title('Slice ' + str(int(5*n/9)), fontsize=10, color='#104E8B')
			ax4.set_title('Slice ' + str(int(6*n/9)), fontsize=10, color='#104E8B')
	
			ax5.set_title('Slice ' + str(int(3*n/9)), fontsize=10, color='#104E8B')
			ax6.set_title('Slice ' + str(int(4*n/9)), fontsize=10, color='#104E8B')
			ax7.set_title('Slice ' + str(int(5*n/9)), fontsize=10, color='#104E8B')
			ax8.set_title('Slice ' + str(int(6*n/9)), fontsize=10, color='#104E8B')
	
			ax1.set_ylabel('Input Volume\n\n',        fontsize=14, color='#104E8B', fontweight='bold')
			ax5.set_ylabel('Pre-whitened Volume\n\n', fontsize=14, color='#104E8B', fontweight='bold')
	
			plt.show()

		# Set the data to be the pre-whitened volume
		data     = dataPW
		if splitVolume == True:
			dataDiff = dataDiffPW
			del dataDiffF, dataDiffPW
		del dataF, dataPW

	else:

		print("\n=======================================================================\n"
				"|                                                                     |\n"
				"|                 ResMap Pre-Whitening (beta) Tool                    |\n"
				"|                                                                     |\n"
				"|                 The volume is of reasonable size.                   |\n"
				"|                                                                     |\n"
				"|        ResMap will run its pre-whitening on the whole volume        |\n"
				"|         by softly masking the background from the particle.         |\n"
				"|                                                                     |\n"
				"|               In split volume mode, ResMap will use                 |\n"
				"|             the difference map instead of a soft mask.              |\n"
				"|                                                                     |\n"
				"=======================================================================")

		if splitVolume == False:
			print '\n= Computing Soft Mask Separating Particle from Background'
			tStart = time()

			# Dilate the mask a bit so that we don't seep into the particle when we blur it later
			boxElement  = np.ones([5, 5, 5])
			dilatedMask = ndimage.morphology.binary_dilation(dataMask, structure=boxElement, iterations=3)
			dilatedMask = np.logical_and(dilatedMask, Rinside)

			# Blur the mask
			softBGmask  = filters.gaussian_filter(np.array(np.logical_not(dilatedMask),dtype='float32'),float(n)*0.02)

			# Get the background
			dataBG      = np.multiply(data,softBGmask)
			del boxElement, dataMask, dilatedMask

			m, s = divmod(time() - tStart, 60)
			print "  :: Time elapsed: %d minutes and %.2f seconds" % (m, s)
		else:
			dataBG = dataDiff

		print '\n= Calculating Spherically Averaged Power Spectra'
		tStart = time()

		# Calculate spectrum of input volume only if downsampled, otherwise use previous computation
		if LPFfactor != 0.0 or n > subVolLPF:
			(dataF, dataPowerSpectrum) = calculatePowerSpectrum(data)

		# Calculate spectrum of background volume
		(dataBGF, dataBGSpect) = calculatePowerSpectrum(dataBG)
		del dataBG

		m, s = divmod(time() - tStart, 60)
		print "  :: Time elapsed: %d minutes and %.2f seconds" % (m, s)
		
		if scipionPrewhitenParams.get('force-stop', False):
			return locals()

		#   While: the user changes the elbow in the Pre-Whitening Interface, repeat: the pre-whitening.
		# 	This loop will stop when the user does NOT change the elbow in the interface.
		#	It is a bit of a hack, but it works completely within matplotlib (which is a relief)
		while newElbowAngstrom != oldElbowAngstrom or oldRampWeight != newRampWeight:

			if splitVolume == False:
				preWhiteningResult = preWhitenVolumeSoftBG(n = n,
										elbowAngstrom = newElbowAngstrom,
										dataBGSpect   = dataBGSpect,
										dataF         = dataF,
										softBGmask    = softBGmask,
										vxSize        = vxSize,
										rampWeight    = newRampWeight)
			else:
				preWhiteningResult = preWhitenCube( n = n,
										vxSize        = vxSize,
										elbowAngstrom = newElbowAngstrom,
										rampWeight    = newRampWeight,
										dataF         = dataF,
										dataBGF       = dataBGF,
										dataBGSpect   = dataBGSpect)

			dataPW   = preWhiteningResult['dataPW']
			if splitVolume == True:
				dataBGPW = preWhiteningResult['dataBGPW']

			oldElbowAngstrom = newElbowAngstrom
			oldRampWeight    = newRampWeight

			if scipionPrewhitenParams.get('display', True):
				newElbowAngstrom, newRampWeight = displayPreWhitening(
									elbowAngstrom = oldElbowAngstrom,
									rampWeight    = oldRampWeight,
									dataSpect     = dataPowerSpectrum,
									dataBGSpect   = dataBGSpect,
									peval         = preWhiteningResult['peval'],
									dataPWSpect   = preWhiteningResult['dataPWSpect'],
									dataPWBGSpect = preWhiteningResult['dataPWBGSpect'],
									vxSize 		  = vxSize,
									dataSlice     = data[int(n/2),:,:],
									dataPWSlice   = dataPW[int(n/2),:,:]
									)

			del preWhiteningResult

		data     = dataPW
		if splitVolume == True:
			dataDiff = dataBGPW
			del dataBGPW
		del dataF, dataBGF, dataPW


	if splitVolume == False:
		return {'data': data, 
				'newElbowAngstrom': newElbowAngstrom,
				'newRampWeight': newRampWeight}
	else:
		return {'data': data, 
				'dataDiff': dataDiff,
				'newElbowAngstrom': newElbowAngstrom,
				'newRampWeight': newRampWeight}



























def computeMask(**kwargs):

	data        = kwargs.get('data', None)
	dataMask    = kwargs.get('dataMask', None)
	LPFfactor   = kwargs.get('LPFfactor', None)
	splitVolume = kwargs.get('splitVolume', False)

	print '\n= Computing Mask Separating Particle from Background'
	tStart = time()

	# Update n in case downsampling was done above
	n = data.shape[0]

	# We assume the particle is at the center of the volume
	# Create spherical mask
	R = createRmatrix(n)
	Rinside = R < n/2 - 1

	# Check to see if mask volume was provided
	if isinstance(dataMask,MRC_Data) == False:
		# Compute mask separating the particle from background
		dataBlurred  = filters.gaussian_filter(data, float(n)*0.02)	# kernel size 2% of n
		dataMask     = dataBlurred > np.max(dataBlurred)*5e-2		# threshold at 5% of max value
		del dataBlurred
	else:
		if LPFfactor == 0.0:
			dataMask = np.array(dataMask.matrix, dtype='bool')
		else:	# Interpolate the given mask
			dataMask = ndimage.interpolation.zoom(dataMask.matrix, LPFfactor, mode='reflect')
			dataMask = filters.gaussian_filter(dataMask, float(n)*0.02)	# kernel size 2% of n
			dataMask = dataMask > np.max(dataMask)*5e-2					# threshold at 5% of max value

	if splitVolume == False:
		mask = np.bitwise_and(dataMask,  R < n/2 - 9)	# backoff 9 voxels from edge (make adaptive later)
	else:
		tmp_box = np.zeros((n,n,n), dtype='bool')	# make cube that goes to 9 voxels to the edge
		tmp_box[9:-9,9:-9,9:-9] = True 						# | this is a hack for Liz Kellog's case
		mask = np.bitwise_and(dataMask, tmp_box)
		del tmp_box

	oldSumOfMask = np.sum(mask)

	m, s = divmod(time() - tStart, 60)
	print "  :: Time elapsed: %d minutes and %.2f seconds" % (m, s)

	return {'mask':mask, 'dataMask':dataMask, 'oldSumOfMask':oldSumOfMask,
					'Rinside': Rinside}


























def testLPF(**kwargs):

	# Get inputs
	data        = kwargs.get('data',     None)
	dataDiff    = kwargs.get('dataDiff', None)
	splitVolume = kwargs.get('splitVolume', False)
	vxSize      = kwargs.get('vxSize', 0.0)
	minRes      = kwargs.get('minRes', 0.0)
	maxRes      = kwargs.get('maxRes', 0.0)
	currentRes  = kwargs.get('currentRes', 0.0)
	subVolLPF   = kwargs.get('subVolLPF', 0.0)

	n = data.shape[0]

	print '\n\n= Testing Whether the Input Volume has been Low-pass Filtered\n'
	tStart = time()

	# If the volume is larger than subVolLPF^3, then do LPF test on smaller volume to save computation
	if n > subVolLPF:

		print ( "=======================================================================\n"
				"|                                                                     |\n"
				"|          The input volume is quite large ( >160 voxels).            |\n"
				"|                                                                     |\n"
				"|         ResMap will run its low-pass filtering test on a            |\n"
				"|       cube of size 160 taken from the center of the volume.         |\n"
				"|                                                                     |\n"
				"|        This is usually not a problem, but please notify the         |\n"
				"|                  authors if something goes awry.                    |\n"
				"|                                                                     |\n"
				"=======================================================================\n")

		mid  = int(n/2)
		midR = subVolLPF/2

		# Extract a cube from the middle of the density
		middleCube = data[mid-midR:mid+midR, mid-midR:mid+midR, mid-midR:mid+midR]

		# Create a 3D hamming window
		hammingWindow1D = signal.hamming(subVolLPF)
		hammingWindow2D = array_outer_product(hammingWindow1D,hammingWindow1D)
		hammingWindow3D = array_outer_product(hammingWindow2D,hammingWindow2D)
		del hammingWindow1D, hammingWindow2D

		# Apply the hamming window to the middle cube
		middleCube = np.multiply(middleCube,hammingWindow3D)

		# Calculate the Fourier spectrum and run the test
		(dataF, dataPowerSpectrum) = calculatePowerSpectrum(middleCube)
		LPFtest                    = isPowerSpectrumLPF(dataPowerSpectrum)
		del middleCube, hammingWindow3D
	else:
		(dataF, dataPowerSpectrum) = calculatePowerSpectrum(data)
		LPFtest                    = isPowerSpectrumLPF(dataPowerSpectrum)

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

		# Down-sample the volume using cubic splines
		data   = ndimage.interpolation.zoom(data, LPFfactor, mode='reflect')
		if splitVolume == True:
			dataDiff = ndimage.interpolation.zoom(dataDiff, LPFfactor, mode='reflect')
		vxSize = float(vxSize)/LPFfactor
	else:
		print ( "=======================================================================\n"
				"|                                                                     |\n"
				"|        The volume does not appear to be low-pass filtered.          |\n"
				"|                                                                     |\n"
				"|                              Great!                                 |\n"
				"|                                                                     |\n"
				"=======================================================================\n")
		LPFfactor = 0.0

	# Calculate min res
	if minRes <= (2.2*vxSize):
		minRes = round((2.2*vxSize)/0.1)*0.1 # round to the nearest 0.1
	currentRes = minRes

	# Calculate max res
	if maxRes == 0.0:
		maxRes = round((4.0*vxSize)/0.5)*0.5 # round to the nearest 0.5

	m, s = divmod(time() - tStart, 60)
	print "  :: Time elapsed: %d minutes and %.2f seconds" % (m, s)

	if splitVolume == False:
		return {'LPFfactor':LPFfactor, 'vxSize':vxSize,
						'minRes':minRes, 'maxRes':maxRes, 'currentRes':currentRes,
						'data':data,
						'dataF':dataF, 'dataPowerSpectrum':dataPowerSpectrum}
	else:
		return {'LPFfactor':LPFfactor, 'vxSize':vxSize,
						'minRes':minRes, 'maxRes':maxRes, 'currentRes':currentRes,
						'data':data, 'dataDiff':dataDiff,
						'dataF':dataF, 'dataPowerSpectrum':dataPowerSpectrum}


















def visualize2Doutput(**kwargs):

	# Get inputs
	dataOrig   = kwargs.get('dataOrig',  None)
	minRes     = kwargs.get('minRes', 0.0)
	maxRes     = kwargs.get('maxRes', 0.0)
	resTOTALma = kwargs.get('resTOTALma',  None)
	resHisto   = kwargs.get('resHisto', None)

	# Grab the volume size (assumed to be a cube)
	orig_n = dataOrig.shape[0]
	n     = resTOTALma.shape[0]

	# Plots
	f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
	f.suptitle('Slices Through Input Volume', fontsize=14, color='#104E8B', fontweight='bold')
	vminData, vmaxData = np.min(dataOrig), np.max(dataOrig)
	ax1.imshow(dataOrig[int(3*orig_n/9),:,:], vmin=vminData, vmax=vmaxData, cmap=plt.cm.gray, interpolation="nearest")
	ax2.imshow(dataOrig[int(4*orig_n/9),:,:], vmin=vminData, vmax=vmaxData, cmap=plt.cm.gray, interpolation="nearest")
	ax3.imshow(dataOrig[int(5*orig_n/9),:,:], vmin=vminData, vmax=vmaxData, cmap=plt.cm.gray, interpolation="nearest")
	ax4.imshow(dataOrig[int(6*orig_n/9),:,:], vmin=vminData, vmax=vmaxData, cmap=plt.cm.gray, interpolation="nearest")

	ax1.set_title('Slice ' + str(int(3*n/9)), fontsize=10, color='#104E8B')
	ax2.set_title('Slice ' + str(int(4*n/9)), fontsize=10, color='#104E8B')
	ax3.set_title('Slice ' + str(int(5*n/9)), fontsize=10, color='#104E8B')
	ax4.set_title('Slice ' + str(int(6*n/9)), fontsize=10, color='#104E8B')

	f2, ((ax21, ax22), (ax23, ax24)) = plt.subplots(2, 2)
	f2.suptitle('Slices Through ResMap Results', fontsize=14, color='#104E8B', fontweight='bold')
	# ax21.imshow(dataOrig[int(3*orig_n/9),:,:], cmap=plt.cm.gray, interpolation="nearest")
	# ax22.imshow(dataOrig[int(4*orig_n/9),:,:], cmap=plt.cm.gray, interpolation="nearest")
	# ax23.imshow(dataOrig[int(5*orig_n/9),:,:], cmap=plt.cm.gray, interpolation="nearest")
	# ax24.imshow(dataOrig[int(6*orig_n/9),:,:], cmap=plt.cm.gray, interpolation="nearest")
	vminRes, vmaxRes = minRes, maxRes
	im = ax21.imshow(resTOTALma[int(3*orig_n/9),:,:], vmin=vminRes, vmax=vmaxRes, cmap=plt.cm.jet, interpolation="nearest")#, alpha=0.25)
	ax22.imshow(     resTOTALma[int(4*orig_n/9),:,:], vmin=vminRes, vmax=vmaxRes, cmap=plt.cm.jet, interpolation="nearest")#, alpha=0.25)
	ax23.imshow(     resTOTALma[int(5*orig_n/9),:,:], vmin=vminRes, vmax=vmaxRes, cmap=plt.cm.jet, interpolation="nearest")#, alpha=0.25)
	ax24.imshow(     resTOTALma[int(6*orig_n/9),:,:], vmin=vminRes, vmax=vmaxRes, cmap=plt.cm.jet, interpolation="nearest")#, alpha=0.25)

	ax21.set_title('Slice ' + str(int(3*n/9)), fontsize=10, color='#104E8B')
	ax22.set_title('Slice ' + str(int(4*n/9)), fontsize=10, color='#104E8B')
	ax23.set_title('Slice ' + str(int(5*n/9)), fontsize=10, color='#104E8B')
	ax24.set_title('Slice ' + str(int(6*n/9)), fontsize=10, color='#104E8B')

	cax = f2.add_axes([0.9, 0.1, 0.03, 0.8])
	f2.colorbar(im, cax=cax)

	# Histogram
	f3   = plt.figure()
	f3.suptitle('Histogram of ResMap Results', fontsize=14, color='#104E8B', fontweight='bold')
	axf3 = f3.add_subplot(111)

	axf3.bar(range(len(resHisto)), resHisto.values(), align='center')
	axf3.set_xlabel('Resolution (Angstroms)')
	axf3.set_xticks(range(len(resHisto)))
	axf3.set_xticklabels(resHisto.keys())
	axf3.set_ylabel('Number of Voxels')

	plt.show()




