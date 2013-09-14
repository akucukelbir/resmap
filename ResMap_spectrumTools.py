'''
ResMap_spectrumToolks: module containing spectral processing functions for ResMap algorithm (Alp Kucukelbir, 2013)

Description of functions:
			   preWhitenVolume: attempts to flatten frequencies beyond a given "elbow" value
		   displayPreWhitening: display the quasi-interactive Pre-Whitening Interface
		  displayPowerSpectrum: debugging tool
  			isPowerSpectrumLPF: attempts to determine whether there is a low-pass drop in the spectrum
	    calculatePowerSpectrum: calculates the radially averaged power spectrum of a volume

Requirements:
	scipy
	numpy
	matplotlib

Please see individual functions for attributions.
'''
from time import time

import numpy as np
from scipy import signal
from scipy import fftpack
from scipy import ndimage

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.widgets import Slider

from ResMap_sphericalProfile import sphericalAverage
from ResMap_helpers import createRmatrix

def createPreWhiteningFilter(**kwargs):

	epsilon = 1e-10

	spectrum      = kwargs.get('spectrum', 0)
	elbowAngstrom = kwargs.get('elbowAngstrom', 0)
	rampWeight    = kwargs.get('rampWeight',1.0)
	vxSize        = kwargs.get('vxSize', 0)
	n             = kwargs.get('n', 0)

	# Create R matrix
	R = createRmatrix(n)

	# Create the x and y variables for the polynomial regression
	xpoly = np.array(range(1,spectrum.size + 1))
	ypoly = np.log(np.sqrt(spectrum))

	# Create the index of frequencies (depends on vxSize)
	Fs     = 1/vxSize
	Findex = 1/( Fs/2 * np.linspace(epsilon, 1, xpoly.size) )

	# Find the points of interest
	indexElbow    = np.argmin((Findex-elbowAngstrom)**2)
	indexStart    = np.argmin((Findex-(1.05*elbowAngstrom))**2)
	indexNyquist  = xpoly[-1]

	# Create the weighting function to do a weighted fit
	wpoly =  np.array(np.bitwise_and(xpoly>indexElbow, xpoly<indexNyquist), dtype='float32')
	wpoly += 0.5*np.array(np.bitwise_and(xpoly>indexStart, xpoly<=indexElbow), dtype='float32')

	# Do the polynomial fit
	pcoef = np.polynomial.polynomial.polyfit(xpoly, ypoly, 2, w=wpoly)
	peval = np.polynomial.polynomial.polyval(xpoly, pcoef)

	# Don't change any frequencies outside of indexStart to indexNyquist
	R[R<indexStart]   = indexStart
	R[R>indexNyquist] = indexNyquist

	# Create the pre-whitening filter
	pWfilter  = np.exp(np.polynomial.polynomial.polyval(R,-1.0*rampWeight*pcoef))

	del R

	return {'peval':peval, 'pcoef':pcoef, 'pWfilter': pWfilter}

def createPreWhiteningFilterFinal(**kwargs):

	epsilon = 1e-10

	spectrum      = kwargs.get('spectrum', 0)
	pcoef         = kwargs.get('pcoef', 0)
	elbowAngstrom = kwargs.get('elbowAngstrom', 0)
	rampWeight    = kwargs.get('rampWeight',1.0)
	vxSize        = kwargs.get('vxSize', 0)
	n             = kwargs.get('n', 0)
	cubeSize      = kwargs.get('cubeSize', 0)

	R = createRmatrix(n) 

	# Create the x and y variables for the polynomial regression
	xpoly = np.array(range(1,spectrum.size + 1))

	# Create the index of frequencies (depends on vxSize)
	Fs     = 1/vxSize
	Findex = 1/( Fs/2 * np.linspace(epsilon, 1, xpoly.size) )

	# Find the points of interest
	indexStart    = np.argmin((Findex-(1.05*elbowAngstrom))**2)
	indexNyquist  = xpoly[-1]

	# Don't change any frequencies outside of indexStart to indexNyquist
	R[R<indexStart]   = indexStart
	R[R>indexNyquist] = indexNyquist

	# Rescale R such that the polynomial from the cube fit makes sense
	R = R/(float(n)/(cubeSize-1))

	# Create the pre-whitening filter
	pWfilter  = np.exp(np.polynomial.polynomial.polyval(R,-1.0*rampWeight*pcoef))

	del R

	return {'pWfilter': pWfilter}	

def preWhitenVolumeSoftBG(**kwargs):

	print '\n= Pre-whitening'
	tStart = time()		

	n             = kwargs.get('n', 0)
	elbowAngstrom = kwargs.get('elbowAngstrom', 0)
	dataBGSpect   = kwargs.get('dataBGSpect', 0)
	dataF         = kwargs.get('dataF', 0)
	softBGmask    = kwargs.get('softBGmask', 0)
	vxSize        = kwargs.get('vxSize', 0)
	rampWeight    = kwargs.get('rampWeight',1.0)

	epsilon = 1e-10

	pWfilter = createPreWhiteningFilter(n             = n,
										spectrum      = dataBGSpect,
										elbowAngstrom = elbowAngstrom,
										rampWeight    = rampWeight,
										vxSize        = vxSize)

	# Apply the pre-whitening filter
	dataF       = np.multiply(pWfilter['pWfilter'],dataF)

	dataPWFabs  = np.abs(dataF)
	dataPWFabs  = dataPWFabs-np.min(dataPWFabs)
	dataPWFabs  = dataPWFabs/np.max(dataPWFabs)
	dataPWSpect = sphericalAverage(dataPWFabs**2) + epsilon

	dataPW = np.real(fftpack.ifftn(fftpack.ifftshift(dataF)))
	del dataF

	dataPWBG     = np.multiply(dataPW,softBGmask)
	dataPWBG     = np.array(fftpack.fftshift(fftpack.fftn(dataPWBG,overwrite_x=True)), dtype='complex64')
	dataPWBGFabs = np.abs(dataPWBG)
	del dataPWBG

	dataPWBGFabs  = dataPWBGFabs-np.min(dataPWBGFabs)
	dataPWBGFabs  = dataPWBGFabs/np.max(dataPWBGFabs)
	dataPWBGSpect = sphericalAverage(dataPWBGFabs**2) + epsilon

	m, s = divmod(time() - tStart, 60)
	print "  :: Time elapsed: %d minutes and %.2f seconds" % (m, s)

	return {'dataPW':dataPW, 'dataPWSpect': dataPWSpect, 'dataPWBGSpect': dataPWBGSpect, 'peval': pWfilter['peval'] }

def preWhitenCube(**kwargs):

	print '\n= Pre-whitening the Cubes'
	tStart = time()		

	n             = kwargs.get('n', 0)
	vxSize        = kwargs.get('vxSize', 0)
	elbowAngstrom = kwargs.get('elbowAngstrom', 0)
	rampWeight    = kwargs.get('rampWeight',1.0)
	dataF         = kwargs.get('dataF', 0)
	dataBGF       = kwargs.get('dataBGF', 0)
	dataBGSpect   = kwargs.get('dataBGSpect', 0)

	epsilon = 1e-10

	pWfilter = createPreWhiteningFilter(n             = n,
										spectrum      = dataBGSpect,
										elbowAngstrom = elbowAngstrom,
										rampWeight    = rampWeight,
										vxSize        = vxSize)

	# Apply the pre-whitening filter to the inside cube
	dataF       = np.multiply(pWfilter['pWfilter'],dataF)

	dataPWFabs  = np.abs(dataF)
	dataPWFabs  = dataPWFabs-np.min(dataPWFabs)
	dataPWFabs  = dataPWFabs/np.max(dataPWFabs)
	dataPWSpect = sphericalAverage(dataPWFabs**2) + epsilon

	dataPW = np.real(fftpack.ifftn(fftpack.ifftshift(dataF)))
	del dataF

	# Apply the pre-whitening filter to the outside cube
	dataBGF       = np.multiply(pWfilter['pWfilter'],dataBGF)

	dataPWBGFabs  = np.abs(dataBGF)
	dataPWBGFabs  = dataPWBGFabs-np.min(dataPWBGFabs)
	dataPWBGFabs  = dataPWBGFabs/np.max(dataPWBGFabs)
	dataPWBGSpect = sphericalAverage(dataPWBGFabs**2) + epsilon

	m, s = divmod(time() - tStart, 60)
	print "  :: Time elapsed: %d minutes and %.2f seconds" % (m, s)

	return {'dataPW':dataPW, 'dataPWSpect': dataPWSpect, 'dataPWBGSpect': dataPWBGSpect, 'peval': pWfilter['peval'], 'pcoef': pWfilter['pcoef'] }

# def preWhitenVolumeFinal(**kwargs):

# 	print '\n= Pre-whitening'
# 	tStart = time()		

# 	n             = kwargs.get('n', 0)
# 	rampWeight    = kwargs.get('rampWeight',1.0)
# 	pcoef         = kwargs.get('pcoef', 0)
# 	dataF         = kwargs.get('dataF', 0)

# 	epsilon = 1e-10

# 	pWfilter = createPreWhiteningFilter(n             = n,
# 										spectrum      = dataBGSpect,
# 										elbowAngstrom = elbowAngstrom,
# 										rampWeight    = rampWeight,
# 										vxSize        = vxSize)

# 	# Apply the pre-whitening filter
# 	dataF  = np.multiply(pWfilter['pWfilter'],dataF)
# 	dataPW = np.real(fftpack.ifftn(fftpack.ifftshift(dataF)))
# 	del dataF

# 	m, s = divmod(time() - tStart, 60)
# 	print "  :: Time elapsed: %d minutes and %.2f seconds" % (m, s)

# 	return {'dataPW':dataPW}

def preWhitenVolumeSplit(x,y,z, **kwargs):

	print '\n= Pre-whitening Volume'
	tStart = time()		

	R = np.sqrt(x**2 + y**2 + z**2)
	del x, y, z

	elbowAngstrom = kwargs.get('elbowAngstrom', 0)
	dataBGSpect   = kwargs.get('dataBGSpect', 0)
	dataAF        = kwargs.get('dataAF', 0)
	dataBF        = kwargs.get('dataBF', 0)
	vxSize        = kwargs.get('vxSize', 0)
	rampWeight    = kwargs.get('rampWeight',1.0)

	epsilon = 1e-20

	# Create the x and y variables for the polynomial regression
	xpoly = np.array(range(1,dataBGSpect.size + 1))
	ypoly = np.log(np.sqrt(dataBGSpect))

	# Find the index at which the spectrum hits certain frequencies
	Fs     = 1/vxSize
	Findex = 1/( Fs/2 * np.linspace(epsilon, 1, xpoly.size) )

	# Find the points of interest
	indexElbow    = np.argmin((Findex-elbowAngstrom)**2)
	indexStart    = np.argmin((Findex-(1.05*elbowAngstrom))**2)
	indexNyquist  = xpoly[-1]

	# Create the weighting function (binary, in this case) to do a weighted fit
	wpoly =  np.array(np.bitwise_and(xpoly>indexElbow, xpoly<indexNyquist), dtype='float32')
	wpoly += 0.5*np.array(np.bitwise_and(xpoly>indexStart, xpoly<=indexElbow), dtype='float32')

	# Do the polynomial fit
	pcoef = np.polynomial.polynomial.polyfit(xpoly, ypoly, 2, w=wpoly)
	peval = np.polynomial.polynomial.polyval(xpoly, pcoef)

	# Don't change any frequencies beyond indexStart to indexNyquist
	R[R<indexStart]   = indexStart
	R[R>indexNyquist] = indexNyquist

	# Create the pre-whitening filter
	pWfilter  = np.exp(np.polynomial.polynomial.polyval(R,-1.0*rampWeight*pcoef))

	# Apply the pre-whitening filter
	dataAF = np.multiply(pWfilter,dataAF)
	dataBF = np.multiply(pWfilter,dataBF)
	del pWfilter, R

	# dataPWFabs  = np.abs(dataF)
	# dataPWFabs  = dataPWFabs-np.min(dataPWFabs)
	# dataPWFabs  = dataPWFabs/np.max(dataPWFabs)
	# dataPWSpect = sphericalAverage(dataPWFabs**2) + epsilon

	dataAPW = np.real(fftpack.ifftn(fftpack.ifftshift(dataAF)))
	dataBPW = np.real(fftpack.ifftn(fftpack.ifftshift(dataBF)))
	del dataAF, dataBF

	# dataPWBG      = np.multiply(dataPW,softBGmask)
	# dataPWBG     = np.array(fftpack.fftshift(fftpack.fftn(dataPWBG,overwrite_x=True)), dtype='complex64')
	# dataPWBGFabs  = np.abs(dataPWBG)
	# del dataPWBG

	# dataPWBGFabs  = dataPWBGFabs-np.min(dataPWBGFabs)
	# dataPWBGFabs  = dataPWBGFabs/np.max(dataPWBGFabs)
	# dataPWBGSpect = sphericalAverage(dataPWBGFabs**2) + epsilon

	m, s = divmod(time() - tStart, 60)
	print "  :: Time elapsed: %d minutes and %.2f seconds" % (m, s)

	return {'dataAPW':dataAPW, 'dataBPW':dataBPW, 'peval':peval, 'xpoly':xpoly}


def displayPreWhitening(**kwargs):

	elbowAngstrom = kwargs.get('elbowAngstrom',0)
	rampWeight    = kwargs.get('rampWeight', 1.0)
	dataSpect     = kwargs.get('dataSpect', 0)
	dataBGSpect   = kwargs.get('dataBGSpect', 0)
	peval         = kwargs.get('peval', 0)
	dataPWSpect   = kwargs.get('dataPWSpect', 0)
	dataPWBGSpect = kwargs.get('dataPWBGSpect', 0)
	vxSize        = kwargs.get('vxSize', 0)
	dataSlice     = kwargs.get('dataSlice', 0)	
	dataPWSlice   = kwargs.get('dataPWSlice', 0)

	xpoly = np.array(range(1,dataBGSpect.size + 1))

	# Figure
	fig = plt.figure(figsize=(18, 9))
	fig.suptitle('\nResMap Pre-Whitening Interface (beta)', fontsize=20, color='#104E8B', fontweight='bold')
	ax1 = plt.subplot2grid((2,3), (0,0), colspan=2)
	ax2 = plt.subplot2grid((2,3), (1, 0))
	ax3 = plt.subplot2grid((2,3), (1, 1))
	axtext = plt.subplot2grid((2,3), (1, 2))
	
	# Slider for elbow
	axcolor = 'lightgoldenrodyellow'
	axelbow = plt.axes([0.7, 0.65, 0.2, 0.03], axisbg=axcolor)
	selbow  = Slider(axelbow, 'Angstrom', 2.1*vxSize, 100, valinit=elbowAngstrom)

	# Slider for rampWeight
	axramp = plt.axes([0.7, 0.55, 0.2, 0.03], axisbg=axcolor)
	sramp  = Slider(axramp, 'Ramp Weight', 0.0, 1.0, valinit=rampWeight)


	# Instructions
	axtext.set_title('INSTRUCTIONS', color='#104E8B', fontweight='bold')
	axtext.get_xaxis().set_visible(False)
	axtext.get_yaxis().set_visible(False)
	axtext.text(0.5, 0.5, 
		'Please check that the green line\nis as straight as possible,\nat least in the high frequencies.\n\nIf not, adjust the sliders\nabove and close the window.\n\nResMap will try to\n pre-whiten the volume again.\n\nTo continue, close the window\n without adjusting anything.',
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=14,
        transform=axtext.transAxes)

	# Spectra
	ax1.plot(xpoly, dataSpect,		lw=2, color='b', label='Input Map')
	ax1.plot(xpoly, dataBGSpect,		lw=2, color='c', label='Background of Input Map')
	ax1.plot(xpoly, np.exp(peval)**2,	lw=2, color='r', linestyle='dashed', label='Fitted Line')
	ax1.plot(xpoly, dataPWSpect,		lw=2, color='m', label='Pre-Whitened Map')
	ax1.plot(xpoly, dataPWBGSpect,	lw=2, color='g', label='Background of Pre-Whitened Map')

	Fs     = 1/vxSize
	tmp    = 1/( Fs/2 * np.linspace(1e-2, 1, int(xpoly.size/6)) ) 
	ax1.set_xticks( np.linspace(1,xpoly.size,tmp.size) )
	ax1.set_xticklabels( ["%.1f" % member for member in tmp]  )
	del tmp 

	tmp = np.concatenate((dataSpect, dataBGSpect, dataPWSpect, dataPWBGSpect))
	ax1.set_ylabel('Power Spectrum (|f|^2)')
	ax1.set_xlabel('Angstrom')
	ax1.set_yscale('log')
	ax1.set_ylim((np.min(tmp), np.max(tmp)))
	ax1.grid(linestyle='dotted')
	ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

	# Slices through volumes
	ax2.imshow(dataSlice,   cmap=plt.cm.gray, interpolation="nearest")
	ax3.imshow(dataPWSlice, cmap=plt.cm.gray, interpolation="nearest")
	ax2.set_title('Middle Slice of Input Map')
	ax3.set_title('Middle Slice of Pre-Whitened Map')

	plt.show()

	return (selbow.val, sramp.val)

def displayPowerSpectrum(*args):

	fig = plt.figure(1)
	ax = fig.add_subplot(111)
	colors = iter(cm.rainbow(np.linspace(0, 1, len(args))))

	for a in args:
		dataPowerSpectrum = calculatePowerSpectrum(a.matrix)

		n     = dataPowerSpectrum.size
		xpoly = np.array(range(1,n + 1))

		p = ax.plot(xpoly, dataPowerSpectrum, color=next(colors), label=a.name)

		Fs     = 1/a.data_step[0]
		tmp    = 1/( Fs/2 * np.linspace(1e-2, 1, int(xpoly.size/6)) ) 
		ax.set_xticks( np.linspace(1,xpoly.size,tmp.size) )
		ax.set_xticklabels( ["%.1f" % member for member in tmp]  )
		del tmp 

	plt.yscale('log')
	plt.grid(linestyle='dotted')
	plt.ylabel('Power Spectrum (|f|^2)')
	plt.xlabel('Frequency')
	plt.legend(loc=3)

	plt.show()

def isPowerSpectrumLPF(dataPowerSpectrum):

	# Calculated derivative of log of dataPowerSpectrum
	# smoothedLogSpectrum  = ndimage.filters.gaussian_filter1d(np.log(dataPowerSpectrum), 0.5, mode='nearest')
	diffLogPowerSpectrum = np.diff(np.log(dataPowerSpectrum))

	# Find positive peaks in the derivative
	peakInd = signal.find_peaks_cwt(-1*diffLogPowerSpectrum, np.arange(1,10), min_snr=2)

	# Pick out the maximum radius index where a peak occurs
	maxInd = np.max(peakInd)

	# print peakInd
	# print maxInd

	# fig = plt.figure(1)
	# ax = fig.add_subplot(111)
	# p = ax.plot(-1*diffLogPowerSpectrum)
	# # plt.yscale('log')
	# plt.grid(linestyle='dotted')
	# plt.ylabel('Power Spectrum (|f|^2)')
	# plt.xlabel('Frequency')
	# plt.show()

	# Calculate the mean and variance of the derivative of the power spectrum beyond maxInd
	m, v   = np.mean(diffLogPowerSpectrum[maxInd+2:]), np.var(diffLogPowerSpectrum[maxInd+2:])

	# If the mean and variance are basically zero after maxInd, it is highly likely that the volume was low-pass filtered
	thr = 1e-4
	if abs(m) < thr and v < thr:
		return {'outcome':True, 'factor': float(maxInd-1)/dataPowerSpectrum.size}
	else:
		return {'outcome':False, 'factor': 0.0}

def calculatePowerSpectrum(data):
	
	epsilon = 1e-10

	dataF     = np.array(fftpack.fftshift(fftpack.fftn(data)), dtype='complex64')
	dataFabs  = np.abs(dataF)
	dataFabs  = dataFabs-np.min(dataFabs)
	dataFabs  = dataFabs/np.max(dataFabs)

	dataPowerSpectrum = sphericalAverage(dataFabs**2) + epsilon
	del dataFabs

	return (dataF, dataPowerSpectrum)

