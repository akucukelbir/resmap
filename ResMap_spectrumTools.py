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

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.widgets import Slider

from ResMap_sphericalProfile import sphericalAverage

def preWhitenVolume(R, Rorig, **kwargs):

	print '\n= Pre-whitening Volume'
	tStart = time()		

	elbowAngstrom = kwargs.get('elbowAngstrom',0)
	dataBGSpect   = kwargs.get('dataBGSpect', 0)
	dataF         = kwargs.get('dataF', 0)
	softBGmask    = kwargs.get('softBGmask', 0)
	vxSize        = kwargs.get('vxSize', 0)

	epsilon = 1e-20

	# Create the x and y variables for the polynomial regression
	xpoly = np.array(range(1,dataBGSpect.size + 1))
	ypoly = np.log(dataBGSpect)

	# Find the index at which the spectrum hits certain frequencies
	Fs     = 1/vxSize
	Findex = 1/( Fs/2 * np.linspace(epsilon, 1, xpoly.size) )

	# Find the points of interest
	indexElbow    = np.argmin((Findex-elbowAngstrom)**2)
	indexStart    = np.argmin((Findex-(1.2*elbowAngstrom))**2)
	indexNyquist  = xpoly[-1]

	# Create the weighting function (binary, in this case) to do a weighted fit
	wpoly =  np.array(np.bitwise_and(xpoly>indexElbow, xpoly<indexNyquist), dtype='float32')
	wpoly += 0.8*np.array(np.bitwise_and(xpoly>indexStart, xpoly<=indexElbow), dtype='float32')

	# Do the polynomial fit
	pcoef = np.polynomial.polynomial.polyfit(xpoly, ypoly, 2, w=wpoly)
	peval = np.polynomial.polynomial.polyval(xpoly, pcoef)

	# Don't change any frequencies beyond indexStart to indexNyquist
	R[Rorig<indexStart]   = indexStart
	R[Rorig>indexNyquist] = indexNyquist

	# Create the pre-whitening filter
	Reval     = np.polynomial.polynomial.polyval(R,-1.0*pcoef)
	pWfilter  = np.exp(Reval)

	# Apply the pre-whitening filter
	dataPWF     = pWfilter*dataF
	dataPWFabs  = np.array(np.abs(dataPWF), dtype='float32')
	dataPWFabs  = dataPWFabs-np.min(dataPWFabs)
	dataPWFabs  = dataPWFabs/np.max(dataPWFabs)
	dataPWSpect = sphericalAverage(dataPWFabs) + epsilon

	dataPW = np.real(np.fft.ifftn(np.fft.ifftshift(dataPWF)))

	dataPWBG      = np.multiply(dataPW,softBGmask)
	dataPWBGF     = np.fft.fftshift(np.fft.fftn(dataPWBG))
	dataPWBGFabs  = np.array(np.abs(dataPWBGF), dtype='float32')
	dataPWBGFabs  = dataPWBGFabs-np.min(dataPWBGFabs)
	dataPWBGFabs  = dataPWBGFabs/np.max(dataPWBGFabs)
	dataPWBGSpect = sphericalAverage(dataPWBGFabs) + epsilon

	m, s = divmod(time() - tStart, 60)
	print "  :: Time elapsed: %d minutes and %.2f seconds" % (m, s)

	return {'dataPW':dataPW, 'dataPWSpect': dataPWSpect, 'dataPWBGSpect': dataPWBGSpect, 'peval':peval, 'xpoly':xpoly}

def displayPreWhitening(**kwargs):

	elbowAngstrom = kwargs.get('elbowAngstrom',0)
	dataSpect     = kwargs.get('dataSpect', 0)
	dataBGSpect   = kwargs.get('dataBGSpect', 0)
	peval         = kwargs.get('peval', 0)
	dataPWSpect   = kwargs.get('dataPWSpect', 0)
	dataPWBGSpect = kwargs.get('dataPWBGSpect', 0)
	xpoly         = kwargs.get('xpoly', 0)
	vxSize        = kwargs.get('vxSize')
	dataSlice     = kwargs.get('dataSlice', 0)	
	dataPWSlice   = kwargs.get('dataPWSlice', 0)	

	# Figure
	fig = plt.figure(figsize=(16, 9))
	fig.suptitle('\nResMap Pre-Whitening Interface (beta)', fontsize=20, color='#104E8B', fontweight='bold')
	ax1 = plt.subplot2grid((2,3), (0,0), colspan=2)
	ax2 = plt.subplot2grid((2,3), (1, 0))
	ax3 = plt.subplot2grid((2,3), (1, 1))
	axtext = plt.subplot2grid((2,3), (1, 2))
	
	# Slider
	axcolor = 'lightgoldenrodyellow'
	axelbow = plt.axes([0.7, 0.65, 0.2, 0.03], axisbg=axcolor)
	selbow  = Slider(axelbow, 'Angstrom', 2.1*vxSize, 60, valinit=elbowAngstrom)

	# Instructions
	axtext.set_title('INSTRUCTIONS', color='#104E8B', fontweight='bold')
	axtext.get_xaxis().set_visible(False)
	axtext.get_yaxis().set_visible(False)
	axtext.text(0.5, 0.5, 
		'Please check that the green line\nis as straight as possible,\nat least in the high frequencies.\n\nIf not, adjust the passband\nabove and close the window.\n\nResMap will try to\n pre-whiten the volume again.\n\nTo continue, close the window\n without adjusting anything.',
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=14,
        transform=axtext.transAxes)

	# Spectra
	ax1.plot(xpoly, dataSpect**2,		lw=2, color='b', label='Input Map')
	ax1.plot(xpoly, dataBGSpect**2,		lw=2, color='c', label='Background of Input Map')
	ax1.plot(xpoly, np.exp(peval)**2,	lw=2, color='r', linestyle='dashed', label='Fitted Line')
	ax1.plot(xpoly, dataPWSpect**2,		lw=2, color='m', label='Pre-Whitened Map')
	ax1.plot(xpoly, dataPWBGSpect**2,	lw=2, color='g', label='Background of Pre-Whitened Map')

	Fs     = 1/vxSize
	tmp    = 1/( Fs/2 * np.linspace(1e-2, 1, int(xpoly.size/6)) ) 
	ax1.set_xticks( np.linspace(1,xpoly.size,tmp.size) )
	ax1.set_xticklabels( ["%.1f" % member for member in tmp]  )
	del tmp 

	ax1.set_ylabel('Power Spectrum (|f|^2)')
	ax1.set_xlabel('Angstrom')
	ax1.set_yscale('log')
	ax1.grid(linestyle='dotted')
	ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

	# Slices through volumes
	ax2.imshow(dataSlice,   cmap=plt.cm.gray, interpolation="nearest")
	ax3.imshow(dataPWSlice, cmap=plt.cm.gray, interpolation="nearest")
	ax2.set_title('Middle Slice of Input Map')
	ax3.set_title('Middle Slice of Pre-Whitened Map')

	plt.show()

	return selbow.val

def displayPowerSpectrum(*args):

	plt.figure(1)
	colors = iter(cm.rainbow(np.linspace(0, 1, len(args))))

	for a in args:
		dataPowerSpectrum = calculatePowerSpectrum(a.matrix)

		n     = dataPowerSpectrum.size
		xpoly = np.array(range(1,n + 1))

		p = plt.plot(xpoly, dataPowerSpectrum, color=next(colors), label=a.name)

	plt.yscale('log')
	plt.ylabel('Power (normalized log scale)')
	plt.xlabel('Frequency')
	plt.legend(loc=3)

	plt.show()

def isPowerSpectrumLPF(dataPowerSpectrum):

	# Calculated derivative of log of dataPowerSpectrum
	diffLogPowerSpectrum = np.diff(np.log(dataPowerSpectrum))

	# plt.figure(2)
	# p2 = plt.plot(diffLogPowerSpectrum,'r')
	# plt.show()

	# Find positive peaks in the derivative
	peakInd = signal.find_peaks_cwt(-1*diffLogPowerSpectrum, np.arange(1,10), min_snr=2)

	# Pick out the maximum radius index where a peak occurs
	maxInd = np.max(peakInd)

	# Calculate the mean and variance of the derivative of the power spectrum beyond maxInd
	m, v   = np.mean(diffLogPowerSpectrum[maxInd+1:]), np.var(diffLogPowerSpectrum[maxInd+1:])

	# If the mean and variance are basically zero after maxInd, it is highly likely that the volume was low-pass filtered
	thr = 1e-6
	if abs(m) < thr and v < thr:
		return {'outcome':True, 'factor': float(maxInd)/dataPowerSpectrum.size}
	else:
		return {'outcome':False, 'factor': 0.0}

def calculatePowerSpectrum(data):
	
	epsilon = 1e-20

	dataF     = np.fft.fftshift(np.fft.fftn(data))
	dataFabs  = np.array(np.abs(dataF), dtype='float32')
	dataFabs  = dataFabs-np.min(dataFabs)
	dataFabs  = dataFabs/np.max(dataFabs)

	dataPowerSpectrum = sphericalAverage(dataFabs**2) + epsilon

	return dataPowerSpectrum

