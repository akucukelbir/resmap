'''
ResMap_spectrumToolks: module containing spectral processing functions
											 for ResMap algorithm (Alp Kucukelbir, 2013)

Description of functions:
	    createPreWhiteningFilter: fits a polynomial to a spectrum and
	    													returns a whitening filter
 createPreWhiteningFilterFinal: take a fitted polynomial and returns a
 																whitening filter

		 preWhitenVolumeSoftBG: attempts to pre-whiten using soft background
		 												mask for noise estimate
		         preWhitenCube: attempts to pre-whiten using a cube taken
		         								from the difference map
		   displayPreWhitening: display the quasi-interactive
		   											Pre-Whitening Interface
		  displayPowerSpectrum: debugging tool
  			isPowerSpectrumLPF: attempts to determine whether there is a
  													low-pass drop in the spectrum

	    calculatePowerSpectrum: calculates the radially averaged
	    												power spectrum of a volume

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

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.widgets import Slider
from matplotlib.widgets import Button
from matplotlib.gridspec import GridSpec

from ResMap_sphericalProfile import sphericalAverage
from ResMap_helpers import createRmatrix

def createPreWhiteningFilter(**kwargs):
	'''
	Creates a pre-whitening filter in 3D. Fits a polynomial to the spectrum
	beyond the "elbowAngstrom" frequency. Returns a whitening filter that
	can be adjusted using the "rampWeight." (Alp Kucukelbir, 2013)

	'''
	epsilon = 1e-10

	spectrum      = kwargs.get('spectrum', 0)
	elbowAngstrom = kwargs.get('elbowAngstrom', 0)
	rampWeight    = kwargs.get('rampWeight',1.0)
	vxSize        = kwargs.get('vxSize', 0)
	n             = kwargs.get('n', 0)

	# Create radius matrix
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
	wpoly =  np.array(np.bitwise_and(xpoly>indexElbow, xpoly<indexNyquist),    dtype='float32')
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
	'''
	Creates a pre-whitening filter in 3D. Expects a fitted polynomial
	defined by its "pcoef". Returns a whitening filter that
	can be adjusted using the "rampWeight." (Alp Kucukelbir, 2013)

	'''
	epsilon = 1e-10

	spectrum      = kwargs.get('spectrum', 0)
	pcoef         = kwargs.get('pcoef', 0)
	elbowAngstrom = kwargs.get('elbowAngstrom', 0)
	rampWeight    = kwargs.get('rampWeight',1.0)
	vxSize        = kwargs.get('vxSize', 0)
	n             = kwargs.get('n', 0)
	cubeSize      = kwargs.get('cubeSize', 0)

	# Create radius matrix
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
	'''
	Pre-whitenening using noise estimates from a soft mask of the background.
	Returns a the pre-whitened volume and various spectra. (Alp Kucukelbir, 2013)

	'''
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
	'''
	Pre-whitenening using noise estimates from a cube taken from the difference map.
	Returns a the pre-whitened volume and various spectra. (Alp Kucukelbir, 2013)

	'''
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

	dataBGPW = np.real(fftpack.ifftn(fftpack.ifftshift(dataBGF)))
	del dataBGF

	m, s = divmod(time() - tStart, 60)
	print "  :: Time elapsed: %d minutes and %.2f seconds" % (m, s)

	return {'dataPW':dataPW, 'dataBGPW':dataBGPW, 'dataPWSpect': dataPWSpect, 'dataPWBGSpect': dataPWBGSpect, 'peval': pWfilter['peval'], 'pcoef': pWfilter['pcoef'] }


def createPrewhiteningFigure(**kwargs):

	def something_changed(val):
		fig.axbutton.cla()
		updcolor = 'firebrick'
		fig.buttonclose = Button(fig.axbutton, label='Click here to Update', color=updcolor, hovercolor=updcolor)
		fig.canvas.draw()

	def quit_figure(event):
		plt.close(event.canvas.figure)
			
	def add_subplot(location, rowspan=1, colspan=1):
		gridspec = GridSpec(2, 3)
		subplotspec = gridspec.new_subplotspec(location, rowspan, colspan)
		return fig.add_subplot(subplotspec)


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
	
	# From Scipion we will create sliders and button in tk
	showSliders   = kwargs.get('showSliders', True)
	showButtons   = kwargs.get('showButtons', True)
	instructions  = kwargs.get('instructions', """'Please check that the green line
is as straight as possible,
at least in the high frequencies.

If not, adjust the sliders
above and press the Update button.

ResMap will try to pre-whiten the
volume again (the window will close).

If you are satisfied
please press Continue below.	
	""")

	xpoly = np.array(range(1,dataBGSpect.size + 1))

	# Accept 'figure' as a keyword argument
	# if not passed, create using plt.figure
	if 'figure' in kwargs:
		fig = kwargs.get('figure')
	else:
		fig = plt.figure(figsize=(18, 9))
	
	fig.suptitle('\nResMap Pre-Whitening Interface (beta)', fontsize=20, color='#104E8B', fontweight='bold')

	axcolor  = 'lightgoldenrodyellow'
	okcolor  = 'seagreen'

	ax1      = add_subplot((0,0), colspan=2)
	ax2      = add_subplot((1, 0))
	ax3      = add_subplot((1, 1))
	axtext   = add_subplot((1, 2))


	if showButtons:
		fig.axbutton = fig.add_axes([0.67, 0.025, 0.23, 0.05])
		# Continue/Update Button
		fig.buttonclose = Button(fig.axbutton, label='Continue', color=okcolor, hovercolor=okcolor)
		fig.buttonclose.on_clicked(quit_figure)

	if showSliders:
		print "adding sliders..."
		# Slider for elbow
		axelbow = fig.add_axes([0.7, 0.65, 0.2, 0.03], axisbg=axcolor)
		fig.selbow  = Slider(axelbow, 'Angstrom', 2.1*vxSize, 100, valinit=elbowAngstrom)
		fig.selbow.on_changed(something_changed)
	
		# Slider for rampWeight
		axramp = fig.add_axes([0.7, 0.55, 0.2, 0.03], axisbg=axcolor)
		fig.sramp  = Slider(axramp, 'Ramp Weight', 0.0, 1.0, valinit=rampWeight)
		fig.sramp.on_changed(something_changed)

	# Instructions
	axtext.set_title('INSTRUCTIONS', color='#104E8B', fontweight='bold')
	axtext.get_xaxis().set_visible(False)
	axtext.get_yaxis().set_visible(False)
	axtext.text(0.5, 0.5, instructions,
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
	
	fig.canvas.draw()
	
	return fig
	
	
def displayPreWhitening(**kwargs):
		
	fig = createPrewhiteningFigure(**kwargs)
	plt.show()

	return (fig.selbow.val, fig.sramp.val)

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

	# fig = plt.figure(1)
	# ax = fig.add_subplot(111)
	# p = ax.plot(-1*diffLogPowerSpectrum)
	# # plt.yscale('log')
	# plt.grid(linestyle='dotted')
	# plt.ylabel('Power Spectrum (|f|^2)')
	# plt.xlabel('Frequency')
	# plt.show()

	# Find positive peaks in the derivative
	peakInd = signal.find_peaks_cwt(-1*diffLogPowerSpectrum, np.arange(1,10), min_snr=2)

	# Pick out the maximum radius index where a peak occurs
	if peakInd:
		maxInd = np.max(peakInd)
	else:
		return {'outcome':False, 'factor': 0.0}

	# print peakInd
	# print maxInd

	# Calculate the mean and variance of the derivative of the power spectrum beyond maxInd
	if maxInd < dataPowerSpectrum.size - 3:
		m, v   = np.mean(diffLogPowerSpectrum[maxInd+2:]), np.var(diffLogPowerSpectrum[maxInd+2:])
	else:
		return {'outcome':False, 'factor': 0.0}

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

