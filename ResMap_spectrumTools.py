'''
ResMap_spectrumToolks: module containing spectral processing functions for ResMap algorithm (Alp Kucukelbir, 2013)

Description of functions:
		  displayPowerSpectrum: debugging tool
  			isPowerSpectrumLPF: determines whether there is a low-pass drop in the spectrum
	    calculatePowerSpectrum: calculates the radially averaged power spectrum of a volume

Requirements:
	scipy
	numpy
	matplotlib

Please see individual functions for attributions.
'''

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from ResMap_sphericalProfile import sphericalAverage

def displayPowerSpectrum(data, data2=None):

	dataPowerSpectrum = calculatePowerSpectrum(data)
	n = dataPowerSpectrum.size
	xpoly = np.array(range(1,n + 1))

	if data2 != None:
		dataPowerSpectrum2 = calculatePowerSpectrum(data2)

	plt.figure(1)
	p1 = plt.plot(xpoly,dataPowerSpectrum,'b')
	p2 = plt.plot(xpoly,dataPowerSpectrum2,'r')

	plt.yscale('log')
	plt.ylabel('Power (normalized log scale)')
	plt.xlabel('Frequency')
	plt.legend( (p1[0], p2[0]), ('Data1', 'Data2') )

	diffTowardsNyquist = np.diff(np.log(dataPowerSpectrum)) #[int(n/2):]

	# plt.figure(2)
	# p2 = plt.plot(diffTowardsNyquist,'r')

	# peakInd = signal.find_peaks_cwt(-1*diffTowardsNyquist, np.arange(1,10), min_snr=2)

	# # print peakInd

	# p3 = plt.plot(peakInd, diffTowardsNyquist[peakInd],'ms')

	# maxInd = np.max(peakInd)

	# print np.mean(diffTowardsNyquist[maxInd+1:])
	# print np.var(diffTowardsNyquist[maxInd+1:])

	plt.show()

def isPowerSpectrumLPF(dataPowerSpectrum):

	# Calculated derivative of log of dataPowerSpectrum
	diffLogPowerSpectrum = np.diff(np.log(dataPowerSpectrum))

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

