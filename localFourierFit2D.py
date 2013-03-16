from __future__ import division
import math
import bisect
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy import linalg
from scipy.stats import f
from time import time

def gaussC(x, y, sigma, center):
	exponent = ((x-center[0])**2 + (y-center[1])**2)/(2*sigma**2)
	return np.exp(-1*exponent)

# def main():
if __name__ == '__main__':

	print '== BEGIN MAIN =='
	tBegin = time()

	# Load files from MATLAB (Temporary)
	# mat   = scipy.io.loadmat('hongwei_withBW.mat')
	mat  = scipy.io.loadmat('emdb1019_down.mat')
	# mat  = scipy.io.loadmat('testLinux1019.mat')
	# mat  = scipy.io.loadmat('scheresRIBOSOME_mildLPF.mat')
	# mat  = scipy.io.loadmat('ctf_synth.mat')

	data = mat["y"]
	mask = mat["BW"]
	n 	 = data.shape[0]

	data = data - data.mean()

	[x,y] = np.mgrid[-n/2:n/2, -n/2:n/2]
	R     = np.sqrt(x**2+y**2)
	eps   = 1e-4;
	[x,y] = np.mgrid[0:n, 0:n]

	angleFou = np.zeros([n,n,int(math.ceil(n/3-1))])
	tStart = time()
	for i in range(n):
		for j in range(n):
			if mask[i,j] == 1:
				for idx in range(1,int(math.ceil(n/3-1))+1):
					ring    = np.logical_and(R<0.5+idx+eps, R>=idx-0.5+eps)
					spx     = 1/(1/2*idx/(n/2-1))
					gMap    = gaussC( x, y, spx/(3*math.sqrt(2*math.log(2))), (i, j) )
					dataG   = data*gMap
					dataGf  = np.fft.fftshift(np.fft.fft2(dataG))
					fouFit  = gMap*np.real(np.fft.ifft2(np.fft.ifftshift(ring*dataGf)))

					angleFou[i,j,idx-1] = ( np.dot(dataG.flatten(),fouFit.flatten()) 
						/ (np.linalg.norm(dataG)*np.linalg.norm(fouFit)) )
	
	m, s = divmod(time() - tStart, 60)
	print "TOTAL :: Time elapsed: %d minutes and %.2f seconds" % (m, s)

	res = np.argmax(angleFou,axis=2)

	# Plot
	f1 = plt.figure()
	f2 = plt.figure()
	ax1 = f1.add_subplot(111)
	ax1.imshow(data, cmap=plt.cm.gray, interpolation="nearest")
	ax2 = f2.add_subplot(111)
	ax2.imshow(res, cmap=plt.cm.jet_r, interpolation="nearest")
	plt.show()

	# scipy.io.savemat('output.mat', {'res':res})
	# scipy.io.savemat('output.mat', {'pyF':F,'pyFAlphaPreComp':fAlphaPreComp,'pyR':R,'pyRes':res})




# if __name__ == '__main__':
	# main()

