import math
# import bisect
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy import linalg
# from scipy.stats import f
from time import time
from itertools import product
from numpy.lib.stride_tricks import as_strided as ast

def rolling_window_lastaxis(a, window):
    """Directly taken from Erik Rigtorp's post to numpy-discussion.
    <http://www.mail-archive.com/numpy-discussion@scipy.org/msg29450.html>"""
    if window < 1:
       raise ValueError, "`window` must be at least 1."
    if window > a.shape[-1]:
       raise ValueError, "`window` is too long."
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def rolling_window(a, window):
    """Takes a numpy array *a* and a sequence of (or single) *window* lengths
    and returns a view of *a* that represents a moving window."""
    if not hasattr(window, '__iter__'):
        return rolling_window_lastaxis(a, window)
    for i, win in enumerate(window):
        if win > 1:
            a = a.swapaxes(i, -1)
            a = rolling_window_lastaxis(a, win)
            a = a.swapaxes(-2, i)
    return a

def make3DsteerableDirections(x, y, z):
	dirs = np.zeros(x.shape + (16,))

	## 6 rotations for G2
	
	# Unit normals to the faces of the dodecahedron
	v = np.array([1, 0, (np.sqrt(5.0)+1)/2.0]);
	v = v/np.linalg.norm(v);
	dirs[:,:,:,0] =  x*v[0] + y*v[1] + z*v[2];
	dirs[:,:,:,1] =  x*v[1] + y*v[2] + z*v[0];
	dirs[:,:,:,2] =  x*v[2] + y*v[0] + z*v[1];
	
	# Flip sign of golden ratio (arbitrary choice, just stay consistent)
	v[2] = -v[2];
	dirs[:,:,:,3] =  x*v[0] + y*v[1] + z*v[2];
	dirs[:,:,:,4] =  x*v[1] + y*v[2] + z*v[0];
	dirs[:,:,:,5] =  x*v[2] + y*v[0] + z*v[1];

	## 10 rotations for H2
	
	# Unit normals to the faces of the icosahedron
	v = np.array([1, (np.sqrt(5.0)+1)/2.0, 2.0/(np.sqrt(5.0)+1)]);
	v = v/np.linalg.norm(v);
	dirs[:,:,:,6] =  x*v[0] + y*v[1] + z*v[2];
	dirs[:,:,:,7] =  x*v[1] + y*v[2] + z*v[0];
	dirs[:,:,:,8] =  x*v[2] + y*v[0] + z*v[1];
	
	# Flip sign of golden ratio (arbitrary choice, just stay consistent)
	v[1] = -v[1];
	dirs[:,:,:,9]  =  x*v[0] + y*v[1] + z*v[2];
	dirs[:,:,:,10] =  x*v[1] + y*v[2] + z*v[0];
	dirs[:,:,:,11] =  x*v[2] + y*v[0] + z*v[1];
	
	# Unit normals to the vertices of the cube
	dirs[:,:,:,12] = 1/np.sqrt(3.0) * ( x    + y + z );
	dirs[:,:,:,13] = 1/np.sqrt(3.0) * ( -1*x + y + z );
	dirs[:,:,:,14] = 1/np.sqrt(3.0) * ( x    - y + z );
	dirs[:,:,:,15] = 1/np.sqrt(3.0) * ( -1*x - y + z );
	return dirs

# def main():
if __name__ == '__main__':

	print '== BEGIN MAIN =='
	tBegin = time()

	# Load files from MATLAB (Temporary)
	mat  = scipy.io.loadmat('volScheres2275.mat')
	
	data = mat["y"]
	mask = np.array(mat["mask"],dtype='bool')
	n 	 = data.shape[0]
	N 	 = int(n)

	# User defined parameters
	k      = 1.77	# voxel size (in Angstroms)
	M      = 4.5	# query resolution (in Angstroms)
	pValue = 0.05	# generally between (0, 0.05]

	if M < 2.5*k:
		print "Please choose a query resolution M > 2.5*k = %.2f" % 2.5*k

	# Create spherical mask
	[x,y,z] = np.mgrid[ -n/2:n/2:complex(0,n),
						-n/2:n/2:complex(0,n),
						-n/2:n/2:complex(0,n) ]
	R       = np.sqrt(x**2 + y**2 + z**2)
	Rinside = R < n/2					 	
	del R

	# Compute window size and form steerable bases
	r = np.ceil(0.5*M/k)  		# number of pixels around center
	s = (2.0*r*k)/M      		# scaling factor to account for overshoot due to k
	l = np.pi*np.sqrt(2.0/5)	# lambda

	# Define range of x, y, z for steerable bases
	# lsp     = np.linspace(-s*l, s*l, 2*r+1)
	# [x,y,z] = np.meshgrid(lsp, lsp, lsp)
	[x,y,z] = np.mgrid[	-s*l:s*l:complex(0,2*r+1),
						-s*l:s*l:complex(0,2*r+1),
						-s*l:s*l:complex(0,2*r+1) ]
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
	# [U, s, V] = linalg.svd(np.dot(np.diag(kernelSqrt),A))
	# Sinv      = linalg.diagsvd(1/s,numberOfPoints,numberOfBases)
	# H = np.dot(A, np.dot(
	# 	np.dot(V.transpose(), np.dot(Sinv.transpose(), U.transpose())),
	# 	np.diag(kernelSqrt)))	
	tmp = np.dot(np.diag(kernelSqrt),A)
	H = np.dot(A, np.dot(np.linalg.pinv(np.dot(np.diag(kernelSqrt),A)), np.diag(kernelSqrt)))

	# Invert weighted Ac matrix via SVD
	# [U, s, V] = linalg.svd(np.dot(np.diag(kernelSqrt),Ac))
	# Sinv      = linalg.diagsvd(1/s,numberOfPoints,numberOfBases)
	# Hc = np.dot(Ac, np.dot(
	# 	np.dot(V.transpose(), np.dot(Sinv.transpose(), U.transpose())),
	# 	np.diag(kernelSqrt)))
	Ack = np.dot(np.diag(kernelSqrt),Ac)	
	Hc = np.dot(Ac, np.dot(Ack.T/(np.linalg.norm(Ack)**2), np.diag(kernelSqrt)))

	# Create LAMBDA matrices that correspond to WRSS = Y^T*LAMBDA*Y
	LAMBDA     = W-np.dot(W,H);
	LAMBDAc    = W-np.dot(W,Hc);
	LAMBDAdiff = LAMBDAc-LAMBDA;

	# Calculate variance estimate
	print 'Calculating variance estimate...',
	tStart = time()
	variance = 0.003
	print 'done.'
	m, s = divmod(time() - tStart, 60)
	print ":: Time elapsed: %d minutes and %.2f seconds" % (m, s)

	## Compute Likelihood Ratio Statistic

	# # Extract data from 3D volume
	# print 'Vectorizing 3D data...',
	# tStart = time()
	# # tmpIdx  = 0
	# # dataVec  = np.zeros([mask.sum(), numberOfPoints], dtype='float32')
	
	# # for idx in range(indexVec.shape[1]):
	# 	# i, j, k = indexVec[:,idx]
	# 	# dataVec[idx,:] = data[	i-r:i+r+1, 
	# 	# 						j-r:j+r+1,
	# 	# 						k-r:k+r+1 ].flatten()
	# # for i, j, k in product(range(N), range(N), range(N)):
	# # 	if mask[i,j,k]:
	# # 		dataVec[tmpIdx,:] = data[	i-r:i+r+1, 
	# # 									j-r:j+r+1,
	# # 									k-r:k+r+1 ].flatten()
	# # 		tmpIdx += 1
	# print 'done.'
	# m, s = divmod(time() - tStart, 60)
	# print ":: Time elapsed: %d minutes and %.2f seconds" % (m, s)

	# Calculate weighted residual sum of squares difference
	print 'Calculating Likelihood Ratio Statistic...',
	tStart   = time()
	indexVec = np.array(np.where(mask))
	WRSSdiff = np.zeros([indexVec.shape[1],1], 	dtype='float32')
	dataCube = np.zeros([numberOfPoints,1], 	dtype='float32')
	for idx in range(indexVec.shape[1]):
		i, j, k = indexVec[:,idx]
		dataCube[:,0] = data[	i-r:i+r+1, 
								j-r:j+r+1,
								k-r:k+r+1 ].flatten()
		WRSSdiff[idx] = np.dot(dataCube.T, np.dot(LAMBDAdiff, dataCube))
	LRSvec = WRSSdiff/variance
	print 'done.'
	m, s = divmod(time() - tStart, 60)
	print ":: Time elapsed: %d minutes and %.2f seconds" % (m, s)

	# Undo reshaping to get LRS in a 3D volume
	print 'Reshaping Likelihood Ratio Statistic into 3D volume...',
	tStart = time()
	LRS    = np.zeros([n,n,n], dtype='float32')
	for idx in range(indexVec.shape[1]):
		i, j, k = indexVec[:,idx]
		LRS[i,j,k] = LRSvec[idx]
	print 'done.'
	m, s = divmod(time() - tStart, 60)
	print ":: Time elapsed: %d minutes and %.2f seconds" % (m, s)

	m, s = divmod(time() - tBegin, 60)
	print "TOTAL :: Time elapsed: %d minutes and %.2f seconds" % (m, s)
	
	# # Calculate Resolution
	# R   = F/Falpha
	# val = np.amax(R,axis=2)
	# res = np.argmax(R,axis=2)
	# res[val<=1] = widthMax;
	
	# # Plot
	# f1 = plt.figure()
	# f2 = plt.figure()
	# ax1 = f1.add_subplot(111)
	# ax1.imshow(data, cmap=plt.cm.gray, interpolation="nearest")
	# ax2 = f2.add_subplot(111)
	# ax2.imshow(res, cmap=plt.cm.jet, interpolation="nearest")
	# plt.show()

	scipy.io.savemat('output.mat', {'LRSpy':LRS, 'LAMBDAdiffpy':LAMBDAdiff})
	# scipy.io.savemat('output.mat', {'pyF':F,'pyFAlphaPreComp':fAlphaPreComp,'pyR':R,'pyRes':res})




# if __name__ == '__main__':
	# main()

