import math
import bisect
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.stats import f


# def main():
if __name__ == '__main__':

	# Load files from MATLAB (Temporary)
	mat  = scipy.io.loadmat('testLinux.mat')
	y    = mat["y"]
	mask = mat["BW"]

	z = y
	n = z.shape[0]

	# User defined parameters
	width0   = 1
	widthMax = 8
	alpha    = 1-1e-3;

	# Precompute F statistic at chosen alpha level
	stepSize = 1e-2;
	maxDegreeOfFreedom = (2*widthMax+1)**2
	fAlphaPreComp = np.zeros([1, maxDegreeOfFreedom]);

	for dof in range(10,maxDegreeOfFreedom):
		# Create F CDF with a given degree of freedom
		fCDF = f.cdf(np.linspace(0,100,100/stepSize), 8, dof-9)

		# Find first point where CDF > alpha
		fAlphaPreComp[0,dof] = stepSize*bisect.bisect(fCDF,alpha)

	dataList  = []
	fStatList = []

	# [x,y] = np.mgrid[-1:1:11j,-1:1:11j]





def kernelWeights(u):
	return np.exp(-3*(u**2));





# if __name__ == '__main__':
	# main()

