'''
ResMap_sphericalProfile: module containing spherical profiling functions
											for ResMap algorithm (Alp Kucukelbir, 2013)

Description of functions:
  		sum_by_group: efficient helper function to sum values in
  									array using another index
	sphericalAverage: calculates the spherically averaged profile of a volume

Requirements:
	numpy

Please see individual functions for attributions.
'''
import numpy as np

def sum_by_group(values, groups):
	"""
	Taken from
	http://stackoverflow.com/questions/4373631/sum-array-by-number-in-numpy

	"""
	order             = np.argsort(groups)
	groups            = groups[order]
	values            = values[order]
	values.cumsum(out=values)
	index             = np.ones(len(groups), 'bool')
	index[:-1]        = groups[1:] != groups[:-1]
	values            = values[index]
	groups            = groups[index]
	values[1:]        = values[1:] - values[:-1]
	return values

def sphericalAverage(image, center=None, binsize=1.0):
	"""
	Calculate the spherically averaged profile.

	image - The 3D image
	center - The [x,y,z] pixel coordinates used as the center. The default is
			 None, which then uses the center of the image (including
			 fractional pixels).
	binsize - size of the averaging bin.  Can lead to strange results if
		non-binsize factors are used to specify the center and the binsize is
		too large

	If a bin contains NO DATA, it will have a NAN value because of the
	divide-by-sum-of-weights component.  I think this is a useful way to denote
	lack of data, but users let me know if an alternative is prefered...

	"""

	n 	    = image.shape[0]

	[x,y,z] = np.mgrid[ -n/2:n/2:complex(0,n),
						-n/2:n/2:complex(0,n),
						-n/2:n/2:complex(0,n) ]
	r       = np.array(np.sqrt(x**2 + y**2 + z**2), dtype='float32')
	del x, y, z

	# the 'bins' as initially defined are lower/upper bounds for each bin
	# so that values will be in [lower,upper)
	nbins  = int(np.round( ((n/2.0) - 1) / binsize))
	maxbin = nbins * binsize
	bins   = np.linspace(0,maxbin,nbins)

	# Find out which radial bin each point in the map belongs to
	whichbin = np.digitize(r.flat,bins)
	del r

	# how many per bin (i.e., histogram)?
	# there are never any in bin 0, because the lowest index
	# returned by digitize is 1
	nr = np.bincount(whichbin)[1:]

	radial_prof = sum_by_group(image.flat, whichbin)/nr

	return radial_prof
