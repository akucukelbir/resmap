from scipy.stats import norm
import numpy as np

def evaluateRuben(c, alpha, weights):
	evaluated = rubenPython(weights,c)
	answer = np.abs(alpha-evaluated[2])
	return answer

def rubenPython(weights, c, mult=None, delta=None, mode=1, maxit=100000, eps=1e-5):
	"""
	Cite the UofMontreal people
	"""

	# Initialize parameters
	n     = weights.size
	gamma = np.zeros_like(weights,dtype='float32')
	theta = np.ones_like(weights,dtype='float32')
	alist = np.array([],dtype='float32')
	blist = np.array([],dtype='float32')

	weights = np.array(weights,dtype='float32')

	# If no multiplicities are given, assume 1 for all
	if mult is None:
		mult = np.ones_like(weights)

	# If no non-centralities are given, assume 0 for all
	if delta is None:
		delta = np.zeros_like(weights)

	# Basic error checking
	if (n<1) or (c<=0) or (maxit<1) or (eps<=0.0):
		dnsty = 0.0
		ifault  = 2
		res     = -2.0
		return (dnsty, ifault, res)
	else:
		tol = -200.0

		bbeta = np.min(weights)
		summ  = np.max(weights)

		# Some more error checking
		if bbeta <= 0 or summ <= 0 or np.any(mult<1) or np.any(delta<0):
			dnsty  = 0.0
			ifault = -1
			res    = -7.0
			return (dnsty, ifault, res)

		# Calculate BetaB
		if mode > 0.0:
			bbeta = mode*bbeta
		else:
			bbeta = 2.0/(1.0/bbeta + 1.0/summ)

		k = 0
		summ = 1.0
		sum1 = 0.0
		for i in range(n):
			hold       = bbeta/weights[i]
			gamma[i]   = 1.0 - hold
			summ       = summ*(hold**mult[i]) #this is ok -- A.K.
			sum1       = sum1 + delta[i]
			k          = k + mult[i]
			# theta[i]   = 1.0

		ao = np.exp(0.5*(np.log(summ)-sum1))
		if ao <= 0.0:
			dnsty  = 0.0
			ifault = 1
			res    = 0.0
			return (dnsty, ifault, res)
		else: # evaluate probability and density of chi-squared on k degrees of freedom. 
			z = c/bbeta

			if np.mod(k,2)==0:
				i    = 2
				lans = -0.5*z
				dans = np.exp(lans)
				pans = 1.0 - dans
			else:
				i    = 1
				lans = -0.5*(z+np.log(z)) - np.log(np.sqrt(np.pi/2))
				dans = np.exp(lans)
				# pans = normcdf(sqrt(z),0,1) - normcdf(-1*sqrt(z),0,1)
				pans = norm.cdf(np.sqrt(z)) - norm.cdf(-1*np.sqrt(z))

			k = k-2
			for j in range(i,int(k+2),2):
				if lans < tol:
					lans = lans + np.log(z/j)
					dans = np.exp(lans)
				else:
					dans = dans*z/j
				pans = pans - dans

			# Evaluate successive terms of expansion
			prbty = pans
			dnsty = dans
			eps2  = eps/ao
			aoinv = 1.0/ao
			summ  = aoinv - 1.0

			ifault = 4
			for m in range(1,maxit):

				sum1 = 0.5*np.sum(theta*gamma*mult + m*delta*(theta-(theta*gamma)))
				theta = theta*gamma

				# b[m] = sum1
				blist = np.append(blist,sum1)	
				if m>1:
					sum1 = sum1 + np.dot(blist[:-1],alist[::-1])

				sum1 = sum1/m
				# a[m] = sum1
				alist = np.append(alist,sum1)	
				k    = k + 2
				if lans < tol:
					lans = lans + np.log(z/k)
					dans = np.exp(lans)
				else:
					dans = dans*z/k

				pans  = pans - dans
				summ  = summ - sum1
				dnsty = dnsty + dans*sum1
				sum1  = pans*sum1
				prbty = prbty + sum1
				if prbty < -aoinv:
					dnsty  = 0.0
					ifault = 3
					res    = -3.0
					return (dnsty, ifault, res)

				if abs(pans*summ) < eps and abs(sum1) < eps2:
					ifault = 0
					break

			dnsty  = ao*dnsty/(bbeta+bbeta)
			prbty  = ao*prbty
			if prbty<0.0 or prbty>1.0:
				ifault = ifault + 5
			else: 
				if dnsty < 0.0:
					ifault = ifault + 6
				res = prbty
			
		return (dnsty, ifault, res)

def make3DsteerableDirections(x, y, z):
	"""
	Cite the ICCASP paper and the Freeman 91 paper
	"""
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