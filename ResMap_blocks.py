'''
ResMap_blocks: module containing blocking functions.
							 All functions courtesy of Sebastien Berg.
		LINK: http://www.mail-archive.com/numpy-discussion@scipy.org/msg38712.html

Description of functions:
		rolling_window: Create a view of `array` which for every point gives the
										n-dimensional neighbourhood of size window.
		permute_axes:   Change the arrays axes order or combine multiple
										axes into one.

Requirements:
		numpy

Please see individual functions for attributions.
'''

import numpy as np

def rolling_window(array, window=(0,), asteps=None, wsteps=None,
										axes=None, intersperse=False):
	"""Create a view of `array` which for every point gives the n-dimensional
	neighbourhood of size window. New dimensions are added at the end of
	`array` or after the corresponding original dimension.

	Parameters
	----------
	array : array_like
			Array to which the rolling window is applied.
	window : int or tuple
			Either a single integer to create a window of only the last axis or a
			tuple to create it for the last len(window) axes. 0 can be used as a
			to ignore a dimension in the window.
	asteps : tuple
			Aligned at the last axis, new steps for the original array, ie. for
			creation of non-overlapping windows. (Equivalent to slicing result)
	wsteps : int or tuple (same size as window)
			steps for the added window dimensions. These can be 0 to repeat values
			along the axis.
	axes: int or tuple
			If given, must have the same size as window. In this case window is
			interpreted as the size in the dimension given by axes. IE. a window
			of (2, 1) is equivalent to window=2 and axis=-2.
	intersperse : bool
			If True, the new dimensions are right after the corresponding original
			dimension, instead of at the end of the array.

	Returns
	-------
	A view on `array` which is smaller to fit the windows and has windows added
	dimensions (0s not counting), ie. every point of `array` is an array of size
	window.

	Examples
	--------
	>>> a = np.arange(9).reshape(3,3)
	>>> rolling_window(a, (2,2))
	array([[[[0, 1],
					 [3, 4]],

					[[1, 2],
					 [4, 5]]],


				 [[[3, 4],
					 [6, 7]],

					[[4, 5],
					 [7, 8]]]])

	Or to create non-overlapping windows, but only along the first dimension:
	>>> rolling_window(a, (2,0), asteps=(2,1))
	array([[[0, 3],
					[1, 4],
					[2, 5]]])

	Note that the 0 is discared, so that the output dimension is 3:
	>>> rolling_window(a, (2,0), asteps=(2,1)).shape
	(1, 3, 2)

	This is useful for example to calculate the maximum in all (overlapping)
	2x2 submatrixes:
	>>> rolling_window(a, (2,2)).max((2,3))
	array([[4, 5],
				 [7, 8]])

	Or delay embedding (3D embedding with delay 2):
	>>> x = np.arange(10)
	>>> rolling_window(x, 3, wsteps=2)
	array([[0, 2, 4],
				 [1, 3, 5],
				 [2, 4, 6],
				 [3, 5, 7],
				 [4, 6, 8],
				 [5, 7, 9]])
	"""
	array = np.asarray(array)
	orig_shape = np.asarray(array.shape)
	window = np.atleast_1d(window).astype(int) # maybe crude to cast to int...

	if axes is not None:
		axes = np.atleast_1d(axes)
		w = np.zeros(array.ndim, dtype=int)
		for axis, size in zip(axes, window):
				w[axis] = size
		window = w

	# Check if window is legal:
	if window.ndim > 1:
		raise ValueError("`window` must be one-dimensional.")
	if np.any(window < 0):
		raise ValueError("All elements of `window` must be larger then 1.")
	if len(array.shape) < len(window):
		raise ValueError("`window` length must be less or equal `array` dimension.")

	_asteps = np.ones_like(orig_shape)
	if asteps is not None:
		asteps = np.atleast_1d(asteps)
		if asteps.ndim != 1:
			raise ValueError("`asteps` must be either a scalar or one dimensional.")
		if len(asteps) > array.ndim:
			raise ValueError("`asteps` cannot be longer then the `array` dimension.")
		# does not enforce alignment, so that steps can be same as window too.
		_asteps[-len(asteps):] = asteps

		if np.any(asteps < 1):
			 raise ValueError("All elements of `asteps` must be larger then 1.")
	asteps = _asteps

	_wsteps = np.ones_like(window)
	if wsteps is not None:
		wsteps = np.atleast_1d(wsteps)
		if wsteps.shape != window.shape:
			raise ValueError("`wsteps` must have the same shape as `window`.")
		if np.any(wsteps < 0):
			raise ValueError("All elements of `wsteps` must be larger then 0.")

		_wsteps[:] = wsteps
		_wsteps[window == 0] = 1 # make sure that steps are 1 for non-existing dims.
	wsteps = _wsteps

	# Check that the window would not be larger then the original:
	if np.any(orig_shape[-len(window):] < window * wsteps):
		raise ValueError("`window` * `wsteps` larger then `array` in at least one dimension.")

	new_shape = orig_shape # just renaming...

	# For calculating the new shape 0s must act like 1s:
	_window = window.copy()
	_window[_window==0] = 1

	new_shape[-len(window):] += wsteps - _window * wsteps
	new_shape = (new_shape + asteps - 1) // asteps
	# make sure the new_shape is at least 1 in any "old" dimension (ie. steps
	# is (too) large, but we do not care.
	new_shape[new_shape < 1] = 1
	shape = new_shape

	strides = np.asarray(array.strides)
	strides *= asteps
	new_strides = array.strides[-len(window):] * wsteps

	# The full new shape and strides:
	if not intersperse:
		new_shape = np.concatenate((shape, window))
		new_strides = np.concatenate((strides, new_strides))
	else:
		_ = np.zeros_like(shape)
		_[-len(window):] = window
		_window = _.copy()
		_[-len(window):] = new_strides
		_new_strides = _

		new_shape = np.zeros(len(shape)*2, dtype=int)
		new_strides = np.zeros(len(shape)*2, dtype=int)

		new_shape[::2] = shape
		new_strides[::2] = strides
		new_shape[1::2] = _window
		new_strides[1::2] = _new_strides

	new_strides = new_strides[new_shape != 0]
	new_shape = new_shape[new_shape != 0]

	return np.lib.stride_tricks.as_strided(array, shape=new_shape, strides=new_strides)


def permute_axes(array, axes, copy=False, order='C'):
	"""Change the arrays axes order or combine multiple axes into one. Creates
	a view if possible, but when axes are combined will usually return a copy.

	Parameters
	----------
	array : array_like
			Array for which to define new axes.
	axes : iterable
			Iterable giving for every new axis which old axis are combined into it.
			np.newaxis/None can be used to insert a new axis. Elements must be
			either ints or iterables of ints identifying each axis.
	copy : bool
			If True a copy is forced.
	order : 'C', 'F', 'A' or 'K'
			Whether new array should have C or Fortran order. See np.copy help for
			details.

	See Also
	--------
	swapaxes, rollaxis, reshape

	Examples
	--------
	>>> a = np.arange(12).reshape(3,2,2)

	Just reverse the axes order:
	>>> permute_axes(a, (2, 1, 0))
	array([[[ 0,  4,  8],
					[ 2,  6, 10]],

				 [[ 1,  5,  9],
					[ 3,  7, 11]]])

	Combine axis 0 and 1 and put the last axis to the front:
	>>> permute_axes(a, (-1, (0, 1)))
	array([[ 0,  2,  4,  6,  8, 10],
				 [ 1,  3,  5,  7,  9, 11]])

	Or going over the first two axes in different order:
	>>> permute_axes(a, (-1, (1, 0)))
	array([[ 0,  4,  8,  2,  6, 10],
				 [ 1,  5,  9,  3,  7, 11]])
	"""

	new_axes = []
	for a in axes:
		if a is None:
			new_axes.append(None)
		else:
			a = np.atleast_1d(a)
			if a.ndim > 1:
					raise ValueError("All items of `axes` must be zero or one dimensional.")
			new_axes.append(a) # array for slicing.

	old_shape = np.asarray(array.shape)
	old_strides = np.asarray(array.strides)

	# Shape and strides for the copy operation:
	tmp_shape = []
	tmp_strides = []

	final_shape = []
	final_strides = [] # only used if no copy is needed.

	check_axes = np.zeros(len(old_shape), dtype=bool)

	must_copy = False

	# create a reordered array first:
	for ax, na in enumerate(new_axes):
		if na is not None:
			if np.any(check_axes[na]) or np.unique(na).shape != na.shape:
				raise ValueError("All axis must at most occure once in the new array")
			check_axes[na] = True

			if len(na) != 0:
				_shapes = old_shape[na]
				_strides = old_strides[na]

				tmp_shape.extend(_shapes)
				tmp_strides.extend(_strides)

				final_strides.append(_strides.min()) # smallest stride...
				final_shape.append(_shapes.prod())

				if not must_copy:
					# If any of the strides do not fit together we must copy:
					prev_stride = _strides[0]
					for stride, shape in zip(_strides[1:], _shapes[1:]):
						if shape == 1:
							# 1 sized dimensions just do not matter, but they
							# also do not matter for legality check.
							continue
						elif prev_stride != stride * shape:
							must_copy = True
							break
						prev_stride = stride

				continue # skip adding empty axis.

		tmp_shape.append(1)
		tmp_strides.append(0)
		final_shape.append(1)
		final_strides.append(0)

	if not must_copy:
		result = np.lib.stride_tricks.as_strided(array, shape=final_shape, strides=final_strides)
		if copy:
			return result.copy(order=order)
		return result

	# No need for explicite copy, as reshape must already create one since
	# must_copy is True.
	copy_from = np.lib.stride_tricks.as_strided(array, shape=tmp_shape, strides=tmp_strides)
	return copy_from.reshape(final_shape, order=order)
