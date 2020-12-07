import cython

def func(double x, double y):
	cdef:
		double z
	z = cfunc(x, y)
	return z