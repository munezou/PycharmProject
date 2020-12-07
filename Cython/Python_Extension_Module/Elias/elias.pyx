ctypedef double real
ctypedef long integral

def displacement(real d0, real v0, real a, real t):
	cdef real d = d0 + (v0 * t) + (0.5 * a * t**2)
	return d