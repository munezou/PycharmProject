cimport cython

@cython.cdivision(True)
def divides(int a, int b):
	return a / b