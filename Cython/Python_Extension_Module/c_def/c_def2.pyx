cimport cython

def remainder(int a, int b):
	with cython.cdivision(True):
		return a % b