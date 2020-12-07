from libc.stdlib cimport malloc, free
from Cython import NULL

cdef class Matrix:
	cdef:
		unsigned int nrows, ncols
		double * _matrix
	
	def __init__(self, nr, nc):
		self.nrows = nr
		self.ncols = nc
		self._matrix = <double*> malloc(nr * nc * sizeof(double))
		
		if self._matrix == NULL:
			raise MemoryError()
	
	def __dealloc__(self):
		if self._matrix != NULL:
			free(self._matrix)
	