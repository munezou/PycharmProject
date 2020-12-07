def print_address(a):
	cdef void *v = <void*>a
	cdef long addr = <long>v
	print('Cython address: {0}\n'.format(addr))
	print('python id = {0}\n'.format(id(a)))