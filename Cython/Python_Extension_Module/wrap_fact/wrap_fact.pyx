cdef long c_fact(long n):
	if n <= 1:
		return 1
	return n * c_fact(n - 1)

def wrap_c_fact(n):
	return c_fact(n)