cpdef long cp_fact(long n):
	if n <= 1:
		return 1
	return n * cp_fact(n - 1)