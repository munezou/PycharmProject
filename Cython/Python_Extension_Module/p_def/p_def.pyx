def py_fact(n):
	if n < 1:
		return 1
	return n * py_fact(n - 1)