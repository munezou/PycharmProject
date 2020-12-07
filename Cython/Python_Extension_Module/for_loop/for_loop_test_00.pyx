cdef unsigned int i, n = 100

for i in range(n):
	if (i % 10) == 0:
		print('i = {0}\n'.format(i))