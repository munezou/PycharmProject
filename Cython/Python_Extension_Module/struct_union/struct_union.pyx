ctypedef struct mycpx:
	float real
	float imag

ctypedef union uu:
	int a
	short b, c

cdef mycpx zz

zz.real = 3.1415
zz.imag = -1.0

print('zz.real = {0}\n'.format(zz.real))
print('zz.imag = {0} \n'.format(zz.imag))