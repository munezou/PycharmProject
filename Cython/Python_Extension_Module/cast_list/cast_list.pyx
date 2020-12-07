def safe_cast_to_list(a):
	cdef list cast_list = <list?>a
	print('type(a) = {0}'.format(type(a)))
	print('type(cast_list) = {0]'.format(cast_list))
	cast_list.append(1)