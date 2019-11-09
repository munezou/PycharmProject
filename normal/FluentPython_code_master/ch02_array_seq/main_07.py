'''
Insertion by bisect.insort
'''

import bisect
import random

SIZE = 7

random.seed(1729)

my_list = []

for i in range(SIZE):
    new_item = random.randrange(SIZE*2)
    bisect.insort(my_list, new_item)
    print('%2d ->' % new_item, my_list)

print()

'''
---------------------------------
2.9.1 Array
--------------------------------
'''

print ('---< 2.9.1 Array >---')
from array import array
from random import random

# Create an array with random 10 ** 7 variables of type double.
floats = array('d', (random() for i in range(10**7)))

# Print the last variable of floats array.
print ('floats[-1] = {0}'.format(floats[-1]))

# Store floats array in file named by floats.bin.
fp = open('floats.bin', 'wb')

# Writes floats array to floats.bin.
floats.tofile(fp)

fp.close()

# Create double type floats2 array with 0 elements.
floats2 = array('d')

# Read a data in floats.bin.
fp = open('floats.bin', 'rb')

# Read data in floats.bin into floats2 array.
floats2.fromfile(fp, 10**7)

fp.close()

# Print the last variable of floats2 array.
print ('floats2[-1] = {0}'.format(floats2[-1]))

# TO confirm whether floats equals floats2.
if floats == floats2:
    result = True
else:
    result = False

print ('Does "floats" equal "floats2"? = {0}'.format(result))
print()

'''
-----------------------------------------
2.9.2 memory view
 contents)
  The built-in class memoryview is a sequence type on shared memory that slices an array without copying bytes.
-----------------------------------------
'''

print('---< 2.9.2 memory view >---')

# Create numbers, which is an array of unsigned short type [-2, -1, 0, 1, 2].
numbers = array('h', [-2, -1, 0, 1, 2])

for i in range(len(numbers)):
    print('numbers[{0}] = {1}'.format(i, numbers[i]))

print ()
memv = memoryview(numbers)

print('---< memv >---')

print('len(memv) = {0}'.format(len(memv)))

print()

for i in range(len(memv)):
    print('memv[{0}] = {1}'.format(i, memv[i]))

print()


print ('---< memv_oct >---')

# translate signed short to unsigned char.
memv_oct = memv.cast('B')

for i in range(len(memv_oct)):
    print('memv_oct[{0}] = {1}'.format(i, memv_oct))

print()


# translate array to list.
print('memv_oct.tolist() = \n{0}'.format(memv_oct.tolist()))

memv_oct[5] = 4

print ('numbers = {0}'.format(numbers))
print()

'''
-----------------------------------------------
2.9.3 Numpy and SciPy
-----------------------------------------------
'''
import numpy as np

print ('---< 2.9.3 Numpy and SciPy >---')
a = np.arange(12)

print ('np.arange(12) = {0}'.format(a))

print('type(a) = {0}'.format(type(a)))

print ('a.shape = {0}'.format(a.shape))

a.shape = 3, 4

print ('a.shape = 3,4 : Result = \n{0}'.format(a))

print()

print ('a[2] = {0}'.format(a[2]))

print ('a[2, 1] = {0}'.format(a[2, 1]))

print ('a[:, 1] = {0}'.format(a[:, 1]))

print ('a.traspose = \n{0}'.format(a.transpose()))

print()

'''
-----------------------------------------------
2.9.4 Deque or other que
-----------------------------------------------
'''

print ('---< 2.9.4 Deque or other que >---')

# How to use deque
from collections import deque
dq = deque(range(10), maxlen=10)

print ('deque(range(10), maxlen=10) = {0}'.format(dq))

dq.rotate(3)
print ('dq.rotate(3) = {0}'.format(dq))

dq.rotate(-4)
print ('dq.rotate(-4) = {0}'.format(dq))

dq.appendleft(-1)
print ('dq.appendleft(-1) = {0}'.format(dq))

dq.extend([11, 22, 33])
print ('dq.extend([11, 22, 33]) = {0}'.format(dq))

dq.appendleft([10, 20, 30, 40])
print ('dq.appendleft([10, 20, 30, 40]) = {0}'.format(dq))

