import bisect
import random

SIZE = 7

random.seed(1729)

my_list = []

'''
bisect is a library that calculates insertion index when adding a certain value to any list.
bisect is as same as bitsect_right.
'''

print('---< bisect.insort >---')
for i in range(SIZE):
    new_item = random.randrange(SIZE*2)
    bisect.insort(my_list, new_item)
    print('%2d ->' % new_item, my_list)

print()

print ('---< bisect.bisect() >---')
A = [1, 2, 3, 3, 3, 4, 4, 6, 6, 6, 6]
x = 5

print ('A = {0}, x = {1}'.format(A, x))
print ('bitsect.bisect_left(A, x) = insert position:{0}'.format(bisect.bisect_left(A, x)))
print()

B = [0, 2, 2, 5, 5, 5, 8, 8, 11, 15, 18]
y = 11

print ('B = {0}, y = {1}'.format(B, y))
print ('bitsect.bisect_left(B, x) = insert position:{0}'.format(bisect.bisect_left(B, y)))
print ('bitsect.bisect_right(B, x) = insert position:{0}'.format(bisect.bisect_right(B, y)))
print ('bitsect.bisect(B, x) = insert position:{0}'.format(bisect.bisect(B, y)))
print()

'''
insort is a library that calculates the insertion index when adding a value to an arbitrary list, and sends it to the insertion operation.
insort is as same as insort_right.
'''
print ('---< bisect.insort() >---')

C = []
z = 4
for i in range(10):
    C.append(i)

print ('C = {0}, z = {1}'.format(C, z))
print ('bitsect.insort_left(B, x) = insert position:{0}'.format(bisect.insort_left(C, z)))
print()
print ('C = {0}, z = {1}'.format(C, z))
print ('bitsect.insort_right(B, x) = insert position:{0}'.format(bisect.insort_right(C, z)))
print()
print ('C = {0}, z = {1}'.format(C, z))
print ('bitsect.bisect(B, x) = insert position:{0}'.format(bisect.bisect(B, y)))
print()

