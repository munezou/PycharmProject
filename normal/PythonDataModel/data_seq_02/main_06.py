'''
------------------------------------------
list.sort and built-in function sorted
------------------------------------------
'''

fruits = ['grape', 'raspberry', 'apple', 'banana']
print('fruits = {0}'.format(fruits))

print('sorted(fruits) = {0}'.format(sorted(fruits)))
print()

print('fruits = {0}'.format(fruits))
print()

print('sorted(fruits, reverse=True) = {0}'.format(sorted(fruits, reverse=True)))
print()

print('sorted(fruits, key=len) = {0}'.format(sorted(fruits, key=len)))
print()

print('sorted(fruits, key=len, reverse=True) = {0}'.format(sorted(fruits, key=len, reverse=True)))
print()

print('fruits = {0}'.format(fruits))
print()

fruits.sort()
print('fruits.sort() = {0}'.format(fruits))
print()

'''
--------------------------------------------
Processing ordered sequences using bisect
--------------------------------------------
'''

# Search by bisect(Array bisection algorithm)
import bisect
import sys

HAYSTACK = [1, 4, 5, 6, 8, 12, 15, 20, 21, 23, 23, 26, 29, 30]
NEEDLES = [0, 1, 2, 5, 8, 10, 22, 23, 29, 30, 31]

'''
-------------------------------------------------------------------------
{0: 2d} represents the number to insert.
{1: 2 d} indicates at which index the number to be inserted is inserted.
-------------------------------------------------------------------------
'''
ROW_FMT = '{0:2d} @ {1:2d}     {2}{0:<2d}'

def demo(bisect_fn):
    for needle in reversed(NEEDLES):
        position = bisect_fn(HAYSTACK, needle)
        offset = position * '  |'
        print (ROW_FMT.format(needle, position, offset))

def grade(score, breakpoints=[60, 70, 80, 90], grades='FDCBA'):
    i = bisect.bisect(breakpoints, score);
    return grades[i]

if __name__ == '__main__':
    if sys.argv[-1] == 'left':
        bisect_fn = bisect.bisect_left
    else:
        bisect_fn = bisect.bisect

    print ('DEMO:', bisect_fn.__name__)
    print ('haystack ->', ' '.join('%2d' % n for n in HAYSTACK))
    demo(bisect_fn)

    print([grade(score) for score in [33, 99, 77, 70, 89, 90, 100]])
