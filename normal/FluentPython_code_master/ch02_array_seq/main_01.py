# Chapter 2

import bisect
import sys
from collections import Hashable

'''
List comprehensions and readability
'''
print('------------------< List comprehensions and readability >-----------------------')
symbols = '$¢£¥€¤'
codes = []

'''
ord()
Returns a string representing a single Unicode character, 
an integer representing the Unicode code point of that character. 
For example, ord ('a') returns the integer 97 and ord ('') (the euro symbol) returns 8364. 
This is the reverse of chr ().
'''

for symbol in symbols:
    codes.append(ord(symbol))

print('code = {0}'.format(codes))

print()

codes = [ord(symbol) for symbol in symbols]

print('code = {0}'.format(codes))

print()

print('---------<  list comprehension and Difference between?map and  filter >---------')

print()

print('---< How to use map function. >---')
result = list(map(lambda x : x + 1, [1, 2, 3]))

print('list(map(lambda x : x + 1, [1, 2, 3])) = {0}'.format(result))

print()

print ('---< How to use filter function. >---')

list_0 = list(range(10))

# Extract an even number from list_0.
list_1 = list(filter(lambda x : x % 2 == 0, list_0))

print('Extract an even number from {0} = {1}.'.format(list_0, list_1))

print()

# Extract odd numbers from list_0.
list_2 = list(filter(lambda y : y % 2 == 1, list_0))

print('Extract a odd number from {0} = {1}.'.format(list_0, list_2))

print()

# Extract elements containing 'n' in list_s.
list_s = ['apple', 'grape', 'banana', 'orange']
list_r = list(filter(lambda s: 'n' in s, list_s))

print('Extract elements containing n in {0}. = {1}'.format(list_s, list_r))

print()

print('----------< Cartesian product >----------')

'''
list is mutable.    ex ['a', 'b', 'c']
tuple is immutable. ex ('a', 'b', 'c')
'''
print ('---< tuple >---')
t = ('tuple', 'list', 'dict', 'set')
print(t)  # ('tuple', 'list', 'dict', 'set')
print(type(t))  # <class 'tuple'>

print()

print('---< list >---')

l = ['tuple', 'list', 'dict', 'set']
print(l)  # ['tuple', 'list', 'dict', 'set']
print(type(l))  # <class 'list'>

print()

print('----< Check if list and tuple are hashable. >----')
print('isinstance([], Hashtable) = {0}'.format(isinstance([], Hashable)))
print('isinstance((), Hashtable) = {0}'.format(isinstance((), Hashable)))

print()

'''
-----------------------------------------------------
Prepare dict with value of building name using latitude and longitude information as key.
(You can not specify list as the key of Hash, but you can specify tuple.)
-----------------------------------------------------
'''
locations = {
    (35.676, 139.744): '?????',
    (34.669, 135.431): '??????',
    (35.039, 135.729): '?????',
}

# output key of location.
for s in locations.keys():
    print('key = {0}'.format(s))

# output values of location.
for s in locations.values():
    print('value = {0}'.format(s))

print()

print('---< Create tuple >---')
t = ('tuple', 'list', 'dict', 'set')
print(' type of {0} is {1}.'.format(t, type(t)))

print()

print('---< Create list. >---')
l = ['tuple', 'list', 'dict', 'set']
print('type of {0} is {1}.'.format(l, type(l)))

print()

print('---< A tuple containing a mixture of strings, numbers and lists >---')
t = ('string', 100, ['a', 'b', 'c'])
print(t)

print('t[0] = {0}.'.format(t[0]))
print('t[1] = {0}.'.format(t[1]))
print('t[2] = {0}.'.format(t[2]))
print('t[2][0] = {0}.'.format(t[2][0]))

print()

'''
---------------------------------------------------------
If you want to create a tuple with only one element (a single element tuple), 
you must always add a comma (,) after the element.
---------------------------------------------------------
'''
a = ('tuple')
print('type of {0} is {1}.'.format(a, type(a)))

b = ('tuple',)
print('type of {0} is {1}.'.format(b, type(b)))

c = (10)
print('type of {0} is {1}.'.format(c, type(c)))

d = (10, )
print('type of {0} is {1}.'.format(d, type(d)))

print()

'''
-----------------------------------------------------------
The parentheses for tuple generation are optional. 
The elements are separated by commas as shown below to generate a tuple.
-----------------------------------------------------------
'''
a = 7
print('type of {0} is {1}.'.format(a, type(a)))

b = 'python3'
print('type of {0} is {1}.'.format(b, type(b)))

c = 'python3', 'JavaScript'
print('type of {0} is {1}.'.format(c, type(c)))

print()

print('---< Acquisition of element >---')
t = ('tuple', 'list', 'dict', 'set')
print('t[0] = {0}'.format(t[0]))
print('t[1] = {0}'.format(t[1]))
print('t[-1] = {0}'.format(t[-1]))

print()

print('---< Acquisition of multiple elements (slice) >---')

'''
----------------------------------------------------------
[Start position: End position] allows you to get part of a tuple.
----------------------------------------------------------
'''
print('t = {0}'.format(t))
print('t[0:3] = {0}'.format(t[0:3]))
print('t[1:3] = {0}'.format(t[1:3]))

print()

'''
----------------------------------------------------------
You can also slice by specifying the number of steps in [Start Position: End Position: Number of Steps].
----------------------------------------------------------
'''
a = ('Python', 'C', 'C++', 'Java', 'PHP', 'Ruby', 'JavaScript')
print('a = {0}'.format(a))

# How to use step
print('a[0:6:2] = {0}'.format(a[0:6:2]))
print('a[2:6:3] = {0}'.format(a[2:6:3]))

# Start and end can be omitted
print('a[0::2] = {0}'.format(a[0::2]))
print('a[::2] = {0}'.format(a[::2]))
print('a[1::3] = {0}'.format(a[1::3]))

print()

# Get the number of elements.
print('length of a tuple is {0}.'.format(len(a)))

print()

# Add / Change / Delete Element
t = ('a', 'b', 'c')

try:
    print('t[0] = d')
    t[0] = 'd'
except Exception as ex:
    print(ex)
    print(type(ex))

print()

try:
    print('del t[0]')
    del t[0]
except Exception as ex:
    print(ex)
    print(type(ex))

print()

'''
--------------------------------------------------------
Tuples thus guarantee element immutability. 
Therefore, it is used as a fixed set value, as a substitute for "constants" in other programming languages.
* Although many programming languages have constants (variables whose values can not be changed),
 Python does not have "constants" in other languages.
--------------------------------------------------------
'''