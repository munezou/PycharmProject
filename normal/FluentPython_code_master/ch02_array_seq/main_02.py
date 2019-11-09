# Cartesian product
colors = ['black', 'white']
sizes = ['S', 'M', 'L']

print('colors are {0}'.format(list(colors)))
print('sizes are {0}'.format(list(sizes)))

print()

# Shirt types are made with a combination of color and size.
print('---< Shirt types are made with a combination of color and size. >---')
tshirts = [(color, size) for color in colors for size in sizes]

print('tshirts type = {0}'.format(list(tshirts)))

print()

print('tshirts type = ')

for tshirts in ('%s %s' % (c, s) for c in colors for s in sizes):
    print(tshirts)


print()

# use for steatment.

print('tshirts type = ')

for color in colors:
    for size in sizes:
        print((color, size))


# Use tapple as a record.
print('---< Use tapple as a record. >---')
lax_coordinate = (33.9425, -118.408056)
city, year, pop, chg, area = ('Tokyo', 2003, 32.450, 0.66, 8014)
'''
pop: population Unit millions
chg: Population change rate
area: area (km^2)
'''
traveler_ids = [('USA', '31195855'), ('BRA', 'CE342567'), ('ESP', 'XDA205856')]

for passport in sorted(traveler_ids):
    print('country/passport number = %s / %s' % passport)

for country, _ in traveler_ids:
    print('country = %s' % country)

# unpack tuple.
print ('---< Unpack tuple >---')
latitude, longitude = lax_coordinate
print ('latitude = %s, longitude = %s' % lax_coordinate)
print ('latitude = {0}'.format(latitude))
print ('longitude = {0}'.format(longitude))

print()

print ('---< An elegant example of tuple unpacking is to exchange variable values without using temporary variables. >---')
a = 23
b = 46
print('a = {0},  b = {1} before exchange'.format(a, b))
b, a = a, b
print('a = {0},  b = {1} after exchange'.format(a, b))

print()

print('---< Tuples can be unpacked by adding "*" at the beginning of the argument when calling the function. >---')
'''
divmod(a, b) = (a//b, a % b)
'''
print('divmod(20, 8) = {0}'.format(divmod(20, 8)))

t = (20, 8)
print('t = {0}'.format(t))
print('divmod(*t) = {0}'.format(divmod(*t)))

print('quotient = %s, remainder = %s' % divmod(*t))

print()

import os
_, filename = os.path.split('/home/luiciano/.ssh/idrsa.pub')
print('file name = {0}'.format(filename))