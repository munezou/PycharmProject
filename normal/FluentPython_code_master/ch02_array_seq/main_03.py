# Acquisition of surplus elements using "*"
print ('---< Acquisition of surplus elements using "*" >---')

print('In case of a, b, *rest = range(5)')

a, b, *rest = range(5)

print ('a, b, rest = {0}, {1}, {2}'.format(a, b, rest))

print()

print ('In case of a, b, *rest = range(3)')

a, b, *rest = range(3)

print ('a, b, rest = {0}, {1}, {2}'.format(a, b, rest))

print()

print ('In case of a, b, *rest = range(2)')

a, b, *rest = range(2)

print ('a, b, rest = {0}, {1}, {2}'.format(a, b, rest))

print()

'''
-----------------------------------------------------------
The "*" prefix can only be used at one place, but it can be used anywhere.
-----------------------------------------------------------
'''
print('In case of a, *body, c, d = range(5)')

a, *body, c, d = range(5)

print ('a, *body, c, d = {0}, {1}, {2}, {3}'.format(a, body, c, d))

print()

print('In case of *body, b, c, d = range(5)')

*body, b, c, d = range(5)

print ('a, *body, c, d = {0}, {1}, {2}, {3}'.format(body, b, c, d))

print()

print ('---< Unpack nested tuples >---')

metro_areas = [('Tokyo', 'jp', 36.933, (35.689722, 139.691667)),
               ('Delhi NCR', 'IN', 21.935, (28.613889, 77.28889)),
               ('Mexico City', 'MX', 20.142, (19.433333, -99.133333)),
               ('New York-Newark', 'US', 20.104, (40.808611, -74.020386)),
               ('Sao Paulo', 'BR', 19.649, (-23.54778, -46.635833))]

# Title output
print('{:15} | {:^9} | {:^9}'.format('', 'lat.', 'long.'))

fmt = '{:15} | {:9.4f} | {:9.4f}'
for name, CC, pop, (latitude, longitude) in metro_areas:
    if longitude <= 0:
        print (fmt.format(name, latitude, longitude))

print()

print('---< Named tuple >---')

from collections import namedtuple

# declare named tuple
City = namedtuple('City', 'name country population cordinates')

tokyo = City ('Tokyo', 'JP', 36.933, (35.68972, 139.691667))

print ('tokyo = {0}'.format(tokyo))

print ('tokyo.population = {0}'.format(tokyo.population))
print ('tokyo.cordinates = {0}'.format(tokyo.cordinates))
print ('tokyo[1] = {0}'.format(tokyo[1]))

print ()

print ('---< Named Tuple Attributes and Methods >---')
print('City.field = {0}'.format(City._fields))

LatLong = namedtuple('LatLong', 'lat long')
delhi_data = ('Delhi NCR', 'IN', 21.935, LatLong(28.613889, 77.208889))
delhi = City._make(delhi_data)
print('delhi._asdict() = {0}'.format(delhi._asdict()))

print()

for key, value in delhi._asdict().items():
    print(key + ':', value)

print()
