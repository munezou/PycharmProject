# common library
from array import array
import math


# class
class Vector2d:
    typecode = 'd'

    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def __iter__(self):
        return (i for i in (self.x, self.y))

    def __repr__(self):
        class_name = type(self).__name__
        return '{}({!r}, {!r})'.format(class_name, *self)

    def __str__(self):
        return str(tuple(self))

    def __bytes__(self):
        return (bytes([ord(self.typecode)]) +
                bytes(array(self.typecode, self)))

    def __eq__(self, other):
        return tuple(self) == tuple(other)

    def __abs__(self):
        return math.hypot(self.x, self.y)

    def __bool__(self):
        return bool(abs(self))

    def angle(self):
        return math.atan2(self.y, self.x)

    def __format__(self, fmt_spec=''):
        if fmt_spec.endswith('p'):
            fmt_spec = fmt_spec[:-1]
            coords = (abs(self), self.angle())
            outer_fmt = '<{}, {}>'
        else:
            coords = self
            outer_fmt = '({}, {})'
        components = (format(c, fmt_spec) for c in coords)
        return outer_fmt.format(*components)

    @classmethod
    def frombytes(cls, octets):
        typecode = chr(octets[0])
        memv = memoryview(octets[1:]).cast(typecode)
        return cls(*memv)

class ShortVector2d(Vector2d):
    typecode = 'f'

print('------------------------------------------------------------------------------------------------\n'
      '             9.8 Overriding class attributes                                                    \n'
      '------------------------------------------------------------------------------------------------\n')

print('------------------------------------------------------------------------------------------------\n'
      ' Customize the instance by setting the typecode attribute that was inherited from the class.    \n'
      '------------------------------------------------------------------------------------------------\n')
v1 = Vector2d(1.1, 2.2)
dumpd = bytes(v1)
print('dmpd = {0}\n'.format(dumpd))
print('len(dmpd) = {0}\n'.format(len(dumpd)))

v1.typecode = 'f'
dumpf = bytes(v1)
print('dmpf = {0}\n'.format(dumpf))
print('len(dmpf) = {0}\n'.format(len(dumpf)))

print('------------------------------------------------------------------------------------------------\n'
      ' ShortVector2d is a Vector2d subclass only for overriding the default typecode.                 \n'
      '------------------------------------------------------------------------------------------------\n')
sv = ShortVector2d(1/11, 1/27)
print('sv = \n{0}\n'.format(sv))
print('len(bytes(sv)) = {0}\n'.format(len(bytes(sv))))