# common library
from datetime import datetime
from array import array
import math

print('------------------------------------------------------------------------------------------------\n'
      '             9.4 classmethod and staticmethod 　　　　　　　　　　　　　　　　　　　　　　　　　\n'
      '------------------------------------------------------------------------------------------------\n')


class Demo:
    @classmethod
    def klassmeth(*args):
        return args

    @staticmethod
    def statmeth(*args):
        return args


print('Demo.klassmeth() = \n{0}\n'.format(Demo.klassmeth()))

print('Demo.klassmeth() = \n{0}\n'.format(Demo.klassmeth('spam')))

print('Demo.statmeth() = \n{0}\n'.format(Demo.statmeth()))

print('Demo.statmeth() = \n{0}\n'.format(Demo.statmeth('spam')))

print('------------------------------------------------------------------------------------------------\n'
      '             9.5 Output format                  　　　　　　　　　　　　　　　　　　　　　　　　\n'
      '------------------------------------------------------------------------------------------------\n')

# flot type specifier
brl = 1 / 2.43
print('brl = {0}\n'.format(brl))

print('format(brl, "0.4f") = {0}\n'.format(format(brl, '0.4f')))

print('1 BRL = {rate:0.2f} USD\n'.format(rate=brl))

# bit type specifier
print('format(42, "b") = {0}\n'.format(format(42, 'b')))

# % Format
print('format(2/3, ".1%") = {0}\n'.format(format(2 / 3, '.1%')))

# current date time
now = datetime.now()
print('now = {0}\n'.format(format(now, '%H:%M:%S')))

print('It is now {:%I:%M %p}\n'.format(now))


# BEGIN VECTOR2D_V0
class Vector2d:
    typecode = 'd'  # <1>

    def __init__(self, x, y):
        self.x = float(x)  # <2>
        self.y = float(y)

    def __iter__(self):
        return (i for i in (self.x, self.y))  # <3>

    def __repr__(self):
        class_name = type(self).__name__
        return '{}({!r}, {!r})'.format(class_name, *self)  # <4>

    def __str__(self):
        return str(tuple(self))  # <5>

    def __bytes__(self):
        return (bytes([ord(self.typecode)]) +  # <6>
                bytes(array(self.typecode, self)))  # <7>

    def __eq__(self, other):
        return tuple(self) == tuple(other)  # <8>

    def __abs__(self):
        return math.hypot(self.x, self.y)  # <9>

    def __bool__(self):
        return bool(abs(self))  # <10>

    def angle(self):
        return math.atan2(self.y, self.x)

    def __format__(self, fmt_spec=''):
        if fmt_spec.endswith('p'):  # <1>
            fmt_spec = fmt_spec[:-1]  # <2>
            coords = (abs(self), self.angle())  # <3>
            outer_fmt = '<{}, {}>'  # <4>
        else:
            coords = self  # <5>
            outer_fmt = '({}, {})'  # <6>
        components = (format(c, fmt_spec) for c in coords)  # <7>
        return outer_fmt.format(*components)  # <8>


# END VECTOR2D_V0

v1 = Vector2d(3, 4)
print('format(v1) = \n{0}\n'.format(format(v1)))

print('format(v1, ".2f") = \n{0}\n'.format(format(format(v1, '.2f'))))

print('format(v1, ".3e") = \n{0}\n'.format(format(v1, '.3e')))

print('format(Vector2d(1, 1), "p") = \n{0}\n'.format(format(Vector2d(1, 1), "p")))

print('format(Vector2d(1, 1), ".3ep") = \n{0}\n'.format(format(Vector2d(1, 1), '.3ep')))

print('format(Vector2d(1, 1), "0.5fp") = \n{0}\n'.format(format(Vector2d(1, 1), '0.5fp')))

print('------------------------------------------------------------------------------------------------\n'
      '             9.6 Hashable Vector2D              　　　　　　　　　　　　　　　　　　　　　　　　\n'
      '------------------------------------------------------------------------------------------------\n')


class Vector2d_hash:
    typecode = 'd'

    def __init__(self, x, y):
        self.__x = float(x)
        self.__y = float(y)

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

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

    def __hash__(self):
        return hash(self.x) ^ hash(self.y)

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


# create instance
v1 = Vector2d_hash(3, 4)

print('({0}. {1})\n'.format(v1.x, v1.y))

try:
    v1.x = 7
except Exception as ex:
    print('v1.x = 7: Error = {0}\n'.format(ex))
    pass
finally:
    pass

# create instance
v1 = Vector2d_hash(3, 4)
v2 = Vector2d_hash(3.1, 4.2)

print('hash(v1) = {0}, hash(v2) = {1}\n'.format(hash(v1), hash(v2)))

try:
    print('set([v1, v2]) = \n{0}\n'.format(set([v1, v2])))
except Exception as ex:
    print(ex)
    print()
    pass
finally:
    pass

