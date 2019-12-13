'''
A multi-dimensional ``Vector`` class, take 2
'''

from array import array
import reprlib
import math
import numbers


class Vector:
    typecode = 'd'

    def __init__(self, components):
        self._components = array(self.typecode, components)

    def __iter__(self):
        return iter(self._components)

    def __repr__(self):
        components = reprlib.repr(self._components)
        components = components[components.find('['):-1]
        return 'Vector({})'.format(components)

    def __str__(self):
        return str(tuple(self))

    def __bytes__(self):
        return (bytes([ord(self.typecode)]) +
                bytes(self._components))

    def __eq__(self, other):
        return tuple(self) == tuple(other)

    def __abs__(self):
        return math.sqrt(sum(x * x for x in self))

    def __bool__(self):
        return bool(abs(self))

# BEGIN VECTOR_V2
    def __len__(self):
        return len(self._components)

    def __getitem__(self, index):
        cls = type(self)  # <1>
        if isinstance(index, slice):  # <2>
            return cls(self._components[index])  # <3>
        elif isinstance(index, numbers.Integral):  # <4>
            return self._components[index]  # <5>
        else:
            msg = '{cls.__name__} indices must be integers'
            raise TypeError(msg.format(cls=cls))  # <6>
# END VECTOR_V2

    @classmethod
    def frombytes(cls, octets):
        typecode = chr(octets[0])
        memv = memoryview(octets[1:]).cast(typecode)
        return cls(memv)

# BEGIN VECTOR_DEMO: take_2
print   (
        '------------------------------------------------------------------------------------------------------------\n'
        '              A ``Vector`` is built from an iterable of numbers::                                           \n'
        '------------------------------------------------------------------------------------------------------------\n'
        )

print('Vector([3.1, 4.2]) = {0}\n'.format(Vector([3.1, 4.2])))

print('Vector((3, 4, 5)) = {0}\n'.format(Vector((3, 4, 5))))

print('Vector(range(10)) = \n{0}\n'.format(Vector(range(10))))

print   (
        '------------------------------------------------------------------------------------------------------------\n'
        '              Tests with 2-dimensions (same results as ``vector2d_v1.py``)::                                \n'
        '------------------------------------------------------------------------------------------------------------\n'
        )

v1 = Vector([3, 4])
x, y = v1
print('x = {0}, y = {1}\n'.format(x, y))

print('v1 = {0}\n'.format(v1))

v1_clone = eval(repr(v1))
print('v1 == v1_clone = {0}\n'.format(v1 == v1_clone))

print('v1 = {0}\n'.format(v1))

octets = bytes(v1)
print('octets = \n{0}\n'.format(octets))

print('abs(v1) = {0}\n'.format(abs(v1)))

print('(bool(v1) = {0}, bool(Vector([0, 0]))) = {1})\n'.format(bool(v1), bool(Vector([0, 0]))))

print   (
        '------------------------------------------------------------------------------------------------------------\n'
        '              Test of ``.frombytes()`` class method:                                                        \n'
        '----------------------------------------------------------------   -----------------------------------------\n'
        )

v1_clone = Vector.frombytes(bytes(v1))
print('v1_clone = {0}\n'.format(v1_clone))

print('v1 == v1_clone = {0}\n'.format(v1 == v1_clone))

print   (
        '------------------------------------------------------------------------------------------------------------\n'
        '              Tests with 3-dimensions::                                                                     \n'
        '------------------------------------------------------------------------------------------------------------\n'
        )

v1 = Vector([3, 4, 5])
x, y, z = v1
print('(x = {0}, y = {1}, z = {2})\n'.format(x, y, z))

print('v1 = {0}\n'.format(v1))

v1_clone = eval(repr(v1))
print('v1 == v1_clone = {0}\n'.format(v1 == v1_clone))

print('v1 = {0}\n'.format(v1))

print('abs(v1) = {0}\n'.format(abs(v1))) # doctest:+ELLIPSIS

print('(bool(v1) = {0}, bool(Vector([0, 0, 0])) = {1})\n'.format(bool(v1), bool(Vector([0, 0, 0]))))

print   (
        '------------------------------------------------------------------------------------------------------------\n'
        '              Tests with many dimensions::                                                                  \n'
        '------------------------------------------------------------------------------------------------------------\n'
        )

v7 = Vector(range(7))
print('v7 = {0}\n'.format(v7))

print('abs(v7) = {0}\n'.format(abs(v7)))  # doctest:+ELLIPSIS

print   (
        '------------------------------------------------------------------------------------------------------------\n'
        '              Test of ``.__bytes__`` and ``.frombytes()`` methods::                                         \n'
        '------------------------------------------------------------------------------------------------------------\n'
        )

v1 = Vector([3, 4, 5])
v1_clone = Vector.frombytes(bytes(v1))
print('v1_clone = {0}\n'.format(v1_clone))

print('v1 == v1_clone = {0}\n'.format(v1 == v1_clone))

print   (
        '------------------------------------------------------------------------------------------------------------\n'
        '              Tests of sequence behavior::                                                                   \n'
        '------------------------------------------------------------------------------------------------------------\n'
        )

v1 = Vector([3, 4, 5])
print('len(v1) = {0}\n'.format(len(v1)))

print('(v1[0] = {0}, v1[len(v1)-1] = {1}, v1[-1] = {2})\n'.format(v1[0], v1[len(v1)-1], v1[-1]))

# END VECTOR_DEMO: take_2

# Test of slicing::

# BEGIN VECTOR_V2_DEMO

v7 = Vector(range(7))
print('v7[-1] = {0}\n'.format(v7[-1])) # <1>

print('v7[1:4] = {0}\n'.format(v7[1:4])) # <2>

print('v7[-1:] = {0}\n'.format(v7[-1:]))

try:
    v7[1,2]
except Exception as ex:
    print(ex)
    pass
finally:
    pass
# END VECTOR_V2_DEMO