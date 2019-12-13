"""
A multi-dimensional ``Vector`` class, take 3
"""

from array import array
import reprlib
import math
import numbers

print(__doc__)

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

    def __len__(self):
        return len(self._components)

    def __getitem__(self, index):
        cls = type(self)
        if isinstance(index, slice):
            return cls(self._components[index])
        elif isinstance(index, numbers.Integral):
            return self._components[index]
        else:
            msg = '{.__name__} indices must be integers'
            raise TypeError(msg.format(cls))

# BEGIN VECTOR_V3_GETATTR
    shortcut_names = 'xyzt'

    def __getattr__(self, name):
        cls = type(self)  # <1>
        if len(name) == 1:  # <2>
            pos = cls.shortcut_names.find(name)  # <3>
            if 0 <= pos < len(self._components):  # <4>
                return self._components[pos]
        msg = '{.__name__!r} object has no attribute {!r}'  # <5>
        raise AttributeError(msg.format(cls, name))
# END VECTOR_V3_GETATTR

# BEGIN VECTOR_V3_SETATTR
    def __setattr__(self, name, value):
        cls = type(self)
        if len(name) == 1:  # <1>
            if name in cls.shortcut_names:  # <2>
                error = 'readonly attribute {attr_name!r}'
            elif name.islower():  # <3>
                error = "can't set attributes 'a' to 'z' in {cls_name!r}"
            else:
                error = ''  # <4>
            if error:  # <5>
                msg = error.format(cls_name=cls.__name__, attr_name=name)
                raise AttributeError(msg)
        super().__setattr__(name, value)  # <6>

# END VECTOR_V3_SETATTR

    @classmethod
    def frombytes(cls, octets):
        typecode = chr(octets[0])
        memv = memoryview(octets[1:]).cast(typecode)
        return cls(memv)


# BEGIN VECTOR_DEMO: take_3
print   (
        '------------------------------------------------------------------------------------------------------------\n'
        '             A ``Vector`` is built from an iterable of numbers::                                            \n'
        '------------------------------------------------------------------------------------------------------------\n'
        )
print('Vector([3.1, 4.2]) = {0}\n'.format(Vector([3.1, 4.2])))

print('Vector((3, 4, 5)) = {0}\n'.format(Vector((3, 4, 5))))

print('Vector(range(10)) = {0}\n'.format(Vector(range(10))))


print   (
        '------------------------------------------------------------------------------------------------------------\n'
        '             Tests with 2-dimensions (same results as ``vector2d_v1.py``)::                                 \n'
        '------------------------------------------------------------------------------------------------------------\n'
        )

v1 = Vector([3, 4])
x, y = v1
print('({0}, {1})\n'.format(x, y))

print('v1 = {0}\n'.format(v1))

v1_clone = eval(repr(v1))
print('v1 == v1_clone = {0}\n'.format(v1 == v1_clone))

print('v1 = {0}\n'.format(v1))

octets = bytes(v1)
print('octets = \n{0}\n'.format(octets))

print('abs(v1) = {0}\n'.format(abs(v1)))

print('({0}, {1})\n'.format(bool(v1), bool(Vector([0, 0]))))

print   (
        '------------------------------------------------------------------------------------------------------------\n'
        '             Test of ``.frombytes()`` class method:                                                         \n'
        '------------------------------------------------------------------------------------------------------------\n'
        )

v1_clone = Vector.frombytes(bytes(v1))
print('v1_clone = {0}\n'.format(v1_clone))

print('v1 == v1_clone = {0}\n'.format(v1 == v1_clone))

print   (
        '------------------------------------------------------------------------------------------------------------\n'
        '             Tests with 3-dimensions::                                                                      \n'
        '------------------------------------------------------------------------------------------------------------\n'
        )

v1 = Vector([3, 4, 5])
x, y, z = v1
print('({0}, {1}, {2})\n'.format(x, y, z))

print('v1 = {0}\n'.format(v1))

v1_clone = eval(repr(v1))
print('v1 == v1_clone = {0}\n'.format(v1 == v1_clone))

print('v1 = {0}\n'.format(v1))

print('abs(v1) = {0}\n'.format(abs(v1))) # doctest:+ELLIPSIS

print('({0}, {1})\n'.format(bool(v1), bool(Vector([0, 0, 0]))))

print   (
        '------------------------------------------------------------------------------------------------------------\n'
        '             Tests with many dimensions::                                                                   \n'
        '------------------------------------------------------------------------------------------------------------\n'
        )

v7 = Vector(range(7))
print('v7 = {0}\n'.format(v7))

print('abs(v7) = {0}\n'.format(abs(v7))) # doctest:+ELLIPSIS

print   (
        '------------------------------------------------------------------------------------------------------------\n'
        '             Test of ``.__bytes__`` and ``.frombytes()`` methods::                                          \n'
        '------------------------------------------------------------------------------------------------------------\n'
        )

v1 = Vector([3, 4, 5])
v1_clone = Vector.frombytes(bytes(v1))
print('v1_clone = {0}\n'.format(v1_clone))

print('v1 == v1_clone = {0}\n'.format(v1 == v1_clone))

print   (
        '------------------------------------------------------------------------------------------------------------\n'
        '                    Tests of sequence behavior::                                                            \n'
        '------------------------------------------------------------------------------------------------------------\n'
        )

v1 = Vector([3, 4, 5])
print('len(v1) = {0}\n'.format(len(v1)))

print('({0}, {1}, {2})\n'.format(v1[0], v1[len(v1)-1], v1[-1]))

print   (
        '------------------------------------------------------------------------------------------------------------\n'
        '                    Test of slicing::                                                                        \n'
        '------------------------------------------------------------------------------------------------------------\n'
        )

v7 = Vector(range(7))
print('v7[-1] = {0}\n'.format(v7[-1]))

print('v7[1:4] = {0}\n'.format(v7[1:4]))

print('v7[-1:] = {0}\n'.format(v7[-1:]))

try:
    print('v7[1,2] = {0}\n'.format(v7[1,2]))
    pass
except Exception as ex:
    print(ex)
    pass
finally:
    pass

print   (
        '------------------------------------------------------------------------------------------------------------\n'
        '                    Tests of dynamic attribute access::                                                                       \n'
        '------------------------------------------------------------------------------------------------------------\n'
        )

v7 = Vector(range(10))
print('v7.x = {0}\n'.format(v7.x))

print('({0}, {1}, {2})\n'.format(v7.y, v7.z, v7.t))

print   (
        '------------------------------------------------------------------------------------------------------------\n'
        '                    Dynamic attribute lookup failures::                                                     \n'
        '------------------------------------------------------------------------------------------------------------\n'
        )

try:
    print('v7.k = {0}\n'.format(v7.k))
    pass
except Exception as ex:
    print(ex)
    pass
else:
    pass
finally:
    pass

v3 = Vector(range(3))

try:
    print('v3.t = {0}\n'.format(v3.t))
    pass
except Exception as ex:
    print(ex)
    pass
finally:
    pass

try:
    print('v3.spam = {0}\n'.format(v3.spam))
    pass
except Exception as ex:
    print(ex)
    pass
finally:
    pass

print   (
        '------------------------------------------------------------------------------------------------------------\n'
        '                  Tests of preventing attributes from ''a'' to ''z''::                                      \n'
        '------------------------------------------------------------------------------------------------------------\n'
        )

try:
    v1.x = 7
    pass
except Exception as ex:
    print(ex)
    pass
finally:
    pass

try:
    v1.w = 7
    pass
except Exception as ex:
    print(ex)
    pass
finally:
    pass

print   (
        '------------------------------------------------------------------------------------------------------------\n'
        '                  Other attributes can be set::                                                             \n'
        '------------------------------------------------------------------------------------------------------------\n'
        )

v1.X = 'albatross'
print('v1.X = {0}\n'.format(v1.X))

v1.ni = 'Ni!'
print('v1.ni = {0}\n'.format(v1.ni))

# END VECTOR_DEMO: take_3