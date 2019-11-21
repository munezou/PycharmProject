# common library
import os, sys
import time
from array import array
import math
from subprocess import check_call

print('------------------------------------------------------------------------------------------------\n'
      '             9.7 Private and "Protect" attributes                                               \n'
      '------------------------------------------------------------------------------------------------\n')


class Hero():
    def __init__(self, name, hp):
        ''' class permision '''
        # public variable
        self.name = name

        # private variable
        self.__hp = hp

    def self_introduction(self):
        print('My name is Brave: {0}.\n'.format(self.name))

    def get_hp(self):
        return self.__hp

    def set_hp(self, new_hp):
        self.__hp = new_hp


# create instance
hero = Hero("ヨシヒロ", 10)

print('hero.name = {0}\n'.format(hero.name))

print('hero.get_hp() = {0}\n'.format(hero.get_hp()))

try:
    print(hero.__hp)
except Exception as ex:
    print(ex)
    pass
finally:
    pass

hero.set_hp(15)
print('hero.get_hp() = {0}\n'.format(hero.get_hp()))

print(''.format(hero.__dict__))


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


v1 = Vector2d_hash(3, 4)

print('v1.__dict__ = {0}\n'.format(v1.__dict__))

print('v1._Vector2d_hash__y = {0}\n'.format(v1._Vector2d_hash__y))

print('------------------------------------------------------------------------------------------------\n'
      '             9.8 Save memory with class attribute __slots__                                     \n'
      '------------------------------------------------------------------------------------------------\n')

PROJECT_ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))


def slots_path(source_id):
    return os.path.join(PROJECT_ROOT_DIR, source_id)


try:
    print('---< excute mem_test(vector2d_v3) command >---')
    start_time = time.time()
    print(check_call(['python3', slots_path('mem_test.py'), 'vector2d_v3.py']))
    finish_time = time.time()

    print()

    lapted_time = finish_time - start_time
    print('lapted time(vector2D_v3) = {0}\n'.format(lapted_time))

    print('---< excute mem_test(vector2d_v3_slots) command >---')
    start_time = time.time()
    print(check_call(['python3', slots_path('mem_test.py'), 'vector2d_v3_slots.py']))
    finish_time = time.time()

    print()

    lapted_time = finish_time - start_time
    print('lapted time(vector2D_v3_slots) = {0}\n'.format(lapted_time))
except Exception as ex:
    print(ex)
    pass
finally:
    pass


