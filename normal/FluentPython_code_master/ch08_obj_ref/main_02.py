# common library
import weakref
from normal.PythonDataModel.ch08_obj_ref.cheese import *

print('-----------------------------------------------------------------\n'
      '                     8.5 del and garbage collection              \n'
      '-----------------------------------------------------------------\n')
s1 = {1, 2, 3}
s2 = s1

def byte():
    print('Gone with the wind.....')

ender = weakref.finalize(s1, byte)
print('ender.alive = {0}'.format(ender.alive))
print()

del s1
print('ender.alive = {0}'.format(ender.alive))
print()

s2 = 'spam'
print()

print('ender.alive = {0}'.format(ender.alive))
print()

print('-----------------------------------------------------------------\n'
      '                     8.6 weak reference　　　　　　              \n'
      '-----------------------------------------------------------------\n')
a_set = {0, 1}

wref = weakref.ref(a_set)
print('wref = {0}'.format(wref))
print()

print('wref() = {0}'.format(wref()))
print()

a_set = {2, 3, 4}
print('wref() = {0}'.format(wref()))
print()

print('wref() is None = {0}'.format(wref() is None))
print()

print('wref() is None = {0}'.format(wref() is None))
print()

print('-----------------------------------------------------------------\n'
      '            8.6.1 Conte "WeakValueDictionary"　　　　             \n'
      '-----------------------------------------------------------------\n')
stock = weakref.WeakValueDictionary()
catalog = [Cheese('Red Leicester'), Cheese('Tilsit'), Cheese('Brie'), Cheese('Parmesan')]

for cheese in catalog:
    stock[cheese.kind] = cheese

print('sorted(stock.keys()) = \n{0}'.format(sorted(stock.keys())))
print()

del catalog
print('sorted(stock.keys()) = \n{0}'.format(sorted(stock.keys())))
print()

del cheese
print('sorted(stock.keys()) = \n{0}'.format(sorted(stock.keys())))
print()
