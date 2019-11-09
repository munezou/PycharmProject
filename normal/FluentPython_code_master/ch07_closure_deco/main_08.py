from normal.PythonDataModel.ch07_closure_deco.clockdeco_demo import snooze
from normal.PythonDataModel.ch07_closure_deco.generic import *

print(htmlize({1, 2, 3}))
print()

print(htmlize((1, 2, 3)))
print()

'''
------------------------------------------------------------------------------------------------------------------------
7.10 Parameterized decorator
------------------------------------------------------------------------------------------------------------------------
'''
from normal.PythonDataModel.ch07_closure_deco.registration import *

print('-----------------------------------------------------------------------------------------------------------------\n'
      '                        　　     7.10 Parameterized decorator    　                                               \n'
      '-----------------------------------------------------------------------------------------------------------------\n')
main()
print()

'''
------------------------------------------------------------------------------------------------------------------------
7.10.1 Registered decorator parameterization
------------------------------------------------------------------------------------------------------------------------
'''
from normal.PythonDataModel.ch07_closure_deco.registration_param import *

print('-----------------------------------------------------------------------------------------------------------------\n'
      '                        7.10.1 Registered decorator parameterization    　                                       \n'
      '-----------------------------------------------------------------------------------------------------------------\n')

print('f1() = {0}'.format(f1))
print('registtry = {0}'.format(registry))
print()

print('f2() = {0}'.format(f2))
print('registtry = {0}'.format(registry))
print()

print('f3() = {0}'.format(f3))
print('registtry = {0}'.format(registry))
print()

print('-----------------------------------------------------------------------------------------------------------------\n'
      '                             7.10.2 clock decorator parameterization    　                                       \n'
      '-----------------------------------------------------------------------------------------------------------------\n')
from normal.PythonDataModel.ch07_closure_deco.clockdeco_param import clock
import time

clock()
clock('{name}: {elapsed}')(time.sleep)(.2)  # doctest: +ELLIPSIS
clock('{name}({args}) dt={elapsed:0.3f}s')(time.sleep)(.2)
