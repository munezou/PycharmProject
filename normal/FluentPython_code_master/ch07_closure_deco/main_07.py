'''
------------------------------------------------------------------------------------------
Revised clockdeco
------------------------------------------------------------------------------------------
'''
import time
from normal.FluentPython_code_master.ch07_closure_deco.clockdeco_modify import clock

@clock
def snooze(second):
    time.sleep(second)

@clock
def factorial(n):
    return 1 if n < 2 else n * factorial(n - 1)

if __name__ == '__main__':
    print('*' * 40, 'Calling snooze(.123)')
    snooze(.123)
    print('*' * 40, 'Calling snooze(6)')
    print('6! =', factorial(6))

'''
------------------------------------------------------------------------------------------------------------------------
7.8 Standard library decorator
------------------------------------------------------------------------------------------------------------------------
'''
print('-----------------------------------------------------------------------------------------------------------------\n'
      '                           7.8.1 Memoization using functools.lru_cache                                           \n'
      '-----------------------------------------------------------------------------------------------------------------\n')
from normal.FluentPython_code_master.ch07_closure_deco.fibo_demo import *

print('---< no using lru_cache >---')
fibonacci(6)
print()

from normal.FluentPython_code_master.ch07_closure_deco.fibo_demo_lru import *

print('---< using lru_cache >---')
fibonacci_lru(30)

print('-----------------------------------------------------------------------------------------------------------------\n'
      '                           7.8.2 Single dispatch generic function                                                \n'
      '-----------------------------------------------------------------------------------------------------------------\n')
import html

def htmlize(obj):
    contents = html.escape(repr(obj))
    return '<pre>{0}</pre>'.format(contents)

print(htmlize({1, 2, 3}))
print()

print(htmlize(abs))
print()

print(htmlize('Heimich & Co.\n- a game'))
print()

print(htmlize(42))
print()

print(htmlize(['alpha', 66, {3, 2, 1}]))
print()



