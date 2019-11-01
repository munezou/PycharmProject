from .clockdeco import *

print('-----------------------------------------------------------------------------------------------------------------\n'
      '                                             7.6 NONLOCAL declaration                                            \n'
      '-----------------------------------------------------------------------------------------------------------------\n')

def make_averager():
    count = 0
    total = 0
    
    def averager(new_value):
        nonlocal count, total
        count += 1
        total += new_value
        return total / count
    
    return averager

ave = make_averager()
print('avg(10) = {0}'.format(ave(10)))
print()

print('-----------------------------------------------------------------------------------------------------------------\n'
      '                                   7.7 Simple decorator implementation                                           \n'
      '-----------------------------------------------------------------------------------------------------------------\n')
print('referance clockdeco_demo.py')