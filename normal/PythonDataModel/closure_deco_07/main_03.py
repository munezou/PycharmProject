
'''
------------------------------------------------------------------------------------------------------------------------
7.4 Variable scope
------------------------------------------------------------------------------------------------------------------------
'''
print('-----------------------------------------------------------------------------------------------------------------\n'
      '                                            7.4 Variable scope                                                   \n'
      '-----------------------------------------------------------------------------------------------------------------\n')

print('---< f1 function >---')
try:
    b1 = 6
    def f1(a):
        print('a = {0}'.format(a))
        print('b1 = {0}'.format(b1))
except Exception as e:
    print(e)
finally:
    pass

f1(3)
print()

print('--< f2 function >---')
try:
    b2 = 6
    def f2(a):
        print('a = {0}'.format(a))
        print('b2 = {0}'.format(b2))
        b2 = 9
except Exception as e:
    print(e)
finally:
    pass

f2(3)
print()

print('---< f3 function >---')
try:
    b3 = 6
    def f3(a):
        global b3
        print('a = {0}'.format(a))
        print('b3 = {0}'.format(b3))
        b3 = 9
except Exception as e:
    print(e)
finally:
    pass

f3(3)
print()

