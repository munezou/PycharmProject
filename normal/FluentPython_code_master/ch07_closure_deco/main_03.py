
'''
------------------------------------------------------------------------------------------------------------------------
7.4 Variable scope
------------------------------------------------------------------------------------------------------------------------
'''
print('-----------------------------------------------------------------------------------------------------------------\n'
      '                                            7.4 Variable scope                                                   \n'
      '-----------------------------------------------------------------------------------------------------------------\n')

print('---< f1 function >---')

b1 = 6
def f1(a):
    print('a = {0}'.format(a))
    try:
        print('b1 = {0}'.format(b1))
    except Exception as e:
        print(e)
        pass
    finally:
        pass

f1(3)
print()

print('--< f2 function >---')

b2 = 6
def f2(a):
    print('a = {0}'.format(a))
    try:
        print('b2 = {0}'.format(b2))
    except Exception as e:
        print(e)
        pass
    finally:
        pass
    b2 = 9


f2(3)
print()

print('---< f3 function >---')

b3 = 6
def f3(a):
    global b3
    print('a = {0}'.format(a))
    try:
        print('b3 = {0}'.format(b3))
    except Exception as e:
        print(e)
    finally:
        pass
    b3 = 9


f3(3)
print()

