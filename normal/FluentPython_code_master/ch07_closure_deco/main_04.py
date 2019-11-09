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
    finally:
        pass

print('-----------------------------------------------------------------------------------------------------------------\n'
      '                                   Error verification by disassembler                                            \n'
      '-----------------------------------------------------------------------------------------------------------------\n')
from dis import dis

print('dis(f1) = \n{0}'.format(dis(f1)))
print()

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
    finally:
        pass
    b2 = 9


print('-----------------------------------------------------------------------------------------------------------------\n'
      '                                   Error verification by disassembler                                            \n'
      '-----------------------------------------------------------------------------------------------------------------\n')

print('dis(f2) = \n{0}'.format(dis(f2)))
print()

f2(3)
print()
