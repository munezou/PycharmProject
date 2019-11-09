'''
--------------------------------------------------
Chapter 8 Object reference, variability, recycling
---------------------------------------------------
'''
print('----------------------------------------------------------\n'
      '               8.1 Variables are not boxes.               \n'
      '----------------------------------------------------------\n')
a = [1, 2, 3]
b = a
a.append(4)

print('b = {0}'.format(b))
print()

try:
    class Gizmo:
        def __init__(self):
            print('Gizmo id: %d' % id(self))

    x = Gizmo()

    y = Gizmo() * 10

except Exception as e:
    print(e)
finally:
    pass


print('dir() = \n{0}'.format(dir()))

print('----------------------------------------------------------\n'
      '               8.2 Identity, equality, alias              \n'
      '----------------------------------------------------------\n')

charles = {'name': 'Charles L. Dodgson', 'born': 1832}
lewis = charles
print('lewis is charles = {0}'.format(lewis is charles))
print('id(chales) = {0}, id(lewis) = {1}'.format(id(charles), id(lewis)))
print()

lewis['baclance'] = 950

print('charles = \n{0}'.format(charles))
print()

alex = {'name': 'Charles L. Dodgson', 'balance': 950, 'born': 1832}
print('(alex == charles) is {0}'.format(alex == charles))
print('(alex is not charles) == {0}'.format(alex is not charles))
print()

print('----------------------------------------------------------\n'
      '        8.2.2 The relative invariance of tuples           \n'
      '----------------------------------------------------------\n')

t1 = (1, 2, [30, 40])
t2 = (1, 2, [30, 40])
print('(t1 == t2) is {0}'.format(t1 == t2))
print()

print('id(t1[-1]) = {0}'.format(id(t1[-1])))
print()

t1[-1].append(99)

print('t1 = {0}'.format(t1))
print('id(t1[-1]) = {0}'.format(id(t1[-1])))
print()

print('(t1 == t2) is {0}'.format(t1 == t2))
print()

print('----------------------------------------------------------\n'
      '          8.3 The default copy is ÅgshallowÅh.              \n'
      '----------------------------------------------------------\n')

l1 = [3, [55, 44], (7, 8, 9)]
l2 = list(l1)
print('l2 = \n{0}'.format(l2))
print()
print('(l2 == l1) is {0}'.format(l1 == l2))
print('(l2 is l1) is {0}'.format(l2 is l1))
print()
