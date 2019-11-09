'''
About Sequence + and *
'''
print ('---< About Sequence + and * >---')

l = [1, 2, 3]

print('l = {0}'.format(l))
print()

print('l * 5 = {0}'.format(l * 5))
print()

m = 5 * 'abcd'
print ('5 * "abcd" = {0}'.format(m))
print()

print ('---< Generate list of lists >---')
print ('Bad examople:')

weired_board = [['_'] * 3] * 3
print ('[["_"] * 3] * 3 = {0}'.format(weired_board))

weired_board[1][2] = '0'

print ('weired_board[1][2] = "0" Result: {0}'.format(weired_board))
print ()

print ('Good example:')

board = []

for i in range(3):
    row = ['_'] * 3
    board.append(row)

print ('board = {0}'.format(board))

board[2][0] = 'X'

print('board[2][0] = "X" Result: {0}'.format(board))
print ()

print ('---< Sequence and cumulative assignment >---')
print ('list example:')
l = [1, 2, 3]
print ('l = {0}, id(l) = {1}'.format(l, id(l)))
print()

l *= 2
print ('l * 2 = {0}, id(l * 2) = {1}'.format(l, id(l)))
print()

print ('tuple example:')

t = (1, 2, 3)
print('t = {0}, id(t) = {1}'.format(t, id(t)))
print()

t *= 2
print('t * 2 = {0}, id(t * 2) = {1}'.format(t, id(t)))
print ()

print ('---< Mystery of substitution by + = >---')
t = (1, 2, [30, 40])
print('t = {0}'.format(t))

t[2] += [50, 60]

print ('t[2] += [50, 60] Result: {0}'.format(t))
print()