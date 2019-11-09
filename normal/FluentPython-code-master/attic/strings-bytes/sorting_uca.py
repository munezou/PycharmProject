from pyuca import Collator

def check(sorted_list):
    return 'CORRECT' if fruits == sorted_list else 'WRONG'

fruits = ['a巽a鱈', 'acerola', 'atemoia', 'caj叩', 'caju']

print('manual_sort', fruits)

plain_sort = sorted(fruits)

print('plain_sort ', plain_sort, check(plain_sort))

coll = Collator()

pyuca_sort = sorted(fruits, key=coll.sort_key)

print('pyuca_sort ', pyuca_sort, check(pyuca_sort))
