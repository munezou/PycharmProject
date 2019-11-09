'''
SLICE
'''
invoice = """
0.....6.................................40..........52.55.......
1909  Pimoroni PiBrella                     $17.50    3  $52.50
1489  6mm Tactile Switch x20                 $4.95    2   $9.90
1510  Panavise Jr. -PV-201                  $28.00    1  $28.00
1601  piTFT Mini Kit 320x240                $34.95    1  $34.95
"""

# slice setting
SKU = slice(0, 6)
DESCRIPTION = slice(6, 40)
UNIT_PRICE  = slice(40, 52)
QUANTITY = slice(52, 55)
ITEM_TOTAL = slice(55, None)

# Setting of effective line
line_items = invoice.split('\n')[2:]

# output list
for item in line_items:
    print('unit_price = {0} , description statement = {1}'.format(item[UNIT_PRICE], item[DESCRIPTION]))

print()

print ('---< Add item to slice. >---')

l = list(range(10))
print ('l = {0}'.format(l))
print()

l[2:5] = [20, 30]

print ('l[2:5] = [20, 30] Result: {0}'.format(l))
print()

del l[5:7]

print ('del l[5:7] Result: {0}'.format(l))
print()

l[3::2] = [11, 22]

print ('l[3::2] = [11, 22] Result: {0}'.format(l))
print()

l[2:5] = [100]

print('l[2:5] = [100] Result: {0}'.format(l))
print()





