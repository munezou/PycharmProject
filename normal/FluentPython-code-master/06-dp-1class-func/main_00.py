from normal.PythonDataModel.ch06_dp_1class_func.classic_strategy import *

'''
------------------------------------------------------------------------------------------------------------------------
6.1 Strategy pattern as a refactoring case study
------------------------------------------------------------------------------------------------------------------------
'''
print('-----------------------------------------------------------------------------------------------------------------\n'
      '                             6.1.1 Typical Strategy pattern                                                      \n'
      '-----------------------------------------------------------------------------------------------------------------\n')

joe = Customer('John Doe', 0)
ann = Customer('Ann Smith', 1100)

cart = [LineItem('banana', 4, .5),
        LineItem('apple', 10, 1.5),
        LineItem('watermellon', 5, 5.0)]

Order_joe = Order(joe, cart, FidelityPromo())
print('Order_joe = \n {0}'.format(Order_joe))
print()

Order_ann = Order(ann, cart, FidelityPromo())
print('Order_ann = \n {0}'.format(Order_ann))
print()

banana_cart = [LineItem('banana', 30, .5),
               LineItem('apple', 10, 1.5)]

Order_joe_2 = Order(joe, banana_cart, BulkItemPromo())
print('Order_joe_2 = \n{0}'.format(Order_joe_2))
print()

long_order = [LineItem(str(item_code), 1, 1.0) for item_code in range(10)]
Order_joe_3 = Order(joe, long_order, LargeOrderPromo())
print('Order_joe_3 = \n{0}'.format(Order_joe_3))
print()

Order_joe_4 = Order(joe, cart, LargeOrderPromo())
print('Order_joe_4 = \n{0}'.format(Order_joe_4))
print()
