'''
------------------------------------------------------------------------------------------------------------------------
7.2 Decorator execution timing
------------------------------------------------------------------------------------------------------------------------
'''
print('-----------------------------------------------------------------------------------------------------------------\n'
      '                                    7.2 Decorator execution timing                                               \n'
      '-----------------------------------------------------------------------------------------------------------------\n')
from .registration import *

'''
------------------------------------------------------------------------------------------------------------------------
7.3 Improving strategy patterns using decorators
------------------------------------------------------------------------------------------------------------------------
'''
print('-----------------------------------------------------------------------------------------------------------------\n'
      '                          7.3 Improving strategy patterns using decorators                                       \n'
      '-----------------------------------------------------------------------------------------------------------------\n')
from .strategy_best4 import *

joe = Customer('John Doe', 0)
ann = Customer('Ann Smith', 1100)
cart = [LineItem('banana', 4, .5),
        LineItem('apple', 10, 1.5),
        LineItem('watermellon', 5, 5.0)]

Order_joe_cart_fidelity = Order(joe, cart, fidelity)
print('Order_joe_cart_fidelity = {0}'.format(Order_joe_cart_fidelity))

Order_ann_cart_fidelity = Order(ann, cart, fidelity)
print('Order_ann_cart_fidelity = {0}'.format(Order_ann_cart_fidelity))

banana_cart = [LineItem('banana', 30, .5),
               LineItem('apple', 10, 1.5)]

Order_joe_banana_bulk = Order(joe, banana_cart, bulk_item)
print('Order_joe_banana_bulk = {0}'.format(Order_joe_banana_bulk))

long_order = [LineItem(str(item_code), 1, 1.0) for item_code in range(10)]

Order_joe_long_large = Order(joe, long_order, large_order)
print('Order_joe_long_large = {0}'.format(Order_joe_long_large))

Order_joe_cart_large = Order(joe, cart, large_order)
print('Order_joe_cart_large = {0}'.format(Order_joe_cart_large))

# BEGIN STRATEGY_BEST_TESTS

Order_joe_long_best = Order(joe, long_order, best_promo)
print('Order_joe_long_best = {0}'.format(Order_joe_long_best))

Order_joe_banana_best = Order(joe, banana_cart, best_promo)
print('Order_joe_banana_best = {0}'.format(Order_joe_banana_best))

Order_ann_cart_best = Order(ann, cart, best_promo)
print('Order_ann_cart_best = {0}'.format(Order_ann_cart_best))

print()

# END STRATEGY_BEST_TESTS