from normal.FluentPython_code_master.ch06_dp_1class_func.strategy_best import *

'''
------------------------------------------------------------------------------------------------------------------------
6.1.3 Choosing the best strategy in a simple way
------------------------------------------------------------------------------------------------------------------------
'''
print('-----------------------------------------------------------------------------------------------------------------\n'
      '                       6.1.3 Choosing the best strategy in a simple way                                          \n'
      '-----------------------------------------------------------------------------------------------------------------\n')
joe = Customer('John Doe', 0)
ann = Customer('Ann Smith', 1100)
cart = [LineItem('banana', 4, .5),
        LineItem('apple', 10, 1.5),
        LineItem('watermellon', 5, 5.0)]

Order_joe_cart_fidelity = Order(joe, cart, fidelity_promo)
print('Order_joe_cart_fidelity = {0}'.format(Order_joe_cart_fidelity))

Order_ann_cart_fidelity = Order(ann, cart, fidelity_promo)
print('Order_ann_cart_fidelity = {0}'.format(Order_ann_cart_fidelity))

banana_cart = [LineItem('banana', 30, .5),
               LineItem('apple', 10, 1.5)]

Order_banana_bulk_item = Order(joe, banana_cart, bulk_item_promo)
print('Order_banana_bulk_item = {0}'.format(Order_banana_bulk_item))

long_order = [LineItem(str(item_code), 1, 1.0) for item_code in range(10)]

Order_long_large_order = Order(joe, long_order, large_order_promo)
print('Order_long_large_order = {0}'.format(Order_long_large_order))

Order_joe_cart_large_order = Order(joe, cart, large_order_promo)
print('Order_joe_cart_large_order = {0}'.format(Order_joe_cart_large_order))

print()

# BEGIN STRATEGY_BEST_TESTS

print('---< BEGIN STRATEGY_BEST_TESTS >---')
Order_joe_long_best = Order(joe, long_order, best_promo)  # <1>
print('Order_joe_long_best = {0}'.format(Order_joe_long_best))

Order_joe_banana_best = Order(joe, banana_cart, best_promo)  # <2>
print('Order_joe_banana_best = {0}'.format(Order_joe_banana_best))

Order_ann_cart_best = Order(ann, cart, best_promo)  # <3>
print('Order_ann_cart_best = {0}'.format(Order_ann_cart_best))
print()


# END STRATEGY_BEST_TESTS