
'''
------------------------------------------------------------------------------------------------------------------------
7 Function decorators and closures
------------------------------------------------------------------------------------------------------------------------
'''
print('-----------------------------------------------------------------------------------------------------------------\n'
      '                                7.1 Decorator basics                                                             \n'
      '-----------------------------------------------------------------------------------------------------------------\n')

def deco(func):
    def wrapper(*args, **kwargs):
        print('\n--start--')
        func(*args, **kwargs)
        print('--end--')
    return wrapper

@deco
def test():
    print('Hello Decorator')

test()

'''
Example of a decorator in a method with a return value
'''

def deco2(func):
    import os
    def wrapper(*args,**kwargs):
        res = os.linesep + '--start--' + os.linesep
        res += func(*args,**kwargs) + '!' + os.linesep
        res += '--end--'
        return print(res)
    return wrapper

@deco2
def test2():
    return('Hello Decorator')

test2()

print()

'''
Nest decorators.
'''
def deco_html(func):
    def wrapper(*args, **kwargs):
        res = '\n<html>'
        res = res + func(*args, **kwargs)
        res = res + '</html>'
        return print(res)
    return wrapper

def deco_body(func):
    def wrapper(*args, **kwargs):
        res = '<body>'
        res = res + func(*args, **kwargs)
        res = res + '</body>'
        return res
    return wrapper

@deco_html
@deco_body
def test3():
    return 'Hello Decorator'

test3()





