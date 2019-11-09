'''
------------------------------------------------------------------------------------------------------------------------
Chapter 5 First Class Functions
------------------------------------------------------------------------------------------------------------------------
'''
import functools
from functools import reduce
from functools import partial
import operator
from operator import add
from operator import mul
from operator import itemgetter
from operator import attrgetter
from operator import methodcaller
from collections import namedtuple
import unicodedata
import random

print('-------------------------------------------------------------------\n'
      '     　　　　　　5.1 Treat functions like objects.　　　　　　　　　　　\n'
      '-------------------------------------------------------------------\n')
def factorial(n):
    '''return n!'''
    return 1 if n < 2 else n * factorial(n - 1)

print('factorial(42) = {0}'.format(factorial(42)))
print()
print('factorial.__doc__ = {0}'.format(factorial.__doc__))
print()
print('type(factorial) = {0}'.format(type(factorial)))
print()

print('---< Pass a function through another as an argument. >---')
fact = factorial
print('fact = {0}'.format(fact))
print()
print('fact(5) = {0}'.format(fact(5)))
print()
map_factorial = map(factorial, range(11))
print('map_factorial = {0}'.format(map_factorial))
print()

map_fact = list(map(fact, range(11)))
print('map_fact = {0}'.format(map_fact))
print()

print('-------------------------------------------------------------------\n'
      '                   5.2 Higher order function                       \n'
      '-------------------------------------------------------------------\n')

print('---< Sort the list of words by length. >---')
fruits = ['strawberry', 'fig', 'apple', 'cherry', 'raspberry', 'banana']
print('sorted(fruits, key=len) = \n{0}'.format(sorted(fruits, key=len)))
print()

print('---< Sort the list of words in reverse spelling. >---')

def reserve(word):
    return word[::-1]

reverse_word = reserve('testing')
print('reverse_word = {0}'.format(reverse_word))
print()
print('sorted(fruits, key=reserve) = {0}'.format(sorted(fruits, key=reserve)))
print()

print('-------------------------------------------------------------------\n'
      '     5.2.1 The latest alternative to map, filter and reduce        \n'
      '-------------------------------------------------------------------\n')
list_map = list(map(fact, range(6)))
print('list_map = {0}'.format(list_map))
print()

list_map_mark = [fact(n) for n in range(6)]
print('list_map_mark = {0}'.format(list_map_mark))
print()

list_map_filter = list(map(fact, filter(lambda n: n %2, range(6))))
print('list_map_filter = {0}'.format(list_map_filter))
print()

list_map_mark_filter = [factorial(n) for n in range(6) if n % 2]
print('list_map_mark_filter = {0}'.format(list_map_mark_filter))
print()

print('---< Use reduce and sum to calculate the sum of integers from 0 to 100. >---')
add_integer_reduce = reduce(add, range(100))
print('add_integer_reduce ={0}'.format(add_integer_reduce))
print()

add_integer_sum = sum(range(100))
print('add_integer_sum ={0}'.format(add_integer_sum))
print()

'''
------------------------------------------------------------------------------------------------------------------------
5.3 lambda(Anonymous function)
------------------------------------------------------------------------------------------------------------------------
'''
print('-------------------------------------------------------------------\n'
      '                   5.3 lambda(Anonymous function)                   \n'
      '-------------------------------------------------------------------\n')
print('---< Use lambda to sort the list by reverse spelling of words. >---')

lambda_sort = sorted(fruits, key=lambda word: word[::-1])
print('lambda_sort ={0}'.format(lambda_sort))
print()

'''
------------------------------------------------------------------------------------------------------------------------
5.4 7 callable objects
------------------------------------------------------------------------------------------------------------------------
'''
print('-------------------------------------------------------------------\n'
      '                     5.4   7 callable objects                      \n'
      '-------------------------------------------------------------------\n')
callable_function = abs, str, 13
print('callable_function = {0}'.format(callable_function))
print()

callable_function_link_type = [callable(obj) for obj in (abs, str, 13)]
print('callable_function_link_type = {0}'.format(callable_function_link_type))
print()

'''
------------------------------------------------------------------------------------------------------------------------
5.5 Callable user-defined types
------------------------------------------------------------------------------------------------------------------------
'''
print('-------------------------------------------------------------------\n'
      '                 5.5 Callable user-defined types                   \n'
      '-------------------------------------------------------------------\n')

print('---< BingoCage is a class that has the ability to select elements from a shuffled list. >---')

class BingoCage:
    def __init__(self, items):
        self._items = list(items)
        random.shuffle(self._items)

    def pick(self):
        try:
            return self._items.pop()
        except IndexError:
            raise LookupError('pick from empty BingoCage!')

    def __call__(self):
        return self.pick()

bingo = BingoCage(range(5))
print('bingo.pick() = {0}'.format(bingo.pick()))
print()
print('bingo.pick() = {0}'.format(bingo.pick()))
print()

print('bingo() = {0}'.format(bingo()))
print()
print('bingo() = {0}'.format(bingo()))
print()

print('callable(bingo) = {0}'.format(callable(bingo)))
print()

'''
------------------------------------------------------------------------------------------------------------------------
5.6 Function introspection
------------------------------------------------------------------------------------------------------------------------
'''
print('-------------------------------------------------------------------\n'
      '                   5.6 Function introspection                      \n'
      '-------------------------------------------------------------------\n')
print('dir(factorial) = \n{0}'.format(dir(factorial)))
print()

print('---< Get function attributes that do not exist in normal instances. >---')
class C: pass

obj = C()
def func(): pass
attribute = sorted(set(dir(func)) - set(dir(obj)))
print('attribute = \n{0}'.format(attribute))
print()

'''
------------------------------------------------------------------------------------------------------------------------
5.7 From positional arguments to keyword-only arguments
------------------------------------------------------------------------------------------------------------------------
'''
print('-------------------------------------------------------------------\n'
      '   5.7 From positional arguments to keyword-only arguments         \n'
      '-------------------------------------------------------------------\n')
def tag(name, *content, cls=None, **attrs):
    """Generate one or more HTML tags"""
    if cls is not None:
        attrs['class'] = cls
    if attrs:
        attr_str = ''.join(' %s="%s"' % (attr, value)
                           for attr, value
                           in sorted(attrs.items()))
    else:
        attr_str = ''
    if content:
        return '\n'.join('<%s%s>%s</%s>' %
                         (name, attr_str, c, name) for c in content)
    else:
        return '<%s%s />' % (name, attr_str)

tag_br = tag('br')
print('tag_br = {0}'.format(tag_br))
print()

tag_p_hello = tag('p', 'hello')
print('tag_p_hello = {0}'.format(tag_p_hello))
print()

tag_p_hello_world = tag('p', 'hello', 'world')
print('tag_p_hello_world = \n{0}'.format(tag_p_hello_world))
print()

tag_p_hello_id = tag('p', 'hello', id=33)
print('tag_p_hello_id = {0}'.format(tag_p_hello_id))
print()

tag_p_hello_world_sidebar = tag('p', 'hello', 'world', cls='sidebar')
print('tag_p_hello_world_sidebar = \n{0}'.format(tag_p_hello_world_sidebar))
print()

tag_content_image = tag(content='testing', name="img")
print('tag_content_image = {0}'.format(tag_content_image))
print()

my_tag = {'name': 'img', 'title': 'Sunset Boulevard', 'src': 'sunset.jpg', 'cls': 'framed'}
tag_my_tag = tag(**my_tag)
print('tag_my_tag = {0}'.format(tag_my_tag))
print()

def f(a, *, b):
    return a, b

print('f(1, b=2) = {0}'.format(f(1, b=2)))
print()

'''
------------------------------------------------------------------------------------------------------------------------
5.8 Get argument information
------------------------------------------------------------------------------------------------------------------------
'''
print('-------------------------------------------------------------------\n'
      '                     5.8 Get argument information                  \n'
      '-------------------------------------------------------------------\n')

print('---< Function to truncate a string >---')

def clip(text, max_len=80):
    """Return text clipped at the last space before or after max_len
    """
    end = None
    if len(text) > max_len:
        space_before = text.rfind(' ', 0, max_len)
        if space_before >= 0:
            end = space_before
        else:
            space_after = text.rfind(' ', max_len)
            if space_after >= 0:
                end = space_after
    if end is None:  # no spaces were found
        end = len(text)
    return text[:end].rstrip()

from clip import clip
print('clip.__default__ = {0}'.format(clip.__defaults__))
print()

print('clip.__code__ = {0}'.format(clip.__code__))
print()

print('clip.__code__.co_varnames = {0}'.format(clip.__code__.co_varnames))
print()

print('---< Extract the function signature. >---')

from inspect import signature
sig = signature(clip)

print('sig = {0}'.format(sig))
print()

for name, param in sig.parameters.items():
    print('{0} : {1} = {2}'.format(param.kind, name, param.default))

print()

print('---< Bind the tag function signature to the argument dict. >---')

import inspect
sig = inspect.signature(tag)
bound_args = sig.bind(**my_tag)

print('bound_args = \n{0}'.format(bound_args))
print()

for name, value in bound_args.arguments.items():
    print('{0} = {1}'.format(name, value))

del my_tag['name']

try:
    bound_args = sig.bind(**my_tag)
except Exception as ex:
    print(ex)
    print(type(ex))
finally:
    pass

'''
------------------------------------------------------------------------------------------------------------------------
5.9 function annotation
------------------------------------------------------------------------------------------------------------------------
'''
print('-------------------------------------------------------------------\n'
      '                       5.9 function annotation                     \n'
      '-------------------------------------------------------------------\n')

print('---< Annotated clip function >---')

'''
Annotation is an English word meaning “annotation”. 
In the IT field, additional information embedded in data and programs using a special notation is often called in this way.
'''

def clip(text:str, max_len:'int > 0'=80) -> str:  # <1>
    """Return text clipped at the last space before or after max_len
    """
    end = None
    if len(text) > max_len:
        space_before = text.rfind(' ', 0, max_len)
        if space_before >= 0:
            end = space_before
        else:
            space_after = text.rfind(' ', max_len)
            if space_after >= 0:
                end = space_after
    if end is None:  # no spaces were found
        end = len(text)
    return text[:end].rstrip()

print('clip.__annotations__ = \n{0}'.format(clip.__annotations__))
print()

sig = signature(clip)
print('sig.return_annotation = {0}'.format(sig.return_annotation))
print()

for param in sig.parameters.values():
    note = repr(param.annotation).ljust(13)
    print('{0} : {1} = {2}'.format(note, param.name, param.default))

print()

'''
------------------------------------------------------------------------------------------------------------------------
5.10 Package for functional programming
------------------------------------------------------------------------------------------------------------------------
'''
print('-------------------------------------------------------------------\n'
      '                       5.10.1 operator module                      \n'
      '-------------------------------------------------------------------\n')
print('---< Factorial calculation implemented with reduce and anonymous function >---')

def fact(n):
    return reduce(lambda a, b: a*b, range(1, n + 1))

print('fact(5) = {0}'.format(fact(5)))
print()

print('---< Factorial calculation implemented with reduce and operator.mul >---')

def fact(n):
    return reduce(mul, range(1, n + 1))

print('fact(5) = {0}'.format(fact(5)))
print()

print('---< Itemgetter demo that sorts a list of tuples >---')

metro_data = [
    ('Tokyo', 'JP', 36.933, (35.689722, 139.691667)),   # <1>
    ('Delhi NCR', 'IN', 21.935, (28.613889, 77.208889)),
    ('Mexico City', 'MX', 20.142, (19.433333, -99.133333)),
    ('New York-Newark', 'US', 20.104, (40.808611, -74.020386)),
    ('Sao Paulo', 'BR', 19.649, (-23.547778, -46.635833)),
]

for city in sorted(metro_data, key=itemgetter(1)):
    print(city)

print()

cc_name = itemgetter(1, 0)
for city in metro_data:
    print(cc_name(city))

print()

print('---< Demo of attrgetter processing tuple list of named metro_data >---')
LatLong = namedtuple('LatLong', 'lat long')
Metropolis = namedtuple('Metropolis', 'name cc pop coord')
metro_areas = [Metropolis(name, cc, pop, LatLong(lat, long)) for name, cc, pop, (lat, long) in metro_data]

print('metro_areas[0] = \n{0}'.format(metro_areas[0]))
print()

print('metro_areas[0].coord.lat = \n{0}'.format(metro_areas[0].coord.lat))
print()

name_lat = attrgetter('name', 'coord.lat')
for city in sorted(metro_areas, key=attrgetter('coord.lat')):
    print(name_lat(city))

print()

operator_function_list = [name for name in dir(operator) if not name.startswith('_')]
print('operator_function_list = \n{0}'.format(operator_function_list))
print()

print('---< methodcaller demo >---')
s = 'The time has come'
upcase = methodcaller('upper')
print('upcase(s) = {0}'.format(upcase(s)))
print()
hiphenate = methodcaller('replace', ' ', '-')
print('hiphenate(s) = {0}'.format(hiphenate(s)))
print()

'''
------------------------------------------------------------------------------------------------------------------------
5.10.2 Argument fixing by functools.partial
------------------------------------------------------------------------------------------------------------------------5.10.2
'''
print('-------------------------------------------------------------------\n'
      '            5.10.2 Argument fixing by functools.partial            \n'
      '-------------------------------------------------------------------\n')
triple = partial(mul, 3)
print('triple(7) = {0}'.format(triple(7)))
print()

list_triple = list(map(triple, range(1, 10)))
print('list_triple = \n{0}'.format(list_triple))
print()

print('---< Construct useful functions for Unicode normalization via partial. >---')

nfc = functools.partial(unicodedata.normalize, 'NFC')
s1 = 'café'
s2 = 'cafe\u0301'
print('({0}, {1})'.format(s1, s2))
print()
print('if s1 == s2 is {0}'.format(s1 == s2))
print()

print('if nfc(s1) == nfc(s2) is {0}'.format(nfc(s1) == nfc(s2)))
print()

print('---< Demo with partial applied to tag function >---')
from tagger import tag
print('tag = {0}'.format(tag))
print()

picture = partial(tag, 'img', cls='pic-frame')
picture(src = 'wumpus.jpeg')
print('picture = {0}'.format(picture))
print()

print('picture.func = {0}'.format(picture.func))
print()

print('picture.args = {0}'.format(picture.args))
print()

print('picture.keywords = {0}'.format(picture.keywords))
print()