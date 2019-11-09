# Summary of special methods

'''
-----------------------------------------------------------------------------------------------------------------
__str__

The __str__ method is called when an object is specified as an argument of a print statement or format statement, 
and returns an "informal" string representation of the object.
-----------------------------------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------------------------------
__repr__

The __repr__ method returns a more formal object content as a string,
and is used when the interpreter checks the equivalence of two objects.
-----------------------------------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------------------------------
How to use __str/__repr__

The usage of __str__ and __repr__ in print statement and format statement is as follows.

-If only __str__ is defined, __str__ is used.
-If only __repr__ is defined, __repr__ is used.
-If both __str__ and __repr__ are defined, __str__ is used.
-----------------------------------------------------------------------------------------------------------------
'''

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __str__(self):
        return '<Name:{0}, Age:{1:3.2f}>'.format(self.name, self.age)

print('---< Person >---')

john = Person("John", 38)
luice = Person("Luice", 32)

print(john)
print(luice)

print()

print('---< PersonModify >---')

class PersonModify:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return 'My name is {0}.'.format(self.name)

john = PersonModify("John")
print('Hello, ' + str(john))

print()

'''
Use the methods keys (), values () and items () of the dictionary object dict to loop over elements of the Python dictionary object dict with a for statement.

keys (): for loop processing for the key key of each element
values (): for loop processing for each element's value value
items (): for loop processing for key key and value of each element
'''

print('---< Python Hash() >---')

# Hash
d = {'key1': 1, 'key2' : 2, 'key3': 3}

'''
You can get the key by turning the dictionary object with the for statement.
'''
for dt in d:
    print(dt)

print()

for dt in d.keys():
    print(dt)

print()

for dt in d.values():
    print(dt)

print()

'''
special methods (__hash__ / __bool__ / __eq__ )
'''

print('----------< special methods (__hash__ / __bool__ / __eq__ ) >---------')
from normal.PythonDataModel.ch01_data_model.human import Human

taro = Human("Taro", 21)
jiro = Human("Jiro", 20)
taro_2 = Human("Taro", 21)
nanashi = Human("", 32)

print("object.__repr__({0}) = {1})".format(taro.name, repr(taro)))

print("object.__str__({0}) = {1}".format(taro.name, str(taro)))

print("obect.__bytes__({0}) = {1}".format(taro.name, bytes(taro)))

print("object.__hash__({0}) = {1}".format(taro.name, hash(taro)))

print("object.__eq__({0}) = {1}".format(taro.name, taro == jiro or taro == taro_2))






