'''StrKeyDict always converts non-string keys to `str`

Test for initializer: keys are converted to `str`.

    >>> d = StrKeyDict([(2, 'two'), ('4', 'four')])
    >>> sorted(d.keys())
    ['2', '4']

Tests for item retrieval using `d[key]` notation::

    >>> d['2']
    'two'
    >>> d[4]
    'four'
    >>> d[1]
    Traceback (most recent call last):
      ...
    KeyError: '1'

Tests for item retrieval using `d.get(key)` notation::

    >>> d.get('2')
    'two'
    >>> d.get(4)
    'four'
    >>> d.get(1, 'N/A')
    'N/A'

Tests for the `in` operator::

    >>> 2 in d
    True
    >>> 1 in d
    False

Test for item assignment using non-string key::

    >>> d[0] = 'zero'
    >>> d['0']
    'zero'

Tests for update using a `dict` or a sequence of pairs::

    >>> d.update({6:'six', '8':'eight'})
    >>> sorted(d.keys())
    ['0', '2', '4', '6', '8']
    >>> d.update([(10, 'ten'), ('12', 'twelve')])
    >>> sorted(d.keys())
    ['0', '10', '12', '2', '4', '6', '8']
    >>> d.update([1, 3, 5])
    Traceback (most recent call last):
      ...
    TypeError: 'int' object is not iterable

'''

# BEGIN STRKEYDICT

import collections

class StrKeyDict(collections.UserDict):  # <1>
    try:
        def __missing__(self, key):  # <2>
            if isinstance(key, str):
                raise KeyError(key)
            return self[str(key)]

        def __contains__(self, key):
            return str(key) in self.data  # <3>

        def __setitem__(self, key, item):
            self.data[str(key)] = item   # <4>
    except Exception as e:
        print(e)
    finally:
        pass

# END STRKEYDICT

print ('----< Test for initializer: keys are converted to `str`. >---')
d = StrKeyDict([(2, 'two'), ('4', 'four')])
print('sorted(d.keys() = {0}'.format(sorted(d.keys())))
print()

print('---< Tests for item retrieval using `d[key]` notation: >---')
print('d["2"] = {0}'.format(d['2']))
print('d[4] = {0}'.format(d[4]))

try:
    print('d[1] = {0}'.format(d[1]))
except Exception as e:
    print(e)
finally:
    pass
print()

print('---< Tests for the `in` operator:: >---')
print('2 in d = {0}'.format(2 in d))
print('1 in d = {0}'.format(1 in d))
print()

print('---< Test for item assignment using non-string key:: >---')
d[0] = 'zero'
print('d["0"] = {0}'.format(d['0']))
print()

print('---< Tests for update using a `dict` or a sequence of pairs:: >---')
d.update({6:'six', '8':'eight'})
print('sorted(d.keys()) = {0}'.format(sorted(d.keys())))
print()
d.update([(10, 'ten'), ('12', 'twelve')])
print('sorted(d.keys()) = {0}'.format(sorted(d.keys())))
print()

try:
    print('d.update([1, 3, 5]) = {0}'.format(d.update([1, 3, 5])))
except Exception as e:
    print(e)
finally:
    pass
print()



