'''
Dictionary and set
'''

print ('---< 3.1 General map type >---')

a = dict(one=1, two=2, three=3)
b = {'one': 1, 'two': 2, 'three': 3}
c = dict(zip(['one', 'two', 'three'], [1, 2, 3]))
d = dict([('two', 2), ('one', 1), ('three', 3)])
e = dict({'three': 3, 'one': 1, 'two': 2})

print('a == b == c == d == e : Result = {0}'.format(a == b == c == d == e))

print ('---< 3.2 Dictionary comprehension >---')

DIAL_CODES = [(86, 'China'), (91, 'India'), (1, 'United States'), (62, 'Indonesia'), (55, 'Brazil'), (92, 'Pakistan'),
              (880, 'Bangladesh'), (234, 'Nigeria'), (7, 'Russia'), (81, 'Japan')]

country_code = {country: code for code, country in DIAL_CODES}

print('country_code = \n{0}'.format(country_code))

country_code2 = {code: country.upper() for country, code in country_code.items() if code < 66}

print('country_code2 = {0}'.format(country_code2))