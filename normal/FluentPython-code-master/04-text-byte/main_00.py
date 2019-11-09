import array
import struct
import sys, locale
import unicodedata
import string
import re
from unicodedata import normalize, name

'''
------------------------------------------------------------------------------------------------------------------------
Character problem
------------------------------------------------------------------------------------------------------------------------
'''
print('--------------------------------------------\n'
      '   character problem\n'
      '--------------------------------------------\n')
s = 'café'
print('len("café") = {0}'.format(len(s)))

b = s.encode('utf8')
print('s.encode("utf8") = {0}'.format(b))
print('len(s.encode("utf8")) = {0}'.format(len(b)))
print('b.decode("utf8") = {0}'.format(b.decode('utf8')))
print()

'''
------------------------------------------------------------------------------------------------------------------------
about byte
------------------------------------------------------------------------------------------------------------------------
'''
print('--------------------------------------------\n'
      '   about byte\n'
      '--------------------------------------------\n')
print('---< about bytes(python 3) >---')
cafe = bytes('café', encoding='utf_8')
print('cafe = {0}'.format(cafe))
print('cafe[0] = {0}'.format(cafe[0]))
print('cafe[:1] = {0}'.format(cafe[:1]))
print()

print('---< about bytearray(python 2) >---')
cafe_arr = bytearray(cafe)
print('cafe_arr = {0}'.format(cafe_arr))
print('cafe_arr[0] = {0}'.format(cafe_arr[0]))
print('cafe_arr[:1] = {0}'.format(cafe_arr[:1]))
print()

print('---< fromhex in binary sequence >---')
print('bytes.fromhex("31 4B CE A9") = {0}'.format(bytes.fromhex('31 4B CE A9')))
print()

print('---< Initialize bytes from raw array data. >---')

numbers = array.array('h', [-2, -1, 0, 1, 2])
print ('number(raw array) = \n{0}'.format(numbers))
octets = bytes(numbers)
print('bytes(number) = {0}'.format(octets))
print()

'''
------------------------------------------------------------------------------------------------------------------------
struct and memory view

struct is Interpret bytes as packed binary data.

------------------------------------------------------------------------------------------------------------------------
'''
print('--------------------------------------------\n'
      '   struct and memory view\n'
      '--------------------------------------------\n')
fmt = '<3s3sHH'
with open('filter.gif', 'rb') as fp:
    img = memoryview(fp.read())

header = img[:10]
print('bytes(header) = {0}'.format(bytes(header)))
print()
print('struct.unpack(fmt, header) = {0}'.format(struct.unpack(fmt, header)))
print()
del header
del img

'''
------------------------------------------------------------------------------------------------------------------------
Basic encoder and decoder
------------------------------------------------------------------------------------------------------------------------
'''
print('--------------------------------------------\n'
      '   Basic encoder and decoder\n'
      '--------------------------------------------\n')
for codec in ['latin_1', 'utf_8', 'utf_16']:
    print(codec, 'El Niño'.encode(codec), sep='\t')

print()

'''
------------------------------------------------------------------------------------------------------------------------
Encoder and decoder problems
------------------------------------------------------------------------------------------------------------------------
'''
print('--------------------------------------------\n'
      '   Encoder and decoder problems\n'
      '--------------------------------------------\n')
# Countermeasures for Unicode Encode Error
print('---< são Paulo >---')
city = 'são Paulo'
print('city.encode("utf_8") = {0}'.format(city.encode('utf_8')))
print()
print('city.encode("utf_16") = {0}'.format(city.encode('utf_16')))
print()
print('city.encode("iso8859_1") = {0}'.format(city.encode('iso8859_1')))
print()
try:
    print('city.encode("cp437") = {0}'.format(city.encode('cp437')))
except Exception as e:
    print(e)
finally:
    pass
print()

print('city.encode("cp437", errors="ignore") = {0}'.format(city.encode('cp437', errors='ignore')))
print()
print('city.encode("cp437", errors="replace") = {0}'.format(city.encode('cp437', errors='replace')))
print()
print('city.encode("cp437", errors="xmlcharrefreplace") = {0}'.format(city.encode('cp437', errors='xmlcharrefreplace')))
print()

# Countermeasures for Unicode decode Error
octets = b'Montr\xe9al'
print('octets.decode("cp1252") = {0}'.format(octets.decode('cp1252')))
print()
print('octets.decode("iso8859_7") = {0}'.format(octets.decode('iso8859_7')))
print()
print('octets.decode("koi8_r") = {0}'.format(octets.decode('koi8_r')))
print()
try:
    print('octets.decode("utf_8") = {0}'.format(octets.decode('utf_8')))
except Exception as e:
    print(e)
finally:
    pass
print()
print('octets.decode("utf_8", errors="replace")) = {0}'.format(octets.decode('utf_8', errors='replace')))
print()

'''
------------------------------------------------------------------------------------------------------------------------
UTF_8 with BOM or without BOM

A BOM is a short code written at the beginning of a document written in Unicode, 
and is used to specify the type of character encoding (character encoding) used and the byte order (endian).

Big endian 
Little endian : intel
------------------------------------------------------------------------------------------------------------------------
'''
print('--------------------------------------------\n'
      '   utf_8 with BOM or without BOM\n'
      '--------------------------------------------\n')
u16 = 'El Niño'.encode('utf_16')
print('u16 = {0}'.format(u16))
print('BOM = \xff\xfe')
print()
print('---< Little endian >---')
print('list(u16) = \n{0}'.format(list(u16)))
print()
print('---< utf_16le: little endian >---')
u16le = 'El Niño'.encode('utf_16le')
print('list(u16le) = \n{0}'.format(list(u16le)))
print()
print('---< big endian >---')
u16be = 'El Niño'.encode('utf_16be')
print('list(u16be) = \n{0}'.format(list(u16be)))
print()

'''
------------------------------------------------------------------------------------------------------------------------
Processing text files
------------------------------------------------------------------------------------------------------------------------
'''
print('--------------------------------------------\n'
      '             Processing text files          \n'
      '--------------------------------------------\n')
# Platform encoding issues
open('cafe.txt', 'w', encoding='utf_8').write('café')

# check contents in cafe.txt.
tmpSt = open('cafe.txt').read()
print('contents = {0}'.format(tmpSt))

print()

print('--------------------------------------------\n'
      '             scrutinize code　　　　          \n'
      '--------------------------------------------\n')
fp = open('cafe.txt', 'w', encoding='utf_8')
print('fp = {0}'.format(fp))
fp.write('café')
fp.close()

import os
stSize = os.stat('cafe.txt').st_size
print('stSize = {0}'.format(stSize))
print()

fp2 = open('cafe.txt')
print('fp2 = {0}'.format(fp2))
print('fp2.encoding = {0}'.format(fp2.encoding))
print('fp2.read() = {0}'.format(fp2.read()))
print()

fp3 = open('cafe.txt', encoding='utf_8')
print('fp3 = {0}'.format(fp3))
print('fp3.read() = {0}'.format(fp3.read()))
print()

fp4 = open('cafe.txt', 'rb')
print('fp4 = {0}'.format(fp4))
print('fp4.read() = {0}'.format(fp4.read()))
print()

'''
------------------------------------------------------------------------------------------------------------------------
Messy default encoding
------------------------------------------------------------------------------------------------------------------------
'''
print('--------------------------------------------\n'
      '         Messy default encodin　　          \n'
      '--------------------------------------------\n')
expressions = """
        locale.getpreferredencoding()
        type(my_file)
        my_file.encoding
        sys.stdout.isatty()
        sys.stdout.encoding
        sys.stdin.isatty()
        sys.stdin.encoding
        sys.stderr.isatty()
        sys.stderr.encoding
        sys.getdefaultencoding()
        sys.getfilesystemencoding()
    """

my_file = open('dummy', 'w')

for expression in expressions.split():
    value = eval(expression)
    print(expression.rjust(30), '->', repr(value))
print()

'''
------------------------------------------------------------------------------------------------------------------------
4.6 Unicode normalization for proper comparison
------------------------------------------------------------------------------------------------------------------------
'''
print('---------------------------------------------------------------------\n'
      '         4.6 Unicode normalization for proper comparison　　          \n'
      '---------------------------------------------------------------------\n')
s1 = 'café'
s2 = 'cafe\u0301'
# The display is the same but is considered not equal.
print('s1 = {0}, s2 = {1}'.format(s1, s2))
print()
print('len(s1) = {0}, len(s2) = {1}'.format(len(s1), len(s2)))
print()
print('s1 == s2, result = {0}'.format(s1 == s2))
print()

print('---------------------------------------------------------------------\n'
      '   Use unicodedata.normalize () as a countermeasure 　　　　　　　　　　\n'
      '　 that is not considered the same even if the display is the same.  \n'
      '---------------------------------------------------------------------\n')
print('len(normalize("NFC", s1)) = {0}, len(normalize("NFC", s2)) = {1}'.format(len(normalize('NFC', s1)), len(normalize('NFC', s2))))
print()
print('len(normalize("NFD", s1)) = {0}, len(normalize("NFD", s2)) = {1}'.format(len(normalize('NFD', s1)), len(normalize('NFD', s2))))
print()
print('normalize("NFC", s1) == normalize("NFC", s2), Result = {0}'.format(normalize('NFC', s1) == normalize('NFC', s2)))
print()
print('normalize("NFD", s1) == normalize("NFD", s2), Result = {0}'.format(normalize('NFD', s1) == normalize('NFD', s2)))
print()

ohm = '\u2126'
print('name(ohm) = {0}'.format(name(ohm)))
print()

ohm_c = normalize('NFC', ohm)
print('name(ohm_c) = {0}'.format(name(ohm_c)))
print()
print('ohm == ohm_c, result = {0}'.format(ohm == ohm_c))
print()
print('normalize("NFC", ohm) == normalize("NFC", ohm_c), result = {0}'.format(normalize('NFC', ohm) == normalize("NFC", ohm_c)))
print()

micro ='μ'
micro_kc = normalize('NFKC', micro)
print('micro = {0}, micro_kc = {1}'.format(micro, micro_kc))
print()
print('ord(micro) = {0}, ord(micro_kc) = {1}'.format(ord(micro), ord(micro_kc)))
print()
print('name(micro) = {0}, name(micro_kc) = {1}'.format(name(micro), name(micro_kc)))
print()

'''
------------------------------------------------------------------------------------------------------------------------
case folding
------------------------------------------------------------------------------------------------------------------------
'''
print('-----------------------------------------\n'
      '         4.6.1  case folding　　          \n'
      '-----------------------------------------\n')
micro = 'μ'
print('micro = {0}, name(micro) = {1}'.format(micro, name(micro)))
print()
micro_cf = micro.casefold()
print('micro_cf = {0}, name(micro_cf) = {1}'.format(micro_cf, name(micro_cf)))
print()
print('micro = {0}, micro_cf = {1}'.format(micro, micro_cf))
print()
eszett = 'β'
print('eszett = {0}, name(eszett) = {1}'.format(eszett, name(eszett)))
print()
eszett_cf = eszett.casefold()
print('eszett = {0}, eszett_cf = {1}'.format(eszett, eszett_cf))
print()

'''
------------------------------------------------------------------------------------------------------------------------
Utility functions for normalized Unicode string comparison.
------------------------------------------------------------------------------------------------------------------------
'''
print('----------------------------------------------------------------------\n'
      '  4.6.2  Utility functions for normalized Unicode string comparison.  \n'
      '----------------------------------------------------------------------\n')


def nfc_equal(str1, str2):
    return normalize('NFC', str1) == normalize('NFC', str2)

def fold_equal(str1, str2):
    return (normalize('NFC', str1).casefold() ==
            normalize('NFC', str2).casefold())

s1 = 'café'
s2 = 'cafe\u0301'
print('s1 == s2, result = {0}'.format(s1 == s2))
print()
print('nfc_equal(s1, s2) = {0}'.format(nfc_equal(s1, s2)))
print()
print('nfc_equal("A", "a") = {0}'.format(nfc_equal('A', 'a')))
print()
s3 = 'Straße'
s4 = 'strasse'
print('s3 == s4, result = {0}'.format(s3 == s4))
print()
print('nfc_equal(s3, s4) = {0}'.format(nfc_equal(s3, s4)))
print()
print('fold_equal(s3, s4) = {0}'.format(fold_equal(s3, s4)))
print()
print('fold_equal(s1, s2) = {0}'.format(fold_equal(s1, s2)))
print()
print('fold_equal("A", "a") = {0}'.format(fold_equal('A', 'a')))
print()

'''
------------------------------------------------------------------------------------------------------------------------
4.6.3 Extreme normalization to remove sign symbols
------------------------------------------------------------------------------------------------------------------------
'''
print('----------------------------------------------------------------------\n'
      '          4.6.3 Extreme normalization to remove sign symbols          \n'
      '----------------------------------------------------------------------\n')
def shave_marks(txt):
    """Remove all diacritic marks"""
    norm_txt = unicodedata.normalize('NFD', txt)
    shaved = ''.join(c for c in norm_txt
                     if not unicodedata.combining(c))
    return unicodedata.normalize('NFC', shaved)

order = '“Herr Voß: • ½ cup of Œtker™ caffè latte • bowl of açaí.”'
print('order = {0}'.format(order))
print('shave_marks(order) = {0}'.format(shave_marks(order)))
print()
greek = 'Ζέφυρος, Zéfiro'
print('greek = {0}'.format(greek))
print('shave_marks(greek) = {0}'.format(shave_marks(greek)))
print()

def shave_marks_latin(txt):
    """Remove all diacritic marks from Latin base characters"""
    norm_txt = unicodedata.normalize('NFD', txt)  # <1>
    latin_base = False
    keepers = []
    for c in norm_txt:
        if unicodedata.combining(c) and latin_base:   # <2>
            continue  # ignore diacritic on Latin base char
        keepers.append(c)                             # <3>
        # if it isn't combining char, it's a new base char
        if not unicodedata.combining(c):              # <4>
            latin_base = c in string.ascii_letters
    shaved = ''.join(keepers)
    return unicodedata.normalize('NFC', shaved)

print('shave_marks_latin(order) = {0}'.format(shave_marks_latin(order)))
print()
print('shave_marks_latin(greek) = {0}'.format(shave_marks_latin(greek)))
print()

'''
------------------------------------------------------------------------------------------------------------------------
Converts some Western photosetting symbols to ASCII characters.
------------------------------------------------------------------------------------------------------------------------
'''
print('----------------------------------------------------------------------\n'
      '   Converts some Western photosetting symbols to ASCII characters.    \n'
      '----------------------------------------------------------------------\n')
single_map = str.maketrans("""‚ƒ„†ˆ‹‘’“”•–—˜›""",  # <1>
                           """'f"*^<''""---~>""")

multi_map = str.maketrans({  # <2>
    '€': '<euro>',
    '…': '...',
    'Œ': 'OE',
    '™': '(TM)',
    'œ': 'oe',
    '‰': '<per mille>',
    '‡': '**',
})

multi_map.update(single_map)  # <3>


def dewinize(txt):
    """Replace Win1252 symbols with ASCII chars or sequences"""
    return txt.translate(multi_map)  # <4>


def asciize(txt):
    no_marks = shave_marks_latin(dewinize(txt))     # <5>
    no_marks = no_marks.replace('ß', 'ss')          # <6>
    return unicodedata.normalize('NFKC', no_marks)  # <7>

order = '“Herr Voß: • ½ cup of Œtker™ caffè latte • bowl of açaí.”'
print('order = {0}'.format(order))
print()
print('dewinize(order) = {0}'.format(dewinize(order)))
print()
print('asciize(order) = {0}'.format(asciize(order)))
print()

greek = 'Ζέφυρος, Zéfiro'
print('greek = {0}'.format(greek))
print()
print('dewinize(greek) = {0}'.format(dewinize(greek)))
print()
print('asciize(greek) = {0}'.format(asciize(greek)))
print()

'''
------------------------------------------------------------------------------------------------------------------------
4.7 Sort unicode text
------------------------------------------------------------------------------------------------------------------------
'''
print('----------------------------------------------------------------------\n'
      '                           4.7 Sort unicode text                      \n'
      '----------------------------------------------------------------------\n')
fruits = ['caju', 'atemoia', 'cajá', 'açaí', 'acerola']
print('fruits = \n{0}'.format(fruits))
print()
print('sorted(fruits) = \n{0}'.format(sorted(fruits)))
print()
try:
    local_setlocale = locale.setlocale(locale.LC_COLLATE, 'pt_BR.UTF-8')
    print('local_setlocale = {0}'.format(local_setlocale))
    print()
except Exception as ex:
    print(ex)
    pass

print()

sorted_fruits = sorted(fruits, key=locale.strxfrm)
print('sorted_fruits = \n{0}'.format(sorted_fruits))
print()

'''
------------------------------------------------------------------------------------------------------------------------
4.8 Unicode data base
------------------------------------------------------------------------------------------------------------------------
'''
print('----------------------------------------------------------------------\n'
      '                         4.8 Unicode data base                        \n'
      '----------------------------------------------------------------------\n')
re_digit = re.compile(r'\d')

sample = '1\xbc\xb2\u0969\u136b\u216b\u2466\u2480\u3285'

for char in sample:
    print('U+%04x' % ord(char),                       # <1>
          char.center(6),                             # <2>
          're_dig' if re_digit.match(char) else '-',  # <3>
          'isdig' if char.isdigit() else '-',         # <4>
          'isnum' if char.isnumeric() else '-',       # <5>
          format(unicodedata.numeric(char), '5.2f'),  # <6>
          unicodedata.name(char),                     # <7>
          sep='\t')

print()

'''
------------------------------------------------------------------------------------------------------------------------
4.9 str/bytes dual mode API
------------------------------------------------------------------------------------------------------------------------
'''
print('----------------------------------------------------------------------\n'
      '       4.9.1 Simple regular expression behavior of str / bytes        \n'
      '----------------------------------------------------------------------\n')
re_numbers_str = re.compile(r'\d+')     # <1>
re_words_str = re.compile(r'\w+')
re_numbers_bytes = re.compile(rb'\d+')  # <2>
re_words_bytes = re.compile(rb'\w+')

text_str = ("Ramanujan saw \u0be7\u0bed\u0be8\u0bef"  # <3>
            " as 1729 = 1³ + 12³ = 9³ + 10³.")        # <4>

text_bytes = text_str.encode('utf_8')  # <5>

print('Text', repr(text_str), sep='\n  ')
print('Numbers')
print('  str  :', re_numbers_str.findall(text_str))      # <6>
print('  bytes:', re_numbers_bytes.findall(text_bytes))  # <7>
print('Words')
print('  str  :', re_words_str.findall(text_str))        # <8>
print('  bytes:', re_words_bytes.findall(text_bytes))    # <9>



