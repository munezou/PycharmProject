# adapted from Alex Martelli's example in "Re-learning Python"
# http://www.aleax.it/Python/accu04_Relearn_Python_alex.pdf
# (slide 41) Ex: lines-by-word file index

# BEGIN INDEX0
"""Build an index mapping word -> list of occurrences"""

import sys
import re

'''
------------------------------------
Compile a regular expression pattern into a regular expression object and use it for matching using the match (), search () 
and other methods described below.

W+
------------------------------------
'''
WORD_RE = re.compile(r'\w+')

index = {}

with open(sys.argv[1], encoding='utf-8') as fp:
    '''
    enumerate:
    Using enumerate functions, you can retrieve element indices and elements simultaneously.
    '''
    for line_no, line in enumerate(fp, 1):
        print ('line_no = {0},  Line = {1}'.format(line_no, line))
        for match in WORD_RE.finditer(line):
            print('match = {0}'.format(match))
            word = match.group()
            print('word = {0}'.format(word))
            column_no = match.start()+1
            location = (line_no, column_no)
            print ('location = ( {0}, {1} )'.format(line_no, column_no))
            # this is ugly; coded like this to make a point
            occurrences = index.get(word, [])  # <1>
            occurrences.append(location)       # <2>
            index[word] = occurrences          # <3>
            print ('index[{0}] = {1}'.format(word, occurrences))
        print ()

# print in alphabetical order
for word in sorted(index, key=str.upper):  # <4>
    print(word, index[word])
# END INDEX0
