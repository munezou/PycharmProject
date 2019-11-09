# adapted from Alex Martelli's example in "Re-learning Python"
# http://www.aleax.it/Python/accu04_Relearn_Python_alex.pdf
# (slide 41) Ex: lines-by-word file index

# BEGIN INDEX_DEFAULT
"""Build an index mapping word -> list of occurrences"""

import sys
import re
import collections

WORD_RE = re.compile(r'\w+')

index = collections.defaultdict(list)     # <1>
with open(sys.argv[1], encoding='utf-8') as fp:
    for line_no, line in enumerate(fp, 1):
        print('-------------------------------------------------------------------')
        print('line_no ={0}, line = {1}'.format(line_no, line))
        print('-------------------------------------------------------------------')
        for match in WORD_RE.finditer(line):
            word = match.group()
            print('word = {0}'.format(word))
            column_no = match.start()+1
            location = (line_no, column_no)
            print('location(Line_no, column_no) = (Line_no:{0}, column_no:{1})'.format(line_no, column_no))
            index[word].append(location)  # <2>

print()
print('---< Before Sorting >---')
# print in alphabetical order
for word in sorted(index, key=str.upper):
    print(word, index[word])
# END INDEX_DEFAULT
