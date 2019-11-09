
"""
Radical folding and text sanitizing.

Handling a string with `cp1252` symbols:

    >>> order = '�gHerr Vos:   cup of tker caffe latte  bowl of acai.�h'
    >>> shave_marks(order)
    '�gHerr Vos:   cup of tker caffe latte  bowl of acai.�h'
    >>> shave_marks_latin(order)
    '�gHerr Vos:   cup of tker caffe latte  bowl of acai.�h'
    >>> dewinize(order)
    '"Herr Vos: -  cup of OEtker(TM) caffe latte - bowl of acai."'
    >>> asciize(order)
    '"Herr Voss: - 12 cup of OEtker(TM) caffe latte - bowl of acai."'

Handling a string with Greek and Latin accented characters:

    >>> greek = '���Ӄ҃σ�, Zefiro'
    >>> shave_marks(greek)
    '���ÃӃ҃σ�, Zefiro'
    >>> shave_marks_latin(greek)
    '���Ӄ҃σ�, Zefiro'
    >>> dewinize(greek)
    '���Ӄ҃σ�, Zefiro'
    >>> asciize(greek)
    '���Ӄ҃σ�, Zefiro'

"""

# BEGIN SHAVE_MARKS
import unicodedata
import string


def shave_marks(txt):
    """Remove all diacritic marks"""
    norm_txt = unicodedata.normalize('NFD', txt)  # <1>
    shaved = ''.join(c for c in norm_txt
                     if not unicodedata.combining(c))  # <2>
    return unicodedata.normalize('NFC', shaved)  # <3>
# END SHAVE_MARKS

# BEGIN SHAVE_MARKS_LATIN
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
    return unicodedata.normalize('NFC', shaved)   # <5>
# END SHAVE_MARKS_LATIN

# BEGIN ASCIIZE
single_map = str.maketrans("""���e�f�g�h""",  # <1>
                           """'f"*^<''""---~>""")

multi_map = str.maketrans({  # <2>
    '': '<euro>',
    '�c': '...',
    '': 'OE',
    '': '(TM)',
    '': 'oe',
    '��': '<per mille>',
    '��': '**',
})

multi_map.update(single_map)  # <3>


def dewinize(txt):
    """Replace Win1252 symbols with ASCII chars or sequences"""
    return txt.translate(multi_map)  # <4>


def asciize(txt):
    no_marks = shave_marks_latin(dewinize(txt))     # <5>
    no_marks = no_marks.replace('s', 'ss')          # <6>
    return unicodedata.normalize('NFKC', no_marks)  # <7>
# END ASCIIZE
