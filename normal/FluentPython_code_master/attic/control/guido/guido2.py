"""
Exemplo adaptado da mensagem do Guido van Rossum em:
https://groups.google.com/forum/#!msg/python-tulip/bmphRrryuFk/aB45sEJUomYJ
http://bit.ly/yieldfrom

    >>> principal_susto(ger1())
    OK
    Bu!

Visualiza�F�Bo no PythonTutor: http://goo.gl/m6p2Bc

"""

def ger1():
    try:
        val = yield 'OK'
    except RuntimeError as exc:
        print(exc)
    else:
        print(val)
    yield  # para evitar o StopIteration


def principal_susto(g):
    print(next(g))
    g.throw(RuntimeError('Bu!'))


# auto-teste
import doctest
doctest.testmod()
