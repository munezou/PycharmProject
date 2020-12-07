import cython
from libc.stdlib cimport malloc, free
cimport from_ndarray
cimport numpy as cnp

def func(cnp.ndarray[double, ndim=1, mode="c"] temp):
    """
    １次元のdouble型のndarrayを引数にとる。
    cfuncにCのポインタを渡して計算させる。
    ちなみに、cfuncはvoid型の関数で行列を書き換える操作をする
    """
    cdef:
        int N
        double *ctemp
        
    N = len(temp)
    #ctemp = <double*>malloc(N*sizeof(double))通常はこうやって配列を作る
    ctemp = <double*> temp.data
    # C function in test.c
    cfunc(N, ctemp)


