# pylint: disable=C0111
# pylint: disable=C0103
# pylint: disable=C0326
import bisect
import contextlib
import time

from typing import Sequence, Any


@contextlib.contextmanager
def stopwatch(prefix: str=''):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f'{prefix:<12}{" = " if prefix else ""}{end - start:.6f}')


def index(seq: Sequence, val: Any, direction: int=1) -> int:
    def _get_func():
        if direction == 1:
            return (bisect.bisect_left, 0)
        return (bisect.bisect_right, -1)

    f, adjust = _get_func()
    i = f(seq, val)

    if i != len(seq) and seq[i + adjust] == val:
        return i + adjust

    raise ValueError(f'{val} is not in sequence')


values = list(range(10000000))
target = 5555555

with stopwatch('list.index'):
    values.index(target)

with stopwatch('bisect_left'):
    index(values, target)

with stopwatch('bisect_right'):
    index(values, target, direction=2)