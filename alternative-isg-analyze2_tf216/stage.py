from enum import IntEnum


class Stage(IntEnum):
    Wake = 0
    REM = 1
    NonREM1 = 2
    NonREM2 = 3
    NonREM3 = 4
    NotScored = 5
