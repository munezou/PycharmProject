from estimator import Estimator
from ai_version import AiVersion
from ai_01.ai_01 import Ai01
from ai_02.ai_02 import Ai02

import numpy as np

from arguments_return_values import AnalyzerArguments


class Analyzer:
    __estimator: Estimator

    def __init__(self, ai_version: AiVersion, arguments: AnalyzerArguments) -> None:
        if ai_version.is_ai_version1():
            self.__estimator = Ai01(arguments=arguments)
        elif ai_version.is_ai_version2():
            self.__estimator = Ai02(arguments=arguments)
        else:
            raise

    def start(self) -> np.ndarray:
        return self.__estimator.start()
