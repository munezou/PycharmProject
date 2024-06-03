from abc import ABCMeta, abstractmethod
from arguments_return_values import AnalyzerArguments
import numpy as np


class Estimator(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, arguments: AnalyzerArguments) -> None:
        pass

    @abstractmethod
    def start(self) -> np.ndarray:
        pass
