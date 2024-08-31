from dataclasses import dataclass

import numpy as np

from ai_version import AiVersion
from user import User


@dataclass
class PreProsessorReturn:
    eeg: dict
    chin: np.ndarray
    light_off: int
    light_on: int
    elect_info: np.ndarray
    duration: int


@dataclass
class AnalyzerArguments:
    eeg: dict
    ai_version: AiVersion
    elect_info: np.ndarray


@dataclass
class PostProcessorArguments:
    ai_version: AiVersion
    certainty: np.ndarray
    scoring_result: np.ndarray
    light_on_epoch: int


@dataclass
class WeakModelArguments:
    certainty2_0: np.ndarray
    scoring_result2_0: np.ndarray
    certainty2_1: np.ndarray
    scoring_result2_1: np.ndarray


@dataclass
class EnsembleModelArguments:
    ai_version: AiVersion
    certainty2_0: np.ndarray
    scoring_result2_0: np.ndarray
    certainty2_1: np.ndarray
    scoring_result2_1: np.ndarray
    elect_info: np.ndarray


@dataclass
class OutputGeneratorArguments:
    output_filenames: dict
    user: User
    scoring_results: np.ndarray
    certainty: np.ndarray
    duration: int
    lightsoff: int
    lightson: int
    edffilterd_name: str
    eeg: dict
    edf_filenames: str
