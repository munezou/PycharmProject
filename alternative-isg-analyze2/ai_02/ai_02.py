from ai_version import AiVersion
from arguments_return_values import AnalyzerArguments, WeakModelArguments, EnsembleModelArguments

import numpy as np

from estimator import Estimator
from ai_02.weak_model import WeakModel
from ai_02.ensemble_model import EnsembleModel
from constants import Constants


class Ai02(Estimator):
    __eeg: dict = dict()
    __ai_version: AiVersion
    __elect_info: np.ndarray

    def __init__(self, arguments: AnalyzerArguments) -> None:
        self.__eeg = arguments.eeg
        self.__ai_version = arguments.ai_version
        self.__elect_info = arguments.elect_info

    def start(self) -> np.ndarray:
        duration = int(len(self.__eeg["fp1_ma"]) / Constants.SAMPLING_FREQUENCE)
        n_epo = int(duration / 30)

        eeg_matrix = np.concatenate(
            [
                self.__eeg["fp1_ma"].reshape(n_epo, -1, 1),
                self.__eeg["fp2_ma"].reshape(n_epo, -1, 1),
                self.__eeg["fp1_fp2"].reshape(n_epo, -1, 1),
                self.__eeg["m1_m2"].reshape(n_epo, -1, 1),
                self.__eeg["fp1_m1"].reshape(n_epo, -1, 1),
                self.__eeg["fp2_m1"].reshape(n_epo, -1, 1),
                self.__eeg["fp1_m2"].reshape(n_epo, -1, 1),
                self.__eeg["fp2_m2"].reshape(n_epo, -1, 1),
                self.__eeg["fp1"].reshape(n_epo, -1, 1),
                self.__eeg["fp2"].reshape(n_epo, -1, 1),
                self.__eeg["a1"].reshape(n_epo, -1, 1),
                self.__eeg["a2"].reshape(n_epo, -1, 1)
            ],
            axis=2
        )

        weak_models = np.array([])
        for i in range(Constants.NUMBER_OF_WEAK_MODELS):
            weak_models = np.append(
                weak_models,
                WeakModel(
                    setting_parameters={
                        "eeg": eeg_matrix,
                        "weak_n": i
                    }
                )
            )

        weak_arguments = WeakModelArguments(
            certainty2_0=np.empty((0, Constants.NUMBER_OF_STAGE)),
            scoring_result2_0=np.empty((0,)),
            certainty2_1=np.empty((0, Constants.NUMBER_OF_STAGE)),
            scoring_result2_1=np.empty((0,))
        )

        for model in weak_models:
            model.start(arguments=weak_arguments)

        ensemble_arguments = EnsembleModelArguments(
            ai_version=self.__ai_version,
            certainty2_0=weak_arguments.certainty2_0,
            scoring_result2_0=weak_arguments.scoring_result2_0,
            certainty2_1=weak_arguments.certainty2_1,
            scoring_result2_1=weak_arguments.scoring_result2_1,
            elect_info=self.__elect_info
        )

        np.savetxt("elect_info.csv", self.__elect_info, delimiter=",", fmt="%.1f")
        ensemble_model = EnsembleModel(arguments=ensemble_arguments)

        return ensemble_model.start()
