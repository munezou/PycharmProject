import os
from arguments_return_values import AnalyzerArguments

import numpy as np

from keras.models import load_model
from keras.models import Model

from ai_version import AiVersion
from common_ai import LayerNormalization
import common_ai
from estimator import Estimator
from constants import Constants


class Ai01(Estimator):
    __eeg: dict = dict()
    __ai_version: AiVersion

    def __init__(self, arguments: AnalyzerArguments) -> None:
        self.__eeg = arguments.eeg
        self.__ai_version = arguments.ai_version

    def start(self) -> np.ndarray:
        npz_files = os.path.join("ai_01", "bins", "0_ph21_noise_mean_std_1base.npz")

        scoring_eeg = self.normalize_signal(npz_files)

        if self.__ai_version.is_ai_version1_1():
            certainty = self.predicte_on_model_ai1_1(scoring_eeg)
            scoring_result = map(np.argmax, certainty)
        elif self.__ai_version.is_ai_version1_2():
            certainty1_1, certainty1_2 = self.predicte_on_model_ai1_2(scoring_eeg)

            certainty = np.concatenate(
                [
                    certainty1_1[: Constants.DUMMY_STAGE_NUM],
                    certainty1_2,
                    certainty1_1[-1 * Constants.DUMMY_STAGE_NUM :],
                ],
                0,
            )

            scoring_result = np.argmax(certainty, axis=1)

        return certainty, scoring_result

    def read_model_ai1(self) -> Model:
        model_files = os.path.join("ai_01", "bins", "0_ph21_noise_1base.h5")
        return load_model(
            model_files,
            custom_objects={
                "LayerNormalization": LayerNormalization,
                "cat_cross": common_ai.cat_cross,
            },
        )

    def predicte_on_model_ai1_1(self, scoring_eeg: np.ndarray) -> np.ndarray:
        model = self.read_model_ai1()
        return model.predict(scoring_eeg)

    def predicte_on_model_ai1_2(self, scoring_eeg: np.ndarray) -> np.ndarray:
        model_files = os.path.join("ai_01", "bins", "0_ph21_noise_1base.h5")
        feature_model_files = os.path.join(
            "ai_01", "bins", "0_ph21_noise_feature_1base.h5"
        )

        model = common_ai.stage_model_main3ch()

        model.load_weights(model_files)

        feature_integ_module = common_ai.feature_integration_module(model)
        certainty1_1, feature_pre_tes = feature_integ_module.predict(scoring_eeg)

        pred_tes = np.zeros(
            [feature_pre_tes.shape[0] - 10, 11, feature_pre_tes.shape[1]]
        )

        for i_epo in range(pred_tes.shape[0]):
            pred_tes[i_epo] = feature_pre_tes[i_epo : i_epo + 11]

        stage_normalization_model = common_ai.stage_normalization_model()
        stage_normalization_model.load_weights(feature_model_files)

        certainty1_2 = stage_normalization_model.predict(pred_tes)

        return certainty1_1, certainty1_2

    def normalize_signal(self, npz_files: str) -> np.ndarray:
        loaded = np.load(npz_files)
        mean_e = loaded["mean_E"]
        std_e = loaded["std_E"]

        fp1_ma = self.__eeg["fp1_ma"].reshape(self.__eeg["num_of_epoch"], -1, 1)
        fp2_ma = self.__eeg["fp2_ma"].reshape(self.__eeg["num_of_epoch"], -1, 1)
        m1_m2 = self.__eeg["m1_m2"].reshape(self.__eeg["num_of_epoch"], -1, 1)

        eeg_concat = np.concatenate([fp1_ma, fp2_ma, m1_m2], axis=2)
        return (eeg_concat - mean_e) / std_e
