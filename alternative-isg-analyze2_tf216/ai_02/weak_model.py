import os
import math

import numpy as np

from keras.models import Model
from keras.layers import Input, MaxPool1D
from keras.layers import Activation, concatenate
from keras.layers import ELU
from keras.layers import GlobalAveragePooling1D

import common_ai
from common_ai import LayerNormalization
from constants import Constants
from arguments_return_values import WeakModelArguments


class WeakModel:
    __eeg: dict = dict()
    __weak_n: int

    def __init__(self, setting_parameters: dict) -> None:
        self.__eeg = setting_parameters["eeg"]
        self.__weak_n = setting_parameters["weak_n"]

    def start(self, arguments: WeakModelArguments) -> None:
        model_files = os.path.join(
            os.getcwd(),
            "ai_02",
            "bins",
            f"0_ph21_noise_50cut2_{Constants.WEAK_NAMES[self.__weak_n]}.h5",
        )

        feature_model_files = os.path.join(
            os.getcwd(),
            "ai_02",
            "bins",
            f"0_ph21_noise_feature_50cut2_{Constants.WEAK_NAMES[self.__weak_n]}.h5",
        )

        scoring_eeg = self.normalize_signal()

        if self.__weak_n in [0, 1, 4, 5]:
            model = common_ai.stage_model_main3ch()
        elif self.__weak_n in [2, 3]:
            model = self.stage_model_2ch()
        else:
            model = self.stage_model_1ch()

        model.load_weights(model_files)

        feature_integ_module = common_ai.feature_integration_module(model)
        certainty2_0, feature_pre_tes = feature_integ_module.predict(scoring_eeg)
        scoring_result2_0 = np.argmax(certainty2_0, axis=1)

        pred_tes = np.zeros(
            [feature_pre_tes.shape[0] - 10, 11, feature_pre_tes.shape[1]]
        )

        for i_epo in range(pred_tes.shape[0]):
            pred_tes[i_epo] = feature_pre_tes[i_epo : i_epo + 11]

        stage_normalization_model = common_ai.stage_normalization_model()
        stage_normalization_model.load_weights(feature_model_files)

        certainty2_1 = stage_normalization_model.predict(pred_tes)
        scoring_result2_1 = np.argmax(certainty2_1, axis=1)

        arguments.certainty2_0 = np.append(arguments.certainty2_0, certainty2_0, axis=0)
        arguments.scoring_result2_0 = np.append(
            arguments.scoring_result2_0, scoring_result2_0, axis=0)
        arguments.certainty2_1 = np.append(arguments.certainty2_1, certainty2_1, axis=0)
        arguments.scoring_result2_1 = np.append(
            arguments.scoring_result2_1, scoring_result2_1, axis=0)

    def stage_model_2ch(self) -> Model:
        eeg_input_main = Input(
            shape=(int(math.floor(6000)), 2), dtype="float32", name="eeg"
        )
        eeg = common_ai.conv_1d(
            fb=32 * Constants.FB,
            bai=50 * Constants.BAI,
            layer_name="fs1",
            input=eeg_input_main,
        )

        eeg = ELU()(eeg)
        eeg = common_ai.conv_1d(
            fb=32 * Constants.FB,
            bai=10 * Constants.BAI,
            layer_name="fs2",
            input=eeg,
        )

        eeg = ELU()(eeg)
        eeg = common_ai.conv_1d(
            fb=16 * Constants.FB,
            bai=10 * Constants.BAI,
            layer_name="fs3",
            input=eeg,
        )
        eeg = ELU()(eeg)
        eeg = MaxPool1D(pool_size=10)(eeg)
        eeg = LayerNormalization()(eeg)

        eeg2 = common_ai.conv_1d(
            fb=32 * Constants.FB,
            bai=50 * Constants.BAI2,
            layer_name="fl1",
            input=eeg_input_main,
        )

        eeg2 = ELU()(eeg2)
        eeg2 = common_ai.conv_1d(
            fb=32 * Constants.FB,
            bai=10 * Constants.BAI2,
            layer_name="fl2",
            input=eeg2,
        )
        eeg2 = ELU()(eeg2)
        eeg2 = common_ai.conv_1d(
            fb=16 * Constants.FB,
            bai=10 * Constants.BAI2,
            layer_name="fl3",
            input=eeg2,
        )
        eeg2 = ELU()(eeg2)
        eeg2 = MaxPool1D(pool_size=10)(eeg2)
        eeg2 = LayerNormalization()(eeg2)

        features = concatenate([eeg, eeg2])
        features = common_ai.conv_1d(fb=5, bai=10, layer_name="fcam", input=features)
        x_main = GlobalAveragePooling1D()(features)
        main_output = Activation("softmax", name="stages")(x_main)

        return Model(inputs=[eeg_input_main], outputs=[main_output])

    def stage_model_1ch(self) -> Model:
        eeg_input_main = Input(
            shape=(int(math.floor(6000)), 1), dtype="float32", name="eeg"
        )
        eeg = common_ai.conv_1d(
            fb=32 * Constants.FB,
            bai=50 * Constants.BAI,
            layer_name="fs1",
            input=eeg_input_main,
        )

        eeg = ELU()(eeg)
        eeg = common_ai.conv_1d(
            fb=32 * Constants.FB,
            bai=10 * Constants.BAI,
            layer_name="fs2",
            input=eeg,
        )
        eeg = ELU()(eeg)
        eeg = common_ai.conv_1d(
            fb=16 * Constants.FB,
            bai=10 * Constants.BAI,
            layer_name="fs3",
            input=eeg,
        )
        eeg = ELU()(eeg)
        eeg = MaxPool1D(pool_size=10)(eeg)
        eeg = LayerNormalization()(eeg)

        eeg2 = common_ai.conv_1d(
            fb=32 * Constants.FB,
            bai=50 * Constants.BAI2,
            layer_name="fl1",
            input=eeg_input_main,
        )

        eeg2 = ELU()(eeg2)
        eeg2 = common_ai.conv_1d(
            fb=32 * Constants.FB,
            bai=10 * Constants.BAI2,
            layer_name="fl2",
            input=eeg2,
        )

        eeg2 = ELU()(eeg2)
        eeg2 = common_ai.conv_1d(
            fb=16 * Constants.FB,
            bai=10 * Constants.BAI2,
            layer_name="fl3",
            input=eeg2,
        )

        eeg2 = ELU()(eeg2)
        eeg2 = MaxPool1D(pool_size=10)(eeg2)
        eeg2 = LayerNormalization()(eeg2)

        features = concatenate([eeg, eeg2])
        features = common_ai.conv_1d(fb=5, bai=10, layer_name="fcam", input=features)
        x_main = GlobalAveragePooling1D()(features)
        main_output = Activation("softmax", name="stages")(x_main)

        return Model(inputs=[eeg_input_main], outputs=[main_output])

    def normalize_signal(self) -> np.ndarray:
        npz_files = os.path.join(
            os.getcwd(),
            "ai_02",
            "bins",
            f"0_ph21_noise_mean_std_50cut2_{Constants.WEAK_NAMES[self.__weak_n]}.npz",
        )

        loaded = np.load(npz_files)
        mean_e = loaded["mean_E"]
        std_e = loaded["std_E"]

        if self.__weak_n == 0:
            chs = [0, 1, 2]
        elif self.__weak_n == 1:
            chs = [0, 1, 3]
        elif self.__weak_n == 2:
            chs = [1, 3]
        elif self.__weak_n == 3:
            chs = [0, 3]
        elif self.__weak_n == 4:
            chs = [6, 7, 2]
        elif self.__weak_n == 5:
            chs = [4, 5, 2]
        elif self.__weak_n == 6:
            chs = [7]
        elif self.__weak_n == 7:
            chs = [5]
        elif self.__weak_n == 8:
            chs = [6]
        elif self.__weak_n == 9:
            chs = [4]
        else:
            raise Exception("Can't found weakN!")

        eeg_p = self.__eeg[:, :, chs]
        return (eeg_p - mean_e) / std_e
