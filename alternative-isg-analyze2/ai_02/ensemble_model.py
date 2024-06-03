from arguments_return_values import EnsembleModelArguments

import numpy as np

from ai_version import AiVersion
from common_ai import WeakModelObject
from constants import Constants, InvalidElectrode


class EnsembleModel:
    __ai_version: AiVersion
    __certainty2_0: np.ndarray
    __scoring_result2_0: np.ndarray
    __certainty2_1: np.ndarray
    __scoring_result2_1: np.ndarray
    __elect_info: np.ndarray

    def __init__(self, arguments: EnsembleModelArguments) -> None:
        self.__ai_version = arguments.ai_version
        self.__certainty2_0 = arguments.certainty2_0
        self.__scoring_result2_0 = arguments.scoring_result2_0
        self.__certainty2_1 = arguments.certainty2_1
        self.__scoring_result2_1 = arguments.scoring_result2_1
        self.__elect_info = arguments.elect_info

    def start(self) -> np.ndarray:
        self.__scoring_result2_0 = self.__scoring_result2_0.reshape(Constants.NUMBER_OF_WEAK_MODELS, -1)
        self.__scoring_result2_1 = self.__scoring_result2_1.reshape(Constants.NUMBER_OF_WEAK_MODELS, -1)

        self.__certainty2_0 = self.__certainty2_0.reshape(Constants.NUMBER_OF_WEAK_MODELS, -1, 5)
        self.__certainty2_1 = self.__certainty2_1.reshape(Constants.NUMBER_OF_WEAK_MODELS, -1, 5)

        centanity_ens9 = np.mean(self.__certainty2_0, axis=0)
        pred_alls_ens9 = np.argmax(centanity_ens9, axis=1)

        centanity2_ens9 = np.mean(self.__certainty2_1, axis=0)
        pred_alls2_ens9 = np.argmax(centanity2_ens9, axis=1)

        arg_np = np.sum(self.__elect_info[:, 1:6] >= 100, axis=0)

        arg_n = np.argmax(np.concatenate([arg_np[0:2], arg_np[3:5]]))

        max_n = np.max(np.concatenate([arg_np[0:2], arg_np[3:5]]))

        if arg_n == InvalidElectrode.R_FP1.value:
            [model1, model2, model3] = [
                WeakModelObject.weak_2xfp1.value,
                WeakModelObject.weak_6xfp1m1.value,
                WeakModelObject.weak_7xfp1m2.value
            ]
        elif arg_n == InvalidElectrode.R_FP2.value:
            [model1, model2, model3] = [
                WeakModelObject.weak_3xfp2.value,
                WeakModelObject.weak_8xfp2m1.value,
                WeakModelObject.weak_9xfp2m2.value
            ]
        elif arg_n == InvalidElectrode.R_A1.value:
            [model1, model2, model3] = [
                WeakModelObject.weak_4xm1.value,
                WeakModelObject.weak_6xfp1m1.value,
                WeakModelObject.weak_8xfp2m1.value
            ]
        elif arg_n == InvalidElectrode.R_A2.value:
            [model1, model2, model3] = [
                WeakModelObject.weak_5xm2.value,
                WeakModelObject.weak_7xfp1m2.value,
                WeakModelObject.weak_9xfp2m2.value
            ]
        else:
            raise Exception("no supported electrode!\n")

        certanity_2_0_remaining3 = \
            (self.__certainty2_0[model1] + self.__certainty2_0[model2] + self.__certainty2_0[model3]) / 3
        scoring_result2_0_remaining3 = np.argmax(certanity_2_0_remaining3, axis=1)

        certanity_2_1_remaining3 = \
            (self.__certainty2_1[model1] + self.__certainty2_1[model2] + self.__certainty2_1[model3]) / 3
        scoring_result2_1_remaining3 = np.argmax(certanity_2_1_remaining3, axis=1)

        ratio = (max_n / self.__elect_info.shape[0])
        ratio = 0
        print(f"ratio: {ratio}")

        if ratio < 0.1:
            print("using 10model type")
            if self.__ai_version.is_ai_version2_0():
                certainty = centanity_ens9
                scoring_result = pred_alls_ens9
            elif self.__ai_version.is_ai_version2_1():
                certainty = np.concatenate(
                    [
                        centanity_ens9[: Constants.DUMMY_STAGE_NUM],
                        centanity2_ens9,
                        centanity_ens9[-1 * Constants.DUMMY_STAGE_NUM:],
                    ],
                    0
                )

                scoring_result = np.concatenate(
                    [
                        pred_alls_ens9[: Constants.DUMMY_STAGE_NUM],
                        pred_alls2_ens9,
                        pred_alls_ens9[-1 * Constants.DUMMY_STAGE_NUM :]
                    ],
                    0,
                )
        else:
            print("using 3model type")
            if self.__ai_version.is_ai_version2_0():
                certainty = certanity_2_0_remaining3
                scoring_result = scoring_result2_0_remaining3
            elif self.__ai_version.is_ai_version2_1():
                certainty = np.concatenate(
                    [
                        certanity_2_0_remaining3[: Constants.DUMMY_STAGE_NUM],
                        certanity_2_1_remaining3,
                        certanity_2_0_remaining3[-1 * Constants.DUMMY_STAGE_NUM :]
                    ],
                    0
                )

                scoring_result = np.concatenate(
                    [
                        scoring_result2_0_remaining3[: Constants.DUMMY_STAGE_NUM],
                        scoring_result2_1_remaining3,
                        scoring_result2_0_remaining3[-1 * Constants.DUMMY_STAGE_NUM :]
                    ],
                    0
                )

        return certainty, scoring_result