import numpy as np
from ai_version import AiVersion
from stage import Stage
from constants import Constants
from arguments_return_values import PostProcessorArguments


class PostProcessor:
    __ai_version: AiVersion
    __scoring_result: list = list()
    __asleep_epoch: int
    __result: int
    __certainty = np.array([])
    __rem_epoc_results: list = list()
    __rem_epoc_results2: list = list()
    __candidates: list = list()
    __median = np.array([])

    def __init__(self, post_processor_parameter: PostProcessorArguments) -> None:
        self.__ai_version = post_processor_parameter.ai_version
        self.__certainty = post_processor_parameter.certainty
        self.__scoring_result = list(post_processor_parameter.scoring_result)

    def start(self):
        self.detect_asleep_epoch()

        if self.__ai_version.is_ai_version1_1():
            self.aasm_filter_1()
            self.sorem_filter()
            self.aasm_filter_2()
        elif self.__ai_version.is_ai_version1_2_0():
            self.aasm_filter_1()
            self.sorem_filter()
            self.aasm_filter_2()
        elif self.__ai_version.is_ai_version1_2_1():
            self.aasm_filter_1()
            self.sorem_filter()
            self.aasm_filter_2()
            self.aasm_filter_3()
        elif self.__ai_version.is_ai_version1_2_2():
            pass
        elif self.__ai_version.is_ai_version1_2_3():
            self.aasm_filter_1()
            self.sorem_filter()
            self.aasm_filter_2()
            self.aasm_filter_3()
            self.aasm_filter_4()
        elif self.__ai_version.is_ai_version1_2_4():
            self.aasm_filter_1()
            self.sorem_filter()
            self.aasm_filter_2()
            self.aasm_filter_3()
            self.aasm_filter_4()
            self.aasm_filter_5()
        elif self.__ai_version.is_ai_version2():
            pass

        return self.__scoring_result

    def second_largest(self, array) -> np.ndarray:
        return np.argsort(array)[-2]

    def first_largest(self, array) -> np.ndarray:
        return np.argsort(array)[-1]

    def detect_asleep_epoch(self):
        if Stage.NonREM1 not in self.__scoring_result and Stage.NonREM2 not in self.__scoring_result:
            self.__asleep_epoch = None
        for i, label in enumerate(self.__scoring_result):
            if label == Stage.NonREM1 or label == Stage.NonREM2:
                self.__asleep_epoch = i
                break

    def calculate_median(self, i) -> None:
        min_index = max(i - Constants.DELTA_N, 0)
        max_index = min(i + Constants.DELTA_N, self.__certainty.shape[0] - 1)
        meauring_array = np.array(self.__certainty[min_index : (max_index + 1)])
        rem_column = meauring_array[:, 1]
        return np.median(rem_column)

    def sorem_filter(self):
        if self.__asleep_epoch is None:
            return

        if self.__ai_version == AiVersion.version1_1:
            for i, label in enumerate(self.__scoring_result):
                if i < Constants.SOREM_FILTER_DURATION and label == Stage.REM:
                    if self.calculate_median(i) <= Constants.REM_THRESHOLD:
                        self.__scoring_result[i] = self.second_largest(self.__certainty[i])
        else:
            for i, label in enumerate(
                self.__scoring_result[self.__asleep_epoch : self.__asleep_epoch + Constants.SOREM_FILTER_DURATION],
                self.__asleep_epoch,
            ):
                if label == Stage.REM:
                    if self.calculate_median(i) <= Constants.REM_THRESHOLD:
                        self.__scoring_result[i] = self.first_largest(self.__certainty[i])

    # N3 -> WK, REM -> WK before asleep epoch
    def aasm_filter_1(self):
        if self.__asleep_epoch is None:
            return

        for i, label in enumerate(list(self.__scoring_result[:self.__asleep_epoch])):
            if label == Stage.NonREM3 or label == Stage.REM:
                self.__scoring_result[i] = Stage.Wake.value

    # [REM, NonREM3, REM] -> [REM, NotScored, REM] after asleep epoch
    def aasm_filter_2(self):
        if self.__asleep_epoch is None:
            return

        start_index = max(1, self.__asleep_epoch)
        last_index = len(self.__scoring_result) - 1

        for i, _ in enumerate(self.__scoring_result[start_index:last_index], start_index):
            if (
                self.__scoring_result[i - 1] == self.__scoring_result[i + 1] == Stage.REM
                and self.__scoring_result[i] == Stage.NonREM3
            ):
                self.__scoring_result[i] = Stage.NotScored

    def transition_counter(self, targets, src, dst):
        self.__result = 0
        for i, v in enumerate(targets[:-1]):
            if v == src and targets[i + 1] == dst:
                self.__result += 1
        return self.__result

    # repeate [NonREM1, REM] or [REM, NonREM1] -> [NotScored] after asleep epoch
    def aasm_filter_3(self):
        scoring_result_length = len(self.__scoring_result) - 1
        ns_candidate_indexes = []

        def is_rewritable_stage(stage):
            return v == Stage.REM or v == Stage.NonREM1

        for i, v in enumerate(self.__scoring_result):
            if not is_rewritable_stage(v):
                continue

            range_min = max(0, i - Constants.DETECT_RANGE)
            range_max = min(i + Constants.DETECT_RANGE, scoring_result_length)
            sr_in_range = self.__scoring_result[range_min : range_max + 1]
            n1_to_rem_counter = self.transition_counter(sr_in_range, Stage.NonREM1, Stage.REM)
            rem_to_n1_counter = self.transition_counter(sr_in_range, Stage.REM, Stage.NonREM1)

            if Constants.TRANSITION_THRESHOLD <= (n1_to_rem_counter + rem_to_n1_counter):
                ns_candidate_indexes.append(i)

        for index in ns_candidate_indexes:
            self.__scoring_result[index] = Stage.NotScored

    # repeate [NonREM2, REM] or [REM, NonREM2] -> [NotScored] after asleep epoch
    def aasm_filter_4(self):
        scoring_result_length = len(self.__scoring_result) - 1
        ns_candidate_indexes = []

        def is_rewritable_stage(stage):
            return v == Stage.REM or v == Stage.NonREM2

        for i, v in enumerate(self.__scoring_result):
            if not is_rewritable_stage(v):
                continue

            range_min = max(0, i - Constants.DETECT_RANGE)
            range_max = min(i + Constants.DETECT_RANGE, scoring_result_length)
            sr_in_range = self.__scoring_result[range_min : range_max + 1]
            n2_to_rem_counter = self.transition_counter(sr_in_range, Stage.NonREM2, Stage.REM)
            rem_to_n2_counter = self.transition_counter(sr_in_range, Stage.REM, Stage.NonREM2)

            if Constants.TRANSITION_THRESHOLD <= (n2_to_rem_counter + rem_to_n2_counter):
                ns_candidate_indexes.append(i)

        for index in ns_candidate_indexes:
            self.__scoring_result[index] = Stage.NotScored

    def detect_single_rem_epoch(self):
        for i in range(1, len(self.__scoring_result) - 1):
            if (
                self.__scoring_result[i - 1] != Stage.REM
                and self.__scoring_result[i] == Stage.REM
                and self.__scoring_result[i + 1] != Stage.REM
            ):
                self.__candidates.append(i)

    def detect_2_consecutive_rem_epoch(self):
        for i in range(1, len(self.__scoring_result) - 2):
            if (
                self.__scoring_result[i - 1] != Stage.REM
                and self.__scoring_result[i] == Stage.REM
                and self.__scoring_result[i + 1] == Stage.REM
                and self.__scoring_result[i + 2] != Stage.REM
            ):
                self.__candidates.append(i)
                self.__candidates.append(i + 1)
        self.__candidates = sorted(set(self.__candidates))

    def aasm_filter_5(self):
        self.detect_single_rem_epoch()
        self.detect_2_consecutive_rem_epoch()
        for index in self.__candidates:
            self.__scoring_result[index] = Stage.NotScored
