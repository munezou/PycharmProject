from enum import Enum

class Constants:
    EDF_BEFORE_FILTER_NAME = "edf_before_filter.edf"
    EDF_AFTER_FILTER_DEFAULT_NAME = "edf_filtered.edf"
    RML_OUTPUT_DEFAULT_NAME = "rml.rml"
    CERTAINTY_OUTPUT_DEFAULT_NAME = "result.csv"
    AI_SCRIPT_DIR_NAME = "Scoring"
    DUMMY_STAGE_NUM = 5
    SOREM_FILTER_DURATION = 30 * 2
    REM_THRESHOLD = 0.8
    DETECT_RANGE = 5
    TRANSITION_THRESHOLD = 4
    SAMPLING_FREQUENCE = 200
    FB = 1
    BAI = 1
    BAI2 = 8
    NUMBER_OF_WEAK_MODELS = 10
    NUMBER_OF_STAGE = 5
    DELTA_N = 3

    WEAK_NAMES = [
        '0base',  # 0
        '1base',  # 1
        '2xfp1',  # 2
        '3xfp2',  # 3
        '4xm1',  # 4
        '5xm2',  # 5
        '6xfp1m1',  # 6
        '7xfp1m2',  # 7
        '8xfp2m1',  # 8
        '9xfp2m2',  # 9
    ]

    EEG_LABELS = {
        "Fp1": "FP1",
        "Fp2": "FP2",
        "A1": "A1",
        "A2": "A2",
        "Light": "Light OFF",
        "R_A1": "R_A1",
        "R_A2": "R_A2",
        "R_Ref": "R_Ref",
        "R_Fp1": "R_FP1",
        "R_Fp2": "R_FP2",
    }

    TEMPLATES = {
        "version": "version.txt",
        "channel_config": "channel.txt",
        "option": "option.txt",
    }

    BRAIN_SIGNAL_NAME = [
        "FP1-Average",
        "FP2-Average",
        "FP1-Fp2",
        "A1-A2",
        "FP1",
        "FP2",
        "A1",
        "A2",
    ]

    ANOTHER_SIGNAL_NAME = ["Light_OFF", "R_A1", "R_A2", "R_Ref", "R_FP1", "R_FP2"]

    ALL_SIGNAL_NAME = BRAIN_SIGNAL_NAME + ANOTHER_SIGNAL_NAME


class InvalidElectrode(Enum):
    R_A1 = 0
    R_FP1 = 1
    R_FP2 = 2
    R_A2 = 3


class InvalidElectrodeStatus(Enum):
    not_analyzed = 0
    analyzed = 1
