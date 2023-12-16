import math
from enum import Enum
from dataclasses import dataclass
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Input, Activation, Conv1D, ELU, concatenate
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, MaxPool1D
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.engine.keras_tensor import KerasTensor

from constants import Constants, InvalidElectrodeStatus


class LayerNormalization(Layer):
    def __init__(self, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.epsilon = 1e-6

    def build(self, input_shape):
        self.built = False

    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        norm = (x - mean) * (1 / (std + self.epsilon))
        return norm

    def compute_output_shape(self, input_shape):
        return input_shape


class WeakModelObject(Enum):
    weak_0base = 0
    weak_1base = 1
    weak_2xfp1 = 2
    weak_3xfp2 = 3
    weak_4xm1 = 4
    weak_5xm2 = 5
    weak_6xfp1m1 = 6
    weak_7xfp1m2 = 7
    weak_8xfp2m1 = 8
    weak_9xfp2m2 = 9


def cat_cross(y_true, y_pred):
    return -tf.reduce_sum(
        y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 1.0))
    )


def ret_eeg_idx(signal_labels: list):
    idx_buffer: dict = {
        Constants.EEG_LABELS["Fp1"]: 0,
        Constants.EEG_LABELS["Fp2"]: 1,
        Constants.EEG_LABELS["A1"]: 2,
        Constants.EEG_LABELS["A2"]: 3,
        Constants.EEG_LABELS["Light"]: 4,
        Constants.EEG_LABELS["R_A1"]: 5,
        Constants.EEG_LABELS["R_Fp1"]: 6,
        Constants.EEG_LABELS["R_Ref"]: 7,
        Constants.EEG_LABELS["R_Fp2"]: 8,
        Constants.EEG_LABELS["R_A2"]: 9,
    }
    for el in Constants.EEG_LABELS:
        for (s, i) in zip(signal_labels, range(len(signal_labels))):
            if Constants.EEG_LABELS[el] == s:
                idx_buffer[Constants.EEG_LABELS[el]] = i
                continue
    return idx_buffer


def feature_integration_module(cnn_model):
    input_eeg = cnn_model.input

    for layer in cnn_model.layers:
        if 'concatenate' in layer.name:
            layer_raw_name = layer.name
        elif 'cam' in layer.name:
            layer_cam_name = layer.name
        elif 'stages' in layer.name:
            layer_stages = layer.name
    feature_raw_short = cnn_model.get_layer(layer_raw_name).output[:, :, :16]
    feature_raw_long = cnn_model.get_layer(layer_raw_name).output[:, :, 16:]
    feature_cam = cnn_model.get_layer(layer_cam_name).output
    GAP_feature_s = GlobalAveragePooling1D()(feature_raw_short)
    GMP_feature_s = GlobalMaxPooling1D()(feature_raw_short)
    integrated_feature_s = tf.concat([GAP_feature_s, GMP_feature_s], axis=-1, name='concatenate_gap_gmp_s')
    GAP_feature_l = GlobalAveragePooling1D()(feature_raw_long)
    GMP_feature_l = GlobalMaxPooling1D()(feature_raw_long)
    integrated_feature_l = tf.concat([GAP_feature_l, GMP_feature_l], axis=-1, name='concatenate_gap_gmp_l')
    GAP_feature_c = GlobalAveragePooling1D()(feature_cam)
    GMP_feature_c = GlobalMaxPooling1D()(feature_cam)
    integrated_feature_c = tf.concat([GAP_feature_c, GMP_feature_c], axis=-1, name='concatenate_gap_gmp_c')
    integrated_feature_concat = tf.concat(
        [integrated_feature_s, integrated_feature_l, integrated_feature_c],
        axis=-1,
        name='integrated_feature'
    )
    basic_certainty = cnn_model.get_layer(layer_stages).output
    return Model(inputs=[input_eeg], outputs=[basic_certainty, integrated_feature_concat])


def conv_1d(
    fb: int, bai: int, layer_name: str, input: KerasTensor
) -> KerasTensor:
    return Conv1D(
        fb,
        bai,
        kernel_regularizer=regularizers.l2(0.001),
        padding="same",
        name=layer_name,
    )(input)


def stage_normalization_model() -> Model:
    dim = int(16 * 2 * 2 + 5 * 2)

    preds_input_main = Input(
        shape=(int(math.floor(11)), dim), dtype="float32", name="pred"
    )
    normed_in = BatchNormalization()(preds_input_main)
    x_main = Flatten()(normed_in)
    x_main = Dense(5)(x_main)
    main_output = Activation("softmax", name="stages")(x_main)

    return Model(inputs=[preds_input_main], outputs=[main_output])


def stage_model_main3ch() -> Model:
    eeg_input_main = Input(
        shape=(int(math.floor(6000)), 3), dtype="float32", name="eeg"
    )

    eeg = conv_1d(
        fb=32 * Constants.FB,
        bai=50 * Constants.BAI,
        layer_name="fsi",
        input=eeg_input_main,
    )

    eeg = ELU()(eeg)
    eeg = conv_1d(
        fb=32 * Constants.FB,
        bai=10 * Constants.BAI,
        layer_name="fs2",
        input=eeg,
    )

    eeg = ELU()(eeg)
    eeg = conv_1d(
        fb=16 * Constants.FB,
        bai=10 * Constants.BAI,
        layer_name="fs3",
        input=eeg,
    )

    eeg = ELU()(eeg)
    eeg = MaxPool1D(pool_size=10)(eeg)
    eeg = LayerNormalization()(eeg)

    eeg2 = conv_1d(
        fb=32 * Constants.FB,
        bai=50 * Constants.BAI2,
        layer_name="fl1",
        input=eeg_input_main,
    )

    eeg2 = ELU()(eeg2)
    eeg2 = conv_1d(
        fb=32 * Constants.FB,
        bai=10 * Constants.BAI2,
        layer_name="fl2",
        input=eeg2,
    )

    eeg2 = ELU()(eeg2)
    eeg2 = conv_1d(
        fb=16 * Constants.FB,
        bai=10 * Constants.BAI2,
        layer_name="fl3",
        input=eeg2,
    )

    eeg2 = ELU()(eeg2)
    eeg2 = MaxPool1D(pool_size=10)(eeg2)
    eeg2 = LayerNormalization()(eeg2)

    features = concatenate([eeg, eeg2])
    features = conv_1d(fb=5, bai=10, layer_name="fcam", input=features)
    x_main = GlobalAveragePooling1D()(features)
    main_output = Activation("softmax", name="stages")(x_main)

    return Model(inputs=[eeg_input_main], outputs=[main_output])


@dataclass
class Electrode:
    ai_analyze_status_fp1: InvalidElectrodeStatus = InvalidElectrodeStatus.not_analyzed
    ai_analyze_status_fp2: InvalidElectrodeStatus = InvalidElectrodeStatus.not_analyzed
    ai_analyze_status_a1: InvalidElectrodeStatus = InvalidElectrodeStatus.not_analyzed
    ai_analyze_status_a2: InvalidElectrodeStatus = InvalidElectrodeStatus.not_analyzed
