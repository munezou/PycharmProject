"""
----------------------------------------------------------------------
ph21_train_func.py

----------------------------------------------------------------------
"""
# using library
import os
import math
import tracemalloc
import traceback
import hashlib
import numpy as np

from scipy import signal
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model
from tensorflow.python.keras.layers import Layer, Activation
from tensorflow.keras.layers import (
	Conv1D,
	Softmax,
	ELU,
	MaxPool1D,
	concatenate,
	GlobalAveragePooling1D
)
import tensorflow.keras.regularizers as regularizers


# variable of related filter
fp = np.array([0.3, 35])     # k-d
fs = np.array([0.15, 70])
gpass = -3    # 3                       #通過域端最大損失[dB]
gstop = -5.5  # 40                      #阻止域端最小損失[dB]

# variable of Event_demodel_3
bai=1
bai2=8
fb=1
sub=1


def format_bytes(size):
	"""
	
	"""
	try:
		power = 2 ** 10  # 2**10 = 1024
		n = 0
		power_labels = ['B', 'MB', 'GB', 'TB']
		while size > power and n <= len(power_labels):
			size /= power
			n += 1
		return 'current used memory: {:.3f} {}'.format(size, power_labels[n])
	except:
		print(f"Error information\n{traceback.format_exc()}")
		pass


def log_memory():
	"""
	
	"""
	try:
		snapshot = tracemalloc.take_snapshot()
		size = sum([stat.size for stat in snapshot.statistics('filename')])
		print(format_bytes(size))
	except:
		print(f"Error information\n{traceback.format_exc()}")
		pass


def delete_duplicate_edf(file_list):
	"""
	Delete duplicate files
	:param file_list:
	:return:
	"""
	try:
		hash_list = {}
		
		for file_path in file_list:
			data = open(file_path, 'rb').read()
			
			# Compute the hash of the file.
			h = hashlib.sha256(data).hexdigest()
			
			if h in hash_list:
				if data == open(hash_list[h], 'rb').read():
					print(hash_list[h] + 'と' + file_path + 'は同じ')
					os.remove(file_path)  # remove duplicated file
			else:
				hash_list[h] = file_path
		
		return hash_list
	except:
		print(f"Error information\n{traceback.format_exc()}")
		pass


# definition of custum objects
class LayerNormalization(Layer):
	"""
	
	"""
	try:
		def __init__(self, **kwargs):
			super(LayerNormalization, self).__init__(**kwargs)
			self.epsilon = 1e-6
		
		def build(self, input_shape):
			self.built = False
		
		def call(self, x, **kwargs):
			mean = K.mean(x, axis=-1, keepdims=True)
			std = K.std(x, axis=-1, keepdims=True)
			norm = (x - mean) * (1 / (std + self.epsilon))
			return norm
		
		def compute_output_shape(self, input_shape):
			return input_shape
	except:
		print(f"Error information\n{traceback.format_exc()}")
		pass


def cat_cross(y_true, y_pred):
	"""
	
	"""
	try:
		return -tf.reduce_sum(y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 1.0)))
	except:
		print(f"Error information\n{traceback.format_exc()}")
		pass


def soft_max_(layer):
	"""
	
	"""
	try:
		def __init__(self, **kwargs):
			super(soft_max_(), self).__init__(**kwargs)
		
		def build(self, input_shape):
			self.build = True  # self.built = True
		
		def call(self, x):
			return K.softmax(x)
		
		def compute_output_shape(self, input_shape):
			return input_shape
	except:
		print(f"Error information\n{traceback.format_exc()}")
		pass


# バターワースフィルタ（バンドパス）
def bandpass(x, samplerate, fp, fs, gpass, gstop):
	"""
	
	"""
	try:
		fn = samplerate / 2  # ナイキスト周波数
		wp = fp / fn  # ナイキスト周波数で通過域端周波数を正規化
		ws = fs / fn  # ナイキスト周波数で阻止域端周波数を正規化
		N, Wn = signal.buttord(wp, ws, gpass, gstop)  # オーダーとバターワースの正規化周波数を計算
		b, a = signal.butter(N, Wn, "band")  # フィルタ伝達関数の分子と分母を計算
		y = signal.filtfilt(b, a, x)  # 信号に対してフィルタをかける
		return y
	except:
		print(f"Error information\n{traceback.format_exc()}")
		pass


def event_dmodel_3():
	"""
	
	"""
	try:
		# MAIN PART
		eeg_input_main = Input(shape=(int(math.floor(6000)), 3), dtype='float32', name='eeg')
		eeg = Conv1D(32 * fb, 50 * bai, kernel_regularizer=regularizers.l2(0.001), padding='same', name='fs1')(
			eeg_input_main)  # convolutional layer 1 50  filter96
		eeg = ELU()(eeg)
		eeg = Conv1D(32 * fb, 10 * bai, kernel_regularizer=regularizers.l2(0.001), padding='same', name='fs2')(eeg)
		eeg = ELU()(eeg)  # exponential linear unit as activation function
		eeg = Conv1D(16 * fb, 10 * bai, kernel_regularizer=regularizers.l2(0.001), padding='same', name='fs3')(eeg)
		eeg = ELU()(eeg)
		eeg = MaxPool1D(pool_size=10)(eeg)  # Max pooling layer
		eeg = LayerNormalization()(eeg)
		
		eeg2 = Conv1D(32 * fb, 50 * bai2, kernel_regularizer=regularizers.l2(0.001), padding='same', name='fl1')(
			eeg_input_main)  # convolutional layer 1 50  filter96
		eeg2 = ELU()(eeg2)
		eeg2 = Conv1D(32 * fb, 10 * bai2, kernel_regularizer=regularizers.l2(0.001), padding='same', name='fl2')(eeg2)
		eeg2 = ELU()(eeg2)  # exponential linear unit as activation function
		eeg2 = Conv1D(16 * fb, 10 * bai2, kernel_regularizer=regularizers.l2(0.001), padding='same', name='fl3')(eeg2)
		eeg2 = ELU()(eeg2)
		eeg2 = MaxPool1D(pool_size=10)(eeg2)  # Max pooling layer
		eeg2 = LayerNormalization()(eeg2)
		
		features = concatenate([eeg, eeg2])  # ([eeg,eog,emg,cam])
		features = Conv1D(5, 10, kernel_regularizer=regularizers.l2(0.001), padding='same', name='fcam')(features)  # 50
		x_main = GlobalAveragePooling1D()(features)
		# x_main = GlobalMaxPooling1D()(features)
		main_output = Activation('softmax', name='stages')(x_main)
		
		return Model(inputs=[eeg_input_main], outputs=[main_output])
	except:
		print(f"Error information\n{traceback.format_exc()}")
		pass