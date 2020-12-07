'''
-------------------------------------------------------------------------------------------------
Point Cloud Analysis 4
	programed by K.munekata

contents)
	This program recognizes objects by analyzing Point Cloud output from TI's mmWave.

using AI)
	DBSCAN + RandomForestClassifier(adjustment High parameter)
-------------------------------------------------------------------------------------------------
'''

# common library
import os
import sys
import csv
import time
import traceback
import collections
import datetime
import copy
from statistics import mean, median,variance,stdev


# numpy
import numpy as np

# pandas
import pandas as pd

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rcParams, cycler
from mpl_toolkits.mplot3d import Axes3D

# scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.decomposition import PCA

print(__doc__)

# Setting current path
PROJECT_ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))

"""
---------------------------------------------------------------------------
Declare functions
---------------------------------------------------------------------------
"""
def CaluclateCenterOfGravity(frameNumber, name, indexId, saveFlag, filePath):
	"""
	contents:

	parameter:
		name:       
		indexId:
		saveFlag:
		filePath:
	"""

	indexId_array = np.array(indexId)
	if indexId_array.shape[0] > 0:
		# Calculation of center of gravity
		# confirm wherether movement or static
		index_static = np.where(indexId_array[:, 3] == 0.0)
		index_movement = np.where(indexId_array[:, 3] != 0.0)
		indexId_static = indexId_array[index_static]
		indexId_movement = indexId_array[index_movement]

		if len(index_movement) > len(index_static):
			object_flag = 1
		else:
			object_flag = 0
		
		# caluculate the center of gravity.(after standardize)
		gravity_c = np.average(indexId, axis=0)
		gravity_min = np.min(indexId, axis=0)
		gravity_max = np.max(indexId, axis=0)
		gravity_range = np.subtract(gravity_max, gravity_min)

		# Replace nan with 0.
		np.nan_to_num(gravity_c, copy=False)
		np.nan_to_num(gravity_min, copy=False)
		np.nan_to_num(gravity_max, copy=False)
		np.nan_to_num(gravity_range, copy=False)

		# caluculate the center of gravity.(before standardize)
		indexId_org = np.multiply(indexId, ratio)
		gravity_c_org = np.multiply(gravity_c, ratio)
		gravity_range_org = np.multiply(gravity_range, ratio)

		# Replace nan with 0.
		np.nan_to_num(indexId_org, copy=False)
		np.nan_to_num(gravity_c_org, copy=False)
		np.nan_to_num(gravity_range_org, copy=False)

		id_name = str(name).split('_')

		print('ID = {0}'.format(id_name[1]))
		print('center of gravity(x={0}, y={1}, z={2}, v={3})'.format(gravity_c_org[0], gravity_c_org[1], gravity_c_org[2], gravity_c_org[4]))
		print('range of gravity(x={0}, y={1}, z={2}, v={3})\n'.format(gravity_range_org[0], gravity_range_org[1], gravity_range_org[2], gravity_range_org[4]))

		for i in range(len(indexId)):
			tmpRow = np.block([frameNumber, object_flag, 0, indexId[i][:], np.subtract(indexId[i][:], gravity_c), gravity_c, gravity_range, indexId_org[i][:], gravity_c_org, gravity_range_org])
			featureID.append(tmpRow)
			if saveFlag == True:
				with open(filePath, 'a', newline='') as f:
					writer = csv.writer(f)
					writer.writerow(tmpRow)
				f.close()
	
def transrateImagePatern(patternData, gravityData, patternId, allRange, bit_x, bit_y, ax):
	try:
		# prepare bit_x bit x bit_y bit array
		bit_id = np.zeros((bit_x, bit_y))

		# caluculate total bits
		total_bits = bit_x * bit_y

		if patternId.shape[0] > 0 and allRange != 0:
			# Extracte gravity_base_x and gravity_base_y.
			pattern_xy_data = np.array(patternId.loc[:, 'gravity_base_x':'gravity_base_y'], dtype=np.float32)

			# transerate pattern_xy_data to 32bit x 32bit array.
			pattern_xy_data[:, 0] = np.round(np.multiply(np.add(pattern_xy_data[:, 0], allRange), bit_x / (2 * allRange)))
			
			pattern_xy_data[:, 1] = np.round(np.multiply(np.subtract(allRange, pattern_xy_data[:, 1]), bit_y / (2 * allRange)))

			for tmpRow in pattern_xy_data:
				tmpRowRound_x = tmpRow[1].astype(np.int)
				if tmpRowRound_x >= (bit_x - 1):
					tmpRowRound_x = bit_x - 1
				
				tmpRowRound_y = tmpRow[0].astype(np.int)
				if tmpRowRound_y >= (bit_y - 1):
					tmpRowRound_y = bit_y - 1

				if tmpRowRound_x >= 0 and tmpRowRound_y >= 0:
					bit_id[tmpRowRound_x, tmpRowRound_y] = 1
			
			# Reshape bit_x bit x  bit_y bit to total bits
			training_reshape_data = bit_id.reshape([total_bits])

			tmp_index = patternId.condition.index[0]
			tmpRow_image = np.block([patternId.loc[tmp_index, 'frame_no'], patternId.loc[tmp_index, 'DBSCAN_ID'], training_reshape_data])
			tmpRow_gravity = np.block([patternId.loc[tmp_index, 'frame_no'], patternId.loc[tmp_index, 'DBSCAN_ID'], patternId.loc[tmp_index, 'gravity_x':'gravity_range_snr'].values])
			
			ax.imshow(bit_id)

			return tmpRow_image, tmpRow_gravity
		else:
			return NaN, NaN
	except Exception as ex:
		print(ex)
		pass
	finally:
		pass

def Transfer_ID(featureID, imageData, gravityData, preProbRatio, model,
				featureID_0, featureID_1, featureID_2, featureID_3, featureID_4,
				featureID_5, featureID_6, featureID_7, featureID_8, featureID_9):
	"""
	-------------------------------------------------------------------------------
	
	
	-------------------------------------------------------------------------------
	"""
	if len(imageData) > 0 and len(gravityData) > 0:
		# Exstract frame_no
		frame_value = imageData[0]

		target_real = imageData[1]

		# create image data only.
		imageData_real = imageData[2: 786]

		past_latest_image_data = featureID.frame_image_data.values.tolist()

		check_test_image = np.array(imageData_real)

		# create gravity data only.
		gravityData_real = gravityData[2: 12]

		# Predict to Which ID the current image data belongs.
		predict_proba = model.predict_proba(check_test_image.reshape(1, 784))

		print('predict_proba of current image data = {0}'.format(predict_proba))

		# Detect the ID with the highest expected probability.
		maxPredProbaId = np.argmax(predict_proba[0])

		# Delete maxPredProbaId in predict_proba
		feature_distribution = np.delete(predict_proba, maxPredProbaId)

		print('feature_distribution = {0}'.format(feature_distribution))

		# compute the mean of the predict probability
		mean_value = mean(feature_distribution)

		# Compute the variance of the predicted probability.
		variance_value = variance(feature_distribution)

		# Calculate the standard deviation of the predicted probability.
		stdev_value = stdev(feature_distribution)

		upper_limt = mean_value + (2 * stdev_value)

		lower_limt = mean_value - (2 * stdev_value)

		print('lower:{0}    upper:{1}\n'.format(lower_limt, upper_limt))

		newFeatureId_flag = False

		for item in feature_distribution:
			if item < lower_limt:
				newFeatureId_flag = True
				break

		# check oldest update time.
		update_time = [
						featureID_0.Check_Update_time(), featureID_1.Check_Update_time(), featureID_2.Check_Update_time(), featureID_3.Check_Update_time(), featureID_4.Check_Update_time(),
						featureID_5.Check_Update_time(), featureID_6.Check_Update_time(), featureID_7.Check_Update_time(), featureID_8.Check_Update_time(), featureID_9.Check_Update_time()
					]
		
		# Find a candidate for an identity change.
		update_time_min_index = update_time.index(min(update_time))

		"""
		------------------------------------------------------------
		no using dummy data

		contents)
		The number of image data in each ID is checked, and if it is less than 50, the image data is added.
		If the number of image data holdings exceeds 50, the oldest image data is deleted and the current image data is added.
		------------------------------------------------------------
		"""
		if (maxPredProbaId == 0) and (predict_proba[0, 0] >= upper_limt):
			if not featureID_0.frame_enable:
				featureID_0.Add_data(frame_value, maxPredProbaId, imageData_real, gravityData_real)
			else:
				featureID_0.Update_data(frame_value, maxPredProbaId, imageData_real, gravityData_real)
			
		elif (maxPredProbaId == 1) and (predict_proba[0, 1] >= upper_limt):
			if not featureID_1.frame_enable:
				featureID_1.Add_data(frame_value, maxPredProbaId, imageData_real, gravityData_real)
			else:
				featureID_1.Update_data(frame_value, maxPredProbaId, imageData_real, gravityData_real)
			
		elif (maxPredProbaId == 2) and (predict_proba[0, 2] >= upper_limt):
			if not featureID_2.frame_enable:
				featureID_2.Add_data(frame_value, maxPredProbaId, imageData_real, gravityData_real)
			else:
				featureID_2.Update_data(frame_value, maxPredProbaId, imageData_real, gravityData_real)
			
		elif (maxPredProbaId == 3) and (predict_proba[0, 3] >= upper_limt):
			if not featureID_3.frame_enable:
				featureID_3.Add_data(frame_value, maxPredProbaId, imageData_real, gravityData_real)
			else:
				featureID_3.Update_data(frame_value, maxPredProbaId, imageData_real, gravityData_real)
			
		elif (maxPredProbaId == 4) and (predict_proba[0, 4] >= upper_limt):
			if not featureID_4.frame_enable:
				featureID_4.Add_data(frame_value, maxPredProbaId, imageData_real, gravityData_real)
			else:
				featureID_4.Update_data(frame_value, maxPredProbaId, imageData_real, gravityData_real)
			
		elif (maxPredProbaId == 5) and (predict_proba[0, 5] >= upper_limt):
			if not featureID_5.frame_enable:
				featureID_5.Add_data(frame_value, maxPredProbaId, imageData_real, gravityData_real)
			else:
				featureID_5.Update_data(frame_value, maxPredProbaId, imageData_real, gravityData_real)
			
		elif (maxPredProbaId == 6) and (predict_proba[0, 6] >= upper_limt):
			if not featureID_6.frame_enable:
				featureID_6.Add_data(frame_value, maxPredProbaId, imageData_real, gravityData_real)
			else:
				featureID_6.Update_data(frame_value, maxPredProbaId, imageData_real, gravityData_real)
			
		elif (maxPredProbaId == 7) and (predict_proba[0, 7] >= upper_limt):
			if not featureID_7.frame_enable:
				featureID_7.Add_data(frame_value, maxPredProbaId, imageData_real, gravityData_real)
			else:
				featureID_7.Update_data(frame_value, maxPredProbaId, imageData_real, gravityData_real)
			
		elif (maxPredProbaId == 8) and (predict_proba[0, 8] >= upper_limt):
			if not featureID_8.frame_enable:
				featureID_8.Add_data(frame_value, maxPredProbaId, imageData_real, gravityData_real)
			else:
				featureID_8.Update_data(frame_value, maxPredProbaId, imageData_real, gravityData_real)
			
		elif (maxPredProbaId == 9) and (predict_proba_0[0, 9] >= upper_limt):
			if not featureID_9.frame_enable:
				featureID_9.Add_data(frame_value, maxPredProbaId, imageData_real, gravityData_real)
			else:
				featureID_9.Update_data(frame_value, maxPredProbaId, imageData_real, gravityData_real)
		else:
			"""
			------------------------------------------------------------------------
			If the expected probability is less than preProbRatio for all feature IDs, 
			the current ID is determined to be the new ID.
			------------------------------------------------------------------------
			"""
			judgement = True

			for tmpFeatureId in parameter_feature:
				judgement = judgement & tmpFeatureId.frame_enable
			
			if judgement == False:
				if newFeatureId_flag == True:
					# check whether empty of feature ID exist or not.
					for tmpFeatureId in parameter_feature:
						if tmpFeatureId.frame_enable == False:
							target_real = predict_proba.shape[1]
							tmpFeatureId.Add_data(frame_value, target_real, imageData_real, gravityData_real)
							break
			else:
				if newFeatureId_flag == True:
					# If an empty feature ID does not exist, 
					# the information on the feature ID with an old feature ID update time is initialized and the data is overwritten.
					if update_time_min_index == 0:
						featureID_0.Clear_data()
						featureID_0.Add_data(frame_value, 0.0, imageData_real, gravityData_real)

					elif update_time_min_index == 1:
						featureID_1.Clear()
						featureID_1.Add_data(frame_value, 1.0, imageData_real, gravityData_real)

					elif update_time_min_index == 2:
						featureID_2.Clear_data()
						featureID_2.Add_data(frame_value, 2.0, imageData_real, gravityData_real)

					elif update_time_min_index == 3:
						FeatureID_3.Clear_data()
						FeatureID_3.Add_data(frame_value, 3.0, imageData_real, gravityData_real)

					elif update_time_min_index == 4:
						featureID_4.Clear_data()
						featureID_4.Add_data(frame_value, 4.0, imageData_real, gravityData_real)

					elif update_time_min_index == 5:
						featureID_5.Clear_data()
						featureID_5.Add_data(frame_value, 5.0, imageData_real, gravityData_real)

					elif update_time_min_index == 6:
						featureID_6.Clear_data()
						featureID_6.Add_data(frame_value, 6.0, imageData_real, gravityData_real)

					elif update_time_min_index == 7:
						featureID_7.Clear_data()
						featureID_7.Add_data(frame_value, 7.0, imageData_real, gravityData_real)

					elif update_time_min_index == 8:
						featureID_8.Clear_data()
						featureID_8.Add_data(frame_value, 8.0, imageData_real, gravityData_real)

					elif update_time_min_index == 9:
						featureID_9.Clear_data()
						featureID_9.Add_data(frame_value, 9.0, imageData_real, gravityData_real)

def TrainImagePlot(idIndex, frameNumber, trainID_num):
	if trainID_num.shape[0] > 0:
		plt.figure(figsize=(14.0, 12.0))

		for i in range(trainID_num.shape[0]):
			max_rows = int((trainID_num.shape[0] / 5) + 1)
			tmp_row = int(i / 5)
			tmp_column = int(i % 5)
			plt.subplot2grid((max_rows, 5), (tmp_row, tmp_column))
			plt.xlabel("frame_number = {}".format(i), color='green', fontsize=8)
			plt.xticks([])
			plt.yticks([])
			plt.imshow(trainID_num[i, :, :], cmap=cm.binary)

		#plt.show()
		plt.show(block=False)

		file_name = 'randum_training_id{0}_f{1}.png'.format(idIndex, frameNumber)
		dir_path = os.path.join(PROJECT_ROOT_DIR, 'Images', 'TrainImages')

		if (os.path.isdir(dir_path)  == False):
			os.mkdir(dir_path)

		plt.savefig(os.path.join(PROJECT_ROOT_DIR, 'Images', 'TrainImages', file_name))

		plt.pause(pause_time)

		plt.close()


"""
--------------------------------------------------------------------------
Declare calss
--------------------------------------------------------------------------
"""
class FeatureID_Store:
	"""
	------------------------------------------------
	contents)

	------------------------------------------------
	"""
	def __init__(self):
		"""
		-----------------------------------------------

		-----------------------------------------------
		"""
		try:
			self.header_image = [
							'00_00_bit', '00_01_bit', '00_02_bit', '00_03_bit', '00_04_bit', '00_05_bit', '00_06_bit', '00_07_bit', '00_08_bit', '00_09_bit', '00_10_bit', '00_11_bit', '00_12_bit', '00_13_bit', 
							'00_14_bit', '00_15_bit', '00_16_bit', '00_17_bit', '00_18_bit', '00_19_bit', '00_20_bit', '00_21_bit', '00_22_bit', '00_23_bit', '00_24_bit', '00_25_bit', '00_26_bit', '00_27_bit',
							'01_00_bit', '01_01_bit', '01_02_bit', '01_03_bit', '01_04_bit', '01_05_bit', '01_06_bit', '01_07_bit', '01_08_bit', '01_09_bit', '01_10_bit', '01_11_bit', '01_12_bit', '01_13_bit',
							'01_14_bit', '01_15_bit', '01_16_bit', '01_17_bit', '01_18_bit', '01_19_bit', '01_20_bit', '01_21_bit', '01_22_bit', '01_23_bit', '01_24_bit', '01_25_bit', '01_26_bit', '01_27_bit',
							'02_00_bit', '02_01_bit', '02_02_bit', '02_03_bit', '02_04_bit', '02_05_bit', '02_06_bit', '02_07_bit', '02_08_bit', '02_09_bit', '02_10_bit', '02_11_bit', '02_12_bit', '02_13_bit',
							'02_14_bit', '02_15_bit', '02_16_bit', '02_17_bit', '02_18_bit', '02_19_bit', '02_20_bit', '02_21_bit', '02_22_bit', '02_23_bit', '02_24_bit', '02_25_bit', '02_26_bit', '02_27_bit',
							'03_00_bit', '03_01_bit', '03_02_bit', '03_03_bit', '03_04_bit', '03_05_bit', '03_06_bit', '03_07_bit', '03_08_bit', '03_09_bit', '03_10_bit', '03_11_bit', '03_12_bit', '03_13_bit', 
							'03_14_bit', '03_15_bit', '03_16_bit', '03_17_bit', '03_18_bit', '03_19_bit', '03_20_bit', '03_21_bit', '03_22_bit', '03_23_bit', '03_24_bit', '03_25_bit', '03_26_bit', '03_27_bit',
							'04_00_bit', '04_01_bit', '04_02_bit', '04_03_bit', '04_04_bit', '04_05_bit', '04_06_bit', '04_07_bit', '04_08_bit', '04_09_bit', '04_10_bit', '04_11_bit', '04_12_bit', '04_13_bit', 
							'04_14_bit', '04_15_bit', '04_16_bit', '04_17_bit', '04_18_bit', '04_19_bit', '04_20_bit', '04_21_bit', '04_22_bit', '04_23_bit', '04_24_bit', '04_25_bit', '04_26_bit', '04_27_bit',
							'05_00_bit', '05_01_bit', '05_02_bit', '05_03_bit', '05_04_bit', '05_05_bit', '05_06_bit', '05_07_bit', '05_08_bit', '05_09_bit', '05_10_bit', '05_11_bit', '05_12_bit', '05_13_bit',
							'05_14_bit', '05_15_bit', '05_16_bit', '05_17_bit', '05_18_bit', '05_19_bit', '05_20_bit', '05_21_bit', '05_22_bit', '05_23_bit', '05_24_bit', '05_25_bit', '05_26_bit', '05_27_bit',
							'06_00_bit', '06_01_bit', '06_02_bit', '06_03_bit', '06_04_bit', '06_05_bit', '06_06_bit', '06_07_bit', '06_08_bit', '06_09_bit', '06_10_bit', '06_11_bit', '06_12_bit', '06_13_bit', 
							'06_14_bit', '06_15_bit', '06_16_bit', '06_17_bit', '06_18_bit', '06_19_bit', '06_20_bit', '06_21_bit', '06_22_bit', '06_23_bit', '06_24_bit', '06_25_bit', '06_26_bit', '06_27_bit',
							'07_00_bit', '07_01_bit', '07_02_bit', '07_03_bit', '07_04_bit', '07_05_bit', '07_06_bit', '07_07_bit', '07_08_bit', '07_09_bit', '07_10_bit', '07_11_bit', '07_12_bit', '07_13_bit',
							'07_14_bit', '07_15_bit', '07_16_bit', '07_17_bit', '07_18_bit', '07_19_bit', '07_20_bit', '07_21_bit', '07_22_bit', '07_23_bit', '07_24_bit', '07_25_bit', '07_26_bit', '07_27_bit',
							'08_00_bit', '08_01_bit', '08_02_bit', '08_03_bit', '08_04_bit', '08_05_bit', '08_06_bit', '08_07_bit', '08_08_bit', '08_09_bit', '08_10_bit', '08_11_bit', '08_12_bit', '08_13_bit', 
							'08_14_bit', '08_15_bit', '08_16_bit', '08_17_bit', '08_18_bit', '08_19_bit', '08_20_bit', '08_21_bit', '08_22_bit', '08_23_bit', '08_24_bit', '08_25_bit', '08_26_bit', '08_27_bit',
							'09_00_bit', '09_01_bit', '09_02_bit', '09_03_bit', '09_04_bit', '09_05_bit', '09_06_bit', '09_07_bit', '09_08_bit', '09_09_bit', '09_10_bit', '09_11_bit', '09_12_bit', '09_13_bit', 
							'09_14_bit', '09_15_bit', '09_16_bit', '09_17_bit', '09_18_bit', '09_19_bit', '09_20_bit', '09_21_bit', '09_22_bit', '09_23_bit', '09_24_bit', '09_25_bit', '09_26_bit', '09_27_bit',
							'10_00_bit', '10_01_bit', '10_02_bit', '10_03_bit', '10_04_bit', '10_05_bit', '10_06_bit', '10_07_bit', '10_08_bit', '10_09_bit', '10_10_bit', '10_11_bit', '10_12_bit', '10_13_bit',
							'10_14_bit', '10_15_bit', '10_16_bit', '10_17_bit', '10_18_bit', '10_19_bit', '10_20_bit', '10_21_bit', '10_22_bit', '10_23_bit', '10_24_bit', '10_25_bit', '10_26_bit', '10_27_bit',
							'11_00_bit', '11_01_bit', '11_02_bit', '11_03_bit', '11_04_bit', '11_05_bit', '11_06_bit', '11_07_bit', '11_08_bit', '11_09_bit', '11_10_bit', '11_11_bit', '11_12_bit', '11_13_bit',
							'11_14_bit', '11_15_bit', '11_16_bit', '11_17_bit', '11_18_bit', '11_19_bit', '11_20_bit', '11_21_bit', '11_22_bit', '11_23_bit', '11_24_bit', '11_25_bit', '11_26_bit', '11_27_bit',
							'12_00_bit', '12_01_bit', '12_02_bit', '12_03_bit', '12_04_bit', '12_05_bit', '12_06_bit', '12_07_bit', '12_08_bit', '12_09_bit', '12_10_bit', '12_11_bit', '12_12_bit', '12_13_bit',
							'12_14_bit', '12_15_bit', '12_16_bit', '12_17_bit', '12_18_bit', '12_19_bit', '12_20_bit', '12_21_bit', '12_22_bit', '12_23_bit', '12_24_bit', '12_25_bit', '12_26_bit', '12_27_bit',
							'13_00_bit', '13_01_bit', '13_02_bit', '13_03_bit', '13_04_bit', '13_05_bit', '13_06_bit', '13_07_bit', '13_08_bit', '13_09_bit', '13_10_bit', '13_11_bit', '13_12_bit', '13_13_bit',
							'13_14_bit', '13_15_bit', '13_16_bit', '13_17_bit', '13_18_bit', '13_19_bit', '13_20_bit', '13_21_bit', '13_22_bit', '13_23_bit', '13_24_bit', '13_25_bit', '13_26_bit', '13_27_bit',
							'14_00_bit', '14_01_bit', '14_02_bit', '14_03_bit', '14_04_bit', '14_05_bit', '14_06_bit', '14_07_bit', '14_08_bit', '14_09_bit', '14_10_bit', '14_11_bit', '14_12_bit', '14_13_bit',
							'14_14_bit', '14_15_bit', '14_16_bit', '14_17_bit', '14_18_bit', '14_19_bit', '14_20_bit', '14_21_bit', '14_22_bit', '14_23_bit', '14_24_bit', '14_25_bit', '14_26_bit', '14_27_bit',
							'15_00_bit', '15_01_bit', '15_02_bit', '15_03_bit', '15_04_bit', '15_05_bit', '15_06_bit', '15_07_bit', '15_08_bit', '15_09_bit', '15_10_bit', '15_11_bit', '15_12_bit', '15_13_bit',
							'15_14_bit', '15_15_bit', '15_16_bit', '15_17_bit', '15_18_bit', '15_19_bit', '15_20_bit', '15_21_bit', '15_22_bit', '15_23_bit', '15_24_bit', '15_25_bit', '15_26_bit', '15_27_bit',
							'16_00_bit', '16_01_bit', '16_02_bit', '16_03_bit', '16_04_bit', '16_05_bit', '16_06_bit', '16_07_bit', '16_08_bit', '16_09_bit', '16_10_bit', '16_11_bit', '16_12_bit', '16_13_bit',
							'16_14_bit', '16_15_bit', '16_16_bit', '16_17_bit', '16_18_bit', '16_19_bit', '16_20_bit', '16_21_bit', '16_22_bit', '16_23_bit', '16_24_bit', '16_25_bit', '16_26_bit', '16_27_bit',
							'17_00_bit', '17_01_bit', '17_02_bit', '17_03_bit', '17_04_bit', '17_05_bit', '17_06_bit', '17_07_bit', '17_08_bit', '17_09_bit', '17_10_bit', '17_11_bit', '17_12_bit', '17_13_bit',
							'17_14_bit', '17_15_bit', '17_16_bit', '17_17_bit', '17_18_bit', '17_19_bit', '17_20_bit', '17_21_bit', '17_22_bit', '17_23_bit', '17_24_bit', '17_25_bit', '17_26_bit', '17_27_bit',
							'18_00_bit', '18_01_bit', '18_02_bit', '18_03_bit', '18_04_bit', '18_05_bit', '18_06_bit', '18_07_bit', '18_08_bit', '18_09_bit', '18_10_bit', '18_11_bit', '18_12_bit', '18_13_bit',
							'18_14_bit', '18_15_bit', '18_16_bit', '18_17_bit', '18_18_bit', '18_19_bit', '18_20_bit', '18_21_bit', '18_22_bit', '18_23_bit', '18_24_bit', '18_25_bit', '18_26_bit', '18_27_bit',
							'19_00_bit', '19_01_bit', '19_02_bit', '19_03_bit', '19_04_bit', '19_05_bit', '19_06_bit', '19_07_bit', '19_08_bit', '19_09_bit', '19_10_bit', '19_11_bit', '19_12_bit', '19_13_bit',
							'19_14_bit', '19_15_bit', '19_16_bit', '19_17_bit', '19_18_bit', '19_19_bit', '19_20_bit', '19_21_bit', '19_22_bit', '19_23_bit', '19_24_bit', '19_25_bit', '19_26_bit', '19_27_bit',
							'20_00_bit', '20_01_bit', '20_02_bit', '20_03_bit', '20_04_bit', '20_05_bit', '20_06_bit', '20_07_bit', '20_08_bit', '20_09_bit', '20_10_bit', '20_11_bit', '20_12_bit', '20_13_bit',
							'20_14_bit', '20_15_bit', '20_16_bit', '20_17_bit', '20_18_bit', '20_19_bit', '20_20_bit', '20_21_bit', '20_22_bit', '20_23_bit', '20_24_bit', '20_25_bit', '20_26_bit', '20_27_bit',
							'21_00_bit', '21_01_bit', '21_02_bit', '21_03_bit', '21_04_bit', '21_05_bit', '21_06_bit', '21_07_bit', '21_08_bit', '21_09_bit', '21_10_bit', '21_11_bit', '21_12_bit', '21_13_bit',
							'21_14_bit', '21_15_bit', '21_16_bit', '21_17_bit', '21_18_bit', '21_19_bit', '21_20_bit', '21_21_bit', '21_22_bit', '21_23_bit', '21_24_bit', '21_25_bit', '21_26_bit', '21_27_bit',
							'22_00_bit', '22_01_bit', '22_02_bit', '22_03_bit', '22_04_bit', '22_05_bit', '22_06_bit', '22_07_bit', '22_08_bit', '22_09_bit', '22_10_bit', '22_11_bit', '22_12_bit', '22_13_bit',
							'22_14_bit', '22_15_bit', '22_16_bit', '22_17_bit', '22_18_bit', '22_19_bit', '22_20_bit', '22_21_bit', '22_22_bit', '22_23_bit', '22_24_bit', '22_25_bit', '22_26_bit', '22_27_bit',
							'23_00_bit', '23_01_bit', '23_02_bit', '23_03_bit', '23_04_bit', '23_05_bit', '23_06_bit', '23_07_bit', '23_08_bit', '23_09_bit', '23_10_bit', '23_11_bit', '23_12_bit', '23_13_bit',
							'23_14_bit', '23_15_bit', '23_16_bit', '23_17_bit', '23_18_bit', '23_19_bit', '23_20_bit', '23_21_bit', '23_22_bit', '23_23_bit', '23_24_bit', '23_25_bit', '23_26_bit', '23_27_bit',
							'24_00_bit', '24_01_bit', '24_02_bit', '24_03_bit', '24_04_bit', '24_05_bit', '24_06_bit', '24_07_bit', '24_08_bit', '24_09_bit', '24_10_bit', '24_11_bit', '24_12_bit', '24_13_bit',
							'24_14_bit', '24_15_bit', '24_16_bit', '24_17_bit', '24_18_bit', '24_19_bit', '24_20_bit', '24_21_bit', '24_22_bit', '24_23_bit', '24_24_bit', '24_25_bit', '24_26_bit', '24_27_bit',
							'25_00_bit', '25_01_bit', '25_02_bit', '25_03_bit', '25_04_bit', '25_05_bit', '25_06_bit', '25_07_bit', '25_08_bit', '25_09_bit', '25_10_bit', '25_11_bit', '25_12_bit', '25_13_bit',
							'25_14_bit', '25_15_bit', '25_16_bit', '25_17_bit', '25_18_bit', '25_19_bit', '25_20_bit', '25_21_bit', '25_22_bit', '25_23_bit', '25_24_bit', '25_25_bit', '25_26_bit', '25_27_bit',
							'26_00_bit', '26_01_bit', '26_02_bit', '26_03_bit', '26_04_bit', '26_05_bit', '26_06_bit', '26_07_bit', '26_08_bit', '26_09_bit', '26_10_bit', '26_11_bit', '26_12_bit', '26_13_bit',
							'26_14_bit', '26_15_bit', '26_16_bit', '26_17_bit', '26_18_bit', '26_19_bit', '26_20_bit', '26_21_bit', '26_22_bit', '26_23_bit', '26_24_bit', '26_25_bit', '26_26_bit', '26_27_bit',
							'27_00_bit', '27_01_bit', '27_02_bit', '27_03_bit', '27_04_bit', '27_05_bit', '27_06_bit', '27_07_bit', '27_08_bit', '27_09_bit', '27_10_bit', '27_11_bit', '27_12_bit', '27_13_bit',
							'27_14_bit', '27_15_bit', '27_16_bit', '27_17_bit', '27_18_bit', '27_19_bit', '27_20_bit', '27_21_bit', '27_22_bit', '27_23_bit', '27_24_bit', '27_25_bit', '27_26_bit', '27_27_bit'
						]
			
			self.header_gravity = [
							'gravity_x',
							'gravity_y',
							'gravity_z',
							'gravity_doppler',
							'gravity_snr',
							'gravity_range_x',
							'gravity_range_y',
							'gravity_range_z',
							'gravity_range_doppler',
							'gravity_range_snr'
						]
			
			self.new_index = [
				'0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
				'10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
				'20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
				'30', '31', '32', '33', '34', '35', '36', '37', '38', '39',
				'40', '41', '42', '43', '44', '45', '46', '47', '48', '49'
			]


			self.num_stock_frame = 0
			self.num_max_stock = 50
			self.frame_enable = False
			self.update_time = datetime.datetime.now()

			self.frame_data = pd.DataFrame(data=None, index=None, columns=['frame_no'], dtype='int64', copy=False)
			self.frame_image_data = pd.DataFrame(data=None, index=None, columns=self.header_image, dtype='int64', copy=False)
			self.frame_gravity_data = pd.DataFrame(data=None, index=None, columns=self.header_gravity, dtype='float64', copy=False)
			self.frame_target_data = pd.DataFrame(data=None, index=None, columns=['target'], dtype='float64', copy=False)
		except Exception as ex:
			print(ex)
			pass
		finally:
			pass

	def Add_data(self, frameNumber, targetData, imageData, gravityData):
		"""
		--------------------------------------------
		contents)
		To add data.

		parameter)
		imageData: 28bit x 28bit(784bit)
		gravity data: a center data of graviry
		--------------------------------------------
		"""
		try:
			if len(imageData) == 784:
				self.frame_data.loc[self.num_stock_frame] = [frameNumber]

				# check the number of stock.
				if self.num_stock_frame >= (self.num_max_stock - 1):
					# Add new image_data
					self.frame_image_data.loc[self.num_stock_frame] = imageData

					# Add new gravity data
					self.frame_gravity_data.loc[self.num_stock_frame] = gravityData

					# Add new target data
					self.frame_target_data.loc[self.num_stock_frame] = targetData

					self.frame_enable = True
				else:
					# Add new image_data
					self.frame_image_data.loc[self.num_stock_frame] = imageData

					# Add new gravity data
					self.frame_gravity_data.loc[self.num_stock_frame] = gravityData

					# Add new target data
					self.frame_target_data.loc[self.num_stock_frame] = targetData

					# Count a number of frame
					self.num_stock_frame += 1

				self.update_time = datetime.datetime.now()
			else:
				print('Error: the number of image data is not 784!')
		except Exception as ex:
			print(ex)
			pass
		finally:
			pass
	
	
	def Update_data(self, frameNumber, targetData, imageData, gravityData):
		"""
		--------------------------------------------
		contents)
		To add data.

		parameter)
		imageData: 28bit x 28bit(784bit)
		gravity data: a center data of graviry
		--------------------------------------------
		"""
		try:
			if len(imageData) == 784:
				# check the number of stock.
				if self.num_stock_frame >= (self.num_max_stock - 1):
					"""
					frame_data processing
					"""
					# remove first row
					self.frame_data = self.frame_data.shift(-1)

					# Change last row
					self.frame_data.loc[self.num_stock_frame] = [frameNumber]

					"""
					image_data processing
					"""
					# remove first row
					self.frame_image_data = self.frame_image_data.shift(-1)

					# Change last value
					self.frame_image_data.loc[self.num_stock_frame ] = imageData

					"""
					gravity data processing
					"""
					# Remove first row
					self.frame_gravity_data = self.frame_gravity_data.shift(-1)

					# Change last row
					self.frame_gravity_data.loc[self.num_stock_frame] = gravityData

					"""
					target data processing
					"""
					# Remove first row
					self.frame_target_data = self.frame_target_data.shift(-1)

					# Change last row
					self.frame_target_data.loc[self.num_stock_frame] = targetData

					self.update_time = datetime.datetime.now()

					self.frame_enable = True
			else:
				print('Error: the number of image data is not 784!')
		except Exception as ex:
			print(ex)
			pass
		finally:
			pass

	def Remove_data(self, index_of_data):
		"""
		--------------------------------------------
		contents)
		To remove the data by specified index.

		parameter)
		index_of_data: specify the removing index of  data.
		"""
		try:
			# remove image_data
			frame_image_data.drop(index_of_data)

			# reset index of frame_image_data
			frame_image_data.reset_index()

			# remove gravity_data
			frame_gravity_data.drop(index_of_data)

			# reset index of frame_gravity_data
			frame_gravity_data.reset_index()
		except Exception as ex:
			print(ex)
			pass
		finally:
			pass

	def Clear_data(self):
		"""
		----------------------------------------------------
		contents)
		Initialize all data.
		----------------------------------------------------
		"""
		try:
			self.num_stock_frame = 0
			self.num_max_stock = 50
			self.frame_enable = False
			self.update_time = datetime.datetime.now()

			self.frame_data.tail(-1)
			self.frame_image_data.tail(-1)
			self.frame_gravity_data.tail(-1)
			self.frame_target_data.tail(-1)
		except Exception as ex:
			print(ex)
			pass
		finally:
			pass

	def Check_Num_of_frame(self):
		""""
		--------------------------------------------------
		return the number of frame
		--------------------------------------------------
		"""

		return self.numstock_frame


	def Check_Update_time(self):
		"""
		--------------------------------------------------
		return update time
		--------------------------------------------------
		"""
		return self.update_time

"""
--------------------------------------------------------------------------
Declare variable
--------------------------------------------------------------------------
"""
# relating flags to csv output.
featureSaveFlag = True
patternSaveFlag = True

# relating flags to plotter.
clusteringPlotFlag = True
traceGravityPlotFlag = True
imagesPaternPlotFlag = True

# relating flags to trainig Image
trainingDataSaveFlag = True

pause_time = 3

# Color coding by noise
lowSNR = 30
midleSNR = 100
highSNR = 150

# declare frame counter
frameCounter = 0

frameNumber = 0
frameNumberTotal = 0

# Disable StandardScaler flag
stdSclFlag = True

# existence of noise display
DrawingNoise_flag = True

# coefficient related DBSCAN
eps_value = 0.72
threshold_ratio = 0.12

# hold temporary data.(moving object)
pos_all = []

x = []
y = []
z = []

# Setting the frame display interval
maxCounter = 50

# Counter for frame display interval
counter = 0

# registe indexID
indexID_0 = []
indexID_1 = []
indexID_2 = []
indexID_3 = []
indexID_4 = []
indexID_5 = []
indexID_6 = []
indexID_7 = []
indexID_8 = []
indexID_9 = []
indexID_noise = []

# feature index for DBSCAN
dbscanID_0 = []
dbscanID_1 = []
dbscanID_2 = []
dbscanID_3 = []
dbscanID_4 = []
dbscanID_5 = []
dbscanID_6 = []
dbscanID_7 = []
dbscanID_8 = []
dbscanID_9 = []
dbscanID_noise = []

# ID Status
ID_status_pass = [False, False, False, False, False, False, False, False, False, False]
ID_status_current = [False, False, False, False, False, False, False, False, False, False]

# Create instance of FeatureID_Store
FeatureID_0 = FeatureID_Store()
FeatureID_1 = FeatureID_Store()
FeatureID_2 = FeatureID_Store()
FeatureID_3 = FeatureID_Store()
FeatureID_4 = FeatureID_Store()
FeatureID_5 = FeatureID_Store()
FeatureID_6 = FeatureID_Store()
FeatureID_7 = FeatureID_Store()
FeatureID_8 = FeatureID_Store()
FeatureID_9 = FeatureID_Store()

# list of FeatureID instance
parameter_feature = [
						FeatureID_0,
						FeatureID_1,
						FeatureID_2,
						FeatureID_3,
						FeatureID_4,
						FeatureID_5,
						FeatureID_6,
						FeatureID_7,
						FeatureID_8,
						FeatureID_9
					]

'''
-----------------------------------------------------------
registe featureID
-----------------------------------------------------------
column number : feature name
			 0: frameCounter
			 1: condition(0:static, 1:moving)
			 2: DBSCAN ID
			 3: point_x
			 4: point_y
			 5: point_z
			 6: point_doppler
			 7: point_snr
			 8: gravity_base_x
			 9: gravity_base_y
			10: gravity_base_z
			11: gravity_base_doppler
			12: gravity_base_snr
			13: gravity_x
			14: gravity_y
			15: gravity_z
			16: gravity_doppler
			17: gravity_snr
			18: gravity_range_x
			19: gravity_range_y
			20: gravity_range_z
			21: gravity_range_doppler
			22: gravity_range_snr
			23: point_x_org
			24: point_y_org'
			25: point_z_org'
			26: point_doppler_org
			27: point_snr_org
			28: gravity_x_org
			29: gravity_y_org
			30: gravity_z_org
			31: gravity_doppler_org
			32: gravity_snr_org
			33: gravity_range_x_org
			34: gravity_range_y_org
			35: gravity_range_z_org
			36: gravity_range_doppler_org
			37: gravity_range_snr_org
------------------------------------------------------------
'''
image_pattern_data = []
gravity_data = []

img_data_0 = []
img_data_1 = []
img_data_2 = []
img_data_3 = []
img_data_4 = []
img_data_5 = []
img_data_6 = []
img_data_7 = []
img_data_8 = []
img_data_9 = []

gvy_data_0 = []
gvy_data_1 = []
gvy_data_2 = []
gvy_data_3 = []
gvy_data_4 = []
gvy_data_5 = []
gvy_data_6 = []
gvy_data_7 = []
gvy_data_8 = []
gvy_data_9 = []

# store a center data of gavity each id
c_gravityID_0 = []
c_gravityID_1 = []
c_gravityID_2 = []
c_gravityID_3 = []
c_gravityID_4 = []
c_gravityID_5 = []
c_gravityID_6 = []
c_gravityID_7 = []
c_gravityID_8 = []
c_gravityID_9 = []

# store a ramge of cluster each id.
gravity_base_x_min = []
gravity_base_x_max = []
gravity_base_y_min = []
gravity_base_y_max = []

# judgement ratio whether enable or not.
predictProbaRatio = 0.70

# instance list of feature Id
featureList = [
		FeatureID_0, FeatureID_1, FeatureID_2, FeatureID_3, FeatureID_4,
		FeatureID_5, FeatureID_6, FeatureID_7, FeatureID_8, FeatureID_9
	]


targetID_0_index = []
targetID_1_index = []
targetID_2_index = []
targetID_3_index = []
targetID_4_index = []
targetID_5_index = []
targetID_6_index = []
targetID_7_index = []
targetID_8_index = []
targetID_9_index = []

search_params = {
    'n_estimators'      : [1000, 1050, 1100],
    'max_features'      : [3, 4, 5, 6, 7, 8, 9, 10],
    'random_state'      : [2525],
    'min_samples_split' : [2, 3],
    'max_depth'         : [20, 30]
}

gs = GridSearchCV(
                    RandomForestClassifier(),           # Target machine learning model
                    search_params,                      # Search parameter dictionary
                    cv=3,                               # Cross validation division number
                    verbose=False,                       # display log
                    n_jobs=-1
                )

newFeatureId_flag = False

# setting path of using data
path_files = os.path.join(PROJECT_ROOT_DIR, 'Data')

# check file names and create list.
files = os.listdir(path_files)

# create existing file path in directory.
files_file = [f for f in files if os.path.isfile(os.path.join(path_files, f))]

# A flag indicating whether or not we are using dummy data
dummyUseFlag = False

"""
-----------------------------------------------------------
Store dumy data.
this data is used classification.
-----------------------------------------------------------
"""
for file_name in files_file:
	if 'dummy_pattern_data' in file_name:
		path_images_file = file_name
	
	if 'dummy_gravity_data' in file_name:
		path_gravitys_file = file_name

# extract dumy data in data directory.
if len(path_images_file) != 0:
	df_dummy_image = pd.read_csv(os.path.join(PROJECT_ROOT_DIR, 'Data', path_images_file))
	# confirm raw data
	print('df_dummy_image.head() = \n{0}\n'.format(df_dummy_image.head()))

if len(path_gravitys_file) != 0:
	df_dummy_gravity = pd.read_csv(os.path.join(PROJECT_ROOT_DIR, 'Data', path_gravitys_file))
	print('df_dummy_gravity.head() = \n{0}\n'.format(df_dummy_gravity.head()))

# Read dummy data
try:
	# Store image data
	for indexId, dummy_featureId in enumerate(parameter_feature):
		# get index inforamation of DBSCAN_ID
		dummyID_image_index = np.where(df_dummy_image['DBSCAN_ID'] == indexId)

		# get image data
		df_dummy_image_data = df_dummy_image.values

		df_dummy_image_data = df_dummy_image_data[dummyID_image_index]

		df_dummy_gravity_data = df_dummy_gravity.values

		df_dummy_gravity_data = df_dummy_gravity_data[dummyID_image_index]

		if (df_dummy_image_data.shape[0] > 0) and (df_dummy_gravity_data.shape[0] > 0):
			for indexTmp in range(df_dummy_image_data.shape[0]):
				# Exstract frame_no
				frame_value = df_dummy_image_data[indexTmp, 0]

				target_real = df_dummy_image_data[indexTmp, 1]

				# create image data only.
				imageData_real = df_dummy_image_data[indexTmp, 2: 786]

				# create gravity data only.
				gravityData_real = df_dummy_gravity_data[indexTmp, 2: 12]

				# Add new data
				dummy_featureId.Add_data(frame_value, target_real, imageData_real, gravityData_real)

except Exception as ex:
	print(ex)
	pass
finally:
	pass


"""
------------------------------------------------------------
Devide the file each 100 frames.(the number of datas is about 10000.)
------------------------------------------------------------
"""
enable_file = []
numeric_number = []

# extract frame numbers in the file names.
for file_name in files_file:
	file_st = file_name.split('_f')
	if len(file_st) >= 2 and file_st[0] == '3d_people_counting_data':
		number = file_st[1].split('.')
		if len(number) >= 2:
			if str.isnumeric(number[0]):
				numeric_number.append(int(number[0]))
				enable_file.append(file_name)

# Sort the frame number.
numeric_number.sort()

# prepare column header
header = [
				'frame_no',
				'point_x',
				'point_y',
				'point_z',
				'point_doppler',
				'point_snr'
			]

firstRowFlag = True

# setting frame_number
frame_number = 0

# create a divided file.
if len(enable_file) == 0:
	with open(os.path.join(PROJECT_ROOT_DIR, 'Data', '3d_people_counting_data.csv')) as f:
		reader = csv.reader(f)
		for row in reader:
			if str.isnumeric(row[0]):
				row_number = int(row[0])
				#print('row_number: {0}'.format(row_number))

				if row_number == 0:
					if frame_number % 100 == 0:
						path_file = os.path.join(PROJECT_ROOT_DIR, 'Data', '3d_people_counting_data_f{0}.csv'.format(frame_number))

						# Create new faile
						with open(path_file, 'w', newline='') as fh:
							writer = csv.writer(fh)
							writer.writerow(header)
						
						# Add data.
						with open(path_file, 'a', newline='') as fr:
							writer = csv.writer(fr)
							writer.writerow(row)
					else:
						# Add data.
						with open(path_file, 'a', newline='') as fr:
							writer = csv.writer(fr)
							writer.writerow(row)
					
					# update frame_number
					frame_number = frame_number + 1
				else:
					# Add data.
					with open(path_file, 'a', newline='') as fr:
						writer = csv.writer(fr)
						writer.writerow(row)
	
	# setting path of using data
	path_files = os.path.join(PROJECT_ROOT_DIR, 'Data')

	# check file names and create list.
	files = os.listdir(path_files)

	# create existing file path in directory.
	files_file = [f for f in files if os.path.isfile(os.path.join(path_files, f))]
	
	enable_file.clear()
	numeric_number.clear()

	# extract frame numbers in the file names.
	for file_name in files_file:
		file_st = file_name.split('_f')
		if len(file_st) >= 2 and file_st[0] == '3d_people_counting_data':
			number = file_st[1].split('.')
			if len(number) >= 2:
				if str.isnumeric(number[0]):
					numeric_number.append(int(number[0]))
					enable_file.append(file_name)

	# Sort the frame number.
	numeric_number.sort()

# featureID in raw data
featureID_raw = []

# featureID after DBSCAN
featureID = []

"""
--------------------------------------------------------------------------------
Read and process the data for each segmented file.
--------------------------------------------------------------------------------
"""
for tmpInt in numeric_number:
	file_name = '3d_people_counting_data_f{0}.csv'.format(tmpInt)
	path_file = os.path.join(PROJECT_ROOT_DIR, 'Data', file_name)

	if os.path.isfile(path_file):
		# feature list
		featureID_raw.clear()

		# create pandas.Dataframe by the csv file.
		df = pd.read_csv(os.path.join(PROJECT_ROOT_DIR, 'Data', path_file))

		# confirm raw data
		print('df.head() = \n{0}\n'.format(df.head()))

		pos_all.clear()

		firstRowFlag = True

		"""
		--------------------------------------------------------------------------------------
		Create all data
		--------------------------------------------------------------------------------------
		"""
		for index_df, item_df in df.iterrows():
			# Initialize data in case of index is 0.
			if item_df['frame_no'] == 0:
				if firstRowFlag == False:
					# Store feature.
					pos_all_ = copy.copy(pos_all)
					featureID_raw.extend([pos_all_])

					# reset pos_all data
					pos_all.clear()
				else:
					firstRowFlag = False

				# Store data
				pos_all.append([item_df['point_x'], item_df['point_y'], item_df['point_z'], item_df['point_doppler'], item_df['point_snr']])
			else:
				# store data
				pos_all.append([item_df['point_x'], item_df['point_y'], item_df['point_z'], item_df['point_doppler'], item_df['point_snr']])
		
		"""
		--------------------------------------------------------------------------------------
		main routine
		--------------------------------------------------------------------------------------
		"""
		df_thinned_frame = pd.DataFrame(featureID_raw)

		for index_df_thinned, item_df_thinned in df_thinned_frame.iterrows():
			# total frame number
			frameNumberTotal = tmpInt + index_df_thinned
			print('frameNumberTotal = {0}'.format(frameNumberTotal))

			# Processing per frame
			row_list = item_df_thinned.values.tolist()
			row_list = [x for x in row_list if x]
			row_numpy = np.array(row_list)

			# Calculate the distribution before normalization.
			before_min = np.min(row_numpy, axis=0)
			before_max = np.max(row_numpy, axis=0)
			before_range = np.subtract(before_max, before_min)

			"""
			----------------------------------------------------------------------------------
			Perform standardization work.
			----------------------------------------------------------------------------------
			"""
			row_numpy_std = StandardScaler(copy=True, with_mean=False, with_std=True).fit_transform(row_numpy.tolist())

			# Calculate the distribution after normalization.
			after_min = np.min(row_numpy_std, axis=0)
			after_max = np.max(row_numpy_std, axis=0)
			after_range = np.subtract(after_max, after_min)

			# calculate ratio.
			ratio = np.divide(before_range, after_range)
			np.nan_to_num(ratio, copy=False)

			"""
			----------------------------------------------------------------------------------
			clustering(DBSCAN)
			----------------------------------------------------------------------------------
			"""
			# compute DBSCAN(remove noise)
			all_threshold = len(row_numpy_std) * threshold_ratio

			db_all = DBSCAN(eps=eps_value, min_samples=all_threshold).fit(row_numpy_std)

			# cluster label
			labels_all = db_all.labels_

			if db_all.components_.shape[0] > 0:
				for i, coreIndex in enumerate(db_all.core_sample_indices_):
					if labels_all[coreIndex] == 0:
						indexID_0.append(db_all.components_[i])
					elif labels_all[coreIndex] == 1:
						indexID_1.append(db_all.components_[i])
					elif labels_all[coreIndex] == 2:
						indexID_2.append(db_all.components_[i])
					elif labels_all[coreIndex] == 3:
						indexID_3.append(db_all.components_[i])
					elif labels_all[coreIndex] == 4:
						indexID_4.append(db_all.components_[i])
					elif labels_all[coreIndex] == 5:
						indexID_5.append(db_all.components_[i])
					elif labels_all[coreIndex] == 6:
						indexID_6.append(db_all.components_[i])
					elif labels_all[coreIndex] == 7:
						indexID_7.append(db_all.components_[i])
					elif labels_all[coreIndex] == 8:
						indexID_8.append(db_all.components_[i])
					elif labels_all[coreIndex] == 9:
						indexID_9.append(db_all.components_[i])
					elif labels_all[coreIndex] == -1:
						indexID_noise.append(db_all.components_[i])
			
			
			'''
			------------------------------------------------------------------------
			Get feature values
			------------------------------------------------------------------------
			'''
			header = [
						'frame_no',
						'condition',
						'DBSCAN_ID',
						'point_x',
						'point_y',
						'point_z',
						'point_doppler',
						'point_snr',
						'gravity_base_x',
						'gravity_base_y',
						'gravity_base_z',
						'gravity_base_doppler',
						'gravity_base_snr',
						'gravity_x',
						'gravity_y',
						'gravity_z',
						'gravity_doppler',
						'gravity_snr',
						'gravity_range_x',
						'gravity_range_y',
						'gravity_range_z',
						'gravity_range_doppler',
						'gravity_range_snr',
						'point_x_org',
						'point_y_org',
						'point_z_org',
						'point_doppler_org',
						'point_snr_org',
						'gravity_x_org',
						'gravity_y_org',
						'gravity_z_org',
						'gravity_doppler_org',
						'gravity_snr_org',
						'gravity_range_x_org',
						'gravity_range_y_org',
						'gravity_range_z_org',
						'gravity_range_doppler_org',
						'gravity_range_snr_org',
					]
			
			# write Csv file
			if featureSaveFlag == True:
				path_feature_analyssis = os.path.join(PROJECT_ROOT_DIR, 'Data', 'feature_analyssis_data_f{0}.csv'.format(tmpInt))
				if not os.path.isfile(path_feature_analyssis):
					with open(path_feature_analyssis, 'w', newline='') as fh:
						writer = csv.writer(fh)
						writer.writerow(header)
			
			if (indexID_0 is not None) and (len(indexID_0) != 0):
				CaluclateCenterOfGravity(frameNumberTotal, 'indexID_0', indexID_0, featureSaveFlag, path_feature_analyssis)

			if (indexID_1 is not None) and (len(indexID_1) != 0):
				CaluclateCenterOfGravity(frameNumberTotal, 'indexID_1', indexID_1, featureSaveFlag, path_feature_analyssis)

			if (indexID_2 is not None) and (len(indexID_2) != 0):
				CaluclateCenterOfGravity(frameNumberTotal, 'indexId_2', indexID_2, featureSaveFlag, path_feature_analyssis)

			if (indexID_3 is not None) and (len(indexID_3) != 0):
				CaluclateCenterOfGravity(frameNumberTotal, 'indexID_3', indexID_3, featureSaveFlag, path_feature_analyssis)

			if (indexID_4 is not None) and (len(indexID_4) != 0):
				CaluclateCenterOfGravity(frameNumberTotal, 'indexID_4', indexID_4, featureSaveFlag, path_feature_analyssis)

			if (indexID_5 is not None) and (len(indexID_5) != 0):
				CaluclateCenterOfGravity(frameNumberTotal, 'indexID_5', indexID_5, featureSaveFlag, path_feature_analyssis)

			if (indexID_6 is not None) and (len(indexID_6) != 0):
				CaluclateCenterOfGravity(frameNumberTotal, 'indexID_6', indexID_6, featureSaveFlag, path_feature_analyssis)

			if (indexID_7 is not None) and (len(indexID_7) != 0):
				CaluclateCenterOfGravity(frameNumberTotal, 'indexID_7', indexID_7, featureSaveFlag, path_feature_analyssis)

			if (indexID_8 is not None) and (len(indexID_8) != 0):
				CaluclateCenterOfGravity(frameNumberTotal, 'indexID_8', indexID_8, featureSaveFlag, path_feature_analyssis)

			if (indexID_9 is not None) and (len(indexID_9) != 0):
				CaluclateCenterOfGravity(frameNumberTotal, 'indexID_9', indexID_9, featureSaveFlag, path_feature_analyssis)

			#CaluclateCenterOfGravity(frameNumberTotal, 'indexID_moise', indexID_noise, featureSaveFlag, path_feature_analyssis)


			if (featureID is not None) and (len(featureID) != 0):
				featureID_array = np.array(featureID)

				# Extract the index of each feature ID.
				dbscanID_0_index = np.where((featureID_array[:, 2] == 0) & (featureID_array[:, 0] == frameNumberTotal))
				dbscanID_1_index = np.where((featureID_array[:, 2] == 1) & (featureID_array[:, 0] == frameNumberTotal))
				dbscanID_2_index = np.where((featureID_array[:, 2] == 2) & (featureID_array[:, 0] == frameNumberTotal))
				dbscanID_3_index = np.where((featureID_array[:, 2] == 3) & (featureID_array[:, 0] == frameNumberTotal))
				dbscanID_4_index = np.where((featureID_array[:, 2] == 4) & (featureID_array[:, 0] == frameNumberTotal))
				dbscanID_5_index = np.where((featureID_array[:, 2] == 5) & (featureID_array[:, 0] == frameNumberTotal))
				dbscanID_6_index = np.where((featureID_array[:, 2] == 6) & (featureID_array[:, 0] == frameNumberTotal))
				dbscanID_7_index = np.where((featureID_array[:, 2] == 7) & (featureID_array[:, 0] == frameNumberTotal))
				dbscanID_8_index = np.where((featureID_array[:, 2] == 8) & (featureID_array[:, 0] == frameNumberTotal))
				dbscanID_9_index = np.where((featureID_array[:, 2] == 9) & (featureID_array[:, 0] == frameNumberTotal))
				#dbscanID_noise_index = np.where((featureID_array[:, 2] == -1) & (featureID_array[:, 0] == frameNumberTotal))

				# A feature matrix is extracted based on the index of each feature amount ID.
				dbscanID_0 = featureID_array[dbscanID_0_index].tolist()
				dbscanID_1 = featureID_array[dbscanID_1_index].tolist()
				dbscanID_2 = featureID_array[dbscanID_2_index].tolist()
				dbscanID_3 = featureID_array[dbscanID_3_index].tolist()
				dbscanID_4 = featureID_array[dbscanID_4_index].tolist()
				dbscanID_5 = featureID_array[dbscanID_5_index].tolist()
				dbscanID_6 = featureID_array[dbscanID_6_index].tolist()
				dbscanID_7 = featureID_array[dbscanID_7_index].tolist()
				dbscanID_8 = featureID_array[dbscanID_8_index].tolist()
				dbscanID_9 = featureID_array[dbscanID_9_index].tolist()
				#dbscanID_noise = featureID_array[dbscanID_noise_index].tolist()
			else:
				print('break!')
				break

			'''
			-------------------------------------------------------------
			cluster tracking (Pattern recognition base)

			Program to track and display individual clusters(28bit x 28bit)
			-------------------------------------------------------------
			'''
			# transfer ndarray to pandas.DataFrame
			df_feature = pd.DataFrame(featureID_array)

			df_feature.columns = header

			# Remove abnormal data.
			df_feature = df_feature[df_feature["point_y_org"] <= 5.0]

			# 
			print('df_feature.head() = \n{0}\n'.format(df_feature.head()))

			# Extract existing frame numbers
			frame_number = df_feature["frame_no"].unique()

			# store a number of frame
			N = frame_number.shape[0]

			if patternSaveFlag == True:
				# write pattern datas(28bit x 28bit) to csv file.
				path_pattern_data = os.path.join(PROJECT_ROOT_DIR, 'Data', 'pattern_data_f{0}.csv'.format(tmpInt))
				if not os.path.isfile(path_pattern_data):
					header_pattern = [
								'frame_no',
								'DBSCAN_ID',
								'00_00_bit', '00_01_bit', '00_02_bit', '00_03_bit', '00_04_bit', '00_05_bit', '00_06_bit', '00_07_bit', '00_08_bit', '00_09_bit', '00_10_bit', '00_11_bit', '00_12_bit', '00_13_bit', 
								'00_14_bit', '00_15_bit', '00_16_bit', '00_17_bit', '00_18_bit', '00_19_bit', '00_20_bit', '00_21_bit', '00_22_bit', '00_23_bit', '00_24_bit', '00_25_bit', '00_26_bit', '00_27_bit',
								'01_00_bit', '01_01_bit', '01_02_bit', '01_03_bit', '01_04_bit', '01_05_bit', '01_06_bit', '01_07_bit', '01_08_bit', '01_09_bit', '01_10_bit', '01_11_bit', '01_12_bit', '01_13_bit',
								'01_14_bit', '01_15_bit', '01_16_bit', '01_17_bit', '01_18_bit', '01_19_bit', '01_20_bit', '01_21_bit', '01_22_bit', '01_23_bit', '01_24_bit', '01_25_bit', '01_26_bit', '01_27_bit',
								'02_00_bit', '02_01_bit', '02_02_bit', '02_03_bit', '02_04_bit', '02_05_bit', '02_06_bit', '02_07_bit', '02_08_bit', '02_09_bit', '02_10_bit', '02_11_bit', '02_12_bit', '02_13_bit',
								'02_14_bit', '02_15_bit', '02_16_bit', '02_17_bit', '02_18_bit', '02_19_bit', '02_20_bit', '02_21_bit', '02_22_bit', '02_23_bit', '02_24_bit', '02_25_bit', '02_26_bit', '02_27_bit',
								'03_00_bit', '03_01_bit', '03_02_bit', '03_03_bit', '03_04_bit', '03_05_bit', '03_06_bit', '03_07_bit', '03_08_bit', '03_09_bit', '03_10_bit', '03_11_bit', '03_12_bit', '03_13_bit', 
								'03_14_bit', '03_15_bit', '03_16_bit', '03_17_bit', '03_18_bit', '03_19_bit', '03_20_bit', '03_21_bit', '03_22_bit', '03_23_bit', '03_24_bit', '03_25_bit', '03_26_bit', '03_27_bit',
								'04_00_bit', '04_01_bit', '04_02_bit', '04_03_bit', '04_04_bit', '04_05_bit', '04_06_bit', '04_07_bit', '04_08_bit', '04_09_bit', '04_10_bit', '04_11_bit', '04_12_bit', '04_13_bit', 
								'04_14_bit', '04_15_bit', '04_16_bit', '04_17_bit', '04_18_bit', '04_19_bit', '04_20_bit', '04_21_bit', '04_22_bit', '04_23_bit', '04_24_bit', '04_25_bit', '04_26_bit', '04_27_bit',
								'05_00_bit', '05_01_bit', '05_02_bit', '05_03_bit', '05_04_bit', '05_05_bit', '05_06_bit', '05_07_bit', '05_08_bit', '05_09_bit', '05_10_bit', '05_11_bit', '05_12_bit', '05_13_bit',
								'05_14_bit', '05_15_bit', '05_16_bit', '05_17_bit', '05_18_bit', '05_19_bit', '05_20_bit', '05_21_bit', '05_22_bit', '05_23_bit', '05_24_bit', '05_25_bit', '05_26_bit', '05_27_bit',
								'06_00_bit', '06_01_bit', '06_02_bit', '06_03_bit', '06_04_bit', '06_05_bit', '06_06_bit', '06_07_bit', '06_08_bit', '06_09_bit', '06_10_bit', '06_11_bit', '06_12_bit', '06_13_bit', 
								'06_14_bit', '06_15_bit', '06_16_bit', '06_17_bit', '06_18_bit', '06_19_bit', '06_20_bit', '06_21_bit', '06_22_bit', '06_23_bit', '06_24_bit', '06_25_bit', '06_26_bit', '06_27_bit',
								'07_00_bit', '07_01_bit', '07_02_bit', '07_03_bit', '07_04_bit', '07_05_bit', '07_06_bit', '07_07_bit', '07_08_bit', '07_09_bit', '07_10_bit', '07_11_bit', '07_12_bit', '07_13_bit',
								'07_14_bit', '07_15_bit', '07_16_bit', '07_17_bit', '07_18_bit', '07_19_bit', '07_20_bit', '07_21_bit', '07_22_bit', '07_23_bit', '07_24_bit', '07_25_bit', '07_26_bit', '07_27_bit',
								'08_00_bit', '08_01_bit', '08_02_bit', '08_03_bit', '08_04_bit', '08_05_bit', '08_06_bit', '08_07_bit', '08_08_bit', '08_09_bit', '08_10_bit', '08_11_bit', '08_12_bit', '08_13_bit', 
								'08_14_bit', '08_15_bit', '08_16_bit', '08_17_bit', '08_18_bit', '08_19_bit', '08_20_bit', '08_21_bit', '08_22_bit', '08_23_bit', '08_24_bit', '08_25_bit', '08_26_bit', '08_27_bit',
								'09_00_bit', '09_01_bit', '09_02_bit', '09_03_bit', '09_04_bit', '09_05_bit', '09_06_bit', '09_07_bit', '09_08_bit', '09_09_bit', '09_10_bit', '09_11_bit', '09_12_bit', '09_13_bit', 
								'09_14_bit', '09_15_bit', '09_16_bit', '09_17_bit', '09_18_bit', '09_19_bit', '09_20_bit', '09_21_bit', '09_22_bit', '09_23_bit', '09_24_bit', '09_25_bit', '09_26_bit', '09_27_bit',
								'10_00_bit', '10_01_bit', '10_02_bit', '10_03_bit', '10_04_bit', '10_05_bit', '10_06_bit', '10_07_bit', '10_08_bit', '10_09_bit', '10_10_bit', '10_11_bit', '10_12_bit', '10_13_bit',
								'10_14_bit', '10_15_bit', '10_16_bit', '10_17_bit', '10_18_bit', '10_19_bit', '10_20_bit', '10_21_bit', '10_22_bit', '10_23_bit', '10_24_bit', '10_25_bit', '10_26_bit', '10_27_bit',
								'11_00_bit', '11_01_bit', '11_02_bit', '11_03_bit', '11_04_bit', '11_05_bit', '11_06_bit', '11_07_bit', '11_08_bit', '11_09_bit', '11_10_bit', '11_11_bit', '11_12_bit', '11_13_bit',
								'11_14_bit', '11_15_bit', '11_16_bit', '11_17_bit', '11_18_bit', '11_19_bit', '11_20_bit', '11_21_bit', '11_22_bit', '11_23_bit', '11_24_bit', '11_25_bit', '11_26_bit', '11_27_bit',
								'12_00_bit', '12_01_bit', '12_02_bit', '12_03_bit', '12_04_bit', '12_05_bit', '12_06_bit', '12_07_bit', '12_08_bit', '12_09_bit', '12_10_bit', '12_11_bit', '12_12_bit', '12_13_bit',
								'12_14_bit', '12_15_bit', '12_16_bit', '12_17_bit', '12_18_bit', '12_19_bit', '12_20_bit', '12_21_bit', '12_22_bit', '12_23_bit', '12_24_bit', '12_25_bit', '12_26_bit', '12_27_bit',
								'13_00_bit', '13_01_bit', '13_02_bit', '13_03_bit', '13_04_bit', '13_05_bit', '13_06_bit', '13_07_bit', '13_08_bit', '13_09_bit', '13_10_bit', '13_11_bit', '13_12_bit', '13_13_bit',
								'13_14_bit', '13_15_bit', '13_16_bit', '13_17_bit', '13_18_bit', '13_19_bit', '13_20_bit', '13_21_bit', '13_22_bit', '13_23_bit', '13_24_bit', '13_25_bit', '13_26_bit', '13_27_bit',
								'14_00_bit', '14_01_bit', '14_02_bit', '14_03_bit', '14_04_bit', '14_05_bit', '14_06_bit', '14_07_bit', '14_08_bit', '14_09_bit', '14_10_bit', '14_11_bit', '14_12_bit', '14_13_bit',
								'14_14_bit', '14_15_bit', '14_16_bit', '14_17_bit', '14_18_bit', '14_19_bit', '14_20_bit', '14_21_bit', '14_22_bit', '14_23_bit', '14_24_bit', '14_25_bit', '14_26_bit', '14_27_bit',
								'15_00_bit', '15_01_bit', '15_02_bit', '15_03_bit', '15_04_bit', '15_05_bit', '15_06_bit', '15_07_bit', '15_08_bit', '15_09_bit', '15_10_bit', '15_11_bit', '15_12_bit', '15_13_bit',
								'15_14_bit', '15_15_bit', '15_16_bit', '15_17_bit', '15_18_bit', '15_19_bit', '15_20_bit', '15_21_bit', '15_22_bit', '15_23_bit', '15_24_bit', '15_25_bit', '15_26_bit', '15_27_bit',
								'16_00_bit', '16_01_bit', '16_02_bit', '16_03_bit', '16_04_bit', '16_05_bit', '16_06_bit', '16_07_bit', '16_08_bit', '16_09_bit', '16_10_bit', '16_11_bit', '16_12_bit', '16_13_bit',
								'16_14_bit', '16_15_bit', '16_16_bit', '16_17_bit', '16_18_bit', '16_19_bit', '16_20_bit', '16_21_bit', '16_22_bit', '16_23_bit', '16_24_bit', '16_25_bit', '16_26_bit', '16_27_bit',
								'17_00_bit', '17_01_bit', '17_02_bit', '17_03_bit', '17_04_bit', '17_05_bit', '17_06_bit', '17_07_bit', '17_08_bit', '17_09_bit', '17_10_bit', '17_11_bit', '17_12_bit', '17_13_bit',
								'17_14_bit', '17_15_bit', '17_16_bit', '17_17_bit', '17_18_bit', '17_19_bit', '17_20_bit', '17_21_bit', '17_22_bit', '17_23_bit', '17_24_bit', '17_25_bit', '17_26_bit', '17_27_bit',
								'18_00_bit', '18_01_bit', '18_02_bit', '18_03_bit', '18_04_bit', '18_05_bit', '18_06_bit', '18_07_bit', '18_08_bit', '18_09_bit', '18_10_bit', '18_11_bit', '18_12_bit', '18_13_bit',
								'18_14_bit', '18_15_bit', '18_16_bit', '18_17_bit', '18_18_bit', '18_19_bit', '18_20_bit', '18_21_bit', '18_22_bit', '18_23_bit', '18_24_bit', '18_25_bit', '18_26_bit', '18_27_bit',
								'19_00_bit', '19_01_bit', '19_02_bit', '19_03_bit', '19_04_bit', '19_05_bit', '19_06_bit', '19_07_bit', '19_08_bit', '19_09_bit', '19_10_bit', '19_11_bit', '19_12_bit', '19_13_bit',
								'19_14_bit', '19_15_bit', '19_16_bit', '19_17_bit', '19_18_bit', '19_19_bit', '19_20_bit', '19_21_bit', '19_22_bit', '19_23_bit', '19_24_bit', '19_25_bit', '19_26_bit', '19_27_bit',
								'20_00_bit', '20_01_bit', '20_02_bit', '20_03_bit', '20_04_bit', '20_05_bit', '20_06_bit', '20_07_bit', '20_08_bit', '20_09_bit', '20_10_bit', '20_11_bit', '20_12_bit', '20_13_bit',
								'20_14_bit', '20_15_bit', '20_16_bit', '20_17_bit', '20_18_bit', '20_19_bit', '20_20_bit', '20_21_bit', '20_22_bit', '20_23_bit', '20_24_bit', '20_25_bit', '20_26_bit', '20_27_bit',
								'21_00_bit', '21_01_bit', '21_02_bit', '21_03_bit', '21_04_bit', '21_05_bit', '21_06_bit', '21_07_bit', '21_08_bit', '21_09_bit', '21_10_bit', '21_11_bit', '21_12_bit', '21_13_bit',
								'21_14_bit', '21_15_bit', '21_16_bit', '21_17_bit', '21_18_bit', '21_19_bit', '21_20_bit', '21_21_bit', '21_22_bit', '21_23_bit', '21_24_bit', '21_25_bit', '21_26_bit', '21_27_bit',
								'22_00_bit', '22_01_bit', '22_02_bit', '22_03_bit', '22_04_bit', '22_05_bit', '22_06_bit', '22_07_bit', '22_08_bit', '22_09_bit', '22_10_bit', '22_11_bit', '22_12_bit', '22_13_bit',
								'22_14_bit', '22_15_bit', '22_16_bit', '22_17_bit', '22_18_bit', '22_19_bit', '22_20_bit', '22_21_bit', '22_22_bit', '22_23_bit', '22_24_bit', '22_25_bit', '22_26_bit', '22_27_bit',
								'23_00_bit', '23_01_bit', '23_02_bit', '23_03_bit', '23_04_bit', '23_05_bit', '23_06_bit', '23_07_bit', '23_08_bit', '23_09_bit', '23_10_bit', '23_11_bit', '23_12_bit', '23_13_bit',
								'23_14_bit', '23_15_bit', '23_16_bit', '23_17_bit', '23_18_bit', '23_19_bit', '23_20_bit', '23_21_bit', '23_22_bit', '23_23_bit', '23_24_bit', '23_25_bit', '23_26_bit', '23_27_bit',
								'24_00_bit', '24_01_bit', '24_02_bit', '24_03_bit', '24_04_bit', '24_05_bit', '24_06_bit', '24_07_bit', '24_08_bit', '24_09_bit', '24_10_bit', '24_11_bit', '24_12_bit', '24_13_bit',
								'24_14_bit', '24_15_bit', '24_16_bit', '24_17_bit', '24_18_bit', '24_19_bit', '24_20_bit', '24_21_bit', '24_22_bit', '24_23_bit', '24_24_bit', '24_25_bit', '24_26_bit', '24_27_bit',
								'25_00_bit', '25_01_bit', '25_02_bit', '25_03_bit', '25_04_bit', '25_05_bit', '25_06_bit', '25_07_bit', '25_08_bit', '25_09_bit', '25_10_bit', '25_11_bit', '25_12_bit', '25_13_bit',
								'25_14_bit', '25_15_bit', '25_16_bit', '25_17_bit', '25_18_bit', '25_19_bit', '25_20_bit', '25_21_bit', '25_22_bit', '25_23_bit', '25_24_bit', '25_25_bit', '25_26_bit', '25_27_bit',
								'26_00_bit', '26_01_bit', '26_02_bit', '26_03_bit', '26_04_bit', '26_05_bit', '26_06_bit', '26_07_bit', '26_08_bit', '26_09_bit', '26_10_bit', '26_11_bit', '26_12_bit', '26_13_bit',
								'26_14_bit', '26_15_bit', '26_16_bit', '26_17_bit', '26_18_bit', '26_19_bit', '26_20_bit', '26_21_bit', '26_22_bit', '26_23_bit', '26_24_bit', '26_25_bit', '26_26_bit', '26_27_bit',
								'27_00_bit', '27_01_bit', '27_02_bit', '27_03_bit', '27_04_bit', '27_05_bit', '27_06_bit', '27_07_bit', '27_08_bit', '27_09_bit', '27_10_bit', '27_11_bit', '27_12_bit', '27_13_bit',
								'27_14_bit', '27_15_bit', '27_16_bit', '27_17_bit', '27_18_bit', '27_19_bit', '27_20_bit', '27_21_bit', '27_22_bit', '27_23_bit', '27_24_bit', '27_25_bit', '27_26_bit', '27_27_bit'
							]
					with open(path_pattern_data, 'w', newline='') as fp:
						writer = csv.writer(fp)
						writer.writerow(header_pattern)
				
				# write the gravity related data to csv file.
				path_gravity_data = os.path.join(PROJECT_ROOT_DIR, 'Data', 'gravity_data_f{0}.csv'.format(tmpInt))
				if not os.path.isfile(path_gravity_data):
					header_gravity = [
								'frame_no',
								'DBSCAN_ID',
								'gravity_x',
								'gravity_y',
								'gravity_z',
								'gravity_doppler',
								'gravity_snr',
								'gravity_range_x',
								'gravity_range_y',
								'gravity_range_z',
								'gravity_range_doppler',
								'gravity_range_snr',
							]

					with open(path_gravity_data, 'w', newline='') as fp:
						writer = csv.writer(fp)
						writer.writerow(header_gravity)

			"""
			--------------------------------------------------------------------------
			Show the status of the point cloud before and after data standardization.
			--------------------------------------------------------------------------
			"""
			if traceGravityPlotFlag == True:
				# Creat 3D plot
				fig = plt.figure(figsize=(5.8, 10.0))

				ax1 = fig.add_subplot(2, 1, 1)
				ax1.set_title('2D cluster after DBSCAN processing')

				cmap = plt.cm.coolwarm
				rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, N)))

				# display point cloud  each frames.
				#target_row = df_feature[(df_feature["frame_no"] == frame_number[0])]
				target_row = df_feature[(df_feature["frame_no"] == frameNumberTotal)]

				gravity_base_x_min.append(target_row["gravity_base_x"].min())
				gravity_base_x_max.append(target_row["gravity_base_x"].max())
				gravity_base_y_min.append(target_row["gravity_base_y"].min())
				gravity_base_y_max.append(target_row["gravity_base_y"].max())

				id_0 = target_row[(target_row["DBSCAN_ID"] == 0)]
				id_1 = target_row[(target_row["DBSCAN_ID"] == 1)]
				id_2 = target_row[(target_row["DBSCAN_ID"] == 2)]
				id_3 = target_row[(target_row["DBSCAN_ID"] == 3)]
				id_4 = target_row[(target_row["DBSCAN_ID"] == 4)]
				id_5 = target_row[(target_row["DBSCAN_ID"] == 5)]
				id_6 = target_row[(target_row["DBSCAN_ID"] == 6)]
				id_7 = target_row[(target_row["DBSCAN_ID"] == 7)]
				id_8 = target_row[(target_row["DBSCAN_ID"] == 8)]
				id_9 = target_row[(target_row["DBSCAN_ID"] == 9)]

				if id_0.shape[0] > 0:
					ax1.scatter(id_0["point_x_org"].values, id_0["point_y_org"].values, marker="o")
					ax1.scatter(id_0["gravity_x_org"].values, id_0["gravity_y_org"].values, marker="x")

					# extract tracking data
					c_gravityID_0.append(id_0.values[0])
				
				if id_1.shape[0] > 0:
					ax1.scatter(id_1["point_x_org"].values, id_1["point_y_org"].values, marker="o")
					ax1.scatter(id_1["gravity_x_org"].values, id_1["gravity_y_org"].values,  marker="x")

					# extract tracking data
					c_gravityID_1.append(id_1.values[0])
				
				if id_2.shape[0] > 0:
					ax1.scatter(id_2["point_x_org"].values, id_2["point_y_org"].values, marker="o")
					ax1.scatter(id_2["gravity_x_org"].values, id_2["gravity_y_org"].values, marker="x")

					# extract tracking data
					c_gravityID_2.append(id_2.values[0])
				
				if id_3.shape[0] > 0:
					ax1.scatter(id_3["point_x_org"].values, id_3["point_y_org"].values, marker="o")
					ax1.scatter(id_3["gravity_x_org"].values, id_3["gravity_y_org"].values, marker="x")

					# extract tracking data
					c_gravityID_3.append(id_3.values[0])
				
				if id_4.shape[0] > 0:
					ax1.scatter(id_4["point_x_org"].values, id_4["point_y_org"].values, marker="o")
					ax1.scatter(id_4["gravity_x_org"].values, id_4["gravity_y_org"].values, marker="x")

					# extract tracking data
					c_gravityID_4.append(id_4.values[0])
				
				if id_5.shape[0] > 0:
					ax1.scatter(id_5["point_x_org"].values, id_5["point_y_org"].values, marker="o")
					ax1.scatter(id_5["gravity_x_org"].values, id_5["gravity_y_org"].values, marker="x")

					# extract tracking data
					c_gravityID_5.append(id_5.values[0])
				
				if id_6.shape[0] > 0:
					ax1.scatter(id_6["point_x_org"].values, id_6["point_y_org"].values, marker="o")
					ax1.scatter(id_6["gravity_x_org"].values, id_6["gravity_y_org"].values, marker="x")

					# extract tracking data
					c_gravityID_6.append(id_6.values[0])
				
				if id_7.shape[0] > 0:
					ax1.scatter(id_7["point_x_org"].values, id_7["point_y_org"].values, marker="o")
					ax1.scatter(id_7["gravity_x_org"].values, id_7["gravity_y_org"].values, marker="x")

					# extract tracking data
					c_gravityID_7.append(id_7.values[0])
				
				if id_8.shape[0] > 0:
					ax1.scatter(id_8["point_x_org"].values, id_8["point_y_org"].values, marker="o")
					ax1.scatter(id_8["gravity_x_org"].values, id_8["gravity_y_org"].values, marker="x")

					# extract tracking data
					c_gravityID_8.append(id_8.values[0])
				
				if id_9.shape[0] > 0:
					ax1.scatter(id_9["point_x_org"].values, id_9["point_y_org"].values, marker="o")
					ax1.scatter(id_9["gravity_x_org"].values, id_9["gravity_y_org"].values, marker="x")

					# extract tracking data
					c_gravityID_9.append(id_9.values[0])

				ax1.set_xlim(-6, 6)
				ax1.set_ylim(0, 7)

				# axis label
				ax1.set_xlabel('x')
				ax1.set_ylabel('y')

				n_color = 9

				ax2 = fig.add_subplot(2, 1, 2)
				ax2.set_title('the center of gravity of the 2D point cloud')

				cmap = plt.cm.coolwarm
				rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, N)))

				# display point cloud  each frames.
				#target_row = df_feature[(df_feature["frame_no"] == frame_number[0])]
				target_row = df_feature[(df_feature["frame_no"] == frameNumberTotal)]

				id_error = target_row[(target_row["point_y_org"] > 5.0)]

				if id_error.shape[0] == 0:
					id_0 = target_row[(target_row["DBSCAN_ID"] == 0)]
					id_1 = target_row[(target_row["DBSCAN_ID"] == 1)]
					id_2 = target_row[(target_row["DBSCAN_ID"] == 2)]
					id_3 = target_row[(target_row["DBSCAN_ID"] == 3)]
					id_4 = target_row[(target_row["DBSCAN_ID"] == 4)]
					id_5 = target_row[(target_row["DBSCAN_ID"] == 5)]
					id_6 = target_row[(target_row["DBSCAN_ID"] == 6)]
					id_7 = target_row[(target_row["DBSCAN_ID"] == 7)]
					id_8 = target_row[(target_row["DBSCAN_ID"] == 8)]
					id_9 = target_row[(target_row["DBSCAN_ID"] == 9)]

					if id_0.shape[0] > 0:
						ax2.scatter(id_0["gravity_base_x"].values, id_0["gravity_base_y"].values, marker="o")
						ax2.scatter(0, 0, marker="x")
					
					if id_1.shape[0] > 0:
						ax2.scatter(id_1["gravity_base_x"].values, id_1["gravity_base_y"].values, marker="o")
						ax2.scatter(0, 0, marker="x")
					
					if id_2.shape[0] > 0:
						ax2.scatter(id_2["gravity_base_x"].values, id_2["gravity_base_y"].values, marker="o")
						ax2.scatter(0, 0, marker="x")
					
					if id_3.shape[0] > 0:
						ax2.scatter(id_3["gravity_base_x"].values, id_3["gravity_base_y"].values, marker="o")
						ax2.scatter(0, 0, marker="x")
					
					if id_4.shape[0] > 0:
						ax2.scatter(id_4["gravity_base_x"].values, id_4["gravity_base_y"].values, marker="o")
						ax2.scatter(0, 0, marker="x")
					
					if id_5.shape[0] > 0:
						ax2.scatter(id_5["gravity_base_x"].values, id_5["gravity_base_y"].values, marker="o")
						ax2.scatter(0, 0, marker="x")
					
					if id_6.shape[0] > 0:
						ax2.scatter(id_6["gravity_base_x"].values, id_6["gravity_base_y"].values, marker="o")
						ax2.scatter(0, 0, marker="x")
					
					if id_7.shape[0] > 0:
						ax2.scatter(id_7["gravity_base_x"].values, id_7["gravity_base_y"].values, marker="o")
						ax2.scatter(0, 0, marker="x")
					
					if id_8.shape[0] > 0:
						ax2.scatter(id_8["gravity_base_x"].values, id_8["gravity_base_y"].values, marker="o")
						ax2.scatter(0, 0, marker="x")
					
					if id_9.shape[0] > 0:
						ax2.scatter(id_9["gravity_base_x"].values, id_9["gravity_base_y"].values, marker="o")
						ax2.scatter(0, 0, marker="x")

				ax2.set_xlim(-3, 3)
				ax2.set_ylim(-3, 3)

				# axis label
				ax2.set_xlabel('Gx')
				ax2.set_ylabel('Gy')

				# Display
				#plt.show()
				plt.show(block=False)

				file_name = 'trace_ center_of_gravity_f{0}.png'.format(frameNumberTotal)
				dir_path = os.path.join(PROJECT_ROOT_DIR, 'Images', 'TracingGravity')

				if (os.path.isdir(dir_path)  == False):
					os.mkdir(dir_path)

				plt.savefig(os.path.join(PROJECT_ROOT_DIR, 'Images', 'TracingGravity', file_name))

				plt.pause(pause_time)

				plt.close()
			
			'''
			---------------------------------------------------------------------------------------------------------------------------
			Transrate x-y coordinate to image(28bit x 28bit).
			---------------------------------------------------------------------------------------------------------------------------
			'''

			pathName_pattern = os.path.join(PROJECT_ROOT_DIR, 'Data', 'pattern_data_f{0}.csv'.format(tmpInt))
			pathName_gravity = os.path.join(PROJECT_ROOT_DIR, 'Data', 'gravity_data_f{0}.csv'.format(tmpInt))

			# Create pateran data  each frames.
			target_row = df_feature[(df_feature["frame_no"] == frameNumberTotal)]

			f_number = frame_number[0].astype('int32')

			# Check the spread of the cluster.
			gravity_base_x_max_array = np.array(gravity_base_x_max)
			gravity_base_x_min_array = np.array(gravity_base_x_min)
			if abs(gravity_base_x_max_array.max()) > abs(gravity_base_x_min_array.min()):
				x_range = abs(gravity_base_x_max_array.max())
			else:
				x_range = abs(gravity_base_x_min_array.min())
			
			gravity_base_y_max_array = np.array(gravity_base_y_max)
			gravity_base_y_min_array = np.array(gravity_base_y_min)
			if abs(gravity_base_y_max_array.max()) > abs(gravity_base_y_min_array.min()):
				y_range = abs(gravity_base_y_max_array.max())
			else:
				y_range = abs(gravity_base_y_min_array.min())
			
			if x_range > y_range:
				all_range = x_range
			else:
				all_range = y_range
			
			if all_range is None:
				print('all_range = {0}'.foramt(all_range))
			
			"""
			---------------------------------------------------------------------
			Show the Point Cloud converted to image data.
			---------------------------------------------------------------------
			"""
			plt.figure(figsize=(14.0, 10.0))

			pattern_id_0 = target_row[(target_row["DBSCAN_ID"] == 0)]
			pattern_id_1 = target_row[(target_row["DBSCAN_ID"] == 1)]
			pattern_id_2 = target_row[(target_row["DBSCAN_ID"] == 2)]
			pattern_id_3 = target_row[(target_row["DBSCAN_ID"] == 3)]
			pattern_id_4 = target_row[(target_row["DBSCAN_ID"] == 4)]
			pattern_id_5 = target_row[(target_row["DBSCAN_ID"] == 5)]
			pattern_id_6 = target_row[(target_row["DBSCAN_ID"] == 6)]
			pattern_id_7 = target_row[(target_row["DBSCAN_ID"] == 7)]
			pattern_id_8 = target_row[(target_row["DBSCAN_ID"] == 8)]
			pattern_id_9 = target_row[(target_row["DBSCAN_ID"] == 9)]

			try:
				axID_0 = plt.subplot2grid((3, 4), (0, 0))
				axID_0.set_title('ID = 0: image(28bit x 28bit)')

				if not pattern_id_0.empty:
					img_data_0, gvy_data_0 = transrateImagePatern(image_pattern_data, gravity_data, pattern_id_0, all_range, 28, 28, axID_0)
					image_pattern_data.append(img_data_0)
					gravity_data.append(gvy_data_0)

					if patternSaveFlag == True:
						with open(pathName_pattern, 'a', newline='') as fi:
							writer = csv.writer(fi)
							writer.writerow(img_data_0)
							time.sleep(2)
						fi.close()

						with open(pathName_gravity, 'a', newline='') as fg:
							writer = csv.writer(fg)
							writer.writerow(gvy_data_0)
							time.sleep(2)
						fg.close()

				axID_1 = plt.subplot2grid((3, 4), (0, 1))
				axID_1.set_title('ID = 1: image(28bit x 28bit)')

				if not pattern_id_1.empty:
					img_data_1, gvy_data_1 = transrateImagePatern(image_pattern_data, gravity_data, pattern_id_1, all_range, 28, 28, axID_1)
					image_pattern_data.append(img_data_1)
					gravity_data.append(gvy_data_1)

					if patternSaveFlag == True:
						with open(pathName_pattern, 'a', newline='') as fi:
							writer = csv.writer(fi)
							writer.writerow(img_data_1)
							time.sleep(2)
						fi.close()

						with open(pathName_gravity, 'a', newline='') as fg:
							writer = csv.writer(fg)
							writer.writerow(gvy_data_1)
							time.sleep(2)
						fg.close()

				axID_2= plt.subplot2grid((3, 4), (0, 2))
				axID_2.set_title('ID = 2: image(28bit x 28bit)')

				if not pattern_id_2.empty:
					img_data_2, gvy_data_2 = transrateImagePatern(image_pattern_data, gravity_data, pattern_id_2, all_range, 28, 28, axID_2)
					image_pattern_data.append(img_data_2)
					gravity_data.append(gvy_data_2)

					if patternSaveFlag == True:
						with open(pathName_pattern, 'a', newline='') as fi:
							writer = csv.writer(fi)
							writer.writerow(img_data_2)
							time.sleep(2)
						fi.close()

						with open(pathName_gravity, 'a', newline='') as fg:
							writer = csv.writer(fg)
							writer.writerow(gvy_data_2)
							time.sleep(2)
						fg.close()

				axID_3= plt.subplot2grid((3, 4), (0, 3))
				axID_3.set_title('ID = 3: image(28bit x 28bit)')

				if not pattern_id_3.empty:
					img_data_3, gvy_data_3 = transrateImagePatern(image_pattern_data, gravity_data, pattern_id_3, all_range, 28, 28, axID_3)
					image_pattern_data.append(img_data_3)
					gravity_data.append(gvy_data_3)

					if patternSaveFlag == True:
						with open(pathName_pattern, 'a', newline='') as fi:
							writer = csv.writer(fi)
							writer.writerow(img_data_3)
							time.sleep(2)
						fi.close()

						with open(pathName_gravity, 'a', newline='') as fg:
							writer = csv.writer(fg)
							writer.writerow(gvy_data_3)
							time.sleep(2)
						fg.close()

				axID_4 = plt.subplot2grid((3, 4), (1, 0))
				axID_4.set_title('ID = 4: image(28bit x 28bit)')

				if not pattern_id_4.empty:
					img_data_4, gvy_data_4 = transrateImagePatern(image_pattern_data, gravity_data, pattern_id_4, all_range, 28, 28, axID_4)
					image_pattern_data.append(img_data_4)
					gravity_data.append(gvy_dat_4)

					if patternSaveFlag == True:
						with open(pathName_pattern, 'a', newline='') as fi:
							writer = csv.writer(fi)
							writer.writerow(img_data_4)
							time.sleep(2)
						fi.close()

						with open(pathName_gravity, 'a', newline='') as fg:
							writer = csv.writer(fg)
							writer.writerow(gvy_data_4)
							time.sleep(2)
						fg.close()

				axID_5 = plt.subplot2grid((3, 4), (1, 1))
				axID_5.set_title('ID = 5: image(28bit x 28bit)')

				if not pattern_id_5.empty:
					img_data_5, gvy_data_5 = transrateImagePatern(image_pattern_data, gravity_data, pattern_id_5, all_range, 28, 28, axID_5)
					image_pattern_data.append(img_data_5)
					gravity_data.append(gvy_data_5)

					if patternSaveFlag == True:
						with open(pathName_pattern, 'a', newline='') as fi:
							writer = csv.writer(fi)
							writer.writerow(img_data_5)
							time.sleep(2)
						fi.close()

						with open(pathName_gravity, 'a', newline='') as fg:
							writer = csv.writer(fg)
							writer.writerow(gvy_data_5)
							time.sleep(2)
						fg.close()

				axID_6 = plt.subplot2grid((3, 4), (1, 2))
				axID_6.set_title('ID = 6: image(28bit x 28bit)')

				if not pattern_id_6.empty:
					img_data_6, gvy_data_6 = transrateImagePatern(image_pattern_data, gravity_data, pattern_id_6, all_range, 28, 28, axID_6)
					image_pattern_data.append(img_data_6)
					gravity_data.append(gvy_data_6)

					if patternSaveFlag == True:
						with open(pathName_pattern, 'a', newline='') as fi:
							writer = csv.writer(fi)
							writer.writerow(img_data_6)
							time.sleep(2)
						fi.close()

						with open(pathName_gravity, 'a', newline='') as fg:
							writer = csv.writer(fg)
							writer.writerow(gvy_data_6)
							time.sleep(2)
						fg.close()

				axID_7 = plt.subplot2grid((3, 4), (1, 3))
				axID_7.set_title('ID = 7: image(28bit x 28bit)')

				if not pattern_id_7.empty:
					img_data_7, gvy_data_7 = transrateImagePatern(image_pattern_data, gravity_data, pattern_id_7, all_range, 28, 28, axID_7)
					image_pattern_data.append(img_data_7)
					gravity_data.append(gvy_data_7)

					if patternSaveFlag == True:
						with open(pathName_pattern, 'a', newline='') as fi:
							writer = csv.writer(fi)
							writer.writerow(img_data_7)
							time.sleep(2)
						fi.close()

						with open(pathName_gravity, 'a', newline='') as fg:
							writer = csv.writer(fg)
							writer.writerow(gvy_data_7)
							time.sleep(2)
						fg.close()

				axID_8 = plt.subplot2grid((3, 4), (2, 0))
				axID_8.set_title('ID = 8: image(28bit x 28bit)')

				if not pattern_id_8.empty:
					img_data_8, gvy_data_8 = transrateImagePatern(image_pattern_data, gravity_data, pattern_id_8, all_range, 28, 28, axID_8)
					image_pattern_data.append(img_data_8)
					gravity_data.append(gvy_data_8)

					if patternSaveFlag == True:
						with open(pathName_pattern, 'a', newline='') as fi:
							writer = csv.writer(fi)
							writer.writerow(img_data_8)
						fi.close()

						with open(pathName_gravity, 'a', newline='') as fg:
							writer = csv.writer(fg)
							writer.writerow(gvy_data_8)
							time.sleep(2)
						fg.close()

				axID_9 = plt.subplot2grid((3, 4), (2, 1))
				axID_9.set_title('ID = 9: image(28bit x 28bit)')

				if not pattern_id_9.empty:
					img_data_9, gvy_data_9 = transrateImagePatern(image_pattern_data, gravity_data, pattern_id_9, all_range, 28, 28, axID_9)
					image_pattern_data.append(img_data_9)
					gravity_data.append(gvy_data_9)

					if patternSaveFlag == True:
						with open(pathName_pattern, 'a', newline='') as fi:
							writer = csv.writer(fi)
							writer.writerow(img_data_9)
							time.sleep(2)
						fi.close()

						with open(pathName_gravity, 'a', newline='') as fg:
							writer = csv.writer(fg)
							writer.writerow(gvy_data_9)
							time.sleep(2)
						fg.close()

			except Exception as ex:
				print(ex)
				pass
			finally:
				pass
			#plt.show()
			plt.show(block=False)

			file_name = 'image_data_f{0}.png'.format(frameNumberTotal)
			dir_path = os.path.join(PROJECT_ROOT_DIR, 'Images', 'PatternImages')

			if (os.path.isdir(dir_path)  == False):
				os.mkdir(dir_path)

			plt.savefig(os.path.join(PROJECT_ROOT_DIR, 'Images', 'PatternImages', file_name))

			plt.pause(pause_time)

			plt.close()

			"""
			----------------------------------------------------------------------
			Check if a classification model exists.
			----------------------------------------------------------------------
			"""
			total_classification_enable = False
			total_classification_enable |= FeatureID_0.frame_enable
			total_classification_enable |= FeatureID_1.frame_enable
			total_classification_enable |= FeatureID_2.frame_enable
			total_classification_enable |= FeatureID_3.frame_enable
			total_classification_enable |= FeatureID_4.frame_enable
			total_classification_enable |= FeatureID_5.frame_enable
			total_classification_enable |= FeatureID_6.frame_enable
			total_classification_enable |= FeatureID_7.frame_enable
			total_classification_enable |= FeatureID_8.frame_enable
			total_classification_enable |= FeatureID_9.frame_enable

			parameter_transfer = [
										[FeatureID_0, img_data_0, gvy_data_0],
										[FeatureID_1, img_data_1, gvy_data_1],
										[FeatureID_2, img_data_2, gvy_data_2],
										[FeatureID_3, img_data_3, gvy_data_3],
										[FeatureID_4, img_data_4, gvy_data_4],
										[FeatureID_5, img_data_5, gvy_data_5],
										[FeatureID_6, img_data_6, gvy_data_6],
										[FeatureID_7, img_data_7, gvy_data_7],
										[FeatureID_8, img_data_8, gvy_data_8],
										[FeatureID_9, img_data_9, gvy_data_9]
									]

			"""
			-----------------------------------------------------------------
			A processing method when the classification process is not performed even once.
			-----------------------------------------------------------------
			"""
			
			dbscanList = [
							dbscanID_0, dbscanID_1, dbscanID_2, dbscanID_3, dbscanID_4,
							dbscanID_5, dbscanID_6, dbscanID_7, dbscanID_8, dbscanID_9
						]
			
			# search a minimum empty id
			maximumEmptyId = 0

			# Check ID_status_current
			for tmpIndex, dbscanId in enumerate(dbscanList):
				if len(dbscanId) > 0:
					ID_status_current[tmpIndex] = True

					if tmpIndex > maximumEmptyId:
						maximumEmptyId = tmpIndex
				else:
					ID_status_current[tmpIndex] = False
						
			"""
			----------------------------------------------------------------------
			prepare classification by using past data
			----------------------------------------------------------------------
			"""
			try:
				# create total_classification_data.
				total_classification_data = []
				total_classification_target = []

				# confirm whether the frame_enable in the instance of FeatureID_Store is true or not.
				for featureId in featureList:
					if featureId.frame_enable:
						tmpList = featureId.frame_image_data.values.tolist()

						# create traing data.
						total_classification_data.extend(tmpList)

						tmpListTarget = featureId.frame_target_data.values.tolist()

						total_classification_target.extend(tmpListTarget)

				total_images = np.array(total_classification_data)
				total_images_reshape = total_images.reshape((total_images.shape[0], 28, 28))

				total_targets = np.array(total_classification_target)

				targetID_0_index = np.where(total_targets[:, 0] == 0)
				targetID_1_index = np.where(total_targets[:, 0] == 1)
				targetID_2_index = np.where(total_targets[:, 0] == 2)
				targetID_3_index = np.where(total_targets[:, 0] == 3)
				targetID_4_index = np.where(total_targets[:, 0] == 4)
				targetID_5_index = np.where(total_targets[:, 0] == 5)
				targetID_6_index = np.where(total_targets[:, 0] == 6)
				targetID_7_index = np.where(total_targets[:, 0] == 7)
				targetID_8_index = np.where(total_targets[:, 0] == 8)
				targetID_9_index = np.where(total_targets[:, 0] == 9)

				trainID_0 = total_images_reshape[targetID_0_index]
				trainID_1 = total_images_reshape[targetID_1_index]
				trainID_2 = total_images_reshape[targetID_2_index]
				trainID_3 = total_images_reshape[targetID_3_index]
				trainID_4 = total_images_reshape[targetID_4_index]
				trainID_5 = total_images_reshape[targetID_5_index]
				trainID_6 = total_images_reshape[targetID_6_index]
				trainID_7 = total_images_reshape[targetID_7_index]
				trainID_8 = total_images_reshape[targetID_8_index]
				trainID_9 = total_images_reshape[targetID_9_index]

				TrainImagePlot(0, frameNumberTotal, trainID_0)
				TrainImagePlot(1, frameNumberTotal, trainID_1)
				TrainImagePlot(2, frameNumberTotal, trainID_2)
				TrainImagePlot(3, frameNumberTotal, trainID_3)
				TrainImagePlot(4, frameNumberTotal, trainID_4)
				TrainImagePlot(5, frameNumberTotal, trainID_5)
				TrainImagePlot(6, frameNumberTotal, trainID_6)
				TrainImagePlot(7, frameNumberTotal, trainID_7)
				TrainImagePlot(8, frameNumberTotal, trainID_8)
				TrainImagePlot(9, frameNumberTotal, trainID_9)

				"""
				------------------------------------------------------------------------
				Distribute total_classification_data and total_classification_target to train and test.
				------------------------------------------------------------------------
				"""
				X_train, X_test, Y_train, Y_test = train_test_split(total_classification_data, total_classification_target, random_state=0)

				print(
						'------------------------------------------------------------------------\n'
						'                 Random Forest Classification                           \n'
						'------------------------------------------------------------------------\n'
					)
				
				train_images = np.array(X_train)
				test_images = np.array(X_test)
				train_labels = np.array(Y_train)
				test_labels = np.array(Y_test)

				max_ft = total_targets.max().astype(int)

				search_params['max_features'] = [max_ft.tolist()]

				"""
				------------------------------------------------------------------------------------------------------------------------
				Declare parameters of random forest classification.
				------------------------------------------------------------------------------------------------------------------------
				"""
				gs.fit(train_images, train_labels)

				"""
				------------------------------------------------------------------------------------------------------------------------
				Result
				------------------------------------------------------------------------------------------------------------------------
				"""
				# Correct rate of learning data
				print('Train best score: {0}'.format(gs.best_score_))

				# Correct answer rate of test data
				print('Train best parameter: {0}\n'.format(gs.best_params_))

				clf = gs.best_estimator_

				clf.fit(train_images, train_labels)

				print('Train Accuracy: {0}'.format(clf.score(train_images, train_labels)))
				print('Test Accuracy: {0}\n'.format(clf.score(test_images, test_labels)))

				'''
				------------------------------------------------------------------------------------------------------------------------
				Visualize result
				------------------------------------------------------------------------------------------------------------------------
				'''
				predicted_labels = clf.predict(test_images)

				plt.close()

				plt.figure(figsize=(14.0, 12.0))

				for i in range(25):
					max_rows = len(test_images / 5) + 1
					tmp_row = int(i / 5)
					tmp_column = int(i % 5)
					plt.subplot2grid((max_rows, 5), (tmp_row, tmp_column))
					if predicted_labels[i] == test_labels[i]:
						color = 'green' # True label color
					else:
						color = 'red'   # False label color
					plt.xlabel("{} True({})".format(predicted_labels[i], test_labels[i]), color=color, fontsize=8)
					plt.xticks([])
					plt.yticks([])
					plt.imshow(test_images[i].reshape((28, 28)), cmap=cm.binary)
				

				#plt.show()
				plt.show(block=False)

				file_name = 'randum_test_f{0}.png'.format(frameNumberTotal)
				dir_path = os.path.join(PROJECT_ROOT_DIR, 'Images', 'TestImages')

				if (os.path.isdir(dir_path)  == False):
					os.mkdir(dir_path)

				plt.savefig(os.path.join(PROJECT_ROOT_DIR, 'Images', 'TestImages', file_name))

				plt.pause(pause_time)

				plt.close()
				
				"""
				------------------------------------------------------------------------------------
				Using the trained model from past image data, 
				we check which Clustering the current image data belongs to.
				------------------------------------------------------------------------------------
				"""
				for featureId, image_data, gravity_data in parameter_transfer:
					Transfer_ID(
									featureId, image_data, gravity_data, predictProbaRatio, clf,
									FeatureID_0, FeatureID_1, FeatureID_2, FeatureID_3, FeatureID_4,
									FeatureID_5, FeatureID_6, FeatureID_7, FeatureID_8, FeatureID_9
								)

				
			except Exception as ex:
				print(ex)
				pass
			finally:
				pass

			indexID_0.clear()
			indexID_1.clear()
			indexID_2.clear()
			indexID_3.clear()
			indexID_4.clear()
			indexID_5.clear()
			indexID_6.clear()
			indexID_7.clear()
			indexID_8.clear()
			indexID_9.clear()

			dbscanID_0.clear()
			dbscanID_1.clear()
			dbscanID_2.clear()
			dbscanID_3.clear()
			dbscanID_4.clear()
			dbscanID_5.clear()
			dbscanID_6.clear()
			dbscanID_7.clear()
			dbscanID_8.clear()
			dbscanID_9.clear()

			dummyUseFlag = False


# calculate raw data each frame
print()

