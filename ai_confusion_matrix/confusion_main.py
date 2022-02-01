"""
confusion_main.py

A program that verifies the stage judgment of clinical engineers and the stage judgment of AI
"""

# necessary library
import os
import glob

import numpy as np
import pandas as pd

from confusion_function import *

from confusion_class import *

from confusion_graph import *

# display title
print(__doc__)


def main():
	"""
	main routine
	
	:return:
	"""
	
	project_root_dir = os.getcwd()
	print(f"project_root_dir: \n{project_root_dir}")
	
	# create necessary folder.
	create_folder(project_root_dir)

	# Check if there is a file to calculate the confusion matrix.
	rml_filter = os.path.join(project_root_dir, 'Data', '*.rml')
	csv_filter = os.path.join(project_root_dir, 'Data', '*.csv')
	
	# create file list
	rml_list = glob.glob(rml_filter)
	csv_list = glob.glob(csv_filter)
	
	# extend file name
	if not len(rml_list):
		# Output message.
		print(f"rml files don't exist!\n")
		# end
		exit()
	else:
		# Delete duplicate files
		rml_hash = delete_duplicate_edf(rml_list)
		
		# rml file with path
		rml_path_list = list(rml_hash.values())
		
		# rml file without path
		rml_list = []
		for rml_file in rml_path_list:
			rml_file = rml_file.split('/')
			rml_file = rml_file[len(rml_file) - 1]
			rml_list.append(rml_file)
		
	if not len(csv_list):
		# output message.
		print(f"csv files don't exist!\n")
		# end
		exit()
	else:
		csv_hash = delete_duplicate_edf(csv_list)
		
		csv_path_list = list(csv_hash.values())
		
		csv_list = []
		for csv_file in csv_path_list:
			csv_file = csv_file.split('/')
			csv_file = csv_file[len(csv_file) - 1]
			csv_list.append(csv_file)
	
	# synthesis confusion matrix
	total_y_real = np.array([])
	total_y_predict =np.array([])

	for tmp_file in rml_list:
		# target file
		base_rml, ext_rml = os.path.splitext(tmp_file)
		print(f"target file name: {base_rml}")
		
		# Check if there is data that the clinical engineer has stage-classified.
		target_name = f"{base_rml}-ct.csv"
		if not (target_name in csv_list):
			print(f"The required csv file does not exist.")
		
		for tmp_name in csv_path_list:
			base_cav = tmp_name.split('/')
			base_cav_name = base_cav[len(base_cav) - 1]
			if target_name == base_cav_name:
				# process base-ct.csv
				df_base_ct = pd.read_csv(tmp_name)
				
				# convert df_base_ct to rml file format.
				df_base_ct['睡眠ステージ'] = df_base_ct['睡眠ステージ'].str.replace('NS', 'NotScored')
				
				df_base_ct['睡眠ステージ'] = df_base_ct['睡眠ステージ'].str.replace('WK', 'Wake')
				
				df_base_ct['睡眠ステージ'] = df_base_ct['睡眠ステージ'].str.replace('N1', 'NonREM1')
				
				df_base_ct['睡眠ステージ'] = df_base_ct['睡眠ステージ'].str.replace('N2', 'NonREM2')
				
				df_base_ct['睡眠ステージ'] = df_base_ct['睡眠ステージ'].str.replace('N3', 'NonREM3')
				
				# create an instance of scoring
				file_name_rml = tmp_name.replace('-ct.csv', '.rml')
				
				# Parse the rml file.
				results = ScoringResult.from_rml(file_name_rml)
				results = list(map(to_list, results))
				df_predict = pd.DataFrame(results, columns=['時刻', '睡眠ステージ'])
				
				image_folder_path = os.path.join(project_root_dir, 'Image')
				
				mode = 0
				
				base_rml = os.path.splitext(os.path.basename(file_name_rml))[0]
				
				tmp_y_real, tmp_y_predict = \
					seaborn_heatmap(image_folder_path, base_rml, df_base_ct['睡眠ステージ'], df_predict['睡眠ステージ'], mode)
				
				total_y_real = np.append(total_y_real, tmp_y_real)
				total_y_predict = np.append(total_y_predict, tmp_y_predict)
				
	mode = 1
	
	seaborn_heatmap(image_folder_path, "synthesis confusion matrix", pd.Series(total_y_real), pd.Series(total_y_predict), mode)


# ガター内の緑色のボタンを押すとスクリプトを実行します。
if __name__ == '__main__':
	"""
	main routine
	"""
	
	main()
