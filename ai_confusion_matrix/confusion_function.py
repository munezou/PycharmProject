"""
---------------------------------------------------
confusion_function.py

Define the functions used in confusion_main.
---------------------------------------------------
"""

# necessary library
import os
import hashlib


def create_folder(path):
	"""
	create necessary folder
	
	:param path: root directory
	:return:
	"""
	
	# create data folder
	data_dir_path = os.path.join(path, 'Data')
	if not os.path.isdir(data_dir_path):
		os.mkdir(data_dir_path)
	
	# create image folder
	image_dir_path = os.path.join(path, 'Image')
	if not os.path.isdir(image_dir_path):
		os.mkdir(image_dir_path)


def delete_duplicate_edf(file_list):
	"""
	Delete duplicate files
	:param file_list:
	:return:
	"""
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
