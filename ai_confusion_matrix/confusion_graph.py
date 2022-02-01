"""
------------------------------------------
confusion_graph.py

------------------------------------------
"""

#
import os
import gc
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import traceback


def seaborn_heatmap(image_path, save_name, real_data, predict_data, mode):
	"""
	Graphing the confusion matrix
	
	:param real_data: df_base_ct['睡眠ステージ']
	:param predict_data: df_predict['睡眠ステージ']
	:return:
	"""
	
	try:
		y_real = real_data.to_list()
		y_predict = predict_data.to_list()
		
		# AI 1.2 evaluates from the 5th to the last 5th.
		if not mode:
			y_real = y_real[5:-5]
			y_predict = y_predict[5:-5]
		
		for i, tmp in enumerate(y_real):
			if tmp == "NotScored":
				del y_real[i]
				del y_predict[i]
		
		figure_labels = [
			"Wake",
			"REM",
			"NonREM1",
			"NonREM2",
			"NonREM3"
		]
		
		data = confusion_matrix(y_real, y_predict, labels=figure_labels)
		# df_cm = pd.DataFrame(data, columns=np.unique(y_real), index=np.unique(y_real))
		df_cm = pd.DataFrame(data, columns=figure_labels, index=figure_labels)
		df_cm.index.name = 'Actual'
		df_cm.columns.name = 'Predicted'
		
		plt.figure(figsize=(14, 7))
		plt.title(f"confusion matrix:{save_name}")
		sn.set(font_scale=1.8)
		sn.heatmap(df_cm/np.sum(df_cm), cmap="Blues", annot=True, fmt='.2%', annot_kws={"size": 16})
		
		store_name = os.path.join(image_path, f"{save_name}.png")
		plt.savefig(store_name)
		
		plt.show(block=False)
		
		plt.pause(5)
		
		plt.close()
		
		gc.collect()
		
		return y_real, y_predict
		
	except:
		print(f"Error information\n{traceback.format_exc()}")
		pass