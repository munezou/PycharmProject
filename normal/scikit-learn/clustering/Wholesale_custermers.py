import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

pd.options.display.max_columns = None

'''
------------------------------------------------------------------------------------------------------------------------
Read data set of Wholesale customers Data Set.
------------------------------------------------------------------------------------------------------------------------
'''
'''
from six.moves import urllib

proxy_support = urllib.request.ProxyHandler({'https': 'http://proxy.kanto.sony.co.jp:10080'})
opener = urllib.request.build_opener(proxy_support)
urllib.request.install_opener(opener)
'''

cust_df = pd.read_csv("https://pythondatascience.plavox.info/wp-content/uploads/2016/05/Wholesale_customers_data.csv")

'''
------------------------------------------------------------------------------------------------------------------------
remove no need columns.
------------------------------------------------------------------------------------------------------------------------
'''
del(cust_df['Channel'])
del(cust_df['Region'])

print('cust_df.head() = \n{0}'.format(cust_df.head()))
print('cust_df.tail() = \n{0}'.format(cust_df.tail()))
print()

# Convert from Pandas data frame to Numpy matrix.
cust_array = np.array([cust_df['Fresh'].tolist(),
                       cust_df['Milk'].tolist(),
                       cust_df['Grocery'].tolist(),
                       cust_df['Frozen'].tolist(),
                       cust_df['Milk'].tolist(),
                       cust_df['Detergents_Paper'].tolist(),
                       cust_df['Delicassen'].tolist()
                       ], np.int32)

cust_array = cust_array.T

# Perform cluster analysis (number of clusters = 4).
pred = KMeans(n_clusters=4).fit_predict(cust_array)
print('predector = \n{0}'.format(pred))
print()

'''
------------------------------------------------------------------------------------------------------------------------
Check the characteristics of each cluster.
------------------------------------------------------------------------------------------------------------------------
'''
# Add cluster number to Pandas data frame.
cust_df['cluster_id']=pred

print('cust_df.head() = \n{0}'.format(cust_df.head()))
print('cust_df.tail() = \n{0}'.format(cust_df.tail()))
print()

# Distribution of the number of samples belonging to each cluster
print('cust_df["cluster_id"].value_counts() = \n{0}'.format(cust_df['cluster_id'].value_counts()))
print()

print('---< Average purchase value of each department product in each cluster >---')
print('average of Cluster ID = 0 \n{0}'.format(cust_df[cust_df['cluster_id']==0].mean()))
print()
print('average of Cluster ID = 1 \n{0}'.format(cust_df[cust_df['cluster_id']==1].mean()))
print()
print('average of Cluster ID = 2 \n{0}'.format(cust_df[cust_df['cluster_id']==2].mean()))
print()
print('average of Cluster ID = 3 \n{0}'.format(cust_df[cust_df['cluster_id']==3].mean()))
print()

'''
------------------------------------------------------------------------------------------------------------------------
Visualize cluster trends with Matplotlib.
------------------------------------------------------------------------------------------------------------------------
'''
import matplotlib.pyplot as plt

clusterinfo = pd.DataFrame()
for i in range(4):
    clusterinfo['cluster' + str(i)] = cust_df[cust_df['cluster_id'] == i].mean()
clusterinfo = clusterinfo.drop('cluster_id')

my_plot = clusterinfo.T.plot(kind='bar', stacked=True, title="Mean Value of 4 Clusters")
my_plot.set_xticklabels(my_plot.xaxis.get_majorticklabels(), rotation=0)

plt.show()
