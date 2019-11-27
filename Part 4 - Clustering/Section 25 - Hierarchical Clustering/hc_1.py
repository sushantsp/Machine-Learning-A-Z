# Hierarchial Clustering

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset with pandas
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

# using the dendogram to find the optimal 
# numver of clusters.

import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()

# fitting HC to the mall dataset

from sklearn.cluster import AgglomerativeClustering  

hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean',linkage = 'ward')
y_hc = hc.fit_predict(X)

# visualizing the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s =100, c ='red', label = 'Careful')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s =100, c ='blue', label = 'Standard')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s =100, c ='green', label = 'Target')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s =100, c ='cyan', label = 'CAreless')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s =100, c ='magenta', label = 'Sensible')

plt.title('clusters of clients')
plt.xlabel('Annumal Income')
plt.ylabel('SPending Score')
plt.legend()
plt.show()