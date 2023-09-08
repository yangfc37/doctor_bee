import struct

import joblib
import pymysql
from itertools import chain
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from matplotlib import ticker
import time
#import datetime
from function import *
from datetime import *
from datetime import date
import pickle

from sklearn.decomposition import PCA

from matplotlib import rcParams

config = {
    "font.family":'Times New Roman',  # 设置字体类型
    "axes.unicode_minus": False #解决负号无法显示的问题
}
rcParams.update(config)

n_clusters = 22
#info = np.load("./kmeans_yf_single_nor_newest/info_original_data.npy", allow_pickle = True)
info = np.load("./kmeans_yf_single_nor_newest/info_table.npy", allow_pickle = True)
points = np.array([x[3] for x in info])
#points = scaler.transform(points)
kmeans = joblib.load('./kmeans_yf_single_nor_newest/''kmeans_yf' + '_' + str(n_clusters) + '.pkl')
scaler = pickle.load(open('scaler_yf_newest' + '.pkl', 'rb'))
points_pre = scaler.transform(points)
y_pred = kmeans.predict(points_pre)
indexs = pd.value_counts(y_pred).index
cluster_centers = kmeans.cluster_centers_[indexs]
pca = PCA(n_components=10)
X_points = pca.fit_transform(points_pre)
cluster_centers_nor = pca.fit_transform(cluster_centers)
b = pca.explained_variance_ratio_
j=0
colors = ['blue','orange','green','red','purple','brown','pink','gray','olive','cyan','royalblue','gold','darkgreen','darkred','indigo','coral',
    'crimson','indianred','springgreen','teal','yellowgreen','blueviolet','peru','magenta', 'yellow', 'palegreen', 'slateblue','teal','deepskyblue','chocolate']
fig, ax = plt.subplots() 
#scatter = ax.scatter(X_points[:,0], X_points[:,1], s=100, c=y_pred, cmap='rainbow')
#scatter.set_alpha(0.1) 
for i in indexs:
    scatter = ax.scatter(X_points[y_pred == i, 0], X_points[y_pred == i, 1], color = colors[j], s=100)
    scatter.set_alpha(0.05) 
    ax.scatter(np.mean(X_points[y_pred == i, 0]), np.mean(X_points[y_pred == i, 1]), color = colors[j], edgecolor='black', label = str(j), s=200) 
    #ax.scatter(cluster_centers_nor[i][0], cluster_centers_nor[i][1], color = colors[j], edgecolor='black', label = str(j), s=200) 
    plt.annotate(str(j), (np.mean(X_points[y_pred == i, 0]), np.mean(X_points[y_pred == i, 1])), textcoords="offset points", xytext=(0,10), ha='center')
    j =j + 1

ax.tick_params(axis='x', labelsize=30)  
ax.tick_params(axis='y', labelsize=30) 
xlabel_name = 'Principle Component 1'
ylabel_name = 'Principle Component 2'
plt.xlabel(f'{xlabel_name} {b[0]:.2%}', fontsize=30)
plt.ylabel(f'{ylabel_name} {b[1]:.2%}', fontsize=30)
ax.grid(True) 
ax.legend(loc='upper right')  

j=0
ax1 = ax.inset_axes([100, 60, 100, 40], transform = ax.transData)
for i in indexs:
    scatter = ax1.scatter(X_points[y_pred == i, 0], X_points[y_pred == i, 1], color = colors[j], s=100)
    scatter.set_alpha(0.05) 
    ax1.scatter(np.mean(X_points[y_pred == i, 0]), np.mean(X_points[y_pred == i, 1]), color = colors[j], edgecolor='black', label = str(j), s=200) 
    #ax.scatter(cluster_centers_nor[i][0], cluster_centers_nor[i][1], color = colors[j], edgecolor='black', label = str(j), s=200) 
    plt.annotate(str(j), (np.mean(X_points[y_pred == i, 0]), np.mean(X_points[y_pred == i, 1])), textcoords="offset points", xytext=(0,10), ha='center')
    j =j + 1
ax1.set(xlim = (-10,30), ylim=(-10,20))
ax.indicate_inset_zoom(ax1)
plt.show()
a= 1

