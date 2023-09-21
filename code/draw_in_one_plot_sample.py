import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
import seaborn as sns
import pickle
import math
from sklearn.decomposition import PCA
from scipy.cluster import hierarchy
from function import *
from matplotlib import rcParams
from matplotlib.pyplot import MultipleLocator

config = {
    "font.family":'Times New Roman',  # 设置字体类型
    "axes.unicode_minus": False #解决负号无法显示的问题
}
rcParams.update(config)

def Max_per_total(n, clusters, kmeans_addr, select_points, individual_sn_nor, individual_sn, min_cluster): #执行n次，看每次结果是否相有重复性
    sn_cnt = np.zeros(len(clusters))
    max_per_total_group = []
    for k in range(0,n):        
        high_per_total = []
        max_per_total = []
        max_index_total = []
        count_total = []
        for i in clusters:
            kmeans = joblib.load(kmeans_addr[k]+'kmeans_yf' + '_' + str(i) + '.pkl') 
            y_pred = kmeans.predict(select_points)
            indexs = pd.value_counts(y_pred).index
            counts = pd.value_counts(y_pred)
            percents = 100 * counts / sum(counts)
            cluster_centers = kmeans.cluster_centers_[indexs]
            #查找哪个分类全部来自于同一个蜂箱
            count_cluster = []
            max_per_sn = []
            max_index_sn = []
            index_sn = indexs.asi8
            for j in range(i):
                count_sn = pd.value_counts(individual_sn_nor[y_pred == index_sn[j]])
                sn_list = count_sn.index.values
                sn_event = count_sn.values
                sn_collect = np.array([sn_list, sn_event])
                sn_collect = np.transpose(sn_collect)
                sn_collect = sn_collect[np.argsort(sn_collect[:,0])]
                count_cluster.append(sn_collect) 
                coefficent = 1
                max_percentage = max(sn_collect[:,1])/sum(sn_collect[:,1]) # 每个分类下，各个蜂箱最大占比
                max_index = np.where(sn_collect[:,1] == np.max(sn_collect[:,1])) # 得到最大占比时的蜂箱编号
                if ((max_percentage) * coefficent>= 0.95):
                    sn_cnt[i-min_cluster] = sn_cnt[i-min_cluster] + 1
                max_per_sn.append(max_percentage) #将该分类下，每种分类的最大百分比存储
                max_index_sn.append(max_index) # 将该分类下，每种最大百分比对应的编号存储
            max_per_index = np.where(max_per_sn == np.max(max_per_sn))
            max_per_total.append(max_per_sn[max_per_index[0][0]])
            max_index_total.append(max_index_sn[max_per_index[0][0]])            
            count_total.append(count_cluster)
            #给出满足条件的sn序号
            if (sn_cnt[i-min_cluster]>= 1): 
                for j in range(i):                  
                    high_per_num = np.where(count_cluster[j][:,1]/sum(count_cluster[j][:,1]) > 0.5)[0]
                    if (any(high_per_num)):
                        high_per_sn = count_cluster[j][:,0][high_per_num]
                        high_per_total.append(high_per_sn)



        #统计满足条件大于50%的sn编号
        individual_sn_total = []
        high_per_total = np.array(high_per_total).reshape(len(high_per_total),)
        high_per_count = pd.value_counts(high_per_total).index.values
        high_per_count = np.sort(high_per_count)
        high_per_count = high_per_count.astype(np.int64)
        for i in range(len(high_per_count)):
            sn_code_position = np.where(individual_sn_nor == high_per_count[i])[0][0]
            individual_sn_origin = individual_sn[sn_code_position]
            individual_sn_total.append(individual_sn_origin)
        print(individual_sn_total)

        max_per_total_group.append(max_per_total)
    return max_per_total_group


def Pre_distribution(n_clusters, kmeans_addr, select_points):
    pred_total = []
    for k in range(0,3):
        kmeans= joblib.load(kmeans_addr[k]+'kmeans_yf' + '_' + str(n_clusters) + '.pkl')
        y_pred = kmeans.predict(select_points)
        y_pred_sort = np.zeros(y_pred.shape)
        indexs = pd.value_counts(y_pred).index
        for i in range(0, len(indexs)):
            n = np.where(y_pred == indexs[i])
            y_pred_sort[n] = i
        y = pd.value_counts(y_pred_sort).values
        y_percent = y/np.sum(y)
        pred_total.append(y_percent)
    return pred_total

def Draw_PCA(select_info, kmeans, colors, ax, label_param_size, labelx_front_size, labely_front_size):
    points = np.array([x[3] for x in select_info])
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
    for i in indexs:
        scatter = ax.scatter(X_points[y_pred == i, 0], X_points[y_pred == i, 1], color = colors[j], s=30)
        scatter.set_alpha(0.05) 
        ax.scatter(np.mean(X_points[y_pred == i, 0]), np.mean(X_points[y_pred == i, 1]), color = colors[j], edgecolor='black', label = str(j+1), s=30) 
        j =j + 1
    ax.tick_params(axis='x', labelsize=label_param_size)  
    ax.tick_params(axis='y', labelsize=label_param_size) 
    xlabel_name = 'Principle Component 1'
    ylabel_name = 'Principle Component 2'
    plt.xlabel(f'{xlabel_name} ({b[0]:.2%})', fontsize=labelx_front_size)
    plt.ylabel(f'{ylabel_name} ({b[1]:.2%})', fontsize=labely_front_size)
    ax.grid(True) 
    ax.legend(loc='upper right', prop = {'size':3.4})  

    j=0
    ax1 = ax.inset_axes([100, 50, 100, 40], transform = ax.transData)
    for i in indexs:
        scatter = ax1.scatter(X_points[y_pred == i, 0], X_points[y_pred == i, 1], color = colors[j], s=30)
        scatter.set_alpha(0.05) 
        ax1.scatter(np.mean(X_points[y_pred == i, 0]), np.mean(X_points[y_pred == i, 1]), color = colors[j], edgecolor='black', label = str(j), s=30) 
        j =j + 1
    ax1.set(xlim = (-10,30), ylim=(-10,20))
    ax1.tick_params(axis='x', labelsize=label_param_size)  
    ax1.tick_params(axis='y', labelsize=label_param_size) 
    ax.indicate_inset_zoom(ax1)


def main():
    #配置相关参数
    batch_size = 10000
    max_cluster = 50
    min_cluster = 2
    clusters = [i for i in range(min_cluster,max_cluster)]

    n = 3
    n_clusters = 22

    labelx_front_size = 10
    labely_front_size = 10
    label_param_size = 10
    legend_front_size = 10

    x_insert = np.arange(0, n_clusters, 1)
    #加载相关数据
    individual_sn_nor = np.loadtxt('./kmeans_yf_single_nor_newest/individual_sn_nor_all.txt', delimiter=',')
    individual_sn = np.loadtxt('./kmeans_yf_single_nor_newest/individual_sn_all.txt', delimiter=',')

    select_info = np.load("./kmeans_yf_single_nor_newest/info_table.npy", allow_pickle = True)
    scaler = pickle.load(open('scaler_yf_newest' + '.pkl', 'rb'))
    select_points = np.array([x[3] for x in select_info])
    select_points = scaler.transform(select_points)
    kmeans_addr = ['./kmeans_yf_single_nor_newest/', './kmeans_yf_single_nor_test1/', './kmeans_yf_single_nor_test2/']
    kmeans = joblib.load('./kmeans_yf_single_nor_newest/''kmeans_yf' + '_' + str(n_clusters) + '.pkl')

    y_pred = kmeans.predict(select_points)
    indexs = pd.value_counts(y_pred).index
    cluster_centers = kmeans.cluster_centers_[indexs]

    max_per_total_group = Max_per_total(n, clusters, kmeans_addr, select_points, individual_sn_nor, individual_sn, min_cluster)
    pred_total = Pre_distribution(n_clusters, kmeans_addr, select_points)



    fig = plt.figure(figsize=(15,15), dpi = 100)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, \
        wspace=0.3, hspace=0.5)
    plt.tight_layout()

    #------------------------------------采样三次，不同聚类数量的样本集中度------------------------
    plt.subplot(4,2,1)
    x = np.arange(min_cluster, max_cluster, 1)
    p1 = plt.plot(x, [i*100 for i in max_per_total_group[0]],linewidth=2.0, color = 'tab:blue', label = 'Trial 3')
    p2 = plt.plot(x, [i*100 for i in max_per_total_group[1]],linewidth=2.0, color = 'tab:green', label = 'Trial 2')
    p3 = plt.plot(x, [i*100 for i in max_per_total_group[2]],linewidth=2.0,color = 'tab:orange', label = 'Trial 1')
    plt.legend(fontsize=legend_front_size)
    plt.xlabel("Cluster Number", fontsize=labelx_front_size)
    plt.ylabel("Maximum Colony Proportion (%)",fontsize=labely_front_size)
    plt.xticks(fontsize=label_param_size)
    plt.yticks(fontsize=label_param_size)
    plt.xlim(2, 50)
    plt.ylim(0, 100)
    plt.grid()

    #-------------------------------采样三次，22类每类的分布情况-------------------------------------
    label = ['Trial 3', 'Trial 2', 'Trial 1', 'Pattern', 'Proportion(%)']

    x = x_insert
    y = [pred_total[2]*100, pred_total[1]*100, pred_total[0]*100]
    labels = [str(i) for i in x]
    x = np.arange(len(labels)) + 1 
    width = 0.2 
    gap = 0.02
    a = y[0].tolist()
    b = y[1].tolist()
    c = y[2].tolist()

    plt.subplot(4,2,2)
    rects1 = plt.bar(x - (width+gap), a, width, label=label[0], color = 'tab:blue')
    rects2 = plt.bar(x , b, width, label=label[1], color = 'tab:green')
    rects3 = plt.bar(x + (width+gap), c, width, label=label[2],color = 'tab:orange')   
    plt.xlabel(label[3], fontsize=labelx_front_size)
    plt.ylabel(label[4],fontsize=labely_front_size)
    plt.xticks(x,fontsize=label_param_size)
    plt.yticks(fontsize=label_param_size)
    plt.legend(fontsize=legend_front_size)

    #----------------------------画出PCA分类图------------------------------------------------

    colors = ['blue','orange','green','red','purple','brown','pink','gray','olive','cyan','royalblue','gold','darkgreen','darkred','indigo','coral',
        'crimson','indianred','springgreen','teal','yellowgreen','blueviolet','peru','magenta', 'yellow', 'palegreen', 'slateblue','teal','deepskyblue','chocolate']
    ax = plt.subplot(4,2,3)
    Draw_PCA(select_info, kmeans, colors, ax, label_param_size, labelx_front_size, labely_front_size)

    #---------------------------------树状图-------------------------------------------
    cluster_centers_test = cluster_centers
    Z = hierarchy.linkage(cluster_centers_test,  method = 'average', metric = 'correlation')

    plt.subplot(4,2,4)
    plt.xlabel('Pattern', fontsize=labelx_front_size)
    plt.ylabel('Correlation Distance',fontsize=labelx_front_size)
    plt.xticks(fontsize=label_param_size, rotation = 0)
    plt.yticks(fontsize=label_param_size)
    hierarchy.dendrogram(Z, color_threshold=0.7, leaf_font_size=label_param_size, leaf_rotation=0, labels = np.arange(len(cluster_centers))+1)
    plt.axhline(y=240, c='grey', lw=1, linestyle='dashed')

    #-----------------------------------热力图-------------------------------------------
    xticks = [j for j in range(8, 520, 8)]
    yticks = ['%s' % (k+1) for k in range(n_clusters)]

    x_ticks_str = [ str(x) for x in xticks]
    xlabels = [ str(x) for x in xticks ]
    ylabels=yticks
    cmap=cm.get_cmap('rainbow',1000)
    ax = plt.subplot(2,1,2)

    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels)
    map=ax.imshow(cluster_centers,interpolation='nearest',cmap=cmap,aspect='auto', vmax = 30, norm=PowerNorm(gamma=0.6))
    cb=plt.colorbar(mappable=map,  cax=None,ax=None,shrink=0.5, label = 'Intensity')
    cb.set_label('Intensity',fontsize=10) 
    cb.ax.tick_params(labelsize=10)
    plt.xlabel('Frequency(Hz)', fontsize=labelx_front_size)
    plt.ylabel('Pattern',fontsize=labely_front_size) 
    plt.xticks(rotation = 90)
    plt.yticks(fontsize=label_param_size)
    plt.xticks(fontsize=label_param_size)
    y_major_locator=MultipleLocator(2)
    ax=plt.gca()
    ax.yaxis.set_major_locator(y_major_locator)
    plt.show()

if __name__=='__main__':
    main()