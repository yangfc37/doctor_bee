import numpy as np
import pandas as pd
from pandas import Series,DataFrame
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import holoviews as hv
from holoviews import opts, dim
import seaborn as sns
import math
import matplotlib.patches as mpatches

from matplotlib import cm 
from matplotlib import axes
from matplotlib.colors import PowerNorm
import matplotlib.dates as mdates

from datetime import *

from IPython import display

def coordinate_nor(x_value, y_value, n_clusters):  
    x_insert = np.arange(0, n_clusters, 1)
    y_insert = np.zeros(n_clusters)
    n=0
    for x in range(0,n_clusters):
        if (x==x_value[n]):
            y_insert[x] = y_value[n]
            n = n + 1
            if (n == y_value.shape[0]):
                break
        else:
            y_insert[x] = 0
    return [x_insert, y_insert]    

def sort_index_based(y_pred, group_pred_count, n_clusters):  #根据y_pred的index编号，对group_pred_count重新排序
    group_pred_list = group_pred_count.index.values
    group_pred_event = group_pred_count.values
    indexs = pd.value_counts(y_pred).index
    group_pred_gather_all = np.zeros([n_clusters,2])
    #将分类列表和事件合并为一个数组
    group_pred_gather = np.array([group_pred_list,group_pred_event])
    group_pred_gather = np.transpose(group_pred_gather)
    #若在此数组种的分类列表能找到相应的分类号，则根据y-pred的分类号进行重新排序
    for i in range(0, n_clusters):
        n = np.where(group_pred_gather[:,0] == indexs.values[i])
        if ((np.array(n)).size >0):
            group_pred_gather_all[i,0] = group_pred_gather[n,0]
            group_pred_gather_all[i,1] = group_pred_gather[n,1]
        else:  #若找不到则置为0
            group_pred_gather_all[i,0] = indexs.values[i]
            group_pred_gather_all[i,1] = 0
    group_pred_list_in = group_pred_gather_all[:,0]
    group_pred_event_in = group_pred_gather_all[:,1]
    return  (group_pred_list_in, group_pred_event_in)

# 输出满足条件的蜂箱编号以及相应的数量
def distr_best_cluster(individual_sn, y_pred, sn_group, batch_size, n_clusters, code_rule):
    group_pred = []
    indexs = pd.value_counts(y_pred).index
    sn_pred_data = np.concatenate((individual_sn[:,np.newaxis],y_pred[:,np.newaxis]),axis=1)
    for i in range(len(sn_group)):
        group_pred_data = sn_pred_data[sn_pred_data[:,0] == sn_group[i]]
        if (group_pred_data.any()):
            group_pred.append(group_pred_data)
    group_pred_arr = np.array(group_pred).reshape(np.array(group_pred).shape[0]*batch_size, 2)
    group_pred_count = pd.value_counts(group_pred_arr[:,1])

    group_pred_list = group_pred_count.index.values
    group_pred_event = group_pred_count.values
    #按照y_pred的分类进行排序
    if (code_rule):
        group_pred_list_in, group_pred_event_in =  sort_index_based(y_pred, group_pred_count, n_clusters)     
    else:
        group_pred_list_in = group_pred_list
        group_pred_event_in = group_pred_event
    
    labels = [i for i in range(n_clusters)]
    group_pred_list_in = labels
    group_pred_collect = np.array([group_pred_list_in,group_pred_event_in])
    group_pred_collect = np.transpose(group_pred_collect)
    #group_pred_collect = group_pred_collect[np.argsort(group_pred_collect[:,0])]
    return group_pred_collect

def plot_cluster_tree(code_rule, y_pred, cluster_centers):
    #y_pred = kmeans.labels_
    indexs = pd.value_counts(y_pred).index
    cluster_centers_test = cluster_centers
    # if code_rule:
    #     cluster_centers_test = kmeans.cluster_centers_[indexs]
    # else:
    #     cluster_centers_test = kmeans.cluster_centers_
    #cluster_centers_test = scaler.inverse_transform(cluster_centers_test)
    #Z = hierarchy.linkage(cluster_centers_test, 'ward')
    #Z = hierarchy.linkage(cluster_centers_test,  method = 'single', metric = 'cosine')
    Z = hierarchy.linkage(cluster_centers_test,  method = 'average', metric = 'correlation')
    #plt.figure(figsize=[16, 9], dpi=100)
    #fig, ax = plt.subplots()
    plt.figure(figsize=(16,9),dpi=100)
    #ax.set_yscale("log")
    #plt.title('Dendrogram of Beehive Category', fontsize=20)
    plt.xlabel('Patterns', fontsize=20)
    #plt.ylabel('distance (Ward)',fontsize=10)
    plt.ylabel('Correlation Distance',fontsize=20)
    plt.xticks(fontsize=20, rotation = 0)
    plt.yticks(fontsize=20)
    hierarchy.dendrogram(Z, color_threshold=0.7, leaf_font_size=20, leaf_rotation=0, labels = np.arange(len(cluster_centers))+1)
    # 画水平线，y纵坐标，c颜色，lw线条粗细，linestyle线形
    plt.axhline(y=240, c='grey', lw=1, linestyle='dashed')
    # color_threshold设定颜色阈值，小于color_threshold根据簇节点为一簇

    # color_threshold设定颜色阈值，小于color_threshold根据簇节点为一簇

def get_info(info, indexs, y_pred):
    individuals_sn_info = np.unique(info[:,1])
    clusters_months = []
    clusters_hours = []
    clusters_temperatures = []
    clusters_individuals = []
    for i in indexs:
        months = info[y_pred == i, 5]
        hours = info[y_pred == i, 7]
        individuals = info[y_pred == i, 1]
        temperatures = info[y_pred == i, 2]
        sample_n = months.shape[0]
        months_count = []
        hours_count = []
        temperatures_count = []
        individuals_count = []
        for k in range(1,13):
            months_count.append(np.sum(months == '%02d'%k)*100/sample_n)
        for k in range(0,24):
            hours_count.append(np.sum(hours == '%02d'%k)*100/sample_n)
        for k in range(-20, 50):
            temperatures_count.append(np.sum(np.floor(temperatures) == k) * 100 / sample_n)
        for k in individuals_sn_info:
            individuals_count.append(np.sum(individuals == k) * 100 / sample_n)
        clusters_months.append(months_count)
        clusters_hours.append(hours_count)
        clusters_temperatures.append(temperatures_count)
        clusters_individuals.append(individuals_count)
    return (clusters_months, clusters_hours, clusters_temperatures, clusters_individuals)

def plot_data(data, plt_mode, n_clusters, xticks, yticks, xlabel_name):
    if plt_mode[0]:
        plt.figure(figsize=[16, 9], dpi=100)
        plt.plot(np.transpose(data))
        plt.xlabel(xlabel_name)
        plt.ylabel('percentage(%)')
        plt.legend(yticks)
        plt.xticks(range(len(xticks)),range(min(xticks), max(xticks)+1))
    if plt_mode[1]:
        plt.figure(figsize=[16, 9], dpi=100)
        sns.heatmap(data, cmap='rainbow', xticklabels=xticks, yticklabels=yticks)
        plt.xlabel(xlabel_name, fontsize=15)
        plt.ylabel('Category', fontsize=15)
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)

        clusters_half = math.ceil(n_clusters/2) + 1
        plt.figure(figsize=[16, 9], dpi=100)
        plt.subplot(1, 2, 1)
        sns.heatmap(data[0:clusters_half], cmap='rainbow', xticklabels=xticks, yticklabels=yticks[0:clusters_half])
        plt.subplot(1, 2, 2)
        sns.heatmap(data[clusters_half:n_clusters], cmap='rainbow', xticklabels=xticks, yticklabels=yticks[clusters_half:n_clusters])
    if plt_mode[2]:
        corr = np.corrcoef(np.transpose(data), rowvar=False)
        plt.figure(figsize=[16, 9], dpi=100)
        sns.heatmap(corr, cmap='rainbow', xticklabels=[j for j in range(0, n_clusters)], yticklabels=[j for j in range(0, n_clusters)])
        plt.xlabel(xlabel_name, fontsize=20)
        plt.ylabel('Category', fontsize=20)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)

        Z = hierarchy.linkage(np.array(data),  method = 'average', metric = 'correlation')
        plt.figure(figsize=[16, 9], dpi=100)
        #fig, ax = plt.subplots()
        #ax.set_yscale("log")
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('cluster number', fontsize=10)
        #plt.ylabel('distance (Ward)',fontsize=10)
        plt.ylabel('distance (Average)',fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        hierarchy.dendrogram(Z, color_threshold=0.2)
        # 画水平线，y纵坐标，c颜色，lw线条粗细，linestyle线形
        plt.axhline(y=240, c='grey', lw=1, linestyle='dashed')

def plot_holoview(transition, n_clusters):
    cluster_value = [i for i in range(0,n_clusters)]
    cluster_value_str = [str(i) for i in range(0,n_clusters)]
    df2 = pd.DataFrame(transition, columns = cluster_value_str)
    df2_insert = df2.insert(loc=0, column='cluster', value=cluster_value_str)
    df2_data = df2.melt(id_vars=['cluster'], 
                value_vars=cluster_value_str
                )
    df2_node = pd.DataFrame(df2_data['cluster'].append(df2_data['variable']).unique(),
                        columns=['node']
                    )
    df2_nodes = hv.Dataset(df2_node, 'node',)
    chord = hv.Chord((df2_data, df2_nodes), 
                    ['cluster', 'variable'], ['value'])

    busiest = node['node'].to_list()
    busiest_airports = chord.select(AirportID=busiest, selection_mode='nodes')

    busiest_airports.opts(
        opts.Chord(cmap='Tab20', edge_color=dim('cluster').str(), 
                height=500,
                width=500,
                labels='node',
                node_color='node',
                edge_visible=True
                ))
    return busiest_airports

class Compact():
    def __init__(self, cluster_centers, n_clusters):
        self.cluster_centers = cluster_centers
        self.n_clusters = n_clusters

    def forward(self,x):
        sum_d_in = 0
        sum_d_out = 0
        compact = []
        for i in range(0, self.n_clusters):
            for j in range(0, x[i].shape[0]):
                d_in = self.eucliDist(x[i][j], self.cluster_centers[i])
                sum_d_in = sum_d_in + d_in
            for k in range(0, self.n_clusters):
                if (i!=k):
                    d_out = self.eucliDist(self.cluster_centers[i], self.cluster_centers[k])
                    sum_d_out = sum_d_out + d_out
            compact_single =  (sum_d_in/x[i].shape[0]) / (sum_d_out/(self.n_clusters-1)) 
            compact.append(compact_single)
        return compact

    def eucliDist(self,A,B):
        return np.sqrt(sum(np.power((A - B), 2)))  


def cluster_redefine(num, points):
    #num = np.array([int(x[1]) for x in info])
    #points = np.array([x[3] for x in info])
    num_count = pd.value_counts(num)
    num_index = np.sort(num_count.index.values)
    cluster = num_count.size
    clusters = []
    centers = []
    for i in range(0, cluster):
        n_index = np.where(num == num_index[i])
        clusters_single = points.take(n_index,0)
        clusters_single = np.reshape(clusters_single, (-1,64))
        centers_single = np.mean(clusters_single, axis=0)
        clusters.append(clusters_single)
        centers.append(centers_single)
    centers = np.array(centers)
    return (cluster, clusters, centers)

def myplot(x, y, label=None, xlimit=None, size=(9, 3),fileName=None):
    display.set_matplotlib_formats('svg')
    if len(x) == len(y):
        #plt.figure(figsize=size)
        if xlimit and isinstance(xlimit, tuple):
            plt.xlim(xlimit)
        plt.scatter(x, y, marker = "x", label=label)
        plt.title("cluster with time", fontsize=20)
        plt.xlabel("Time", fontsize=20)
        plt.ylabel("Clusters",fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        if label and isinstance(label, str):
            plt.legend()
        if fileName:
            plt.savefig(fileName)
        plt.show()
    else:
        raise ValueError("x 和 y 的长度不一致！")

def myplot_twoCurves(x1, y1, x2, y2, label_1=None, label_2 = None, xlimit=None, size=(9, 3),fileName=None):
    display.set_matplotlib_formats('svg')
    if (len(x1) == len(y1)) and (len(x2) == len(y2)):
        plt.figure(figsize=size)
        if xlimit and isinstance(xlimit, tuple):
            plt.xlim(xlimit)
        plt.scatter(x1, y1, marker = "x", label=label_1)
        plt.scatter(x2, y2, marker = "o", label=label_2)
        plt.title("cluster with time", fontsize=20)
        plt.xlabel("Time", fontsize=20)
        plt.ylabel("Clusters",fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        if (label_1 and label_2) and isinstance(label_1, str):
            plt.legend()
        if fileName:
            plt.savefig(fileName)
        plt.show()
    else:
        raise ValueError("x 和 y 的长度不一致！")

def myplot_threeCurves(x1, y1, x2, y2, x3, y3, label_1=None, label_2 = None, label_3 = None, xlimit=None, size=(9, 3),fileName=None):
    display.set_matplotlib_formats('svg')
    if (len(x1) == len(y1)) and (len(x2) == len(y2)):
        plt.figure(figsize=size)
        if xlimit and isinstance(xlimit, tuple):
            plt.xlim(xlimit)
        plt.scatter(x1, y1, marker = "x", label=label_1)
        plt.scatter(x2, y2, marker = "o", label=label_2)
        plt.scatter(x3, y3, marker = "v", label=label_3)
        plt.title("cluster with time", fontsize=20)
        plt.xlabel("Time", fontsize=20)
        plt.ylabel("Clusters",fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        if (label_1 and label_2) and isinstance(label_1, str):
            plt.legend()
        if fileName:
            plt.savefig(fileName)
        plt.show()
    else:
        raise ValueError("x 和 y 的长度不一致！")

def cm2inch(x,y):
    return(x/2.54,y/2.54)

def plot_twin(x, x_ff, y1, y2, y3, y4, y3_ff, x_label, y_labels, number):
    # 构造多个ax
    fig,ax1 = plt.subplots(figsize = cm2inch(16,9))
    # 绘制
    if (number == 2):
        ax2 = ax1.twinx()
        img1, = ax1.plot(x,y1,c = 'tab:blue')
        img2, = ax2.plot(x,y2,marker='o', ls='None', c = 'tab:green')
        #img3_add, = ax2.plot(x_ff,y3_ff,marker='v', ls='None', c = 'tab:red')
            #获取对应折线图颜色给到spine ylabel yticks yticklabels
        axs = [ax1,ax2]
        imgs = [img1,img2]
        #y_labels = ['temperature', 'weight','category', 'category_ff', 'Entrance_counts']
        for  i in range(len(axs)): 
            axs[i].spines['right'].set_color(imgs[i].get_color())
            #axs[i].set_ylabel('y{}'.format(i+1),c = imgs[i].get_color())
            axs[i].set_ylabel(y_labels[i],c = imgs[i].get_color() , fontsize=20)
            axs[i].tick_params(axis = 'y', color = imgs[i].get_color(), labelcolor = imgs[i].get_color(), labelsize = 12)            
            axs[i].spines['left'].set_color(img1.get_color())#注意ax1是left
        # 设置其他细节
        ax1.set_xlabel(x_label, fontsize=20)
        ax1.tick_params(axis = 'x',labelsize = 12)
        ax1.set_ylim(-20,60)
        ax2.set_ylim(0,23)
        ax2.yaxis.set_minor_locator(plt.MultipleLocator(1))
        #plt.legend([img1, img2, img3_add], ["temperature", "Category", "Category_ff"], loc='upper left')
        plt.legend([img1, img2], ["Temperature", "Category"], loc='upper left')
        year_total = []
        plot_set = 3  #设置为根据日落日出调整背景色
        backGround_color(x, plot_set, ax2)  #设置背景色
        #ax4.set_ylim(0,10000)
        plt.tight_layout()

        plt.show()
    elif(number == 4):
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()
        ax4 = ax1.twinx()
        # 将构造的ax右侧的spine向右偏移
        ax3.spines['right'].set_position(('outward',60))
        ax4.spines['right'].set_position(('outward',120))
        img1, = ax1.plot(x,y1,c = 'tab:orange')
        img2, = ax2.plot(x,y2,marker='o', ls='None', c = 'tab:green')        
        img3, = ax3.plot(x,y3,c = 'tab:blue')
        img4 = ax4.bar(x,y4, color = 'tab:purple', alpha = 0.5, width = 0.03)
        #ax4.set_yscale("log", base = 2)
        #img3_add, = ax2.plot(x_ff,y3_ff,marker='v', ls='None', c = 'tab:red')
        #获取对应折线图颜色给到spine ylabel yticks yticklabels
        axs = [ax1,ax2,ax3, ax4]
        imgs = [img1,img2,img3, img4]
        #y_labels = ['temperature', 'weight','category', 'category_ff', 'Entrance_counts']
        for  i in range(len(axs)): 
            if (i!=3):
                axs[i].spines['right'].set_color(imgs[i].get_color())
                #axs[i].set_ylabel('y{}'.format(i+1),c = imgs[i].get_color())
                axs[i].set_ylabel(y_labels[i],c = imgs[i].get_color(), fontsize=12)
                axs[i].tick_params(axis = 'y', color = imgs[i].get_color(), labelcolor = imgs[i].get_color(), labelsize = 12)
                axs[i].spines['left'].set_color(img1.get_color())#注意ax1是left
            else:
                axs[i].spines['right'].set_color('purple')
                axs[i].set_ylabel(y_labels[i],c = 'purple', fontsize=12)
                axs[i].tick_params(axis = 'y', color = 'purple', labelcolor = 'purple', labelsize = 12)     
        # 设置其他细节
        ax1.set_xlabel(x_label, fontsize=12)
        ax1.set_ylim(-20,60)
        ax1.tick_params(axis = 'x',labelsize = 12)
        ax2.set_ylim(0,30)
        ax3.set_ylim(min(y3),max(y3))
        ax4.set_ylim(min(y4),max(y4))
        plt.legend([img1, img2, img3, img4], ["Temperature", "Category", "Weight", "Entrance_counts"], loc='upper left')
        #ax4.set_ylim(0,10000)
        plt.tight_layout()
        plt.show()

def draw_heatmap(data, xticks, yticks):
    xlabels = [ str(x) for x in xticks ]
    ylabels=yticks
    cmap=cm.get_cmap('rainbow',1000)
    figure=plt.figure(facecolor='w')
    ax=figure.add_subplot(1,1,1)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels)
    map=ax.imshow(data,interpolation='nearest',cmap=cmap,aspect='auto', vmax = 30, norm=PowerNorm(gamma=0.6))
    cb=plt.colorbar(mappable=map,  cax=None,ax=None,shrink=0.5, label = 'Intensity')
    cb.set_label('Intensity',fontsize=20) 
    cb.ax.tick_params(labelsize=15)
    plt.xlabel('Frequency(Hz)', fontsize=20)
    #plt.ylabel('distance (Ward)',fontsize=10)
    plt.ylabel('Pattern',fontsize=20) 
    plt.xticks(rotation = 90)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.show()

def draw_bar_four(x, y, number, label):
    labels = [str(i) for i in x]
    #labels = ['第一周', '第二周', '第三周', '第四周']

    x = np.arange(len(labels)) + 1  # 标签位置
    width = 0.08  # 柱状图的宽度，可以根据自己的需求和审美来改
    gap = 0.01

    #plt.figure(figsize=(16,9),dpi=100)
    fig, ax = plt.subplots()
    if (number == 8):
        a = y[0].tolist()
        b = y[1].tolist()
        c = y[2].tolist()
        d = y[3].tolist()
        e = y[4].tolist()
        f = y[5].tolist()
        g = y[6].tolist()
        h = y[7].tolist()
        rects5 = ax.bar(x + (width+gap)*(1/2), e, width, label=label[4])
        rects6 = ax.bar(x + (width+gap)*(3/2), f, width, label=label[5])
        rects7 = ax.bar(x + (width+gap)*(5/2), g, width, label=label[6])
        rects8 = ax.bar(x + (width+gap)*(7/2), h, width, label=label[7])
        rects1 = ax.bar(x - (width+gap)*(7/2), a, width, label=label[0])
        rects2 = ax.bar(x - (width+gap)*(5/2), b, width, label=label[1])
        rects3 = ax.bar(x - (width+gap)*(3/2), c, width, label=label[2])
        rects4 = ax.bar(x - (width+gap)*(1/2), d, width, label=label[3])        

    if (number == 4):
        a = y[0].tolist()
        b = y[1].tolist()
        c = y[2].tolist()
        d = y[3].tolist()
        rects1 = ax.bar(x - (width+gap)*(3/2), a, width, label=label[0])
        rects2 = ax.bar(x - (width+gap)*(1/2), b, width, label=label[1])
        rects3 = ax.bar(x + (width+gap)*(1/2), c, width, label=label[2])
        rects4 = ax.bar(x + (width+gap)*(3/2), d, width, label=label[3])
    elif (number == 3):
        a = y[0].tolist()
        b = y[1].tolist()
        c = y[2].tolist()
        rects1 = plt.bar(x - (width+gap), a, width, label=label[0], color = 'tab:blue')
        rects2 = plt.bar(x , b, width, label=label[1], color = 'tab:green')
        rects3 = plt.bar(x + (width+gap), c, width, label=label[2],color = 'tab:orange')     
    #rects5 = ax.bar(x + width*2 + 0.04, e, width, label='e')


    # 为y轴、标题和x轴等添加一些文本。
    #ax.set_ylabel(label[3], fontsize=30)
    #ax.set_xlabel(label[4], fontsize=30)
    plt.xlabel('Pattern', fontsize=30)
    plt.ylabel('Proportion (%)',fontsize=30)
    #ax.set_title(title)
    plt.xticks(x,fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(fontsize=20)
    #ax.set_xticks(x)
    #ax.set_xticklabels(labels)
    #plt.grid()
    #ax.legend()

def Counts_Data_To_Table(data, x_labels, n_clusters, file_name):
    data_group = []
    data_group_sort = []
    sum_group = []
    for i in range (n_clusters):
        data_index = np.concatenate((np.array(data[i])[:,np.newaxis], np.array(x_labels)[:,np.newaxis]),axis=1)
        data_sort = data_index[np.argsort(-data_index[:,0])]
        sum_percent = 0
        n = 0
        data_cluster = []
        while sum_percent<66.7:
            sum_percent = sum_percent + data_sort[n][0]
            data_cluster.append(data_sort[n][1])
            n = n + 1
        sum_group.append(sum_percent)
        date_cluster_sort = sorted(data_cluster)
        data_group.append(data_cluster)
        data_group_sort.append(date_cluster_sort)
    #data_group_sort = data_group.sort()
    labels = [i for i in range(n_clusters)]
    l1 = labels
    l2 = data_group
    l3 = data_group_sort
    l4 = sum_group

    df = DataFrame({'类别': l1, '数据之和大于66.7%(从大到小排列)': l2, '数据之和大于66.7%(顺序排列)': l3, '百分比之和': l4})
    df.to_excel(file_name, sheet_name='sheet1', index=False)

def plot_temp_cat(point_select, x_time, temp, category, x_time_ff, y_num_ff, plot_req):
    skipped_point = point_select[0]
    pre_seleted_points = point_select[1]
    start_point = skipped_point + pre_seleted_points
    temp_selected_average_group = []
    temp_selected_std_group = []
    temp_selected_min_group = []
    temp_selected_max_group = []
    for i in range(start_point, len(temp)):
        temp_selected_average = np.mean(temp[i-24:i])
        temp_selected_std = np.std(temp[i-24:i])
        temp_selected_min = np.min(temp[i-24:i])
        temp_selected_max = np.max(temp[i-24:i])
        temp_selected_average_group.append(temp_selected_average)
        temp_selected_std_group.append(temp_selected_std)
        temp_selected_min_group.append(temp_selected_min)
        temp_selected_max_group.append(temp_selected_max)
    temp_min = np.array(temp_selected_min_group)
    temp_max = np.array(temp_selected_max_group)
    temp_average = np.array(temp_selected_average_group)
    temp_std = np.array(temp_selected_std_group)
    temp_error_downLimit = temp_min - temp_average 
    temp_error_upLimit = temp_max - temp_average 
    temp_error = np.concatenate((temp_error_upLimit[:,np.newaxis], temp_error_downLimit[:,np.newaxis]), axis=1)
    temp_selected_data = temp[start_point:len(temp)]
    temp_selected_data = np.expand_dims(temp_selected_data, axis=-1)
    temp_average = np.expand_dims(temp_average, axis=-1)
    data = temp_average + temp_error

    # 将 time 扩展为 10 列一样的数据
    time_array = x_time[start_point:len(temp)]
    x_cate = x_time[start_point:len(temp)]
    for i in range(temp_error.shape[1] - 1):
        time_array = np.column_stack((time_array, x_time[start_point:len(temp)]))

    # 将 time 和 signal 平铺为两列数据，且一一对应
    time_array = time_array.flatten()  # (630,)
    data = data.flatten()

    data = np.column_stack((time_array, data))
    df = pd.DataFrame(data, columns=['time', 'signal'])
    df.signal = df.signal.astype(float)

    #x_cate = x_time[start_point:len(temp)]
    cate = category[start_point:len(temp)]
    # 绘图
    #fig,ax1 = plt.subplots(figsize = cm2inch(16,9))
    img1 = sns.lineplot(x='time', y='signal', data=df, ax = ax1, color = 'tab:blue')
    sns.set(style="darkgrid")
    #fig_add = plt.gcf()
    #ax = fig.axes[0]
    ax1.set_zorder(1)
    ax2 = plt.twinx()
    img2, = ax2.plot(x_cate, cate, marker='o', ls='None', c = 'tab:green', label = 'cate')
    if plot_req[0]:
        img3_add, = ax2.plot(x_time_ff,y_num_ff,marker='v', ls='None', c = 'tab:red')
    ax1.set_ylabel('Temperature($^\circ$C)', fontsize=16)
    ax1.set_xlabel('Time', fontsize=16)
    ax1.tick_params(axis = 'x', labelsize = 12)
    ax2.set_ylabel('Category', fontsize=16)
    ax2.set_zorder(2)
    y_labels = ['Temperature($^\circ$C)','Category']
    axs = [ax1,ax2]
    imgs = [img1,img2]
    color = ['tab:blue', 'green']
    for  i in range(len(axs)): 
        axs[i].spines['right'].set_color(color[i])
        #axs[i].set_ylabel('y{}'.format(i+1),c = imgs[i].get_color())
        axs[i].set_ylabel(y_labels[i],c = color[i])
        axs[i].tick_params(axis = 'y', color = color[i], labelcolor = color[i], labelsize = 12)
        axs[i].spines['left'].set_color(color[0])#注意ax1是le

    red_patch = mpatches.Patch(label='Temperature Data', color = 'tab:blue')
    if plot_req[0]:
        plt.legend(handles=[red_patch, img2, img3_add], labels = ["Temperature","Category", "Category_ff"], loc='upper left')
    else:
        plt.legend(handles=[red_patch, img2], labels = ["Temperature","Category"], loc='upper left')

    # spring_group = [[],[],[],[],[]]
    # summer_group = [[],[],[],[],[]]
    # autum_group = [[],[],[],[],[]]
    # winter_group = [[],[],[],[],[]]
    # for i in range (0, 5):
    #     spring_group[i] = []
    #     summer_group[i] = []
    #     autum_group[i] = []
    #     winter_group[i] = []

    # year_total = [2019, 2020, 2021, 2022, 2023]
    # year_append = []
    # for j in range(0, len(x_time)):
    #     date = x_time[j].strftime('%Y-%m-%d')
    #     month =int(date[5:7])
    #     day = int(date[8:11])
    #     year = int(date[0:4])
    #     for k in range (0, len(year_total)):
    #         if (year == year_total[k]):
    #             if (month > 3 and  month <= 6):
    #                 spring_group[k].append(x_time[j])
    #             elif (month > 6 and  month <= 9):
    #                 summer_group[k].append(x_time[j])
    #             elif(month > 9 and  month <= 12):
    #                 autum_group[k].append(x_time[j])
    #             else:
    #                 winter_group[k].append(x_time[j])
    # season_group = [spring_group, summer_group, autum_group, winter_group]

    # for i in range(0, len(season_group)):
    #     for j in range(0, len(year_total)):
    #         if len(season_group[i][j]):
    #             if (season_group[i][j] == spring_group[j]):
    #                 ax2.axvspan(season_group[i][j][0], season_group[i][j][len(season_group[i][j])-1], facecolor='green', alpha=0.2)
    #             elif (season_group[i][j] == summer_group[j]):
    #                 ax2.axvspan(season_group[i][j][0], season_group[i][j][len(season_group[i][j])-1], facecolor='red', alpha=0.2)
    #             elif (season_group[i][j] == autum_group[j]):
    #                 ax2.axvspan(season_group[i][j][0], season_group[i][j][len(season_group[i][j])-1], facecolor='yellow', alpha=0.2)     
    #             elif (season_group[i][j] == winter_group[j]):
    #                 ax2.axvspan(season_group[i][j][0], season_group[i][j][len(season_group[i][j])-1], facecolor='blue', alpha=0.2)   
    plot_set = 2 
    backGround_color(x_time, plot_set, ax2)
    plt.show()

def backGround_color(x_time, plot_set, ax):
    if (plot_set == 1): #按四季进行背景色设置
        spring_group = [[],[],[],[],[]]
        summer_group = [[],[],[],[],[]]
        autum_group = [[],[],[],[],[]]
        winter_group = [[],[],[],[],[]]
        for i in range (0, 5):
            spring_group[i] = []
            summer_group[i] = []
            autum_group[i] = []
            winter_group[i] = []

        year_total = [2019, 2020, 2021, 2022, 2023]
        year_append = []
        for j in range(0, len(x_time)):
            date = x_time[j].strftime('%Y-%m-%d')
            month =int(date[5:7])
            day = int(date[8:11])
            year = int(date[0:4])
            for k in range (0, len(year_total)):
                if (year == year_total[k]):
                    if (month > 3 and  month <= 6):
                        spring_group[k].append(x_time[j])
                    elif (month > 6 and  month <= 9):
                        summer_group[k].append(x_time[j])
                    elif(month > 9 and  month <= 12):
                        autum_group[k].append(x_time[j])
                    else:
                        winter_group[k].append(x_time[j])
        season_group = [spring_group, summer_group, autum_group, winter_group]

        for i in range(0, len(season_group)):
            for j in range(0, len(year_total)):
                if len(season_group[i][j]):
                    if (season_group[i][j] == spring_group[j]):
                        ax.axvspan(season_group[i][j][0], season_group[i][j][len(season_group[i][j])-1], facecolor='green', alpha=0.2)
                    elif (season_group[i][j] == summer_group[j]):
                        ax.axvspan(season_group[i][j][0], season_group[i][j][len(season_group[i][j])-1], facecolor='red', alpha=0.2)
                    elif (season_group[i][j] == autum_group[j]):
                        ax.axvspan(season_group[i][j][0], season_group[i][j][len(season_group[i][j])-1], facecolor='yellow', alpha=0.2)     
                    elif (season_group[i][j] == winter_group[j]):
                        ax.axvspan(season_group[i][j][0], season_group[i][j][len(season_group[i][j])-1], facecolor='blue', alpha=0.2) 
    elif (plot_set == 2):
        # 定义每个季节的起始月份  
        seasons = {  
            'spring': range(3, 6),  
            'summer': range(6, 9),  
            'autumn': range(9, 12),  
            'winter': [*range(1, 3), 12]  
        }  
        season_color = {
            'spring': 'green',  
            'summer': 'red',  
            'autumn': 'yellow',  
            'winter': 'blue'  
        }
        
        # 创建一个空列表，用于存储各个日期对应的季节  
        dates = []  
        
        # 遍历一年的日期，并确定每个日期的季节  
        start_date = x_time[0]  # 当年的1月1日作为起始日期  
        end_date = x_time[len(x_time) -1]  # 当年的12月31日作为结束日期  
        current_date = start_date  
        
        while current_date <= end_date:  
            current_season = [season for season, month_range in seasons.items() if current_date.month in month_range][0]  
            dates.append((current_date, current_season))  
            current_date += timedelta(days=1)  
        
        # 设置图形大小  
        #plt.figure(figsize=(12, 4)) 
        
        # 绘制日期区间对应的季节背景色  
        for i, (date, season) in enumerate(dates):  
            if i < len(dates) - 1:  
                #if season != dates[i + 1][1]:  
                ax.axvspan(dates[i][0], dates[i + 1][0], facecolor=season_color[season], alpha=0.1)  

    elif (plot_set == 3):
        # 定义每个昼夜的起始时间  
        day_nights = {  
            'day': range(6, 18),  
            'night': [*range(18, 24), *range(0, 6)] 
        }  
        day_night_color = {
            'day': 'white',  
            'night': 'gray'  
        }
        
        # 创建一个空列表，用于存储各个日期对应的季节  
        dates = []  
        
        # 遍历一年的日期，并确定每个日期的季节  
        start_date = x_time[0]  # 当年的1月1日作为起始日期  
        end_date = x_time[len(x_time) -1]  # 当年的12月31日作为结束日期  
        current_date = start_date  
        
        while current_date <= end_date:  
            current_day_night = [day_night for day_night, time_range in day_nights.items() if current_date.hour in time_range][0]  
            dates.append((current_date, current_day_night))  
            current_date += timedelta(hours=0.5)  
        
        # 设置图形大小  
        #plt.figure(figsize=(12, 4)) 
        
        # 绘制日期区间对应的季节背景色  
        for i, (date, day_night) in enumerate(dates):  
            if i < len(dates) - 1:  
                #if season != dates[i + 1][1]:  
                ax.axvspan(dates[i][0], dates[i + 1][0], facecolor=day_night_color[day_night], alpha=0.1)   
                
    else:  #按日落日出进行背景色设置
        day_start_time = datetime.strptime("06:00", "%H:%M")  
        night_start_time = datetime.strptime("18:00", "%H:%M")     
        for i in range(0, len(x_time)):
            times_adjusted = x_time[i].replace(year=day_start_time.year)
            if (i!= (len(x_time)-1)): 
                times_adjusted_next = x_time[i+1].replace(year=day_start_time.year)
                times_delta = times_adjusted_next - times_adjusted
                if (times_delta == timedelta(hours=0)):
                    i = i+1
                    continue
                else:
                    if (times_adjusted.time() >= day_start_time.time() and times_adjusted.time() < night_start_time.time()):
                        ax.axvspan(x_time[i], x_time[i] + timedelta(hours=1), facecolor='red', alpha=0.1)  
                    else:
                        ax.axvspan(x_time[i], x_time[i] + timedelta(hours=1), facecolor='blue', alpha=0.1) 

