import joblib
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
import time
import datetime
import seaborn as sns
import pickle
import math

from datetime import datetime
from datetime import date
import matplotlib as mpl

from function import *

from matplotlib import rcParams
from matplotlib.pyplot import MultipleLocator
from matplotlib.font_manager import FontProperties


config = {
    "font.family":'Times New Roman',  # 设置字体类型
    "axes.unicode_minus": False #解决负号无法显示的问题
}
rcParams.update(config)


def plot_subplots_twin(x,y1,y2,ax1,label_font, y_labels):
    label_font_size = label_font[0]
    label_param_size = label_font[1]
    ax2 = ax1.twinx()
    img1, = ax1.plot(x,y1,c = 'tab:blue')
    img2, = ax2.plot(x,y2,marker='o', ls='None', c = 'tab:green', markersize=2)
    axs = [ax1,ax2]
    imgs = [img1,img2]
    for  i in range(len(axs)): 
        axs[i].spines['right'].set_color(imgs[i].get_color())
        axs[i].set_ylabel(y_labels[i],c = imgs[i].get_color() , fontsize=label_font_size)
        axs[i].tick_params(axis = 'y', color = imgs[i].get_color(), labelcolor = imgs[i].get_color(), labelsize = label_param_size)            
        axs[i].spines['left'].set_color(img1.get_color())#注意ax1是left
    # 设置其他细节
    ax1.set_xlabel('Time', fontsize=label_font_size)
    ax1.tick_params(axis = 'x',labelsize = label_param_size)
    ax1.set_ylim(-20,60)
    ax2.set_ylim(0,22)
    ax2.yaxis.set_minor_locator(plt.MultipleLocator(1))
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.legend([img1, img2], ["Temperature", "Pattern"], loc='upper left')
    year_total = []
    plot_set = 3  #设置为根据日落日出调整背景色
    backGround_color(x, plot_set, ax2)  #设置背景色

def plot_twin(point_select, x_time, temp, category, ax1, label_font_size, legend_size, label_param_size):
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
    time_array = time_array.flatten()  
    data = data.flatten()

    data = np.column_stack((time_array, data))
    df = pd.DataFrame(data, columns=['time', 'signal'])
    df.signal = df.signal.astype(float)

    legend_font = {
        'family': 'Times New Roman',  
        'style': 'normal',
        'weight': "normal",  
    }

    cate = category[start_point:len(temp)]
    # 绘图
    img1 = sns.lineplot(x='time', y='signal', data=df, ax = ax1, color = 'tab:blue')
    sns.set(style="darkgrid")
    ax1.set_zorder(1)
    ax2 = plt.twinx()
    img2, = ax2.plot(x_cate, cate, marker='o', ls='None', c = 'tab:green', label = 'cate', markersize = 2)
    ax1.set_ylabel('Temperature (℃)', fontsize=label_font_size)
    ax1.set_xlabel('Time', fontsize=label_font_size)
    ax1.tick_params(axis = 'x', labelsize = label_param_size)

    ax1.set_xticks(['2020-09','2020-12', '2021-03','2021-06', '2021-09', '2021-12', '2022-03', '2022-06', '2022-09', '2022-12', '2023-03', '2023-06'])
    ax2.set_ylabel('Pattern', fontsize=label_font_size, font = legend_font)
    font = FontProperties(family='Times New Roman', size=12)  
    ax2.set_yticklabels(ax2.get_yticks(), fontproperties=font)  
    ax2.set_zorder(2)

    ax1.set_ylim(-20,60)
    ax2.set_ylim(0,25)
    y_labels = ['Temperature (℃)','Pattern']
    axs = [ax1,ax2]
    imgs = [img1,img2]
    color = ['tab:blue', 'green']
    for  i in range(len(axs)): 
        axs[i].spines['right'].set_color(color[i])
        axs[i].set_ylabel(y_labels[i],c = color[i])
        axs[i].tick_params(axis = 'y', color = color[i], labelcolor = color[i], labelsize = 10)
        axs[i].spines['left'].set_color(color[0])

    red_patch = mpatches.Patch(label='Temperature Data', color = 'tab:blue')
    plt.legend(handles=[red_patch, img2], labels = ["Temperature","Pattern"], loc='upper left', prop=legend_font)

    plot_set = 2 
    backGround_color(x_time, plot_set, ax2)

    mpl.rcParams.update(mpl.rcParamsDefault)
    config = {
        "font.family":'Times New Roman',  # 设置字体类型
        "axes.unicode_minus": False #解决负号无法显示的问题
    }
    rcParams.update(config)

def cal_xy_value(sn_num, individual_sn_yf, info_ypred, event_info):
    sn_event_split_group = []
    info_event_split_group = []
    time_index_group = []
    sn_unique = sn_num

    index = np.where(individual_sn_yf == sn_unique)   #找出分蜂时间发生的蜂箱编号在所有意蜂数据中的index索引
    # 若索引不为0，则将该部分蜂箱的数据info_event_sn从总数据info_ypred中提取出来。并且在原始数据中查找分蜂蜂箱对应的时间点，将这部分子集info_event_split进行提取
    if (index[0].size!=0):
        sn_event_split_group.append(sn_unique)
        info_event_sn = info_ypred.take(index,0)
        info_event_sn = np.reshape(info_event_sn, (-1,info_ypred.shape[1]))
        info_event_sn_sort = info_event_sn[np.argsort(info_event_sn[:,0])]
        info_event_split_group.append(info_event_sn)
        for j in range(0, len(event_info)):
            if (sn_unique == int(event_info[j][1])):
                time_index = np.where(info_event_sn_sort[:,0] == event_info[j][0][0:16])
                if (time_index[0].size!=0):
                    time_index_group.append(time_index[0][0])
                    time_index_arr = np.array(time_index_group)
                    time_index_arr  = np.unique(time_index_arr)
                    info_event_split = info_event_sn_sort[time_index_arr]
                    info_event_split_sort = info_event_split[np.argsort(info_event_split[:,0])]


    time = info_event_sn_sort[:,0]
    y_num = info_event_sn_sort[:,13]
    x_time= [datetime.strptime(d, '%Y-%m-%d %H:%M') for d in time]
    temp = info_event_sn_sort[:,2]
    category = y_num + 1
    return (x_time, temp, category)


def main():
    #配置参数和加载数据
    n_clusters = 22
    sn_num = 1572636598
    label_font_size = 10
    legend_size = 10
    label_param_size = 10

    event_info = np.load("./kmeans_yf_single_nor_newest/event_info_add.npy", allow_pickle = True)
    info = np.load("./kmeans_yf_single_nor_newest/info_original_data.npy", allow_pickle = True)
    individual_sn_yf = np.array([x[1] for x in info])
    points = np.array([x[3] for x in info])
    scaler = pickle.load(open('scaler_yf_newest' + '.pkl', 'rb'))
    points = scaler.transform(points)
    kmeans = joblib.load('./kmeans_yf_single_nor_newest/''kmeans_yf' + '_' + str(n_clusters) + '.pkl')
    y_pred = kmeans.predict(points)
    y_pred_sort = np.zeros(y_pred.shape)
    indexs = pd.value_counts(y_pred).index
    for i in range(0, len(indexs)):
        n = np.where(y_pred == indexs[i])
        y_pred_sort[n] = i

    info_ypred = np.concatenate((info, y_pred_sort[:,np.newaxis]),axis=1)
    x_time, temp, category = cal_xy_value(sn_num, individual_sn_yf, info_ypred, event_info)

    y_labels = ['Temperature (℃)','Pattern', 'weight(g)', 'Entrance_counts']

    skipped_points = 0
    pre_seleted_points = 24       
    point_select = [skipped_points, pre_seleted_points]
    season_shadow = 1    #是否每个季节背景需要添加 1：添加 0： 不添加
    ff_event = 0        #是否需要加入分蜂事件 1：加入 0：不加入
    plot_req = [ff_event, season_shadow]
    #------------------------------------------------------画图1----------------------------------------------
    fig = plt.subplots(4,2)

    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, \
        wspace=None, hspace=0.5)
    ax1 = plt.subplot(2,1,1)
    plot_twin(point_select, x_time, temp,  category, ax1, label_font_size, legend_size, label_param_size)

    #---------------------------------------------画图2---------------------------------
    ax2 = plt.subplot(4,2,5)
    spring_start_points = 11742
    spring_end_points = 11910
    x = x_time[spring_start_points:spring_end_points]
    y1 = temp[spring_start_points:spring_end_points]
    y2 = category[spring_start_points:spring_end_points]
    label_font = [label_font_size, label_param_size]
    plot_subplots_twin(x,y1,y2,ax2,label_font, y_labels)

    # ------------------------------------------------------画图3-------------------------------------------
    ax3 = plt.subplot(4,2,6)
    summer_start_points = 13594
    summer_end_points = 13778
    x = x_time[summer_start_points:summer_end_points]
    y1 = temp[summer_start_points:summer_end_points]
    y2 = category[summer_start_points:summer_end_points]
    label_font = [label_font_size, label_param_size]
    plot_subplots_twin(x,y1,y2,ax3,label_font, y_labels)

    #------------------------------------------------------画图4---------------------------------------------
    ax4 = plt.subplot(4,2,7)
    autumn_start_points = 15946
    autumn_end_points = 16114
    x = x_time[autumn_start_points:autumn_end_points]
    y1 = temp[autumn_start_points:autumn_end_points]
    y2 = category[autumn_start_points:autumn_end_points]
    label_font = [label_font_size, label_param_size]
    plot_subplots_twin(x,y1,y2,ax4,label_font, y_labels)

    #------------------------------------------------------画图5---------------------------------------------
    ax5 = plt.subplot(4,2,8)
    winter_start_points = 17207
    winter_end_points = 17375
    x = x_time[winter_start_points:winter_end_points]
    y1 = temp[winter_start_points:winter_end_points]
    y2 = category[winter_start_points:winter_end_points]
    label_font = [label_font_size, label_param_size]
    plot_subplots_twin(x,y1,y2,ax5,label_font, y_labels)

    plt.show()

if __name__ == '__main__':
    main()
    a=1