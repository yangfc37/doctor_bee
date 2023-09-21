import joblib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
import pickle
import math
from function import *
from matplotlib import rcParams
from matplotlib.pyplot import MultipleLocator

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
    time_ff = info_event_split_sort[:,0]
    y_num_ff = info_event_split_sort[:,13]
    x_time_ff= [datetime.strptime(d, '%Y-%m-%d %H:%M') for d in time_ff]

    temp = info_event_sn_sort[:,2]
    weight_left = info_event_sn_sort[:,9]
    weight_right = info_event_sn_sort[:,10]
    switch_left = info_event_sn_sort[:,11]
    switch_right = info_event_sn_sort[:,12]
    Entrance_counts =  switch_left + switch_right     
    weight_sum = weight_left + weight_right
    weight_sum[weight_sum>20000] = 0
    weight_sum[weight_sum<0] = 0

    weight = weight_sum*2
    category = y_num + 1
    return (x_time, x_time_ff, y_num_ff, weight, Entrance_counts, temp, category)


def plot_swarming(ax1, x, y1, y2, y3, y4, x_label, y_labels):
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax4 = ax1.twinx()
    # 将构造的ax右侧的spine向右偏移
    ax3.spines['right'].set_position(('outward',50))
    ax4.spines['right'].set_position(('outward',100))
    img1, = ax1.plot(x,y1,c = 'tab:blue')
    img2, = ax2.plot(x,y2,marker='o', ls='None', c = 'tab:green')   
    img3, = ax3.plot(x,y3/1000,c = 'tab:orange')
    img4 = ax4.bar(x,y4, color = 'tab:purple', alpha = 0.5, width = 0.03)
    axs = [ax1,ax2,ax3, ax4]
    imgs = [img1,img2,img3, img4]
    for  i in range(len(axs)): 
        if (i!=3):
            axs[i].spines['right'].set_color(imgs[i].get_color())
            axs[i].set_ylabel(y_labels[i],c = imgs[i].get_color(), fontsize=10)
            axs[i].tick_params(axis = 'y', color = imgs[i].get_color(), labelcolor = imgs[i].get_color(), labelsize = 10)
            axs[i].spines['left'].set_color(img1.get_color())
        else:
            axs[i].spines['right'].set_color('purple')
            axs[i].set_ylabel(y_labels[i],c = 'purple', fontsize=10)
            axs[i].tick_params(axis = 'y', color = 'purple', labelcolor = 'purple', labelsize = 10)     
    # 设置其他细节
    ax1.set_xlabel(x_label, fontsize=10)
    ax1.set_ylim(-20,60)
    ax1.tick_params(axis = 'x',labelsize = 10)
    c = ['00','01','02','03','04','05','06','07','08','09','10', '11','12','13','14','15','16','17','18','19','20','21','22','23']
    plt.xticks(x, c)
    ax2.set_ylim(0,23)
    ax3.set_ylim(10,18)
    ax4.set_ylim(0,60000)
    plt.legend([img1, img2, img3, img4], ["Temperature (℃)", "Pattern", "Weight (kg)", "Entrance (Counts)"], loc='upper left')
    plt.subplots_adjust(left=None, right=None, top=None, bottom=None, wspace=0.3, hspace=0.6)

def swarming_extract(info_ypred, sn_time):
    patterns = []
    ff_event_info = []
    for i in range(0, len(sn_time)):
        index = np.where((info_ypred [:,0] == sn_time[i][0][0:16]) & (info_ypred [:,1] == int(sn_time[i][1])))
        if (index[0].size!=0):
            pattern = int(float(info_ypred [:,13][index][0]))
            ff_event_info.append(info_ypred [index[0][0]])
            patterns.append(pattern)
    return (ff_event_info, patterns)    


def get_data(ff_event_info):
    ff_event_info = np.array(ff_event_info)
    individuals_sn_info = np.unique(ff_event_info[:,1])
    clusters_months = []
    clusters_hours = []
    clusters_temperatures = []

    months = ff_event_info[:,5]
    hours = ff_event_info[:,7]
    temperatures = ff_event_info[:,2]
    sample_n = months.shape[0]
    months_count = []
    hours_count = []
    temperatures_count = []
    for k in range(1,13):
        months_count.append(np.sum(months == '%02d'%k)*100/sample_n)
    for k in range(0,24):
        hours_count.append(np.sum(hours == '%02d'%k)*100/sample_n)
    for k in range(-20, 50):
        temperatures_count.append(np.sum(np.floor(temperatures) == k) * 100 / sample_n)
    clusters_months.append(months_count)
    clusters_hours.append(hours_count)
    clusters_temperatures.append(temperatures_count)
    #clusters_individuals.append(individuals_count)
    return (clusters_months, clusters_hours, clusters_temperatures)

def plot_swarming_bar(data_pre_average, data_after_average, data_pre_std, data_after_std, std_zero):
    error_attri = {"elinewidth": 2, "ecolor":"black", "capsize":6 }  # 误差棒的属性字典
    bar_width = 0.4 # 柱形的宽度
    tick_label = [i for i in range(1, 23)]  # 横坐标的标签
    x = np.arange(22)
    # 创建图形
    plt.bar(x, data_pre_average* 100,
            bar_width,
            color='tab:blue',
            align="center",
            error_kw=error_attri,
            label="Before",
            alpha=1)
    plt.errorbar(x, data_pre_average* 100,
                yerr=[std_zero.tolist(), data_pre_std.tolist()],
                capsize = 8,
                fmt='.', color="k") 

    plt.bar(x+bar_width, data_after_average*100,  # 若没有没有向右侧增加一个bar_width的宽度的话，第一个柱体就会被遮挡住
            bar_width,
            color='tab:orange',
            error_kw=error_attri,
            label="After",
            alpha=1)
    plt.errorbar(x+bar_width, data_after_average* 100,
                yerr=[std_zero.tolist(), data_after_std.tolist()],
                capsize = 8,
                fmt='.', color="k") 

    plt.xlabel("Pattern ", fontsize=10)
    plt.ylabel("Count Percent (%)", fontsize=10)
    plt.xticks(x+bar_width/2, tick_label, fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10)

def main():
    event_info = np.load("./kmeans_yf_single_nor/event_info_1572247262.npy", allow_pickle = True)
    sn_num = 1572247262
    n_clusters = 22
    info = np.load("./kmeans_yf_single_nor_newest/info_original_data.npy", allow_pickle = True)
    kmeans = joblib.load('./kmeans_yf_single_nor_newest/''kmeans_yf' + '_' + str(n_clusters) + '.pkl')
    data_pre_group = np.load("./kmeans_yf_single_nor_newest/data_pre_group.npy", allow_pickle = True)
    data_after_group = np.load("./kmeans_yf_single_nor_newest/data_after_group.npy", allow_pickle = True)
    individual_sn_yf = np.array([x[1] for x in info])
    points = np.array([x[3] for x in info])
    scaler = pickle.load(open('scaler_yf_newest' + '.pkl', 'rb'))
    points = scaler.transform(points)
    y_pred = kmeans.predict(points)
    y_pred_sort = np.zeros(y_pred.shape)

    labels = [i for i in range(n_clusters)]
    counts_all = pd.value_counts(y_pred)
    percents_all = 100 * counts_all / sum(counts_all)
    indexs = counts_all.index

    for i in range(0, len(indexs)):
        n = np.where(y_pred == indexs[i])
        y_pred_sort[n] = i

    event_info = np.load("./kmeans_yf_single_nor_newest/event_info_add.npy", allow_pickle = True)
    sn_time = np.array([(x[0],x[1]) for x in event_info])

    info_ypred = np.concatenate((info, y_pred_sort[:,np.newaxis]),axis=1)
    ff_event_info, patterns = swarming_extract(info_ypred, sn_time)

    x_time, x_time_ff, y_num_ff, weight, Entrance_counts, temp, category = cal_xy_value(sn_num, individual_sn_yf, info_ypred, event_info)

    x_label = 'Time'
    y_labels = ['Temperature($^\circ$C)','Pattern', 'weight (kg)', 'Entrance (Counts)']
    plot_number = 4

    fig = plt.figure(figsize=[15, 15], dpi=100)
    grid = plt.GridSpec(4,4)

    x  = x_time[13954:13978]
    x_ff = x_time_ff[13954:13978]
    y1 = temp[13954:13978]
    y2 = category[13954:13978]
    y3 = weight[13954:13978]
    y4 = Entrance_counts[13954:13978]
    y3_ff =  y_num_ff[13954:13978]
    number = plot_number
    ax1 = plt.subplot(grid[0,0:2])
    plot_swarming(ax1, x, y1, y2, y3, y4, x_label, y_labels)

    #--------------------------------------第五个图--------------------------------
    patterns = np.array(patterns)
    patterns_count = []
    patterns_list = range(0, n_clusters)
    for k in labels:
        patterns_count.append(np.sum(patterns == k))
    patterns_percent = 100 * (patterns_count/sum(patterns_count))
    patterns_percent_relative = 100*((patterns_percent - percents_all)/percents_all)

    ax5 = plt.subplot(grid[2,2:4])
    h1 = ax5.bar(range(n_clusters), np.array(patterns_percent_relative),color = 'r')
    ax5.set_xlabel('Pattern', fontsize=10)
    ax5.set_ylabel('Relative Proportion(%)', fontsize=10)
    ax5.set_xticks(range(n_clusters))
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)

    #-----------------------------------第二个图---------------------------------------
    clusters_months, clusters_hours, clusters_temperatures = get_data(ff_event_info)

    month_xticks = [j for j in range(1, 13)]
    ax2 = plt.subplot(grid[0,3])
    ax2.plot(np.transpose(clusters_months), marker = 'o')
    plt.xlabel('Month')
    plt.ylabel('Proportion (%)')
    plt.xticks(range(len(month_xticks)),range(min(month_xticks), max(month_xticks)+1))

    #------------------------------------第三个图---------------------------------
    temp_xticks = [j for j in range(-20, 50)]
    ax3 = plt.subplot(grid[1,0:4])
    ax3.plot(np.transpose(clusters_temperatures), marker = 'o')
    plt.xlabel('Temperature ($^\circ$C)')
    plt.ylabel('Proportion (%)')
    plt.xticks(range(len(temp_xticks)),range(min(temp_xticks), max(temp_xticks)+1), rotation = 90)

    #----------------------------------第四个图-----------------------------------------
    hour_xticks = [j for j in range(0, 24)]
    ax4 = plt.subplot(grid[2,0:2])
    ax4.plot(np.transpose(clusters_hours), marker = 'o')
    plt.xlabel('Hour')
    plt.ylabel('Proportion (%)')
    plt.xticks(hour_xticks)


    #-------------------------------第六个图--------------------------------------------
    ax6 = plt.subplot(grid[3,0:4])

    data_pre_average = np.mean(data_pre_group, axis = 1) 
    data_pre_std = np.std(data_pre_group, axis = 1) 
    data_pre_std = data_pre_std*100
    std_zero = np.zeros(22)
    data_after_average = np.mean(data_after_group, axis = 1) 
    data_after_std = np.std(data_after_group, axis = 1) 
    data_after_std = data_after_std*100
    plot_swarming_bar(data_pre_average, data_after_average, data_pre_std, data_after_std, std_zero)
    plt.show()
    
if __name__ == '__main__':
    main()
    a = 1