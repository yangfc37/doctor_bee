import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
from function import *
from matplotlib import rcParams

config = {
    "font.family":'Times New Roman',  # 设置字体类型
    "axes.unicode_minus": False #解决负号无法显示的问题
}
rcParams.update(config)

def month_processing(clusters_months,ax, n_clusters):
    month_xticks = [j for j in range(1, 13)]
    month_yticks = ['%s' % (k+1) for k in range(n_clusters)]
    month_percentage = 0
    month_heatmap = 1
    month_corr = 0
    month_plt_mode = [month_percentage, month_heatmap, month_corr]

    ax = sns.heatmap(clusters_months, cmap='rainbow', xticklabels=month_xticks, yticklabels=month_yticks)
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=10)
    cbar = ax.collections[0].colorbar
    cbar.set_label('Probability (%)')
    plt.xlabel('Month', fontsize=10)
    plt.ylabel('Pattern', fontsize=10)
    plt.yticks(fontsize=10, rotation = 0)
    plt.xticks(fontsize=10, rotation = 0)

def hour_processing(clusters_hours, ax, n_clusters):
    hour_xticks = [j for j in range(0, 24)]
    hour_yticks = ['%s' % (k+1) for k in range(n_clusters)]
    hour_percentage = 0
    hour_heatmap = 1
    hour_corr = 0
    hour_plt_mode = [hour_percentage, hour_heatmap, hour_corr]
    ax = sns.heatmap(clusters_hours, cmap='rainbow', xticklabels=hour_xticks, yticklabels=hour_yticks)
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=10)
    cbar = ax.collections[0].colorbar
    cbar.set_label('Probability (%)')
    plt.xlabel('Hour', fontsize=10)
    plt.ylabel('Pattern', fontsize=10)
    plt.yticks(fontsize=10, rotation = 0)
    plt.xticks(fontsize=10, rotation = 0)

def temp_processing(clusters_temperatures, ax, n_clusters):
    temp_xticks = [j for j in range(-20, 50)]
    temp_yticks = ['%s' % (k+1) for k in range(n_clusters)]
    temp_percentage = 1
    temp_heatmap = 1
    temp_corr = 0
    temp_plt_mode = [temp_percentage, temp_heatmap, temp_corr]
    ax = sns.heatmap(clusters_temperatures, cmap='rainbow', xticklabels=temp_xticks, yticklabels=temp_yticks)
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=10)
    cbar = ax.collections[0].colorbar
    cbar.set_label('Probability (%)')
    plt.xlabel('Temperatures (℃)', fontsize=10)
    plt.ylabel('Pattern', fontsize=10)
    plt.yticks(fontsize=10, rotation = 0)
    plt.xticks(fontsize=10, rotation = 90)

def main():
    #设定参数，加载数据
    n_clusters = 22
    kmeans = joblib.load('./kmeans_yf_single_nor_newest/''kmeans_yf' + '_' + str(n_clusters) + '.pkl')
    info = np.load("./kmeans_yf_single_nor_newest/info_original_data.npy", allow_pickle = True)
    points = np.array([x[3] for x in info])
    scaler = pickle.load(open('scaler_yf_newest' + '.pkl', 'rb'))
    points = scaler.transform(points)

    y_pred = kmeans.predict(points)
    indexs = pd.value_counts(y_pred).index
    counts = pd.value_counts(y_pred)
    clusters_months, clusters_hours, clusters_temperatures, clusters_individuals = get_info(info, indexs, y_pred)

    fig = plt.figure(figsize=[15, 15], dpi=100)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, \
        wspace=None, hspace=None)
    plt.tight_layout()
    grid = plt.GridSpec(2,3)
    # 月度分析
    ax1 = plt.subplot(grid[0,0])
    month_processing(clusters_months,ax1, n_clusters)

    # 小时分析
    ax2 = plt.subplot(grid[0,1:3])
    hour_processing(clusters_hours, ax2, n_clusters)

    # 温度分析
    ax3 = plt.subplot(grid[1,0:3])
    temp_processing(clusters_temperatures, ax3, n_clusters)

    plt.show()

if __name__ == '__main__':
    main()

