import os
from utils.utils import *
import matplotlib.pyplot as plt
from matplotlib import rcParams


config = {
    "font.family": 'Times New Roman',
    "font.size": 10,
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)


def plot_graph(x_data, y_data, user_x_data, user_y_data, xlabel, ylabel, title, range_, save_path):

    marker = ['^', '3', 'P', 'x', 'D']
    color = ['coral', 'yellow', 'lawngreen', 'cyan', 'dodgerblue']

    labels = [f'v{i+1}.0' for i in range(len(x_data[0]))]

    for i, (x, y) in enumerate(zip(x_data, y_data)):
        plt.plot(x, y, color=color[i], marker=marker[i], label=f'ai vs user{i+1}')  # 绘制折线图

    for x_data, y_data in zip(user_x_data, user_y_data):
        
        for i, (x, y) in enumerate(zip(x_data, y_data)):
            plt.scatter(x, y, color='violet', marker='0', s=100, alpha=0.5)

    plt.xticks(x, labels)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=15)
    plt.legend()

    plt.grid(linestyle="--", alpha=0.2)  # 设置背景网格线为虚线

    plt.ylim(range_[0], range_[1])

    plt.savefig(save_path)
    plt.close()


def draw(Data, ranges, save_path):

    xlabel = 'version'
    ylabels = ['miou', 'loss_rate', 'wrong_rate']

    for i, ylabel in enumerate(ylabels):

        y_data, user_y_data = [], []

        for key, value in Data.items():

            if '0 vs' in key:
                y_data.append(value[ylabel])

            else:
                user_y_data.append(value[ylabel])

        x_data, user_x_data = range(len(value[ylabel]))*5, range(len(value[ylabel]))*4
        plot_graph(x_data, y_data, user_x_data, user_y_data, xlabel, ylabel, ylabel, ranges[i], save_path+f'_{ylabel}.png')


def add_dict(res, data):

    not_need = ['num_miss', 'size_mask', 'num_wrong', 'size_pred'] 

    for key, value in data.items():

        if key == 'user name':

            user_name = value

            if user_name not in res:
                res[user_name] = {}

        elif key not in not_need:

            if key not in res[user_name]:
                res[user_name][key] = [value]
            
            else:
                res[user_name][key].append(value)

    return res


def draw_metrics(root_path, ranges, save_path):
    
    Data = {}

    for csv_path in os.scandir(root_path):
        
        datas = read_csv(csv_path.path)

        for data in datas:

            Data = add_dict(Data, data)

    draw(Data, ranges, save_path)