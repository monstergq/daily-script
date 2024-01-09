import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.pyplot import MultipleLocator

config = {
    "font.family": 'Times New Roman',
    "font.size": 10,
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)


def plot_line_graph(x, y, save_path, title="Line Graph", xlabel="X-axis", ylabel="Y-axis", grid=True, color="blue", marker="o"):

    """
    使用matplotlib绘制折线图的函数。

    参数:
    x (list): X轴的数据。
    y (list): Y轴的数据。
    title (str): 图表的标题。
    xlabel (str): X轴的标签。
    ylabel (str): Y轴的标签。
    grid (bool): 是否显示网格线。
    color (str): 折线的颜色。
    marker (str): 数据点的标记样式。
    """

    plt.figure(figsize=(10, 6))  # 设置图形的大小
    plt.plot(x, y, color=color, marker=marker)  # 绘制折线图

    # 添加标题和标签
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)

    # 如果指定，则显示网格线
    if grid:
        plt.grid(True)

    # 显示图表
    plt.show()
    plt.savefig(save_path)


# def plot_graph(x_data, y_data, user_x_data, user_y_data, color, marker, user_color, user_marker, save_path,  title="Line Graph", xlabel="X-axis", ylabel="Y-axis", grid=True):

#     """
#     使用matplotlib绘制折线图的函数。

#     参数:
#     x (list): X轴的数据。
#     y (list): Y轴的数据。
#     title (str): 图表的标题。
#     xlabel (str): X轴的标签。
#     ylabel (str): Y轴的标签。
#     grid (bool): 是否显示网格线。
#     color (str): 折线的颜色。
#     marker (str): 数据点的标记样式。
#     """

#     plt.figure(figsize=(10, 6))  # 设置图形的大小

#     for i, (x, y) in enumerate(zip(x_data, y_data)):

#         plt.plot(x, y, color=color[i], marker=marker[i], label=f'ai vs user{i+1}')  # 绘制折线图

#     for x_data, y_data in zip(user_x_data, user_y_data):
        
#         for i, (x, y) in enumerate(zip(x_data, y_data)):
#             plt.scatter(x, y, color=user_color[i], marker=user_marker[i])

#     # 添加标题和标签
#     plt.title(title, fontsize=16)
#     plt.xlabel(xlabel, fontsize=14)
#     plt.ylabel(ylabel, fontsize=14)

#     # 如果指定，则显示网格线
#     if grid:
#         plt.grid(True)

#     # 显示图表
#     plt.show()
#     plt.savefig(save_path)


def plot_graph(x_data, y_data, user_x_data, user_y_data, color, marker, user_color, user_marker, save_path,  title="Line Graph", xlabel="X-axis", ylabel="Y-axis"):

    labels = ['v1.0', 'v2.0', 'v3.0']
    fig = plt.figure(figsize=(8, 5), dpi=300)

    for i, (x, y) in enumerate(zip(x_data, y_data)):
        plt.plot(x, y, color=color[i], marker=marker[i], label=f'ai vs user{i+1}')  # 绘制折线图

    for x_data, y_data in zip(user_x_data, user_y_data):
        
        for i, (x, y) in enumerate(zip(x_data, y_data)):
            plt.scatter(x, y, color=user_color[i], marker=user_marker[i], s=100, alpha=0.5)

    plt.xticks(x, labels)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=15)
    plt.legend()

    plt.grid(linestyle="--", alpha=0.2)  # 设置背景网格线为虚线

    plt.ylim(0.75, 0.87)
    # y_major_locator = MultipleLocator(0.05)
    # fig.yaxis.set_major_locator(y_major_locator)

    plt.savefig(save_path)
    # plt.show()
    # plt.close(fig)


if __name__ == '__main__':

    title = 'Miou'  # 图表的标题
    xlabel = 'Version'  # X轴的标签
    ylabel = 'Miou'  # Y轴的标签

    save_path = './graph/甲状腺红细胞_miou.png'  # 图表的保存路径

    marker = ['^', '3', 'P', 'x', 'D']
    color = ['coral', 'yellow', 'lawngreen', 'cyan', 'dodgerblue']
    x_data = [[1, 2, 3]] * 5
    y_data = [[0.767, 0.842, 0.848],
              [0.783, 0.841, 0.850], 
              [0.774, 0.836, 0.848],
              [0.760, 0.848, 0.851],
              [0.758, 0.819, 0.827]]
    
    user_marker = ['o'] * 5
    user_color = ['violet'] * 5
    user_x_data = [[[1, 2, 3]] * 4] * 5
    user_y_data = [[[0.856]*3, [0.860]*3, [0.843]*3, [0.842]*3],
                   [[0.856]*3, [0.861]*3, [0.833]*3, [0.852]*3],
                   [[0.860]*3, [0.861]*3, [0.833]*3, [0.852]*3],
                   [[0.860]*3, [0.861]*3, [0.833]*3, [0.841]*3],
                   [[0.842]*3, [0.852]*3, [0.841]*3, [0.821]*3]]

    # 使用示例数据调用函数
    plot_graph(x_data, y_data, user_x_data, user_y_data, color, marker, user_color, user_marker, save_path, title, xlabel, ylabel)