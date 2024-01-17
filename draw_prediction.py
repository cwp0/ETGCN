import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


def draw1(lx, dy, title):
    """画双轴折线图
    :param lx x轴数据集合
    :param dy y轴数据字典
    """
    # 设置图片可以显示中文和特殊字符
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    df = pd.DataFrame(dy, index=lx)
    fyt = list(dy.keys())[0]
    syt = list(dy.keys())[1]
    ax = df.plot(secondary_y=[syt], x_compat=True, grid=True, linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel(fyt)
    ax.grid(linestyle="--", alpha=0.3)
    ax.right_ax.set_ylabel(syt)
    plt.show()


def draw2(lx, dy, title):
    """画双轴折线图
    :param lx x轴数据集合
    :param dy y轴数据字典
    """
    # 设置图片可以显示中文和特殊字符
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fyt = list(dy.keys())[0]
    syt = list(dy.keys())[1]
    plt.figure(figsize=(8, 5), dpi=120)
    plt.plot(lx, dy.get(fyt), label=fyt, marker='s', linewidth=0.8, color='black')
    plt.grid(linestyle="--", alpha=0.3)
    # plt.title(title, fontsize=12)
    plt.ylabel(fyt, fontsize=10)
    plt.xlabel("T/min", fontsize=10)
    plt.ylim(3.7, 4.4)
    plt.legend(loc='upper left')
    # 调用twinx后可绘制次坐标轴
    ax = plt.twinx()
    plt.plot(lx, dy.get(syt), label=syt, marker='s', linewidth=0.8, color='red')
    plt.ylabel(syt, fontsize=10)
    plt.ylim(0.750, 0.790)
    plt.legend(loc='upper right')
    # 设置x轴刻度
    ax.spines['right'].set_color('red')
    ax.xaxis.label.set_color('red')
    ax.tick_params(axis='y', colors='red')
    # ax.set_xlabel('X-axis')
    plt.xticks(range(0, 4), lx)
    # plt.show()
    plt.savefig(fname="long-term.eps", format="eps")


def main():
    """主函数"""
    lx = [i for i in range(3, 13, 3)]
    y1 = [3.976, 3.996, 4.006, 4.020]
    y2 = [0.775, 0.774, 0.773, 0.772]
    dy = {'RMSE': y1, 'Accuracy': y2}
    title = 'Prediction'
    # draw1(lx, dy, title)
    lxx = [str(i) for i in list(lx)]
    draw2(lxx, dy, title)


if __name__ == '__main__':
    main()
