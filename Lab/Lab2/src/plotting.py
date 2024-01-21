import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


if __name__ == "__main__":
    # 读取csv文件
    df = pd.read_csv("./tmp/output_original.csv")
    # 画图
    sns.set(style="darkgrid")
    sns.lineplot(x="epoch", y="loss", data=df, palette="tab10", linewidth=2.5)
    # 左下角对齐原点
    # plt.margins(x=0,y=0) 
    # 设置标签大小
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # 设置坐标轴名称字体大小
    plt.xlabel("epoch", fontsize=18)
    plt.ylabel("loss", fontsize=18)
    plt.show()
    sns.lineplot(x="epoch", y="acc", data=df, palette="tab10", linewidth=2.5)
    plt.ylim(0, 1)
    # 设置标签大小
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # 设置坐标轴名称字体大小
    plt.xlabel("epoch", fontsize=18)
    plt.ylabel("accuracy", fontsize=18)
    plt.show()

