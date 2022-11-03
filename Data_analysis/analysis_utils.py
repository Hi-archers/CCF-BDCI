import os
import json
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def load_data(data_file='train.json'):
    file_path = os.path.join('../data_1', data_file)
    json_data = []
    for line in open(file_path, 'r', encoding='utf-8'):
        json_data.append(json.loads(line))
    return json_data


def return_length(data):
    counts = []
    for text in data:
        counts.append(len(text))
    return counts


def plot_bar(num_list, title='', xlabel='', ylabel=''):
    plt.bar(range(len(num_list)), num_list)
    plt.title(title)
    # plt.axvline(num_list[int(len(num_list) * 0.95)], color='red')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(r'./Graph/{}.jpg'.format(title), dpi=600)
    plt.show()


def find_all(text, sub='。'):
    pos = []
    start = 0
    while True:
        start = text.find(sub, start)
        if start == -1:
            return pos
        pos.append(start)
        start += len(sub)


def plot_bar_count(dic, title='', xlabel='', ylabel='', xlim=True, low=0, up=10, step=1):
    dic = sorted(dic.items(), key=lambda x: x[0])
    X, Y = [], []
    for item in dic:
        X.append(item[0])
        Y.append(item[1])
    idx = range(len(X))
    plt.bar(x=X, height=Y)
    plt.xlabel(xlabel)  # 设置X轴Y轴名称
    plt.ylabel(ylabel)
    plt.title(title)
    if xlim:
        plt.xlim((-1, dic[-1][0] + 2))
    else:
        my_x_ticks = np.arange(low, X[-1] + 1, step)  # 原始数据有13个点，故此处为设置从0开始，间隔为1
        plt.xticks(my_x_ticks)
    # 使用text显示数值
    for a, b in zip(X, Y):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=11)
    plt.savefig(r'./Graph/{}.jpg'.format(title), dpi=600)
    plt.show()


def check_entity_times(entity, sent):
    # 测试实体在一个句子中是否多次出现
    num = 0
    start = 0
    while True:
        start = sent.find(entity, start)
        if start == -1:
            break
        num += 1
        start += 1
    return num

