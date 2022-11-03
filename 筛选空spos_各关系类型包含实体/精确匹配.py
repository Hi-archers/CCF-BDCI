"""该文件主要用于“检测工具”关系类型在测试集上的精确匹配"""
import re
import os.path as osp
import json
import pandas as pd


def load_data(test_file, relation_root, relation_file):
    # 读取测试集
    json_data = []
    for line in open(test_file, 'r', encoding='utf-8'):
        json_data.append(json.loads(line))
    # 读取关系类型对应的实体
    data = pd.read_excel(osp.join(relation_root, relation_file)).values

    return json_data, data


def matching(test_file, relation_root, relation_file):
    json_data, data = load_data(test_file, relation_root, relation_file)
    print()
    for j_data in json_data:
        id, texts = j_data['ID'], j_data['text']
        text = texts  # 还未句子划分
        for d in data:
            h, r, t = d
            if h in text and t in text:
                # 头尾实体都出现在text中
                # 返回头尾实体所有下标
                h_list = [(m.start(), m.end()) for m in re.finditer(h, text)]
                t_list = [(m.start(), m.end()) for m in re.finditer(t, text)]
                print(1)


if __name__ == '__main__':
    test_file = 'evalA.json'
    relation_root = './关系类型对应实体类型统计'
    relation_file = '检测工具.xlsx'
    matching(test_file, relation_root, relation_file)
