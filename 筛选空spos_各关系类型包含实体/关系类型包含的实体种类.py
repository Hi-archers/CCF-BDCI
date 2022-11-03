"""该文件主要用于关系类型（比如：检测工具）对应的实体类型的统计"""
import json, os
import os.path as osp
import pandas as pd



def count_relation_entity(json_root, save_root, json_file, relation_type):
    """主要用于统计某种关系类型对应的实体统计
    json_root: str json_file 的存储路径
    save_root: str 统计结果存储路径
    json_file: str
    relation_type: list"""
    with open(osp.join(json_root, json_file), encoding='utf-8') as f:
        samples = json.load(f)
    result = {}
    for relation in relation_type:
        result[relation] = []
    for sample in samples:
        for i_spos in sample['spos']:
            head_ent, rel, tail_ent = i_spos  # 头实体，关系，尾实体
            if [head_ent[-1], rel, tail_ent[-1]] not in result[rel]:
                result[rel].append([head_ent[-1], rel, tail_ent[-1]])

    for relation in relation_type:
        tmp = result[relation]
        data = pd.DataFrame(tmp, columns=['头实体', '关系类型', '尾实体'])  #
        print(data.head())
        data.to_excel(osp.join(save_root, relation + '.xlsx'), index=None)


if __name__ == '__main__':
    save_root = './关系类型对应实体类型统计'
    json_root = './处理空spos'
    os.makedirs(save_root, exist_ok=True)
    json_file = ['train_0_spos_nonempty.json', 'train_123_spos_nonempty.json']
    relation_type = [['部件故障'], ['性能故障', '检测工具', '组成']]
    for j_file, r_type in zip(json_file, relation_type):
        count_relation_entity(json_root, save_root, j_file, r_type)
