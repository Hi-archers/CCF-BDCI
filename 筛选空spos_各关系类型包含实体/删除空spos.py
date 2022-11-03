import json
import os
import os.path as osp


def filter_empty_spos(root='./', file_name='train_0.json'):
    non_empty = []
    spos_empty = []
    with open(osp.join(root, file_name), encoding='utf-8') as f:
        result = json.load(f)
        print(f'样本个数：{len(result)}')
        for i in result:
            if len(i['spos']) != 0:
                non_empty.append(i)
            else:
                spos_empty.append(i)
        print(f'spos非空样本个数：{len(non_empty)}')
        print(f'spos为空样本个数：{len(spos_empty)}')
    name = file_name.split('.')[0]
    with open(osp.join(root, name + '_spos_nonempty.json'), 'w', encoding='utf-8') as f:
        json.dump(non_empty, f, ensure_ascii=False, indent=2)

    with open(osp.join(root, name + '_spos_empty.json'), 'w', encoding='utf-8') as f:
        json.dump(spos_empty, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    file_name = ['train_0.json', 'train_123.json']
    root = './处理空spos'
    os.makedirs(root, exist_ok=True)
    for name in file_name:
        filter_empty_spos(root=root, file_name=name)
