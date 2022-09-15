import json

from analysis_utils import *
from collections import Counter


def train_length():
    # 1. 训练集的样本长度
    train_json = load_data(data_file='train.json')
    ## 样本文本长度
    train_text = []
    spo_list = []
    for sample in train_json:
        train_text.append(sample['text'])
        spo_list.append(sample['spo_list'])

    text_len = return_length(train_text)
    sorted_text_len = sorted(text_len)
    print(f'Train 最短text长度: {sorted_text_len[0]}')
    print(f'Train 最长text长度: {sorted_text_len[-1]}')
    plot_bar(sorted_text_len, title='Train_text_length', xlabel='', ylabel='Text Length')

    ## 句子长度
    sent_lists = [list(filter(lambda x: x, text.split('。'))) for text in train_text]
    sent_list = []
    for sent in sent_lists:
        sent_list = sent_list + sent
    sent_len = return_length(sent_list)
    sorted_sent_len = sorted(sent_len)
    print(f'Train 最短句子长度: {sorted_sent_len[0]}')
    print(f'Train 最长句子长度: {sorted_sent_len[-1]}')
    plot_bar(sorted_sent_len, title='Train_sentence_length', xlabel='', ylabel='Sent Length')


def valid_length():
    # 2. 验证集的样本长度
    valid_json = load_data(data_file='evalA.json')
    ## 样本文本长度
    valid_text = [sample['text'] for sample in valid_json]

    text_len = return_length(valid_text)
    sorted_text_len = sorted(text_len)
    print(f'Valid 最短text长度: {sorted_text_len[0]}')
    print(f'Valid 最长text长度: {sorted_text_len[-1]}')
    plot_bar(sorted_text_len, title='Valid_text_length', xlabel='', ylabel='Text Length')

    ## 句子长度
    sent_lists = [list(filter(lambda x: x, text.split('。'))) for text in valid_text]
    sent_list = []
    for sent in sent_lists:
        sent_list = sent_list + sent
    sent_len = return_length(sent_list)
    sorted_sent_len = sorted(sent_len)
    print(f'Valid 最短句子长度: {sorted_sent_len[0]}')
    print(f'Valid 最长句子长度: {sorted_sent_len[-1]}')
    plot_bar(sorted_sent_len, title='Valid_sentence_length', xlabel='', ylabel='Sent Length')


def entity_pos():
    """验证每个关系（头实体，尾实体，关系）只在一个句子中出现"""
    data_file = 'train.json'
    train_json = load_data(data_file)
    sent_rela = {}  # 记录每个句子包含三元组的情况
    sent_entity = {}  # 记录头尾实体出现在几个句子中
    save_sent_relation = {}
    save_sent_entity = {}
    relations = []
    hname_list = []  #
    tname_list = []
    entity_dic = {}
    hnum_cnt = []
    tnum_cnt = []
    for sample in train_json:
        sents = list(filter(lambda x: x, sample['text'].split('。')))
        # train_dic[id]['period'] = find_all(sample['text'], sub='。')  # 返回text中所有句号的位置
        # train_dic[id]['num_sent'] = len(train_dic[id]['sent'])  # 句子个数
        entity = {}
        for id_i, sent in enumerate(sents):
            rela_num = 0  # 该句子中关系的个数
            now_id = sample['ID'] + '_' + str(id_i)
            for idx, spo_dic in enumerate(sample['spo_list']):
                h_name, h_pos = spo_dic['h']['name'], spo_dic['h']['pos']
                t_name, t_pos = spo_dic['t']['name'], spo_dic['t']['pos']
                hname_list.append(h_name)
                tname_list.append(t_name)
                relations.append(spo_dic['relation'])
                if sent.find(h_name) != -1 and sent.find(t_name) != -1:
                    # 头尾实体在同一个句子内
                    rela_num += 1
                    entity[idx] = entity.setdefault(idx, 0) + 1

                    # 在头尾实体在同一个句子内情况下，判断一个实体在一个句子中是否多次出现
                    h_num = check_entity_times(h_name, sent)
                    t_num = check_entity_times(t_name, sent)

                    hnum_cnt.append(h_num)
                    tnum_cnt.append(t_num)
                    if h_num >= 2 or t_num >= 2:
                        # 一个实体在一个句子中出现那两次以上
                        entity_dic[now_id] = {}
                        if h_num >= 2:
                            entity_dic[now_id].setdefault('多次出现的头实体', [])
                            entity_dic[now_id]['多次出现的头实体'].append(
                                {"hname": h_name, "times": h_num, "sent": sent})
                        if t_num >= 2:
                            entity_dic[now_id].setdefault('多次出现尾实体', [])
                            entity_dic[now_id]['多次出现尾实体'].append({"tname": t_name, "times": t_num, "sent": sent})

            if rela_num >= 2:
                save_sent_relation[now_id] = {}
                save_sent_relation[now_id]['sentence'] = sent
                save_sent_relation[now_id]['spo_list'] = sample["spo_list"]
            for key, value in entity.items():
                if value >= 2:
                    save_sent_entity[now_id] = {}
                    save_sent_entity[now_id]['text'] = sample['text']
                    save_sent_entity[now_id]['spo'] = sample['spo_list'][key]
            sent_rela[rela_num] = sent_rela.setdefault(rela_num, 0) + 1
            for cnt in entity.values():
                sent_entity[cnt] = sent_entity.setdefault(cnt, 0) + 1

    json_str = json.dumps(save_sent_relation, indent=4, ensure_ascii=False)
    with open('./save_file/一个句子包含多个关系.json', 'w', encoding='utf-8') as f:
        f.write(json_str)
        print('success save file!')

    json_str = json.dumps(save_sent_entity, indent=4, ensure_ascii=False)
    with open('./save_file/一个实体关系蕴含在多个句子中.json', 'w', encoding='utf-8') as f:
        f.write(json_str)
        print('success save file!')

    with open('./save_file/训练集中包含的relations.txt', 'w', encoding='utf-8') as f:
        relations = list(set(relations))
        f.write(f'一共有{len(relations)}种关系。\n')
        for rel in relations:
            f.write(rel)
            f.write('\n')
        print('success save file!')

    # 对头尾实体去重

    hname_dic = dict(Counter([len(h) for h in list(set(hname_list))]))
    tname_dic = dict(Counter([len(t) for t in list(set(tname_list))]))
    plot_bar_count(hname_dic, title='训练集头实体长度(去重)', xlabel='头实体长度', ylabel='个数', xlim=False, low=0,
                   step=1)
    plot_bar_count(tname_dic, title='训练集尾实体长度(去重)', xlabel='尾实体长度', ylabel='个数', xlim=False, low=0,
                   step=1)
    plot_bar_count(sent_rela, title='训练集中句子包含三元组的情况', xlabel='包含三元组个数', ylabel='句子个数')
    plot_bar_count(sent_entity, title='头尾实体出现在几个句子中', xlabel='出现在句子中的个数', ylabel='实体个数')

    plot_bar_count(dict(Counter(hnum_cnt)), title='头尾实体在同一个句子内情况下，头实体在一个句子中出现次数',
                   xlabel='在一个句子中出现次数', ylabel='Count')
    plot_bar_count(dict(Counter(tnum_cnt)), title='头尾实体在同一个句子内情况下，尾实体在一个句子中出现次数',
                   xlabel='在一个句子中出现次数', ylabel='Count')


def check_entity_sentence():
    """检查是否有跨实体出现的情况"""
    data_file = 'train.json'
    train_json = load_data(data_file)
    cnt_num = {}
    error = []
    for data in train_json:
        text = data['text']
        spo_list = data['spo_list']
        for spo in spo_list:
            sent = text[spo['h']['pos'][0]:spo['t']['pos'][1]]
            if sent.find('。') != -1:
                # 在实体之间找到其他句号，即两个实体出现在不同的句子内
                cnt_num['跨越句子'] = cnt_num.setdefault('跨越句子', 0) + 1
                error.append([data['ID'], text, str(spo)])
            else:
                cnt_num['句子内'] = cnt_num.setdefault('句子内', 0) + 1
    with open('save_file/实体跨越句子.txt', 'w', encoding='utf-8') as f:
        for e in error:
            f.write(','.join(e))
            f.write('\n')
    plt.bar(x=[0, 1], height=list(cnt_num.values()), tick_label=['实体在句子内部个数', '实体跨越句子个数'])
    for a, b in zip([0, 1], list(cnt_num.values())):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=11)
    plt.ylabel('实体个数')
    plt.savefig('./Graph/实体跨越句子统计.jpg', dpi=600)
    plt.show()


def check_entity_intersection():
    """检查是否有实体嵌套的情况出现"""
    data_file = 'train.json'
    train_json = load_data(data_file)
    cnt = {}
    for data in train_json:
        spo_list = data['spo_list']
        spos = [(spo['h']['pos'], spo['t']['pos']) for spo in spo_list]
        if len(spos) >= 2:
            for i in range(len(spos)):
                spoi = spos[i]
                for j in range(i + 1, len(spos)):
                    spoj = spos[j]
                    # 完全嵌套情况
                    if (spoi[0][0] >= spoj[0][0] and spoi[1][1] <= spoj[1][1]) or (
                            spoi[0][0] <= spoj[0][0] and spoi[1][1] >= spoj[1][1]):
                        cnt['完全嵌套'] = cnt.setdefault('完全嵌套', 0) + 1
                    elif (spoi[0][1] < spoj[0][0] and spoi[1][0] >= spoj[0][1] and spoi[1][1] <= spoj[1][0]) or (
                            spoi[0][0] > spoj[0][1] and spoi[0][0] < spoj[1][0] and spoi[1][0] > spoj[1][1]):
                        cnt['部分嵌套'] = cnt.setdefault('部分嵌套', 0) + 1
                    else:
                        cnt['没有嵌套'] = cnt.setdefault('没有嵌套', 0) + 1
        else:
            cnt['没有嵌套'] = cnt.setdefault('没有嵌套', 0) + 1
    x = [0, 1, 2]
    y = list(cnt.values())
    tick_label = list(cnt.keys())
    plt.bar(x=x, height=y, tick_label=tick_label)
    for a, b in zip(x, y):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=11)
    plt.ylabel('实体个数')
    plt.savefig('./Graph/实体嵌套情况.jpg', dpi=600)
    plt.show()


if __name__ == '__main__':
    # train_length()
    # valid_length()
    # entity_pos()
    # check_entity_sentence()
    check_entity_intersection()
