# -*- coding: utf-8 -*-

"""该文件主要用于“检测工具”关系类型在测试集上的精确匹配"""
import os, sys, re
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


def check(h, t):
    h_begin, h_end, t_begin, t_end = h[0], h[1], t[0], t[1]
    if h_end <= t_begin:
        return True
    elif h_begin >= t_end:
        return True
    else:
        return False  # 有交集


def h_t_index(h_list, t_list):
    for h in h_list:
        for t in t_list:
            if check(h, t):
                return h, t
            else:
                continue
    return None, None


def matching(test_file, relation_root, relation_file):
    json_data, data = load_data(test_file, relation_root, relation_file)
    match_result = []
    for j_data in json_data:
        id, origin_texts = j_data['ID'], j_data['text']
        tmp_result = {"ID": id, 'text': origin_texts, "spo_list": []}
        for d in data:
            h, r, t = d
            if h in origin_texts and t in origin_texts:
                texts = origin_texts.split('。')  # 还未句子划分
                texts = [text + '。' for text in texts]
                # 头尾实体都出现在长文本中
                # 返回头尾实体所有下标
                # h_list = [(m.start(), m.end()) for m in re.finditer(h, text)]
                # t_list = [(m.start(), m.end()) for m in re.finditer(t, text)]
                now_len = 0  # 记录当前句子之前的句子长度
                for i in range(len(texts)):
                    text = texts[i]
                    if h in text and t in text:
                        # 头尾实体都在一个句子中
                        # 把头尾实体在句子中出现的第一个位置作为index
                        h_list = [(m.start() + now_len, m.end() + now_len) for m in re.finditer(h, text)]
                        t_list = [(m.start() + now_len, m.end() + now_len) for m in re.finditer(t, text)]
                        # 有些尾实体会嵌入在头实体内部 比如 钳形电流表——电流
                        # 如果头尾实体存在交集，遍历下一个尾实体
                        h_index, t_index = h_t_index(h_list, t_list)  # #每个句子只返回一组相同的头尾实体，如果要修改，函数里改为list即可
                        if h_index != None:
                            tmp_spo = {"h": {"name": h, "pos": h_index}, "t": {"name": t, "pos": t_index}, "relation": r}
                            tmp_result["spo_list"].append(tmp_spo)
                    now_len += len(text)
        if len(tmp_result["spo_list"]) > 0:
            match_result.append(tmp_result)
    print(f'共找出{len(match_result)}个样本')
    with open('./精确匹配结果.json', 'w', encoding='utf8') as f:
        for i in match_result:
            json.dump(i, f, ensure_ascii=False)
            f.write('\n')


if __name__ == '__main__':
    test_file = 'evalA.json'
    relation_root = './关系类型对应实体类型统计'
    relation_file = '检测工具.xlsx'
    matching(test_file, relation_root, relation_file)


# {"ID": "AE0025", "text": "三、故障排除(1)发生漏电故障后,安装剩余电流动作保护器的线路会跳闸,其他线路漏电时漏电相电流剧增,发生跳闸后抢修人员应立即查明故障原因。(2)及时由 95598抢修平台发布停电信息。(3)查找故障原因。线路跳闸一般由线路短路、混线、断线和漏电故障引起,先对故障线路进行巡视,如未发现短路、混线、断线等异常现象,可以对剩余电流动作保护器装置试送电一次,确定线路是否漏电。(4)查找故障点。先对线路进行巡视有无明显漏电点,如未发现异常可用钳形电流表进行查找,查找电网漏电故障点的方法有不断电检查法和断电查零线法两种。1)不断电检查法:保持电网供电,先测出电网总剩余电流,可用钳形电流表测量变压器中性接地线或总出线上(A、B、C、N三相四线一起钳进钳形电流表钳口);然后去测量各分支线路的漏电电流,查到有漏电的分支再查该分支上每一户的漏电电流,直至查到故障点。该方法适用于电网中有重要用户不能停电,或电网总剩余电流较大(≥500mA)以致于保护器无法投运的线路。其优点是电网不断电,用户的用电不受影响;缺点是检查范围大、工作量大、速度慢。2)断电查零线法:先断电,然后拆下零线,在零线中施加测试电流。具体方法是在零线与配电柜(箱)内有电相线之间串上75W左右的电阻,如电烙铁、灯泡等。a.用钳形电流表测量出零线当中总的测试电流,此电流可以测变压器的中性接地线进行确认;然后爬杆测量零线中测试电流的走向,爬杆测量时A、B、C、N线都要测量,并记下数据以便分析,相线上有测试电流时该相线就有漏电故障点。b.断电查零线法适用于电网总漏电电流小但保护器仍需要频繁跳闸的线路(如单相电机启动保护器就动作);也适用于电网总剩余电流大,保护器不能投运的线路。优点是可根据测量的数据分析出故障电流的走向,具有目标性,可快速查出漏电故障点;零线、相线漏电故障点,零线、相线漏电故障点可同时查出;缺点是电网需停电,影响用户的用电,且需要爬杆测量。c.注意事项:①断电查零线时,断开的为电网的相线和零线,严禁断开变压器的中性接地线;零线断开后电网决不可送电。②在断下的零线上要接通一个250~500mA的测试电流,测试电流不宜过大,登杆测量不安全,且会引起误判。③登杆测量时对A、B、C、N线都要测量,并记下数据,当发现相线上有测试电流时,该路相线一定存在漏电故障点,要仔细测量分析。④查到故障点后,应将故障点的零线和相线同时断开,否则会影响其他故障点的判断。⑤有多个分支的关键点需多次进行测量,线路故障点可能有多处,查到并处理一个故障点后再登杆测量一次,这样可以确定其他故障点的方向。⑥当电网A、B、C三相无测试电流,零线电流呈发射状不往一个方向流时,查找基本可以结束,恢复送电(5)故障点处理安全措施实施。工作许可人应做好线路停电、验电、挂设接地线、悬挂标志牌等安全技术措施,向工作负责人办理许可手续。(6)故障点处理。使用合格的工器具,做好防触电、防高处坠落、防电杆倾倒伤人、防高处坠物伤人等安全措施。漏电处理的步骤:①清除异物,登杆前检查;②登杆处理漏电点。(7)工作负责人对漏电点处理质量进行验收,并符合配电网运行规程要求。(8)拆除现场安全设施,收回工器具、材料并清理现场,工作负责人召开站班会,组织工作人员撤离。(9)工作负责人向工作许可人汇报工作结束,工作许可人拆除所有安全措施后,按操作步骤进行送电。 (10)工作终结,并由95598 抢修平台发布送电信息。"}
