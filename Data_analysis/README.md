# 原始数据集的基本数据分析

**数据分析可能代码有错，有问题及时反应**

## 1. 训练集的句子长度

部分样本的Text文本过长

<div align=center><img src="./Graph/Train_text_length.jpg" width=50% height=50% alt="训练集文本(Text)长度"></div>

因此对Text文本按照"。"划分，得到句子的长度：
<div align=center><img src="./Graph/Train_sentence_length.jpg" width = 50% height = 50% /></div>

## 2. 验证集的句子长度

验证集Text长度
<div align=center><img src="./Graph/Valid_text_length.jpg" width=50% height=50% alt=""></div>

验证集Sentence长度
<div align=center><img src="./Graph/Valid_sentence_length.jpg" width=50% height=50% alt=""></div>


<!-- ## 3. 验证头尾实体是否只会出现在一个完整的句子中
<div align=center><img src="./Graph/%E5%A4%B4%E5%B0%BE%E5%AE%9E%E4%BD%93%E5%87%BA%E7%8E%B0%E5%9C%A8%E5%87%A0%E4%B8%AA%E5%8F%A5%E5%AD%90%E4%B8%AD.jpg" width=70% height=70% alt=""></div>

上图的含义：每个三元组的头尾实体，出现在一段Text中的几个句子里。从图中可以看出，每个三元组的头尾实体至少会出现在Text里的一个句子里。
同时，**每个三元组可能会出现在一段Text的多个句子中**
[具体文件](./save_file/一个实体关系蕴含在多个句子中.json) -->

<!-- ## 4. 同一个句子中实体是否多次出现

当头尾实体在同一个句子中时，确实会出现实体在该句内多次出现
<div align=center><img src="./Graph/头尾实体在同一个句子内情况下，头实体在一个句子中出现次数.jpg" width=70% height=70% alt=""></div>
<div align=center><img src="./Graph/头尾实体在同一个句子内情况下，尾实体在一个句子中出现次数.jpg" width=70% height=70% alt=""></div> -->

## 5. 训练集中一共包含的relation种类

[具体文件](./save_file/训练集中包含的relations.txt)

> 一共有4种关系。
> 检测工具
> 性能故障
> 组成
> 部件故障

## 6. 头尾实体的长度

**Notes: 这是去重后统计的结果**
<div align=center><img src="./Graph/训练集头实体长度(去重).jpg" width=50% height=50% alt=""></div>
<div align=center><img src="./Graph/训练集尾实体长度(去重).jpg" width=50% height=50% alt=""></div>

<!-- # 7. 是不是每个句子都有一种关系

有大量句子不包含三元组

> [具体文件](./save_file/一个句子包含多个关系.json)

<div align=center><img src="./Graph/训练集中句子包含三元组的情况.jpg" width=70% height=70% alt=""></div> -->

## 8. 检查是否有跨实体出现的情况
<div align=center><img src="./Graph/实体跨越句子统计.jpg" width=50% height=50% alt=""></div>

## 9. 检查是否有实体嵌套的情况出现
<div align=center><img src="./Graph/实体嵌套情况.jpg" width=50% height=50% alt=""></div>
具体检查嵌套关系的思路
<div align=center><img src="./Graph/嵌套关系.jpg" width=50% height=50% alt=""></div>
