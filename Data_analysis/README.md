# 原始数据集的基本数据分析
## 1. 训练集的句子长度
部分样本的Text文本过长
![训练集文本(Text)长度](./Graph/Train_text_length.jpg "训练集文本(Text)长度")
因此对Text文本按照"。"划分，得到句子的长度：
![训练集句子(Sentence)长度](./Graph/Train_sentence_length.jpg)

## 2. 验证集的句子长度
验证集Text长度
![](./Graph/Valid_text_length.jpg)
验证集Sentence长度
![](./Graph/Valid_sentence_length.jpg)

## 3. 验证头尾实体是否只会出现在一个完整的句子中
![](./Graph/%E5%A4%B4%E5%B0%BE%E5%AE%9E%E4%BD%93%E5%87%BA%E7%8E%B0%E5%9C%A8%E5%87%A0%E4%B8%AA%E5%8F%A5%E5%AD%90%E4%B8%AD.jpg)
上图的含义：每个三元组的头尾实体，出现在一段Text中的几个句子里。从图中可以看出，每个三元组的头尾实体至少会出现在Text里的一个句子里。

## 4. 同一个句子中实体是否多次出现
当头尾实体在同一个句子中时，确实会出现实体在该句内多次出现
![](./Graph/头尾实体在同一个句子内情况下，头实体在一个句子中出现次数.jpg)
![](./Graph/头尾实体在同一个句子内情况下，尾实体在一个句子中出现次数.jpg)

## 5. 训练集中一共包含的relation种类
[具体文件](./save_file/训练集中包含的relations.txt)
>一共有4种关系。  
检测工具  
性能故障  
组成  
部件故障

## 6. 头尾实体的长度
**Notes: 这是去重后统计的结果**
![](./Graph/训练集头实体长度(去重).jpg)
![](./Graph/训练集尾实体长度(去重).jpg)

# 7. 是不是每个句子都有一种关系
有大量句子不包含三元组
![](./Graph/训练集中句子包含三元组的情况.jpg)
