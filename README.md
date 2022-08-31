# CCF-BDCI
高端装备制造知识图谱自动化构建技术评测任务代码

## 模型
代码基本模型框架为：BART[^1], 该框架为Encoder-Decoder模块，采用如下方式预训练：
![BART](./BART.png)

但是因为BART为英文模型，因而本文实际使用的模型结构为: bart-chinese[^2][^3],该模型为复旦大学基于BART原模型实现的中文版本。

## 数据
将原始数据里面的"text"取出作为输入数据，将"spo_list"中的"name"分别取出作为输出数据。
```
raw_data:
	"ID": "AT0001",
	"text":"故障现象:车速到100迈以上发动机盖后部随着车速抖动。故障原因简要分析:经技术人员试车；怀疑发动机盖锁或发动机盖铰链松旷。",
	"spo_list":[
		{"h": {"name": "发动机盖", "pos": [14, 18]},
		"t": {"name": "抖动", "pos": [24, 26]},
		"relation": "部件故障"},
		{"h": {"name": "发动机盖锁", "pos": [46, 51]},
		"t": {"name": "松旷", "pos": [58, 60]},
		"relation": "部件故障"},
		{"h": {"name": "发动机盖铰链", "pos": [52, 58]},
		"t": {"name": "松旷", "pos": [58, 60]},
		"relation":"部件故障"}
	]

input_text:
故障现象:车速到100迈以上发动机盖后部随着车速抖动。故障原因简要分析:经技术人员试车；怀疑发动机盖锁或发动机盖铰链松旷。
output_text:
头:发动机盖$$关系:部件故障$$尾:抖动
```

ToDo:

 - 统计头实体，尾实体以及关系长度。
 - 将输入句子按照句号划分个多个字句，然后分别进行预测和识别。


## 参考

[^1]: [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461)

[^2]: [CPT: A Pre-Trained Unbalanced Transformer for Both Chinese Language Understanding and Generation](https://arxiv.org/pdf/2109.05729.pdf)

[^3]: [fnlp/bart-base-chinese](https://huggingface.co/fnlp/bart-base-chinese)
