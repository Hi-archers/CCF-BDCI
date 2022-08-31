import json
import numpy as np

from transformers import BertTokenizer, BartForConditionalGeneration

import matplotlib.pyplot as plt

data_path = "../dataset/train.json"

split_len = np.zeros([700])

start_pos = np.zeros([1300])
end_pos = np.zeros([1300])

max_len = 0

with open(data_path) as f:
    for i,line in enumerate(f):
        one_data = json.loads(line)

        input_text = one_data['text']
        input_texts = input_text.split('。')

        tmp = 0

        for _ in input_texts:
            split_len[len(_)] += 1
            if len(_) >= 512:
                print(_)

        continue

        spos = one_data['spo_list']

        output_text = []

        for spo in spos:
            head = spo['h']['name']
            tail = spo['t']['name']
            relation = spo['relation']

            start_pos[spo['h']['pos'][0]] += 1
            end_pos[spo['h']['pos'][1]] += 1

            start_pos[spo['t']['pos'][0]] += 1
            end_pos[spo['t']['pos'][1]] += 1

            output_text.append("头:" + head + "$$关系:" + relation + "$$尾:" + tail)

print(split_len[512:])

plt.plot(split_len)
plt.title("split sentence len")
plt.show()
