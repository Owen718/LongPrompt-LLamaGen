import os
import json
from glob import glob
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class SimpleCustomDataset(Dataset):
    def __init__(self, data_dir, length=None):
        self.data = []
        for json_file in glob(os.path.join(data_dir, '**/*.json'), recursive=True):
            with open(json_file, 'r') as f:
                data = json.load(f)
                caption = data.get('long_caption', '')
                # caption = data.get('short_caption', '')
                code_dir = os.path.relpath(os.path.dirname(json_file), data_dir)
                code_name = os.path.splitext(os.path.basename(json_file))[0]
                self.data.append((caption, code_dir, code_name))
            if length is not None:
                if len(self.data) > length:
                    break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

dataset = SimpleCustomDataset('/home/yetian/Datasets/synthetic-dataset-1m-dalle3-high-quality-captions/synthetic-dataset-1m-dalle3-high-quality-captions/data/extracted_data', length=20000)


# ... existing code ...

from tqdm import tqdm
from transformers import T5Tokenizer
import numpy as np

symbol_lengths = []
count = 0
for data in tqdm(dataset):
    # 计算字符数量（包括空格和标点符号）
    symbol_length = len(data[0])
    symbol_lengths.append(symbol_length)
    count += 1

# 计算统计数据
average_length = np.mean(symbol_lengths) if count > 0 else 0
std_dev = np.std(symbol_lengths)

print(f"平均caption符号长度: {average_length:.2f}", "标准差:", std_dev)

# 绘制柱状图
plt.figure(figsize=(12, 6))
plt.hist(symbol_lengths, bins=50, edgecolor='black')
plt.title('Caption Symbol Length Distribution')
plt.xlabel('Symbol Length')
plt.ylabel('Frequency')

# 添加平均值和标准差的标注
plt.axvline(average_length, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {average_length:.2f}')
plt.axvline(average_length + std_dev, color='g', linestyle='dotted', linewidth=2, label=f'Mean + Std Dev: {average_length + std_dev:.2f}')
plt.axvline(average_length - std_dev, color='g', linestyle='dotted', linewidth=2, label=f'Mean - Std Dev: {average_length - std_dev:.2f}')

plt.legend()
plt.tight_layout()
plt.savefig('symbol_length_distribution.png')
plt.close()

# ... rest of the existing code ...







# # 初始化T5Tokenizer
# tokenizer = T5Tokenizer.from_pretrained("/home/yetian/Project/LlamaGen/pretrained_models/t5-ckpt")

# token_lengths = []
# word_lengths = []
# count = 0

# for data in tqdm(dataset):
#     # if count%1000 == 0:
#     #     print(data[0])
#     # 使用T5Tokenizer计算token长度
#     text_tokens = tokenizer(
#             data[0],
#             max_length=1024,
#             truncation=True,
#             return_attention_mask=False,
#             add_special_tokens=True,
#             return_tensors='pt'
#         )['input_ids']
#     # import pdb; pdb.set_trace()
#     word_length = len(data[0].split(' '))
#     token_length = text_tokens.flatten().size()[0]
#     word_lengths.append(word_length)
#     token_lengths.append(token_length)
#     count += 1

# # 计算统计数据
# average_length = np.mean(token_lengths) if count > 0 else 0
# std_dev = np.std(token_lengths)

# print(f"平均caption token长度: {average_length:.2f}", "标准差:", std_dev)
# print(f"平均caption单词长度: {np.mean(word_lengths):.2f}", "标准差:", np.std(word_lengths))

# # 绘制柱状图
# plt.figure(figsize=(12, 6))
# plt.hist(token_lengths, bins=50, edgecolor='black')
# plt.title('Caption Token Length Distribution')
# plt.xlabel('Token Length')
# plt.ylabel('Frequency')

# # 添加平均值和标准差的标注
# plt.axvline(average_length, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {average_length:.2f}')
# plt.axvline(average_length + std_dev, color='g', linestyle='dotted', linewidth=2, label=f'Mean + Std Dev: {average_length + std_dev:.2f}')
# plt.axvline(average_length - std_dev, color='g', linestyle='dotted', linewidth=2, label=f'Mean - Std Dev: {average_length - std_dev:.2f}')

# plt.legend()
# plt.tight_layout()
# plt.savefig('token_length_distribution.png')
# plt.close()



# total_words = 0
# word_count_list = []
# count = 0
# from tqdm import tqdm
# import re
# for data in tqdm(dataset):
#     line = data[0]
#     word_count = len(re.findall(r'\w+', line))
#     total_words += word_count
#     word_count_list.append(word_count)
#     count += 1

#     if word_count < 150:
#         import pdb; pdb.set_trace()
#         print(data[0])

# average_word_count = total_words / count if count > 0 else 0
# import numpy as np

# print(f"平均caption单词数: {average_word_count:.2f}", "标准差:", np.std(word_count_list))
