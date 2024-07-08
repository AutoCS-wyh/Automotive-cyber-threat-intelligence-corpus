#!/usr/bin/env python
# coding: utf-8

# #### 1、导包

# In[ ]:


import paddle
import time
import os
from tqdm import tqdm
import paddle.nn.functional as F
import paddle.nn as nn
from paddlenlp.datasets import load_dataset
import paddlenlp
from paddlenlp.datasets import MapDataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.layers import LinearChainCrf, ViterbiDecoder, LinearChainCrfLoss
# from paddlenlp.metrics import ChunkEvaluator
import numpy as np
from paddle.optimizer import RMSProp
# 导入自定义的工具类
from utils import *
from ChunkEvaluator import *
# In[ ]:
# 从dev.txt和train.txt文件生成dic
# gernate_dic('data1/dev.txt', 'data1/train.txt', 'data1/test.txt','data1/tag.dic')
gernate_dic('data1/dev.txt', 'data1/train.txt','data1/tag.dic')
id2label,label2id,label_list = load_dicts('data1/tag.dic')
print(id2label)
print(label2id)
print(label_list)

# 输出所有的实体类别，分类评估时候会用到
entity_types = set([i.split('_')[1] for i in label_list if i!='O'])
print(entity_types)
# In[ ]:
# 输出所有关系，分类评估会用到
relation_types = set([i.split('_')[2] for i in label_list if i!='O' and len(i.split('_'))>2])
print(relation_types)

label_map=label2id
def read(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        # 跳过列名
        for line in f.readlines():
            words, labels = line.strip('\n').split('\t')
            words = words.split()
            labels = labels.split()
            labels=[label_map[item] for item in labels]
            yield {'tokens': words, 'labels': labels}

# data_path为read()方法的参数
train_ds = load_dataset(read, data_path='data1/train.txt',lazy=False)
dev_ds = load_dataset(read, data_path='data1/dev.txt',lazy=True)
# test_ds = load_dataset(read, data_path='data1/test.txt',lazy=True)

# #### 2、构建词典

# In[ ]:
label_vocab = {label:label_id for label_id, label in enumerate(label_list)}
words = set()
word_vocab = []
for item in train_ds:
    word_vocab += item['tokens'] 
word_vocab = {k:v+2 for v,k in enumerate(set(word_vocab))}
word_vocab['PAD'] = 0
word_vocab['OOV'] = 1

chars = set()
for item in train_ds:
    for word in item['tokens']:
        chars.update(word)
char_vocab = {k: v+2 for v, k in enumerate(sorted(chars))}
char_vocab['PAD'] = 0
char_vocab['OOV'] = 1
# #### 3、把文本和label映射成id，处理成模型的输入的形式

# In[ ]:
# 1、转换tokens为ids并生成mask
def convert_tokens_to_ids_and_mask(tokens, vocab, oov_token='OOV'):
    token_ids = []
    mask = []
    oov_id = vocab.get(oov_token) if oov_token else None
    for token in tokens:
        token_id = vocab.get(token, oov_id)
        token_ids.append(token_id)
        mask.append(int(token_id is not None))
    return token_ids, mask

max_char_len = 15  # you can adjust this based on your actual data

def convert_word_to_char_ids_and_mask(word, char_vocab, oov_char='OOV'):
    char_ids = [char_vocab.get(char, char_vocab[oov_char]) for char in word[:max_char_len]]
    char_mask = [int(char_id is not None) for char_id in char_ids]
    # If the word is shorter than max_char_len, pad it
    char_ids += [char_vocab['PAD']] * (max_char_len - len(char_ids))
    char_mask += [0] * (max_char_len - len(char_mask))
    return char_ids, char_mask

# 3、修改convert_example
def convert_example(example):
    tokens, labels = example['tokens'], example['labels']
    token_ids, token_mask = convert_tokens_to_ids_and_mask(tokens, word_vocab, 'OOV')
    char_ids_and_mask = [convert_word_to_char_ids_and_mask(word, char_vocab, 'OOV') for word in tokens]
    char_ids, char_mask = zip(*char_ids_and_mask)
    label_ids = labels
    return np.array(token_ids), np.array(token_mask), np.array(char_ids), np.array(char_mask), len(token_ids), np.array(label_ids)

train_ds.map(convert_example)
dev_ds.map(convert_example)
# test_ds.map(convert_example)


# #### 4、加载数据集train_loader和dev_loader和test_loader

# In[ ]:
# 4、修改batchify_fn
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=word_vocab.get('OOV')),  # token_ids
    Pad(axis=0, pad_val=0),  # token_mask
    Pad(axis=0, pad_val=char_vocab.get('PAD')),  # char_ids
    Pad(axis=0, pad_val=0),  # char_mask
    Stack(),  # seq_len
    Pad(axis=0, pad_val=label_vocab.get('O'))  # label_ids
): fn(samples)

train_loader = paddle.io.DataLoader(
        dataset=train_ds,
        batch_size=16,
        shuffle=True,
        drop_last=True,
        return_list=True,
        collate_fn=batchify_fn)

dev_loader = paddle.io.DataLoader(
        dataset=dev_ds,
        batch_size=16,
        drop_last=True,
        return_list=True,
        collate_fn=batchify_fn)

# test_loader = paddle.io.DataLoader(
#         dataset=test_ds,
#         batch_size=16,
#         drop_last=True,
#         return_list=True,
#         collate_fn=batchify_fn)


# #### 5、定义模型初始化方法

class SelfAttention(nn.Layer):
    def __init__(self, hidden_size, init_scale=0.01):
        super(SelfAttention, self).__init__()
        self.query = self.create_parameter(shape=[hidden_size, hidden_size], dtype="float32",
                                           default_initializer=nn.initializer.Uniform(low=-init_scale, high=init_scale))
        self.key = self.create_parameter(shape=[hidden_size, hidden_size], dtype="float32",
                                         default_initializer=nn.initializer.Uniform(low=-init_scale, high=init_scale))
        self.value = self.create_parameter(shape=[hidden_size, hidden_size], dtype="float32",
                                           default_initializer=nn.initializer.Uniform(low=-init_scale, high=init_scale))
        self.scale = hidden_size ** 0.5

    def forward(self, inputs, mask):
        # inputs: [batch_size, seq_len, hidden_size]
        queries = paddle.matmul(inputs, self.query)  # [batch_size, seq_len, hidden_size]
        keys = paddle.matmul(inputs, self.key)  # [batch_size, seq_len, hidden_size]
        values = paddle.matmul(inputs, self.value)  # [batch_size, seq_len, hidden_size]

        # Scaled dot-product attention
        attn_weights = paddle.matmul(queries, keys, transpose_y=True) / self.scale  # [batch_size, seq_len, seq_len]

        # Apply mask
        attn_weights = paddle.where(mask.unsqueeze(1) == 0, paddle.full_like(attn_weights, float('-inf')), attn_weights)

        # Softmax
        attn_weights = F.softmax(attn_weights, axis=-1)  # [batch_size, seq_len, seq_len]

        # Weighted sum of values
        attn_output = paddle.matmul(attn_weights, values)  # [batch_size, seq_len, hidden_size]

        return attn_output


    # 6、修改BiLSTMCRF
alpha_value = 10# 设置你的超参数值
class BiLSTMCRF(nn.Layer):
    def __init__(self, emb_size, hidden_size, word_num, label_num, char_vocab_size, char_emb_dim, use_w2v_emb=True, alpha=alpha_value):
        super(BiLSTMCRF, self).__init__()
        self.word_emb = nn.Embedding(word_num, emb_size)
        self.char_emb = nn.Embedding(char_vocab_size, char_emb_dim)
        self.kernel_size = 5
        self.padding = (self.kernel_size - 1) // 2
        self.cnn = nn.Conv2D(in_channels=char_emb_dim, out_channels=char_emb_dim, kernel_size=self.kernel_size,
                             padding=self.padding)
        self.bilstm = nn.LSTM(emb_size + char_emb_dim, hidden_size, num_layers=1, direction='bidirectional', dropout=0.1)
        self.att = SelfAttention(hidden_size * 2)
        self.bigru = nn.GRU(hidden_size * 2, hidden_size, num_layers=1, direction='forward', dropout=0.1)
        self.lstm = nn.LSTM(hidden_size, hidden_size* 2, num_layers=1, direction='forward', dropout=0.1)
        self.fc = nn.Linear(hidden_size* 2, label_num)
        self.o_label_index = 195  # you should set this according to your label dictionary
        self.alpha = alpha

    def forward(self, input_ids, input_mask, char_input_ids, char_mask, lengths=None, labels=None):
        word_embs = self.word_emb(input_ids)
        char_embs = self.char_emb(char_input_ids)
        cnn_output = self.cnn(char_embs.transpose([0, 2, 1, 3]))
        # Change the shape back to [batch_size, seq_len, seq_len, char_emb_dim]
        cnn_output = cnn_output.transpose([0, 2, 3, 1])
        # We assume that the output of cnn is [batch_size, seq_len, seq_len, char_emb_dim]
        # We take the maximum value over the third dimension to get the final character embeddings
        char_embs = paddle.max(cnn_output, axis=2)
        concat_output = paddle.concat([word_embs, char_embs], axis=-1)
        output, _ = self.bilstm(concat_output)
        output = self.att(output, input_mask)
        output, _ = self.bigru(output)
        output, _ = self.lstm(output)
        output = self.fc(output)

        if labels is not None:
            # loss = self.new_loss(output, lengths, labels, self.alpha)
            loss = self.custom_loss_fn(output, labels)
            return loss
        else:
            predictions = paddle.argmax(output, axis=-1)
            return predictions, lengths

    def custom_loss_fn(self, logits, labels):
        self.loss_fn = paddle.nn.CrossEntropyLoss(reduction='none')  # Initialize CrossEntropyLoss
        standard_loss = self.loss_fn(logits, labels)  # Compute standard CrossEntropyLoss
        is_o_label = (labels == self.o_label_index).astype(paddle.float32)  # Identify 'O' labels
        adjusted_loss = (1 - is_o_label) * self.alpha * standard_loss + is_o_label * standard_loss  # Adjust loss
        return adjusted_loss.sum()

char_vocab_size = len(char_vocab)
char_emb_dim = 15
#model = BiLSTMCRF(200, 200, len(word_vocab), len(label_vocab), char_vocab_size, char_emb_dim)

model = BiLSTMCRF(300, 300, len(word_vocab), len(label_vocab), char_vocab_size, char_emb_dim, alpha=alpha_value)
# optimizer = paddle.optimizer.RMSProp(learning_rate=0.001, parameters=model.parameters(), rho=0.99, epsilon=1e-06)

# In[ ]:
# 优化器使用Adam算法
optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
# 设置评估函数
metric = ChunkEvaluator(label_list=label_vocab.keys(), suffix=True)
# #### 6、模型训练、保存和评估
def predict(text):
    tokens = text.split()
    lens = len(tokens)
    token_ids, token_mask, char_ids, char_mask, len_token_ids, label_ids = convert_example(
        {'tokens': tokens, 'labels': [0] * lens})

    model.eval()
    output = model(paddle.to_tensor(np.array([token_ids])),
                   paddle.to_tensor(np.array([token_mask])),
                   paddle.to_tensor(np.array([char_ids])),
                   paddle.to_tensor(np.array([char_mask])))

    output = output[0]
    preds = output.numpy().tolist()[0]

    # 将预测的整数编码转换为字符串标签
    #print(label_map)
    #print(preds)
    # 创建一个反向标签映射字典
    reverse_label_map = id2label
    #reverse_label_map = {v: k for k, v in label_map.items()}
    preds = [reverse_label_map[pred] for pred in preds]
   # preds = [label_map[pred] for pred in preds]
    # 返回输入的标记序列和预测的标签
    return tokens, preds

# 遍历所有实体类型，构建每个实体类别的TP，FP，FN
entity_counts = {}
for entity_type in entity_types:
    entity_counts[entity_type] = {"TP": 0, "FP": 0, "FN": 0}

# In[ ]:


# 遍历所有关系类型，构建每个实体类别的TP，FP，FN
relation_counts = {}
for relation_type in relation_types:
    relation_counts[relation_type] = {"TP": 0, "FP": 0, "FN": 0}

# In[ ]:


entity_counts

# In[ ]:


relation_counts


# In[ ]:


# 分类评估
def evaluate_classification(model, file_path):
    model.eval()
    # 读取所有数据
    lines = [i.strip('\n') for i in open(file_path, 'r', encoding='utf-8').readlines()]
    # 逐行遍历
    for line in tqdm(lines):
        tmp_list = line.split('\t')
        sentence = tmp_list[0]
        tag = tmp_list[1]
        # 第一步：取出标注的标签，相当于y
        sentence_list = sentence.split()
        tag_list = tag.split()

        # 第二步，取出模型预测的y
        sent, tags = predict(' '.join(sentence_list))
        tag_list = tag_list[0:len(tags)]
        # 遍历，二者进行比较
        for index, tag in enumerate(tag_list):
            if tag == 'O' and tags[index] == 'O':
                continue
            else:
                # 先评估实体
                # FP：标注是O，预测的不是O，则属于FP
                if tag == 'O':
                    entity_type1 = tags[index].split('_')[1]
                    entity_counts[entity_type1]['FP'] += 1
                # FN：预测是O，标注是不是O，，则属于FN
                elif tags[index] == 'O':
                    entity_type2 = tag.split('_')[1]
                    entity_counts[entity_type2]['FN'] += 1
                elif tag.split('_')[1] == tags[index].split('_')[1]:  # TP：标注和预测一样
                    entity_type = tag.split('_')[1]
                    entity_counts[entity_type]['TP'] += 1
                else:
                    # 标注和预测都不是O且二者不一样
                    entity_type3 = tags[index].split('_')[1]
                    entity_counts[entity_type3]['FN'] += 1

                    entity_type4 = tag.split('_')[1]
                    entity_counts[entity_type4]['FP'] += 1

                # 评估关系
                # TP：标注和预测一样
                if len(tag.split('_')) >= 3 and len(tags[index].split('_')) >= 3:  # 标注的和预测的都有关系
                    if tag.split('_')[2] == tags[index].split('_')[2]:
                        relation_type = tag.split('_')[2]
                        relation_counts[relation_type]['TP'] += 1
                    else:
                        # FP：标注是O，预测的不是O，则属于FP
                        if tag == 'O':
                            relation_type1 = tags[index].split('_')[2]
                            relation_counts[relation_type1]['FP'] += 1
                        # FN：预测是O，标注是不是O，，则属于FN
                        elif tags[index] == 'O':
                            relation_type2 = tag.split('_')[2]
                            relation_counts[relation_type2]['FN'] += 1
                        else:
                            # 标注和预测都不是O且二者不一样
                            relation_type3 = tags[index].split('_')[2]
                            relation_counts[relation_type3]['FN'] += 1

                            relation_type4 = tag.split('_')[2]
                            relation_counts[relation_type4]['FP'] += 1
                # 标注没有关系，预测有关系
                elif len(tag.split('_')) == 2 and len(tags[index].split('_')) >= 3:
                    relation_type1 = tags[index].split('_')[2]
                    relation_counts[relation_type1]['FP'] += 1
                # 标注有关系，预测没有
                elif len(tag.split('_')) >= 3 and len(tags[index].split('_')) == 2:
                    relation_type2 = tag.split('_')[2]
                    relation_counts[relation_type2]['FN'] += 1
    evaluation_parameters = {}
    # 计算每种实体类型的精确率、召回率和F1值，并输出结果
    print("-----------------开始评估实体---------------------")
    for entity_type in entity_counts:
        TP = entity_counts[entity_type]["TP"]
        FP = entity_counts[entity_type]["FP"]
        FN = entity_counts[entity_type]["FN"]
        if TP + FP == 0:
            precision = 0.0
        else:
            precision = TP / (TP + FP)
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        if precision + recall == 0:
            F1 = 0.0
        else:
            F1 = 2 * precision * recall / (precision + recall)
        evaluation_parameters[entity_type] = []
        evaluation_parameters[entity_type].extend([precision * 100, recall * 100, F1])
        print("{}: precision={:.2f}%, recall={:.2f}%, F1={:.2f}%".format(entity_type, precision * 100, recall * 100,
                                                                         F1 * 100))
    print("-----------------评估实体结束---------------------")
    # 计算每种关系类型的精确率、召回率和F1值，并输出结果
    print("-----------------开始评估关系---------------------")
    for relation_type in relation_counts:
        TP = relation_counts[relation_type]["TP"]
        FP = relation_counts[relation_type]["FP"]
        FN = relation_counts[relation_type]["FN"]
        if TP + FP == 0:
            precision = 0.0
        else:
            precision = TP / (TP + FP)
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        if precision + recall == 0:
            F1 = 0.0
        else:
            F1 = 2 * precision * recall / (precision + recall)
        evaluation_parameters[relation_type] = []
        evaluation_parameters[relation_type].extend([precision * 100, recall * 100, F1])
        print("{}: precision={:.2f}%, recall={:.2f}%, F1={:.2f}%".format(relation_type, precision * 100, recall * 100,
                                                                         F1 * 100))
    print("-----------------评估关系结束---------------------")
    model.train()

@paddle.no_grad()
# 10、修改evaluate_v1
def evaluate_v1(model, criterion, metric, data_loader):
    model.eval()
    metric.reset()
    losses = []
    for input_ids, input_mask, char_ids, char_mask, lens, labels in data_loader:
        loss = model(input_ids, input_mask, char_ids, char_mask, lens, labels)
        losses.append(loss.numpy())
        preds, lengths = model(input_ids, input_mask, char_ids, char_mask, lens, None)
        n_infer, n_label, n_correct = metric.compute(lens, preds, labels)
        metric.update(n_infer.numpy(), n_label.numpy(), n_correct.numpy())
    precision, recall, f1_score = metric.accumulate()
    print(f'[EVAL] Precision: {precision:.6f} - Recall: {recall:.6f} - F1: {f1_score:.6f}')
    avg_loss = np.mean(losses)
    return avg_loss, precision, recall, f1_score

global_step=0
best_f1 = 0
num_train_epochs = 150
logging_steps = 100
output_dir = 'model'
tic_train = time.time()
# 开始训练
for epoch in range(num_train_epochs):
    for step, batch in enumerate(train_loader):
        global_step += 1
        input_ids, input_mask, char_ids, char_mask, lens, labels = batch
        #print("Labels shape:", labels.shape)
        model.train()
        loss = model(input_ids, input_mask, char_ids, char_mask, lens, labels)
        avg_loss = paddle.mean(loss)
        if global_step % logging_steps == 0:
            print("global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                  % (global_step, epoch, step, avg_loss.numpy(),
                     logging_steps / (time.time() - tic_train)))
            tic_train = time.time()
        avg_loss.backward()
        optimizer.step()
        optimizer.clear_grad()
    print("-----------------评估总体数据---------------------")
    criterion = paddle.nn.CrossEntropyLoss()
    avg_loss, precision, recall, f1 = evaluate_v1(model, criterion, metric, dev_loader)
    if f1 > best_f1 and f1 >= 0.4:
        evaluate_classification(model, 'data1/dev.txt')
        model_path = os.path.join(output_dir, "model.pdparams")
        paddle.save(model.state_dict(), model_path)
        best_f1 = f1

    # 单独评估test
    #avg_loss, precision, recall, f1 = evaluate_v1(model, criterion, metric, test_loader)
    #print(f'[EVAL] Precision: {precision:.6f} - Recall: {recall:.6f} - F1: {f1:.6f}')





