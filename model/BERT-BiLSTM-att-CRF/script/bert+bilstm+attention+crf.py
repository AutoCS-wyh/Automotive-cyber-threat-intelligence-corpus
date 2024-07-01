#!/usr/bin/env python
# coding: utf-8

# ### 1、导包

# In[ ]:


import os
import time
from functools import partial
from tqdm import tqdm
# 导入paddle库
import paddle
import paddle.nn.functional as F
import paddle.nn as nn
from paddlenlp.transformers import  BertModel
from paddle.io import DataLoader
# 导入paddlenlp的库
from paddlenlp.transformers import LinearDecayWithWarmup
#from paddlenlp.metrics import ChunkEvaluator
from paddlenlp.transformers import BertTokenizer,BertPretrainedModel
from paddlenlp.data import Stack, Tuple, Pad, Dict
import numpy as np


# 读取数据
from paddlenlp.datasets import load_dataset
from paddlenlp.layers.crf import LinearChainCrf, ViterbiDecoder, LinearChainCrfLoss

# 导入自定义的工具类
from utils import * 
from ChunkEvaluator import *

# In[ ]:


# 从dev.txt和train.txt文件生成dic
gernate_dic('data1/dev.txt', 'data1/train.txt','data1/tag.dic')
id2label,label2id,label_list = load_dicts('data1/tag.dic')
id2label[183] = 'no_entity'
label2id['no_entity'] = 183
label_list = [id2label[i] for i in sorted(id2label.keys())]
print(id2label)
print(label2id)
print(label_list)


# In[ ]:


# 输出所有的实体类别，分类评估时候会用到
entity_types = set([i.split('_')[1] for i in label_list if i!='O'])
print(entity_types)


# In[ ]:


relation_types = set([i.split('_')[2] for i in label_list if i!='O' and len(i.split('_'))>2])
print(relation_types)


# ### 2、把文本和label映射成id，处理成模型的输入的形式

# In[ ]:


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


# In[ ]:


def convert_example_to_feature(example, tokenizer, no_entity_id,
                              max_seq_len=128):
    labels = example['labels']
    example = example['tokens']
    tokenized_input = tokenizer(
        example,
        return_length=True,
        is_split_into_words=True,
        max_seq_len=max_seq_len)

    # -2 for [CLS] and [SEP]
    if len(tokenized_input['input_ids']) - 2 < len(labels):
        labels = labels[:len(tokenized_input['input_ids']) - 2]
    tokenized_input['labels'] = [no_entity_id] + labels + [no_entity_id]
    tokenized_input['labels'] += [no_entity_id] * (
        len(tokenized_input['input_ids']) - len(tokenized_input['labels']))
    return tokenized_input


# ### 3、加载数据集train_data_loader和dev_data_loader

# In[ ]:128


max_seq_length = 128
batch_size = 16
label_num = len(label_list)
#no_entity_id = label_num-1
no_entity_id = 183


#以下4个参数在在bilstm用到
lstm_hidden_size, num_layers, num_classes, dropout =800,1,len(list(id2label)),0.1

#model_name='bert-large-cased'
model_name = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)

trans_func = partial(
        convert_example_to_feature,
        tokenizer=tokenizer,
        no_entity_id=no_entity_id,
        max_seq_len=max_seq_length)
        
train_ds = train_ds.map(trans_func)
dev_ds = dev_ds.map(trans_func)

#ignore_label = label_num -1
ignore_label = 183


batchify_fn = lambda samples, fn=Dict({
        'input_ids': Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int32'),  # input
        'token_type_ids': Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int32'),  # segment
        'seq_len': Stack(dtype='int64'),  # seq_len
        'labels': Pad(axis=0, pad_val=ignore_label, dtype='int64')  # label
    }): fn(samples)


train_batch_sampler = paddle.io.DistributedBatchSampler(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

train_data_loader = DataLoader(dataset=train_ds,collate_fn=batchify_fn,num_workers=0,batch_sampler=train_batch_sampler,
        return_list=True)
dev_data_loader = DataLoader(dataset=dev_ds,collate_fn=batchify_fn,num_workers=0,batch_size=batch_size,return_list=True)


# ### 4、定义模型初始化方法

# In[ ]:


# class AttentionLayer(paddle.nn.Layer):
#     def __init__(self, hidden_size, init_scale=0.01):
#         super(AttentionLayer, self).__init__()
#         self.w = paddle.create_parameter(shape=[hidden_size, hidden_size], dtype="float32")
#         self.v = paddle.create_parameter(shape=[hidden_size, hidden_size], dtype="float32")
#
#     def forward(self, inputs):
#         # inputs:  [batch_size, seq_len, hidden_size]
#         last_layers_hiddens = inputs
#         # transposed inputs: [batch_size, hidden_size, seq_len]
#         inputs = paddle.transpose(inputs, perm=[0, 2, 1])
#         # inputs: [batch_size, hidden_size, seq_len]
#         inputs = paddle.tanh(paddle.matmul(self.w, inputs))
#         # attn_weights: [batch_size, 1, seq_len]
#         attn_weights = paddle.matmul(self.v, inputs)
#         # softmax数值归一化
#         attn_weights = F.softmax(attn_weights, axis=-1)
#         # 通过attention后的向量值, attn_vectors: [batch_size, 1, hidden_size]
#         # attn_vectors = paddle.matmul(last_layers_hiddens, attn_weights)
#
#         return last_layers_hiddens + paddle.transpose(attn_weights, perm=[0, 2, 1])

# class AttentionLayer(paddle.nn.Layer):
#     def __init__(self, hidden_size, init_scale=0.01):
#         super(AttentionLayer, self).__init__()
#         self.w = paddle.create_parameter(shape=[hidden_size, hidden_size], dtype="float32")
#         self.v = paddle.create_parameter(shape=[hidden_size, 1], dtype="float32")
#
#     def forward(self, inputs):
#         # Applying the tanh activation after multiplying with weight 'w'
#         transformed = paddle.tanh(paddle.matmul(inputs, self.w))
#
#         # Compute attention scores
#         attn_scores = paddle.matmul(transformed, self.v)
#         attn_scores = paddle.squeeze(attn_scores, axis=-1)
#
#         # Compute attention weights using softmax
#         attn_weights = F.softmax(attn_scores, axis=-1)
#         attn_weights = attn_weights.unsqueeze(2)
#
#         # Compute weighted sequence of inputs using attention weights
#         attn_output = attn_weights * inputs  # This is the key change
#
#         return attn_output
class AttentionLayer(paddle.nn.Layer):
    def __init__(self, hidden_size, init_scale=0.01):
        super(AttentionLayer, self).__init__()
        self.wq = paddle.create_parameter(shape=[hidden_size, hidden_size], dtype="float32")
        self.wk = paddle.create_parameter(shape=[hidden_size, hidden_size], dtype="float32")
        self.wv = paddle.create_parameter(shape=[hidden_size, hidden_size], dtype="float32")
        self.scale = paddle.to_tensor(hidden_size ** 0.5)

    def forward(self, inputs):
        # Generating Q, K, V
        Q = paddle.matmul(inputs, self.wq)
        K = paddle.matmul(inputs, self.wk)
        V = paddle.matmul(inputs, self.wv)

        # Compute scaled dot-product attention scores
        attn_scores = paddle.matmul(Q, paddle.transpose(K, [0, 2, 1]))
        attn_scores = attn_scores / self.scale

        # Compute attention weights using softmax
        attn_weights = F.softmax(attn_scores, axis=-1)

        # Compute weighted sequence of V using attention weights
        attn_output = paddle.matmul(attn_weights, V)

        return attn_output


# In[ ]:


class Bert_BiLSTM_att_crf(BertPretrainedModel):

    def __init__(self, bert,num_classes, lstm_hidden_size, num_layers,dropout):
        super(Bert_BiLSTM_att_crf, self).__init__()
        self.num_classes = num_classes
        # 初始化bert
        self.bert = bert
        # 初始化双向的lstm
        self.lstm = nn.LSTM(self.bert.config["hidden_size"], lstm_hidden_size, num_layers,direction='bidirectional',dropout=dropout)
        self.attention = AttentionLayer(hidden_size=lstm_hidden_size*2)

        self.fc = paddle.nn.Linear(in_features=lstm_hidden_size*2,out_features=self.num_classes)
        # crf层
        self.crf = LinearChainCrf(
            self.num_classes, crf_lr=1, with_start_stop_tag=False)

        self.crf_loss = LinearChainCrfLoss(self.crf)

        self.viterbi_decoder = ViterbiDecoder(
            self.crf.transitions, with_start_stop_tag=False)

    def forward(self,
                input_ids,
                token_type_ids,
                lengths=None,
                labels=None):
        sequence_out, _ = self.bert(input_ids,
                                    token_type_ids=token_type_ids)
        lstm_output, _ = self.lstm(sequence_out)

        attention_w = self.attention(lstm_output)
        emission = self.fc(attention_w)

        if labels is not None:
            loss = self.crf_loss(emission, lengths, labels)
            return loss
        else:
            _, prediction = self.viterbi_decoder(emission, lengths)
            return prediction

# ### 5、配置全局参数

# In[ ]:


# 设置epoch
num_train_epochs=85
warmup_steps=0

max_steps=-1
# 优化器的超参数
learning_rate=5e-5
adam_epsilon=1e-8
weight_decay=1e-4

global_step = 0
# 日志输出的step数
logging_steps=50

tic_train = time.time()
save_steps=100
output_dir='model'
os.makedirs(output_dir,exist_ok=True)

# Define the model netword and its loss
last_step = num_train_epochs * len(train_data_loader)

num_training_steps = max_steps if max_steps > 0 else len(train_data_loader) * num_train_epochs
lr_scheduler = LinearDecayWithWarmup(learning_rate, num_training_steps,
                                         warmup_steps)


# ### 6、初始化模型

# In[ ]:


bert = BertModel.from_pretrained(model_name)
bert_bilstm_att_crf = Bert_BiLSTM_att_crf(bert,label_num,lstm_hidden_size,num_layers,dropout)

decay_params = [
        p.name for n, p in bert_bilstm_att_crf.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
# 设置优化器
optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        epsilon=adam_epsilon,
        parameters=bert_bilstm_att_crf.parameters(),
        weight_decay=weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)
# 设置损失函数
loss_fct = nn.loss.CrossEntropyLoss(ignore_index=ignore_label)
# 设置评估函数
metric = ChunkEvaluator(label_list=label_list,suffix=False)


# ### 7、模型评估方法

# In[ ]:


def predict(text, tokenizer=tokenizer, no_entity_id=no_entity_id,max_seq_len=max_seq_length):
    example=text.split()
    example1 = tokenizer(
        example,
        return_length=True,
        is_split_into_words=True,
        max_seq_len=max_seq_len)
    bert_bilstm_att_crf.eval()
   
    input_ids=paddle.to_tensor([example1['input_ids']])
    token_type_ids=paddle.to_tensor([example1['token_type_ids']])
    length=example1['seq_len']
    logits = bert_bilstm_att_crf(input_ids, token_type_ids, paddle.to_tensor(length))
    pred = logits.numpy()
    tags = [id2label[x] for x in pred[0][1:length-1]]
    return example,tags


# In[ ]:


# 先取出所有的实体类别,评估时候会用到
entity_counts = {}
for entity_type in entity_types:
    entity_counts[entity_type] = {"TP": 0, "FP": 0, "FN": 0}


# In[ ]:


# 先取出所有的实体类别,评估时候会用到
relation_counts = {}
for relation_type in relation_types:
    relation_counts[relation_type] = {"TP": 0, "FP": 0, "FN": 0}


# In[ ]:


entity_counts


# In[ ]:


relation_counts


# In[ ]:

def get_entity_and_relation(label):
    parts = label.split('_')
    if label == 'O':  # 直接处理'O'标签的情况
        return 'O', ''
    elif len(parts) == 2:
        return label, ''
    elif len(parts) > 2:
        return '_'.join(parts[:-2]), parts[-2]
    else:
        return None, None  # 对于不合法的标签格式



def evaluate_classification(model, file_path):
    model.eval()
    # 读取所有数据
    lines = [i.strip('\n') for i in open(file_path, 'r', encoding='utf-8').readlines()]
    # 逐行遍历
    for line in tqdm(lines):
        tmp_list = line.split('\t')
        sentence = tmp_list[0]
        tag = tmp_list[1]
        sentence_list = sentence.split()
        tag_list = tag.split()

        # 第二步，取出模型预测的y
        sent, tags = predict(' '.join(sentence_list))
        tag_list = tag_list[0:len(tags)]

        # 遍历，二者进行比较
        for index, tag in enumerate(tag_list):
            #print(f"Original Tag: {tag}")  # 打印原始标签

            true_entity, true_relation = get_entity_and_relation(tag)
            #print(f"True Entity: {true_entity}, True Relation: {true_relation}")  # 打印解析后的实体和关系

            predicted_entity, predicted_relation = get_entity_and_relation(tags[index])
            #print(f"Predicted Entity: {predicted_entity}, Predicted Relation: {predicted_relation}")  # 打印预测的实体和关系

            if true_entity is None or predicted_entity is None:
                continue
            true_entity_type = true_entity.split('_')[-1] if true_entity != 'O' else ''
            predicted_entity_type = predicted_entity.split('_')[-1] if predicted_entity != 'O' else ''


            # 确保键存在
            if true_entity_type not in entity_counts and true_entity != 'O':
                entity_counts[true_entity_type] = {"TP": 0, "FP": 0, "FN": 0}
            if predicted_entity_type not in entity_counts and predicted_entity != 'O':
                entity_counts[predicted_entity_type] = {"TP": 0, "FP": 0, "FN": 0}


            # 实体评估
            # if tag != 'O':  # 真实标签不是'O'，表示有实体
            #     if true_entity == predicted_entity:
            #         entity_counts[true_entity_type]['TP'] += 1  # 真正例：正确预测实体类型
            #     else:
            #         if predicted_entity != '':
            #             entity_counts[predicted_entity_type]['FP'] += 1  # 假正例：错误预测了实体类型
            #         entity_counts[true_entity_type]['FN'] += 1  # 假负例：未能正确预测实体
            # elif predicted_entity != '':
            #     entity_counts[predicted_entity_type]['FP'] += 1  # 假正例：预测了不存在的实体

            if true_entity != 'O':  # 如果有真实实体
                if true_entity == predicted_entity:
                    entity_counts[true_entity_type]['TP'] += 1  # 真正例
                else:
                    entity_counts[true_entity_type]['FN'] += 1  # 假负例：未能正确识别真实实体
            else:
                if predicted_entity != 'O':
                    entity_counts[predicted_entity_type]['FP'] += 1  # 假正例：错误识别或预测了不存在的实体

            # 确保关系类型的键也存在
            if true_relation not in relation_counts and true_relation != '':
                relation_counts[true_relation] = {"TP": 0, "FP": 0, "FN": 0}
            if predicted_relation not in relation_counts and predicted_relation != '':
                relation_counts[predicted_relation] = {"TP": 0, "FP": 0, "FN": 0}

            # 关系评估
            if true_relation != '':
                if true_relation == predicted_relation:
                    relation_counts[true_relation]['TP'] += 1  # 真正例：正确预测关系类型
                else:
                    relation_counts[true_relation]['FN'] += 1
            else:
                if predicted_relation != '':
                    relation_counts[predicted_relation]['FP'] += 1


            # if true_relation == predicted_relation and true_relation != '':
            #     relation_counts[true_relation]['TP'] += 1  # 真正例：正确预测关系类型
            # else:
            #     if predicted_relation != '':
            #         if true_relation == '':
            #             relation_counts[predicted_relation]['FP'] += 1  # 假正例：预测了不存在的关系
            #         else:
            #             relation_counts[predicted_relation]['FP'] += 1  # 假正例：错误预测了关系类型
            #             relation_counts[true_relation]['FN'] += 1  # 假负例：未能正确预测真实存在的关系
            #     elif true_relation != '':
            #         relation_counts[true_relation]['FN'] += 1  # 假负例：未能预测真实存在的关系

    evaluation_parameters = {}
    # 计算每种实体类型的精确率、召回率和F1值，并输出结果
    print("-----------------开始评估实体---------------------")
    for true_entity_type in entity_counts:
        TP = entity_counts[true_entity_type]["TP"]
        FP = entity_counts[true_entity_type]["FP"]
        FN = entity_counts[true_entity_type]["FN"]
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        F1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        evaluation_parameters[true_entity_type] = [precision * 100, recall * 100, F1 * 100]
        print("{}: precision={:.2f}%, recall={:.2f}%, F1={:.2f}%".format(true_entity_type, precision * 100, recall * 100,
                                                                         F1 * 100))
    print("-----------------评估实体结束---------------------")

    # 计算每种关系类型的精确率、召回率和F1值，并输出结果
    print("-----------------开始评估关系---------------------")
    for relation_type in relation_counts:
        TP = relation_counts[relation_type]["TP"]
        FP = relation_counts[relation_type]["FP"]
        FN = relation_counts[relation_type]["FN"]
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        F1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        evaluation_parameters[relation_type] = [precision * 100, recall * 100, F1 * 100]
        print("{}: precision={:.2f}%, recall={:.2f}%, F1={:.2f}%".format(relation_type, precision * 100, recall * 100,
                                                                         F1 * 100))
    print("-----------------评估关系结束---------------------")
    model.train()
# 分类评估
# def evaluate_classification(model,file_path):
#     model.eval()
#     # 读取所有数据
#     lines = [i.strip('\n') for i in open(file_path,'r',encoding='utf-8').readlines()]
#     # 逐行遍历
#     for line in tqdm(lines):
#         tmp_list = line.split('\t')
#         sentence =tmp_list[0]
#         tag = tmp_list[1]
#         # 第一步：取出标注的标签，相当于y
#         sentence_list = sentence.split()
#         tag_list = tag.split()
#
#         # 第二步，取出模型预测的y
#         sent,tags = predict(' '.join(sentence_list))
#         tag_list=tag_list[0:len(tags)]
#         # 遍历，二者进行比较
#         for index,tag in enumerate(tag_list):
#             if tag=='O' and tags[index]=='O':
#                 continue
#             else:
#                 # 先评估实体
#                 # FP：标注是O，预测的不是O，则属于FP
#                 if tag=='O':
#                     entity_type1 = tags[index].split('_')[1]
#                     entity_counts[entity_type1]['FP'] +=1
#                 # FN：预测是O，标注是不是O，，则属于FN
#                 elif tags[index]=='O':
#                     entity_type2 = tag.split('_')[1]
#                     entity_counts[entity_type2]['FN'] +=1
#                 elif tag.split('_')[1]==tags[index].split('_')[1]: # TP：标注和预测一样
#                     entity_type = tag.split('_')[1]
#                     entity_counts[entity_type]['TP'] +=1
#                 else:
#                     # 标注和预测都不是O且二者不一样
#                     entity_type3 = tags[index].split('_')[1]
#                     entity_counts[entity_type3]['FN'] +=1
#
#                     entity_type4 = tag.split('_')[1]
#                     entity_counts[entity_type4]['FP'] +=1
#
#
#
#                 # 评估关系
#                 # TP：标注和预测一样
#                 if len(tag.split('_'))>=3 and len(tags[index].split('_'))>=3: #标注的和预测的都有关系
#                     if tag.split('_')[2]==tags[index].split('_')[2]:
#                         relation_type = tag.split('_')[2]
#                         relation_counts[relation_type]['TP'] +=1
#                     else:
#                         # FP：标注是O，预测的不是O，则属于FP
#                         if tag=='O':
#                             relation_type1 = tags[index].split('_')[2]
#                             relation_counts[relation_type1]['FP'] +=1
#                         # FN：预测是O，标注是不是O，，则属于FN
#                         elif tags[index]=='O':
#                             relation_type2 = tag.split('_')[2]
#                             relation_counts[relation_type2]['FN'] +=1
#                         else:
#                             # 标注和预测都不是O且二者不一样
#                             relation_type3 = tags[index].split('_')[2]
#                             relation_counts[relation_type3]['FN'] +=1
#
#                             relation_type4 = tag.split('_')[2]
#                             relation_counts[relation_type4]['FP'] +=1
#                 # 标注没有关系，预测有关系
#                 elif len(tag.split('_'))==2 and len(tags[index].split('_'))>=3:
#                     relation_type1 = tags[index].split('_')[2]
#                     relation_counts[relation_type1]['FP'] +=1
#                 # 标注有关系，预测没有
#                 elif len(tag.split('_'))>=3 and len(tags[index].split('_'))==2:
#                     relation_type2 = tag.split('_')[2]
#                     relation_counts[relation_type2]['FN'] +=1
#     evaluation_parameters = {}
#     # 计算每种实体类型的精确率、召回率和F1值，并输出结果
#     print("-----------------开始评估实体---------------------")
#     for entity_type in entity_counts:
#         TP = entity_counts[entity_type]["TP"]
#         FP = entity_counts[entity_type]["FP"]
#         FN = entity_counts[entity_type]["FN"]
#         if TP + FP == 0:
#             precision = 0.0
#         else:
#             precision = TP / (TP + FP)
#         recall = TP / (TP + FN) if (TP + FN)>0 else 0
#         if precision + recall == 0:
#             F1 = 0.0
#         else:
#             F1 = 2 * precision * recall / (precision + recall)
#         evaluation_parameters[entity_type]=[]
#         evaluation_parameters[entity_type].extend([precision*100,recall*100,F1])
#         print("{}: precision={:.2f}%, recall={:.2f}%, F1={:.2f}%".format(entity_type, precision*100, recall*100, F1*100))
#     print("-----------------评估实体结束---------------------")
#     # 计算每种关系类型的精确率、召回率和F1值，并输出结果
#     print("-----------------开始评估关系---------------------")
#     for relation_type in relation_counts:
#         TP = relation_counts[relation_type]["TP"]
#         FP = relation_counts[relation_type]["FP"]
#         FN = relation_counts[relation_type]["FN"]
#         if TP + FP == 0:
#             precision = 0.0
#         else:
#             precision = TP / (TP + FP)
#         recall = TP / (TP + FN) if (TP + FN)>0 else 0
#         if precision + recall == 0:
#             F1 = 0.0
#         else:
#             F1 = 2 * precision * recall / (precision + recall)
#         evaluation_parameters[relation_type]=[]
#         evaluation_parameters[relation_type].extend([precision*100,recall*100,F1])
#         print("{}: precision={:.2f}%, recall={:.2f}%, F1={:.2f}%".format(relation_type, precision*100, recall*100, F1*100))
#     print("-----------------评估关系结束---------------------")
#     model.train()


# In[ ]:


@paddle.no_grad()
def evaluate_v1(model, metric, data_loader):
    model.eval()
    metric.reset()
    for input_ids, seg_ids, lens, labels in data_loader:
        preds = model(input_ids, seg_ids, lengths=lens)
        n_infer, n_label, n_correct = metric.compute(lens, preds, labels)
        metric.update(n_infer.numpy(), n_label.numpy(), n_correct.numpy())
        precision, recall, f1_score = metric.accumulate()
    print("[EVAL] Precision: %f - Recall: %f - F1: %f" % (precision, recall, f1_score))
    model.train()
    return f1_score




# ### 8、模型训练

# In[ ]:


global_step=0
best_f1 = 0
for epoch in range(num_train_epochs):
    for step, batch in enumerate(train_data_loader):
        global_step += 1
        input_ids, token_type_ids, lengths, labels = batch
        loss = bert_bilstm_att_crf(input_ids, token_type_ids, lengths=lengths, labels=labels)
        avg_loss = paddle.mean(loss)
        if global_step % logging_steps == 0:
            print("global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                    % (global_step, epoch, step, avg_loss,
                       logging_steps / (time.time() - tic_train)))
            tic_train = time.time()
        avg_loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.clear_grad()
    print("-----------------评估总体数据---------------------")
    f1 = evaluate_v1(bert_bilstm_att_crf, metric, dev_data_loader)
    # 如果模型效果比历史记录好，则保存模型，评估各实体的参数
    # 把评估放在这里，可以节约时间，epoch=5之前，模型效果都不好，所以不用评估
    #if f1>best_f1 and epoch>=40:
    # if f1 > best_f1:
    #     if f1 >= 0.55:
    #         evaluate_classification(bert_bilstm_att_crf, 'data1/dev.txt')
    #         model_path = os.path.join(output_dir, "bert_bilstm_att_crf.pdparams")
    #         paddle.save(bert_bilstm_att_crf.state_dict(), model_path)
    #     best_f1 = f1
    if f1 > best_f1:
        # 保存模型
        model_path = os.path.join(output_dir, "bert_bilstm_att_crf.pdparams")
        paddle.save(bert_bilstm_att_crf.state_dict(), model_path)
        best_f1 = f1

        # 当 f1 分数大于等于 0.55 时，执行额外的评估
        if f1 >= 0.5:
            evaluate_classification(bert_bilstm_att_crf, 'data1/dev.txt')
tokenizer.save_pretrained(output_dir)


# ### 9、查看tp，fp

# In[ ]:

print(entity_counts)



# In[ ]:

print(relation_counts)



# In[ ]:




