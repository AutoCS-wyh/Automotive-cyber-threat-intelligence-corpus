# Automotive-cyber-threat-intelligence-corpus
A dataset for cyber threat intelligence modeling of  connected autonomous vehicles

1.**Experimental environment:**
NVIDIA GeForce RTX 3090 GPU
Python 3.7
CUDA 11.2
PaddlePaddle-GPU 2.3.2 
paddlenlp 2.1.1

**2.Data description**
Raw data: unstructured cybersecurity data (.txt files)
Brat annotation data：annotation data files using brat tool (.ann files) 
BIOES："BIOES" - "entity type" - "relation type" - "entity role" joint annotation data (.txt files)

**3.Source code description**
a.format conversion:BIOES joint annotation.py
b.preprocessing:read.py;preprocess.py
c.deep learning model training:BERT-BiLSTM-att-CRF;BiLSTM-dynamic-att-LSTM

**4.Brat tool**
https://github.com/nlplab/brat/archive/refs/tags/v1.3p1.tar.gz. 
