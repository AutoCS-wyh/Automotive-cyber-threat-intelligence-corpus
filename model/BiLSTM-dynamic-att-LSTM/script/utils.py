# 读取训练数据集和验证数据集中的lables，去重后组成字典
def gernate_dic(source_filename1, source_filename2, target_filename):
    data_set=set()

    with open(source_filename1, 'r', encoding='utf-8') as f:
        lines=f.readlines()

    for line in lines:
        if line != '\n':
            labels=line.strip('\n').split('\t')[1].split()
            for label in labels:
                data_set.add(label+'\n')

    with open(source_filename2, 'r', encoding='utf-8') as f:
        lines=f.readlines()

    for line in lines:
        if line != '\n':
            labels=line.strip('\n').split('\t')[1].split()
            for label in labels:
                data_set.add(label+'\n')
    with open(target_filename, 'w', encoding='utf-8') as f:
        lines=f.writelines(data_set)
# def gernate_dic(source_filename1, source_filename2, source_filename3,target_filename):
#     data_set = set()
#
#     with open(source_filename1, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#
#     for line in lines:
#         if line != '\n':
#             labels = line.strip('\n').split('\t')[1].split()
#             for label in labels:
#                 data_set.add(label + '\n')
#
#     with open(source_filename2, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#
#     for line in lines:
#         if line != '\n':
#             labels = line.strip('\n').split('\t')[1].split()
#             for label in labels:
#                 data_set.add(label + '\n')
#
#     with open(source_filename3, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#
#     for line in lines:
#         if line != '\n':
#             labels = line.strip('\n').split('\t')[1].split()
#             for label in labels:
#                 data_set.add(label + '\n')
#
#     with open(target_filename, 'w', encoding='utf-8') as f:
#         lines = f.writelines(data_set)

    # 从文件中读取字典
def load_dicts(dict_path):
    id2label = {}
    label2id = {}
    label_list = []
    for line in open(dict_path, 'r', encoding='utf-8'):
        key = line.strip('\n')
        label_list.append(key)
    label_list.sort()
    i = 0
    for label in label_list:
        label2id[label] = i
        id2label[i] = label
        i+=1
    return id2label,label2id,label_list

    
def checkbioes(biotags):
    """
    bio转bioes
    :param biotags:
    :return:
    """
    bioestags = []
    for i, tag in enumerate(biotags):
        if tag == 'O' or tag.split('-')[0] == 'S':
            # 直接保留，不变化
            bioestags.append(tag)
        elif tag.split('-')[0] == 'E':
            if (i + 1) < len(biotags) and biotags[i + 1].split('-')[0] == 'E':
                # 直接保留，不变化
                bioestags.append(tag.replace('E-', 'I-'))
            else:
                bioestags.append(tag)
        elif tag.split('-')[0] == 'B':
            if (i - 1) < len(biotags) and biotags[i - 1].split('-')[0] == 'B':
                # B前面是B，则第二个B换成I
                bioestags.append(tag.replace('B-', 'I-'))
                # B后面有I的情况
            elif (i + 1) < len(biotags) and biotags[i + 1].split('-')[0] == 'I':
                # 直接保留，不变化
                bioestags.append(tag)
            elif (i + 1) < len(biotags) and biotags[i + 1].split('-')[0] == 'E':
                bioestags.append(tag)
            elif (i + 1) < len(biotags) and biotags[i + 1].split('-')[0] == 'B':
                bioestags.append(tag)
            else:
                # 单独一个B，转为S
                bioestags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            # I后面有I的情况
            if (i + 1) < len(biotags) and biotags[i + 1].split('-')[0] == 'I':
                # 直接保留，不变化
                bioestags.append(tag)
            elif (i + 1) < len(biotags) and biotags[i + 1].split('-')[0] == 'E':
                # 直接保留，不变化
                bioestags.append(tag)
            else:
                # 最后一个I，转为E
                bioestags.append(tag.replace('I-', 'E-'))
        else:
            # 非BIO编码转为O
            bioestags.append('O')
    return bioestags

  