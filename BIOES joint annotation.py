import os
import re


def process(word_index_dict,id,key,entity,entity_type,entity_postion):
    if word_index_dict[id].split('_')[2] != key[0:-4]:
        if word_index_dict[id].split('_')[3] != entity_postion:
            word_index_dict[id] = '_'.join([entity_type, entity[1], 'M', 'm'])
            print(file_name)
        else:
            print(file_name+'-----------------')
            word_index_dict[id] = '_'.join([entity_type, entity[1], 'M', entity_postion])
    else:
        word_index_dict[id] = '_'.join([entity_type, entity[1], key[0:-4], 'm'])



if __name__ == '__main__':
    # 读取data下的文件
    files = os.listdir('data')
    for file_name in files:
        # 跳过配置文件
        if file_name.endswith('conf'):
            continue
        # 读取ann文件
        elif file_name.endswith('ann'):
            # 先读取ann文件对应的txt
            txt_file_name = file_name.replace('.ann','.txt')
            if txt_file_name not in files:
                continue
            txt_content = [ i for i in open(os.path.join('data',txt_file_name),'r',encoding='utf-8').readlines()]
            # 如果txt中内容为空，则跳过
            if len(txt_content)==0:
                continue
            else:
                # 实体字典，存放所有实体的信息
                entity_dict = {}
                relation_dict = {}
                word_index_dict = {}
                # 读取ann
                ann_content = [i.strip('\n') for i in open(os.path.join('data',file_name),'r',encoding='utf-8').readlines()]
                for line in ann_content:
                    # 解析关系
                    if line.startswith('R'):
                        # 取出内容，如consists-of Arg1:T4 Arg2:T3
                        tmp_list = line.split('\t')[1].split()
                        # 关系名称
                        relation_name = tmp_list[0]
                        # 第一个实体
                        arg1 = tmp_list[1].replace('Arg1:','')
                        # 第二个实体
                        arg2 = tmp_list[2].replace('Arg2:','')
                        tmp_list = []
                        tmp_list.append(entity_dict[arg1])
                        tmp_list.append(entity_dict[arg2])
                        relation_dict[relation_name+arg1+arg2] = tmp_list.copy()
                        tmp_list.clear()
                    # 解析实体
                    elif line.startswith('T'):
                        tmp_list = []
                        tmp_list.append(line.split('\t')[0])
                        tmp_list.extend(line.split('\t')[1].split())
                        tmp_list.append(line.split('\t')[2])
                        entity_dict[tmp_list[0]] = tmp_list.copy()
                        tmp_list.clear()
                    # 其他非实体非关系则跳过
                    else:
                        continue

                # 所有行遍历完成后，开始写结果
                # 只有实体，没有关系则暂时跳过
                if len(relation_dict)==0:
                    # 逐一遍历关系
                    for key, value in entity_dict.items():
                        # 只有一个单词
                        if len(value[4].split()) == 1:
                            id = value[4] + '_' + value[2]
                            word_index_dict[id] = '_'.join(['S', value[1]])
                        # 多个单词
                        else:
                            num = 0
                            for i, word in enumerate(value[4].split()):
                                if i == 0:
                                    id = word + '_' + value[2]
                                    word_index_dict[id] = '_'.join(['B', value[1]])
                                    num = int(value[2]) + len(word) + 1
                                    # 最后一个单词
                                elif i==len(value[4].split())-1:
                                    id = word + '_' + str(num)
                                    word_index_dict[id] = '_'.join(['E', value[1]])
                                    num = num + len(word) + 1
                                else:
                                    id = word + '_' + str(num)
                                    word_index_dict[id] = '_'.join(['I', value[1]])
                                    num = num + len(word) + 1
                else:

                    # 逐一遍历关系
                    for key,value in relation_dict.items():
                        # 处理第一个实体
                        entity1 = value[0]
                        # 只有一个单词
                        if len(entity1[4].split()) == 1:
                            id = entity1[4]+'_'+entity1[2]
                            if id in word_index_dict:
                                process(word_index_dict,id,key,entity1,'S','1')
                            else:
                                relation_name = key.split('T')[0]
                                word_index_dict[id] = '_'.join(['S', entity1[1], relation_name, '1'])
                                # word_index_dict[id] = '_'.join(['S', entity1[1], key[0:-4], '1'])
                        # 多个单词
                        else:
                            num = 0
                            for i,word in enumerate(entity1[4].split()):
                                if i ==0:
                                    id = word + '_' + entity1[2]
                                    if id in word_index_dict:
                                        process(word_index_dict,id,key,entity1,'B','1')
                                    else:
                                        relation_name = key.split('T')[0]
                                        word_index_dict[id] = '_'.join(['B', entity1[1], relation_name, '1'])
                                        # word_index_dict[id] = '_'.join(['B', entity1[1], key[0:-4], '1'])
                                    num = int(entity1[2]) + len(word) +1
                                elif i==len(entity1[4].split())-1:
                                    id = word + '_' + str(num)
                                    if id in word_index_dict:
                                        process(word_index_dict,id,key,entity1,'E','1')
                                    else:
                                        relation_name = key.split('T')[0]
                                        word_index_dict[id] = '_'.join(['E', entity1[1], relation_name, '1'])
                                        # word_index_dict[id] = '_'.join(['E', entity1[1], key[0:-4], '1'])
                                    num = num + len(word) + 1
                                else:
                                    id = word + '_' + str(num)
                                    if id in word_index_dict:
                                        process(word_index_dict,id,key,entity1,'I','1')
                                    else:
                                        relation_name = key.split('T')[0]
                                        word_index_dict[id] = '_'.join(['I', entity1[1], relation_name, '1'])
                                        #word_index_dict[id] = '_'.join(['I', entity1[1], key[0:-4], '1'])
                                    num = num + len(word) + 1

                        entity2 = value[1]
                        if len(entity2[4].split()) == 1:
                            id = entity2[4] + '_' + entity2[2]
                            if id in word_index_dict:
                                process(word_index_dict,id,key,entity2,'S','2')
                            else:
                                relation_name = key.split('T')[0]
                                word_index_dict[id] = '_'.join(['S', entity2[1], relation_name, '1'])
                                #word_index_dict[id] = '_'.join(['S', entity2[1], key[0:-4], '2'])

                        # 多个单词
                        else:
                            num = 0
                            for i, word in enumerate(entity2[4].split()):
                                if i == 0:
                                    id = word + '_' + entity2[2]
                                    if id in word_index_dict:
                                        process(word_index_dict, id, key, entity2, 'B', '2')
                                    else:
                                        relation_name = key.split('T')[0]
                                        word_index_dict[id] = '_'.join(['B', entity2[1], relation_name, '1'])
                                        #word_index_dict[id] = '_'.join(['B', entity2[1], key[0:-4], '2'])
                                    num = int(entity2[2]) + len(word) +1
                                elif i==len(entity2[4].split())-1:
                                    id = word + '_' + str(num)
                                    if id in word_index_dict:
                                        process(word_index_dict, id, key, entity2, 'E', '2')
                                    else:
                                        relation_name = key.split('T')[0]
                                        word_index_dict[id] = '_'.join(['E', entity2[1], relation_name, '1'])
                                        #word_index_dict[id] = '_'.join(['E', entity2[1], key[0:-4], '2'])
                                    num = num + len(word) + 1
                                else:
                                    id = word + '_' + str(num)
                                    if id in word_index_dict:
                                        process(word_index_dict, id, key, entity2, 'I', '2')
                                    else:
                                        relation_name = key.split('T')[0]
                                        word_index_dict[id] = '_'.join(['I', entity2[1], relation_name, '1'])
                                        #word_index_dict[id] = '_'.join(['I', entity2[1], key[0:-4], '2'])
                                    num = num + len(word) + 1
                    # 逐一遍历实体
                    for key, value in entity_dict.items():
                        # 只有一个单词
                        if len(value[4].split()) == 1:
                            id = value[4] + '_' + value[2]
                            if id not in word_index_dict:
                                word_index_dict[id] = '_'.join(['S', value[1]])
                        # 多个单词
                        else:
                            num = 0
                            for i, word in enumerate(value[4].split()):
                                if i == 0:
                                    id = word + '_' + value[2]
                                    if id not in word_index_dict:
                                        word_index_dict[id] = '_'.join(['B', value[1]])
                                    num = int(value[2]) + len(word) + 1
                                # 最后一个单词
                                elif i == len(value[4].split()) - 1:
                                    id = word + '_' + str(num)
                                    if id not in word_index_dict:
                                        word_index_dict[id] = '_'.join(['E', value[1]])
                                    num = num + len(word) + 1
                                else:
                                    id = word + '_' + str(num)
                                    if id not in word_index_dict:
                                        word_index_dict[id] = '_'.join(['I', value[1]])
                                    num = num + len(word) + 1
            word_index = 0
            bio_file = open(os.path.join('BIOES',file_name.replace('.ann','_bio.txt')),'w',encoding='utf-8')
            for word in [i for i in txt_content[0].split()]:
                # 用单词名称以及单词下标来标识一个单词
                #word_id = word.replace(',', '').replace('.', '') + '_' + str(word_index)
                #word = word.replace(',','').replace('.','').replace(')','').replace('\"','')
                if re.match(r"[\w']+[\.,!?;\)\"']$", word):
                    word_id = word[:-1] + '_' + str(word_index)  # 去除标点后的单词id
                    if word_id in word_index_dict:
                        # 如果单词在字典中，写入去除标点后的单词和其对应的标签
                        bio_file.write(word[:-1] + ' ' + word_index_dict[word_id] + '\n')
                    else:
                        # 如果单词不在字典中，标签为O
                        bio_file.write(word[:-1] + ' O' + '\n')
                    # 对于结束的标点符号，其标签总是'O'
                    bio_file.write(word[-1] + ' O' + '\n')
                    if word[-1] == '.':
                        # 如果结束标点为句号，则额外空一行
                        bio_file.write('\n')
                    word_index = word_index + len(word) + 1
                else:
                    word_id = word + '_' + str(word_index)
                    if word_id in word_index_dict:
                        bio_file.write(word + ' ' + word_index_dict[word_id] + '\n')
                    else:
                        bio_file.write(word + ' O' + '\n')
                    word_index = word_index + len(word) + 1
        else:
            continue
                #     #word_id = word[:-1] + '_' + str(word_index)
                #     word_id = word + '_' + str(word_index)
                #     if word_id in word_index_dict:
                #         bio_file.write(word[:-1] + ' ' + word_index_dict[word_id] + '\n')
                #         bio_file.write(word[-1] + ' O' + '\n')
                #     else:
                #         bio_file.write(word[:-1] + ' O' + '\n')
                #         bio_file.write(word[-1] + ' O' + '\n')
                #         if word[-1] == '.':
                #             bio_file.write('\n')
                #     word_index = word_index + len(word) + 1
                #
            #     else:
            #         word_id = word + '_' + str(word_index)
            #         if word_id in word_index_dict:
            #             bio_file.write(word + ' ' + word_index_dict[word_id] + '\n')
            #         else:
            #             bio_file.write(word + ' O' + '\n')
            #         word_index = word_index + len(word) + 1
            # bio_file.close()
        # else:
        #     continue












