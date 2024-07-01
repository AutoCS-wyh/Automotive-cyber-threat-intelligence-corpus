import random
def format_data(source_filename, target_filename1, target_filename2):
    datalist=[]
    with open(source_filename, 'r', encoding='utf-8') as f:
        lines=f.readlines()
    words=''
    labels=''
    flag=0
    for line in lines:
        if line == '\n':
            item=words+'\t'+labels+'\n'
            if len(set(labels))>2:
                datalist.append(item)
            words=''
            labels=''
            flag=0
            continue
        word, label = line.strip('\n').split(' ')
        if flag==1:
            words=words+' '+word
            labels=labels+' '+label
        else:
            words=words+word
            labels=labels+label
            flag=1

    random.shuffle(datalist)
    slice_index1 = int(len(datalist)*0.8)  # 训练数据集取80%
    slice_index2 = slice_index1 + int(len(datalist)*0.2)  # 验证数据集取20%

    with open(target_filename1, 'w', encoding='utf-8') as f:
        lines=f.writelines(datalist[0:slice_index1])
    print(f'{source_filename}文件格式转换完毕，训练数据集保存为{target_filename1}')

    with open(target_filename2, 'w', encoding='utf-8') as f1:
        lines=f1.writelines(datalist[slice_index1:slice_index2])
    print(f'{source_filename}文件格式转换完毕，验证数据集保存为{target_filename2}')



if __name__ == '__main__':
    format_data('data1/BIOES.txt', 'data1/train.txt', 'data1/dev.txt')

