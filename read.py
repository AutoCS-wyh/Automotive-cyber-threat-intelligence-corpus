import os

# 指定目标文件夹路径
folder_path = "BIOES"
# 指定输出文件名
output_file_name = "BIOES.txt"

# 获取文件夹中所有.txt文件的列表
txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

# 打开输出文件
with open(output_file_name, 'w', encoding='utf-8') as outfile:
    # 遍历每一个.txt文件
    for txt_file in txt_files:
        # 获取完整文件路径
        file_path = os.path.join(folder_path, txt_file)
        # 打开并读取文件内容
        with open(file_path, 'r', encoding='utf-8') as infile:
            # 将文件内容写入输出文件
            outfile.write(infile.read())
            # 在每个文件内容之后添加一个换行，保证内容之间的分隔
            #outfile.write("\n")
