# ------------------------------------------------------------------------------------ #
# (1) 由于 train2017 图片太多, 故利用脚本 '划分出coco小型数据集.py' 生成了 mini_train2017
# (2) 为了 训练集 和 验证集 比例合理，故在此脚本中对生成的 coco_train.txt 、coco_val.txt
#     中的条目进行再划分，比例为 8:2
# ------------------------------------------------------------------------------------ #
import os
import random


def merge(file1, file2):
    f1 = open(file1, 'a+', encoding='utf-8')
    with open(file2, 'r', encoding='utf-8') as f2:
        for i in f2:
            f1.write(i)


merge('coco_train.txt', 'coco_val.txt')  # 合并两个文件到第一个文件

# 读取合并后的 coco_train.txt 行数
count = 0  # 计数
with open('coco_train.txt', 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        count += 1
print('文件行数为：', count)
count += 1

# 开始随机分配 每行内容 到两个文件，按 8:2 的比例
oldf = open('coco_train.txt', 'r', encoding='utf-8')  # 要被抽取的文件coco_train.txt
newf = open('coco_val.txt', 'w', encoding='utf-8')  # 抽取的 8:2 行写入 coco_val.txt
f_temp = open('temp.txt', 'w', encoding='utf-8')  # 中间文件

# # sample(x,y)函数的作用是从序列x中，随机选择y个不重复的元素
resultList = random.sample(range(1, count), int((count - 1) * 0.2))  # 0.2
print(resultList, len(resultList))

num = 0
for line in oldf.readlines():
    num += 1
    if num in resultList:
        newf.writelines(line)
        continue
    f_temp.writelines(line)

oldf.close()
newf.close()
f_temp.close()

os.remove("coco_train.txt")
os.rename("temp.txt", "coco_train.txt")
