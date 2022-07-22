# -------------------------------------------------------------------------------#
#  如果在pycharm中运行时提示：
#  FileNotFoundError: [WinError 3] No such file or directory: './Annotations'
#  这是pycharm运行目录的问题，换成绝对路径即可。
# -------------------------------------------------------------------------------#
import os
import random


def voc_yolo():
    random.seed(0)
    xmlfilepath = r'./VOC2012/Annotations'
    saveBasePath = r"./VOC2012/ImageSets/Main/"

    # ----------------------------------------------------------------------#
    #   想要改变数据集分配比例：
    #     修改trainval_percent、train_percent、的比例
    # ----------------------------------------------------------------------#
    trainval_percent = 0.7
    train_percent = 0.6

    temp_xml = os.listdir(xmlfilepath)
    total_xml = []
    for xml in temp_xml:
        if xml.endswith(".xml"):
            total_xml.append(xml)

    num = len(total_xml)
    trainval = random.sample(range(num), int(trainval_percent * num))
    train = random.sample(trainval, int(train_percent * num))

    print("total size: ", num)
    print("train and val size", int(trainval_percent * num))
    print("train size", int(train_percent * num))

    file_trainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
    file_test = open(os.path.join(saveBasePath, 'test.txt'), 'w')
    file_train = open(os.path.join(saveBasePath, 'train.txt'), 'w')
    file_val = open(os.path.join(saveBasePath, 'val.txt'), 'w')

    for i in range(num):
        name = total_xml[i][:-4] + '\n'
        if i in trainval:
            file_trainval.write(name)
            if i in train:
                file_train.write(name)
            else:
                file_val.write(name)
        else:
            file_test.write(name)

    file_trainval.close()
    file_train.close()
    file_val.close()
    file_test.close()


if __name__ == '__main__':
    voc_yolo()
