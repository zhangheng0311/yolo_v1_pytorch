# ----------------------注意--------------------------------------------#
#   1. 必须确保路径中没有中文！！！！！！！！！
#   2. 程序运行之前会先运行VOC2012/ImageSets/Main文件夹下的voc_yolo.py
#      voc_yolo.py只将数据集进行划分，并未写入标注信息与图片路径
#   3. 运行后一定要查看classes输出是否格式正常
#   4. 运行后一定要查看生成的三个.txt文件中每行格式是否如下：
#      image_name.jpg x y w h c
#      image_name.jpg x y w h c x y w h c 这样表示一张图片中有两个目标
# ---------------------------------------------------------------------#
import xml.etree.ElementTree as ET
from os import getcwd
from voc_yolo import voc_yolo

sets = ['train', 'val', 'test']

classes = []
classes_file = open('./VOC2012/Labels/labels.txt').read().strip().split()
for label in classes_file:
    classes.append(label)

print("classes: ", classes)


# ---------------------------------------------------------------#
#  convert函数主要功能：
#    1. 将bbox的左上角点、右下角点坐标的格式，转换为bbox中心点+bbox的w,h的
#    格式，并进行归一化(建议使用归一化，方便后面进行数据转换成方便计算LOSS的形式)
#    2. 若不选择归一化请将64行注释，并删除65行注释符
# ---------------------------------------------------------------#
def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(image_id, list_file):
    in_file = open('./VOC2012/Annotations/%s.xml' % image_id, encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()

    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') != None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
             int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        bb = convert((w, h), b)
        # bb = (b[0], b[1], b[2] - b[0], b[3] - b[1])
        list_file.write(" " + " ".join([str(a) for a in bb]) + " " + str(cls_id))


def voc_annotation():
    for image_set in sets:
        image_ids = open('./VOC2012/ImageSets/Main/%s.txt' % image_set,
                         encoding='utf-8').read().strip().split()
        list_file = open('%s.txt' % image_set, 'w', encoding='utf-8')
        for image_id in image_ids:
            list_file.write('%s/VOC2012/JPEGImages/%s.jpg' % (getcwd(), image_id))
            convert_annotation(image_id, list_file)
            list_file.write('\n')
        list_file.close()


if __name__ == '__main__':
    voc_yolo()
    voc_annotation()
