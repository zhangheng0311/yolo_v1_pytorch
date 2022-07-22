VOC2012
---

**2021年9月7日**  
VOC2012存放的是每一种类型的challenge对应的图像数据。  
&emsp; Annotations 标注文件.xml格式  
&emsp; ImageSets 每个类型任务对应的图像数据名单 .txt格式  
&emsp; JPEGImages 放置原始图片 .jpg格式  
&emsp; SegmentationClass 按类别进行图像分割，同一类别的物体会被标注为相同颜色  
&emsp; SegmentationObject 按对象进行图像分割，即使是同一类别的物体会被标注为不同的颜色


训练自己的数据集
---
##voc_annotation.py
&emsp; 将各个数据集中的图片路径，标注的bounding boxes位置以及类别id写入训练数据  
&emsp; 每一行对应其**图片路径**、**bounding boxes位置**以及**类别ID**

##VOC2012/voc_yolo.py
&emsp; 读取Annotations文件夹下的所有文件名，同时去掉.xml写入ImageSets/Main下的train.txt、
trainval.txt、val.txt、test.txt四个文件中，完成训练集、验证集和测试集的分类（读取JPEGImages文件夹下的所有文件名
也是可以，但是注意保证所有图片都有对应的标注文件，否则容易出问题。建议选用读取Annotations文件夹下
的所有文件名）
