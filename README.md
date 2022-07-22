YOLO:Pytorch实现
---

**2021年9月7日**

## 训练前准备
1. 将图像数据放在./data/VOC2012/JPEGImages文件夹下
2. 将标注数据放在./data/VOC2012/Annotations文件夹下
3. 将只含有类别标签名字的文件放在./data文件夹下

## 训练步骤
1. 本文使用VOC格式进行训练。  
2. 训练前将标签文件放在./data/VOC2012/Annotations 文件夹下  
3. 训练前将图片文件放在./data/VOC2012/JPEGImages 文件夹下  
4. 运行./data 文件夹下的voc_annotation.py生成训练数据（建议运行前查阅与其同目录下的Readme文件）
5. 运行train.py 文件开始训练

## 更新日志
**2021年9月19日**  
    DataLoader.py中的加入数据增强函数  



## 常见问题
1. 测试集预测几乎都是同一个标签：  
    主要是因为训练的迭代次数太少，分类目标的类别太多，以及训练数据集数据量太少。
   
2. 出现显存不够问题：  
    修改DataLoader()函数中batch_size、输入大小、num_workers


## 参考目录
1. [动手学习深度学习pytorch版——从零开始实现YOLOv1](https://blog.csdn.net/weixin_41424926/article/details/105383064)
2. [YOLO详解](https://zhuanlan.zhihu.com/p/25236464)
3. [目标检测（九）--YOLO v1,v2,v3](https://blog.csdn.net/app_12062011/article/details/77554288)
