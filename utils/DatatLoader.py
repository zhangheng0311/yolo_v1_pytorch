import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


class YoloDataset(Dataset):
    def __init__(self, root, image_size, pattern):
        super(YoloDataset, self).__init__()
        """
        :param root: train.txt、val.txt、test.txt三个文件的root路径
        :param pattern: 判断是训练集、验证集还是测试集
        :param mosaic: 判断是否数据增强
        """
        self.root = root
        self.image_size = image_size
        self.pattern = pattern
        self.lines = []
        self.img_path = []
        self.boxes = []
        self.labels = []

        # 判断self.pattern参数是否是str类型
        if isinstance(self.pattern, str):
            if self.pattern == "train":
                self.lines = open(root + "train.txt", 'r', encoding='utf-8').readlines()
                np.random.shuffle(self.lines)
                print("train set total: ", len(self.lines))
            elif self.pattern == "val":
                self.lines = open(root + "val.txt", 'r', encoding='utf-8').readlines()
                print("val set total: ", len(self.lines))
            elif self.pattern == "test":
                self.lines = open(root + "test.txt", 'r', encoding='utf-8').readlines()
                print("test set total: ", len(self.lines))
            else:
                print("YoloDataset函数pattern参数内容有误，请仔细检查！！！")
        else:
            print("YoloDataset函数pattern参数格式有误，请仔细检查！！！")

        for line in self.lines:
            """
            三个.txt文件中每行格式是否如下：
                image_name.jpg x y w h c
                image_name.jpg x y w h c x y w h c 这样表示一张图片中有两个目标
            """
            split = line.strip().split(' ')
            self.img_path.append(split[0])
            num_boxes = (len(split) - 1) // 5
            box = []
            label = []
            for num in range(num_boxes):
                x = float(split[1 + 5 * num])
                y = float(split[2 + 5 * num])
                w = float(split[3 + 5 * num])
                h = float(split[4 + 5 * num])
                cls = split[5 + 5 * num]
                box.append([x, y, w, h])
                label.append(int(cls) + 1)
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))

    def __len__(self):
        """
        获得进来的参数的长度
        """
        return len(self.boxes)

    @staticmethod
    def bgr_hsv(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    def random_scale(self, image, boxes):
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            height, width, c = image.shape
            bgr = cv2.resize(image, (int(width * scale), height))
            scale_tensor = torch.FloatTensor([[scale, 1, scale, 1]]).expand_as(image)
            boxes = boxes * scale_tensor
            return bgr, boxes
        return image, boxes

    # --------------------------------------------------------------------------#
    #  将归一化后的bbox的(center_x,center_y,w,h,cls)数据转换为训练时方便计算Loss的数据形式(S,S,5*B+C)
    #   SxS是grid的尺寸
    #   B是每个grid cel预测B个（bbox，confidence)
    #   C是类别的数目
    #   在yoloV1中限定的是,S=7, B=2, C=20
    #  注意:输入的bbox的信息是(x,y,w,h)格式的，转换为labels后，bbox的信息转换为了(px,py,w,h)格式
    # --------------------------------------------------------------------------#
    @staticmethod
    def boxlabels_target(bbox, label):
        """
        boxes (tensor) [[x1,y1,w1,h1],[x2,y2,w2,h2],[]]
        labels (tensor) [...]
        return SxSx(B*5 + C)
        """
        grid_num = 7
        grid_size = 1. / grid_num
        target = np.zeros((grid_num, grid_num, 30))

        for num in range(len(bbox)):
            # 取值范围在0-6，x对应在哪一列，y对应在哪一行
            grid_x = int(bbox[num][0] // grid_size)  # 当前bbox中心落在第grid_x个网格,列
            grid_y = int(bbox[num][1] // grid_size)  # 当前bbox中心落在第grid_y个网格,行
            # print(grid_x, grid_y)

            # 未进行归一化时时这样计算，(bbox中心坐标 - 网格左上角点的坐标)/网格大小  ==> bbox中心点的相对位置
            # 归一化后，(bbox中心坐标 / 网格大小) - 所在网格的编号 ==> bbox中心点相对网格的坐标
            grid_px = bbox[num][0] / grid_size - grid_x
            grid_py = bbox[num][1] / grid_size - grid_y

            #  将第grid_y行，grid_x列的网格设置为负责当前ground truth的预测，置信度和对应类别概率均置为1
            target[grid_y, grid_x, 0:5] = torch.tensor([grid_px, grid_py, bbox[num][2], bbox[num][3], 1])
            target[grid_y, grid_x, 5:10] = torch.tensor([grid_px, grid_py, bbox[num][2], bbox[num][3], 1])
            # 最后[10:30]是one-host的标签格式，因为标签ID是从1开始，所以要减去1，不然会out range
            target[grid_y, grid_x, 10 + int(label[num]) - 1] = 1
        return target

    # ----------------------------------------------------------------------------#
    #    凡是在类中定义了这个__getitem__ 方法，那么它的实例对象（假定为p），可以像这样p[key]取值，
    #  当实例对象做p[key] 运算时，会调用类中的方法__getitem__。
    #    一般如果想使用索引访问元素时，就可以在类中定义这个方法（getitem(self, key) ）。
    # ————————————————------------------------------------------------------------#
    def __getitem__(self, index):
        """
        当实例对象通过[]运算符取值时，会调用它的方法__getitem__
        @example:
            data = YoloDataset()
            print(data[data.image_size])
            此时实例对象做的是p[key]运算，即data[data.image_size]运算，会调用类中的__getitem__
        @example:
            data = YoloDataset(lines[:100])
            此时实例对象 不会 调用类中的__getitem__
        """
        img_path = self.img_path[index]
        # print("img_path", img_path)
        image = cv2.imread(os.path.join(img_path))
        image = cv2.resize(image, self.image_size)
        image = image.reshape(3, self.image_size[0], self.image_size[1]).astype(np.float32)

        box = self.boxes[index]
        label = self.labels[index]
        target = self.boxlabels_target(box, label)

        return image, target


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    data_root = "./"
    dataset = YoloDataset(root=data_root, image_size=(224, 224), pattern='val', mosaic=True)
    print("dataset init ok !!!!")
    # DataLoader函数在运行过程中会进行dataset[]操作，所以会调用YoloDataset中的__getitem__
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    data_iter = iter(data_loader)
    for i in range(10):
        img, target = next(data_iter)

        show_image = img.numpy()
        show_image = show_image.reshape(224, 224, 3).astype(np.uint8)
        target = target.squeeze(dim=0)
        for m in range(7):
            for n in range(7):
                if target[m, n, 4] == 1:
                    # print(target[m, n, 10:])
                    box = target[m, n, 0:5]
                    # print(box)
                    p_xy = ((box[0] + n) / 7 - box[2] / 2, (box[1] + m) / 7 - box[3] / 2)
                    p_wh = (box[2], box[3])
                    print(p_xy)
                    print(p_wh)
                    x = int(224 * ((box[0] + n) / 7 - box[2] / 2))
                    y = int(224 * ((box[1] + m) / 7 - box[3] / 2))
                    w = int(224 * box[2])
                    h = int(224 * box[3])
                    cv2.rectangle(show_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("image", show_image)

        cv2.waitKey(0)
