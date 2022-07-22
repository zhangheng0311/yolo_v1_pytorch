import cv2
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from utils.DatatLoader import YoloDataset

from torch.autograd import Variable
from network.yolov1.yolo_loss import YoloV1Loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

# 用来标识20个类别的bbox颜色，可自行设定
COLOR = [(255, 0, 0), (255, 125, 0), (255, 255, 0), (255, 0, 125), (255, 0, 250),
         (255, 125, 125), (255, 125, 250), (125, 125, 0), (0, 255, 125), (255, 0, 0),
         (0, 0, 255), (125, 0, 255), (0, 125, 255), (0, 255, 255), (125, 125, 255),
         (0, 255, 0), (125, 255, 125), (255, 255, 255), (100, 100, 100), (0, 0, 0), ]

classes = []
classes_file = open('./data/VOC2012/Labels/labels.txt').read().strip().split()
for label in classes_file:
    classes.append(label)


def target_bbox(predict):
    """
    将每个grid中预测的bbox提取出来，共有2*7*7个，变成(98, len(bbox)+len(labels)),即(98, 25)
    :param predict 输入的预测结果为(7,7,30)格式
    :return nms处理后的结果
    """
    if predict.size()[0:2] != (7, 7):
        raise ValueError("Error: Wrong labels size:", predict.size())

    num_grid_x, num_grid_y = predict.size()[0:2]
    bboxes = torch.zeros((98, 25))
    for m in range(num_grid_y):  # num_grid_y代表行
        for n in range(num_grid_x):  # num_grid_x代表列
            bboxes[2 * (m * 7 + n), 0:4] = torch.Tensor([(predict[m, n, 0] + n) / 7 - predict[m, n, 2] / 2,
                                                         (predict[m, n, 1] + m) / 7 - predict[m, n, 3] / 2,
                                                         (predict[m, n, 0] + n) / 7 + predict[m, n, 2] / 2,
                                                         (predict[m, n, 1] + m) / 7 + predict[m, n, 3] / 2])
            bboxes[2 * (m * 7 + n), 4] = predict[m, n, 4]
            bboxes[2 * (m * 7 + n), 5:] = predict[m, n, 10:]
            bboxes[2 * (m * 7 + n) + 1, 0:4] = torch.Tensor([(predict[m, n, 5] + n) / 7 - predict[m, n, 7] / 2,
                                                             (predict[m, n, 6] + m) / 7 - predict[m, n, 8] / 2,
                                                             (predict[m, n, 5] + n) / 7 + predict[m, n, 7] / 2,
                                                             (predict[m, n, 6] + m) / 7 + predict[m, n, 8] / 2])
            bboxes[2 * (m * 7 + n) + 1, 4] = predict[m, n, 9]
            bboxes[2 * (m * 7 + n) + 1, 5:] = predict[m, n, 10:]

    return nms(bboxes)


# 非极大值抑制算法
def nms(bboxes, conf_thresh=0.1, iou_thresh=0.3):
    num = bboxes.size()[0]

    bboxes_prob = bboxes[:, 5:]  # 预测类别的条件概率，P(Class|Object)
    bboxes_object_confidence = bboxes[:, 4].clone().unsqueeze(1).expand_as(bboxes_prob)  # 预有无Object的置信度，即P(Object)*IOU
    bboxes_class_conf = bboxes_object_confidence * bboxes_prob  # Object置信度*类别条件概率=具体类别的置信分数。P(Class)*IOU
    bboxes_class_conf[bboxes_class_conf <= conf_thresh] = 0  # 将具体类别的置信低于阈值的bbox忽略 (98, 20)

    result = []
    max_class_conf = torch.max(bboxes_class_conf, 0)[0]  # 寻找每个类别的最大置信度,从20列中找出每列的最大值
    print(max_class_conf)
    for cls in range(20):
        if max_class_conf[cls] != 0:  # 找出每类是否有最大置信度
            max_class_conf_list = torch.max(bboxes_class_conf, 0)[1]
            max_class_conf_index = max_class_conf_list[cls]
            # print(max_class_conf_index)
            # print(bboxes_class_conf[max_class_conf_index, cls])
            max_conf_box = [bboxes[max_class_conf_index, 0], bboxes[max_class_conf_index, 1],
                           bboxes[max_class_conf_index, 2], bboxes[max_class_conf_index, 3]]
            # print(max_conf_box)
            for n in range(num):
                if bboxes_class_conf[n, cls] != 0 and n != max_class_conf_index:
                    box = [bboxes[n, 0], bboxes[n, 1],
                           bboxes[n, 2], bboxes[n, 3]]
                    iou = YoloV1Loss.calculate_iou(box, max_conf_box)
                    if iou < iou_thresh:
                        result.append([box, cls, bboxes_class_conf[n, cls]])
            result.append([max_conf_box, cls, bboxes_class_conf[max_class_conf_index, cls]])

        else:
            continue

    return result


def draw_box(img, box):
    h, w = img.shape[0:2]
    # print(len(box))
    for i in range(len(box)):
        p1 = (int(w * box[i][0][0]), int(h * box[i][0][1]))
        p2 = (int(w * box[i][0][2]), int(h * box[i][0][3]))

        cls_name = classes[int(box[i][1])]
        confidence = str(box[i][2])
        cv2.rectangle(img, p1, p2, COLOR[int(box[i][1])], 2)
        cv2.putText(img, cls_name, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255))

    cv2.imshow("bbox", img)
    cv2.waitKey(0)


if __name__ == '__main__':
    input_size = (224, 224)
    batch_size = 1

    root = "./data/"
    test_dataset = YoloDataset(root=root, image_size=input_size, pattern='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = torch.load(r'./models/yolov1/yolov1_epoch30.pth').cuda()

    data_iter = iter(test_loader)
    for i in range(len(test_loader)):
        image, target = next(data_iter)
        # Input = Variable(image).cuda()
        # test_predict = model(Input)
        # test_predict = test_predict.squeeze(dim=0)
        # bbox = target_bbox(test_predict)
        test_label = target.squeeze(dim=0)
        bbox = target_bbox(test_label)

        show_image = image.numpy()
        show_image = show_image.reshape(224, 224, 3).astype(np.uint8)
        draw_box(show_image, bbox)
