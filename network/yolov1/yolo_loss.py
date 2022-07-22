import torch
import torch.nn as nn
from network.yolov1.yolov1 import yolov1_net


class YoloV1Loss(nn.Module):
    def __init__(self, lamda_coor=None, lamda_noobj=None):
        super(YoloV1Loss, self).__init__()
        if lamda_coor is not None:
            self.lamda_coor = lamda_coor
        if lamda_noobj is not None:
            self.lamda_noobj = lamda_noobj

    def forward(self, predict, ground_truth):
        """
        :param predict: 预测的网络输出(batch, 7, 7, 30)
        :param ground_truth: 样本真实标签(batch, 7， 7, 30)
        :return: yoloV1_loss yoloV1的损失
        """
        batch = ground_truth.size()[0]
        num_grid_x, num_grid_y = ground_truth.size()[1:3]
        coordinate_loss = 0
        confidence_loss = 0
        classes_loss = 0

        for i in range(batch):
            for m in range(num_grid_y):  # num_grid_y代表行
                for n in range(num_grid_x):  # num_grid_x代表列
                    if ground_truth[i, m, n, 4] == 1:  # 如果包含obj
                        ground_truth_bbox = self.target_bbox(ground_truth[i, m, n, 0:4],
                                                             (num_grid_x, num_grid_y), (m, n))
                        # print("ground_truth_bbox:", ground_truth_bbox)
                        bbox_1 = self.target_bbox(predict[i, m, n, 0:4], (num_grid_x, num_grid_y), (m, n))
                        bbox_2 = self.target_bbox(predict[i, m, n, 5:10], (num_grid_x, num_grid_y), (m, n))

                        iou_1 = self.calculate_iou(bbox_1, ground_truth_bbox)
                        iou_2 = self.calculate_iou(bbox_2, ground_truth_bbox)

                        if iou_1 >= iou_2:
                            coordinate_loss = coordinate_loss \
                                + (torch.sum((predict[i, m, n, 0:2] - ground_truth[i, m, n, 0:2]) ** 2)
                                   + torch.sum((predict[i, m, n, 2:4].sqrt() - ground_truth[i, m, n, 2:4].sqrt()) ** 2))
                            # iou小的那个是看作noobj的，所以在算的confidence_loss的时候需要两个confidence_loss
                            # 因为iou_1 > iou_2,所以iou_1计算是obj的confidence_loss，iou_2计算是noobj的confidence_loss
                            confidence_loss = confidence_loss + (predict[i, m, n, 4] - iou_1) ** 2 \
                                              + self.lamda_noobj * (predict[i, m, n, 9] - iou_2) ** 2
                        else:
                            coordinate_loss = coordinate_loss \
                                + (torch.sum((predict[i, m, n, 5:7] - ground_truth[i, m, n, 5:7]) ** 2)
                                   + torch.sum((predict[i, m, n, 7:9].sqrt() - ground_truth[i, m, n, 7:9].sqrt()) ** 2))
                            # 同理，因为iou_1 < iou_2,所以iou_2计算是obj的confidence_loss，iou_1计算是noobj的confidence_loss
                            confidence_loss = confidence_loss + (predict[i, m, n, 9] - iou_2) ** 2 \
                                              + self.lamda_noobj * (predict[i, m, n, 4] - iou_1) ** 2

                        label = torch.max(predict[i, m, n, 10:], 0)[1]
                        print("predict: ", label)
                        true_label = torch.max(ground_truth[i, m, n, 10:], 0)[1]
                        print("ground_truth: ", true_label)
                        classes_loss = classes_loss + torch.sum((predict[i, m, n, 10:] - ground_truth[i, m, n, 10:])**2)
                    else:
                        # grid中不包含obj，所以预测的两个bbox都是计算noobj的confidence_loss
                        confidence_loss = confidence_loss + self.lamda_noobj * torch.sum(predict[i, m, n, [4, 9]] ** 2)

        return self.lamda_coor * coordinate_loss + confidence_loss + classes_loss

    @staticmethod
    def target_bbox(target, num_grid, grid):
        """
        将px,py转换为cx,cy，即相对网格的位置转换为原来相对整张图归一化后实际的bbox中心位置cx,xy,所以此处(x,y,w,h)依旧都是0-1之间值
        然后再利用(cx-w/2,cy-h/2,cx+w/2,cy+h/2)转换为(x1,y1,x2,y2)形式，用于计算iou
        :param target: (px,py,w,h)
        :param num_grid: (num_grid_y, num_grid_x),行与列的grid数目
        :param grid: 选定的是第几个grid
        :return: bbox: (x1,y1,x2,y2)
        """
        bbox = [0, 0, 0, 0]
        bbox[0] = (target[0] + grid[1]) / num_grid[1] - target[2] / 2
        bbox[1] = (target[1] + grid[0]) / num_grid[0] - target[3] / 2
        bbox[2] = (target[0] + grid[1]) / num_grid[1] + target[2] / 2
        bbox[3] = (target[1] + grid[0]) / num_grid[0] + target[3] / 2

        return bbox

    @staticmethod
    def calculate_iou(bbox_1, bbox_2):
        """
        计算bbox_1与bbox_2的交并比
        :param bbox_1: (x1, y1, x2, y2)
        :param bbox_2: (x1, y1, x2, y2)
        :return:
        """
        intersect_bbox = [0, 0, 0, 0]  # 两个bbox的交集
        if bbox_1[2] < bbox_2[0] or bbox_1[3] < bbox_2[1] or bbox_1[0] > bbox_2[2] or bbox_1[1] > bbox_2[3]:
            # pass
            return 0
        else:
            intersect_bbox[0] = max(bbox_1[0], bbox_2[0])
            intersect_bbox[1] = max(bbox_1[1], bbox_2[1])
            intersect_bbox[2] = min(bbox_1[2], bbox_2[2])
            intersect_bbox[3] = min(bbox_1[3], bbox_2[3])

        area_1 = (bbox_1[2] - bbox_1[0]) * (bbox_1[3] - bbox_1[1])  # bbox1面积
        area_2 = (bbox_2[2] - bbox_2[0]) * (bbox_2[3] - bbox_2[1])  # bbox2面积
        area_intersect = (intersect_bbox[2] - intersect_bbox[0]) * (intersect_bbox[3] - intersect_bbox[1])  # 交集面积

        if area_intersect > 0:
            return area_intersect / (area_1 + area_2 - area_intersect)

        else:
            return 0


if __name__ == '__main__':
    label = torch.randn((1, 7, 7, 30))
    x = torch.randn((1, 3, 448, 448))
    net = yolov1_net()
    output = net.forward(x)
    yolo_loss = YoloV1Loss(lamda_coor=5, lamda_noobj=0.5)
    loss = yolo_loss.forward(output, label)
    print(loss)
