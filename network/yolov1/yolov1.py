# -------------------------------------------------------------------------------------------------#
#     原论文中是采用自己设计的20层卷积层在ImageNet上预训练了一周，完成特征提取部分的训练。学习者复现不易，因此对原文结构进
#  行一点改变。yolov1的前20层(16个卷积层+4个最大池化层)是用于特征提取，也就是随便替换成一个分类网络（除去全连接层）即可。
#  此处用ResNet34的网络作为特征提取部分。
#     pytorch的torchvision提供了ResNet34的与训练模型，训练集也是ImageNet，现成训练好的模型可直接使用，免去特征提取部
#  分的训练时间。
# -------------------------------------------------------------------------------------------------#

import torchvision.models as model
import torch.nn as nn
import torch


class yolov1_net(nn.Module):
    def __init__(self):
        super(yolov1_net, self).__init__()
        self.NUM_BBOX = 2
        self.CLASSES = 20

        resnet = model.resnet34(pretrained=True)
        resnet_out_channel = resnet.fc.in_features
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # 去除resnet的后两层
        # 后接yolov1剩下的4个卷积层+2个全连接层
        self.Conv_layers = nn.Sequential(
            nn.Conv2d(resnet_out_channel, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),  # 为了加快训练，这里增加了BN层，原论文里YOLOv1是没有的
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU()
        )
        self.Conn_layers = nn.Sequential(
            nn.Linear(4 * 4 * 1024, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 7 * 7 * 30),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.resnet(input)
        output = self.Conv_layers(output)
        output = output.view(output.size()[0], -1)
        output = self.Conn_layers(output)
        return output.reshape(-1, 7, 7, (5 * self.NUM_BBOX + self.CLASSES))  # 记住最后要reshape一下输出数据


if __name__ == '__main__':
    x = torch.randn((1, 3, 224, 224))
    net = yolov1_net()
    print(net)
    y = net.forward(x)
    print(y.shape)
