import timeit
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from network.yolov1.yolov1 import yolov1_net
from network.yolov1.yolo_loss import YoloV1Loss
from utils.DatatLoader import YoloDataset

if __name__ == '__main__':
    # 训练配置参数
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)
    input_size = (224, 224)  # 显存比较大可以使用608x608
    batch_size = 5
    epochs = 30
    lr = 0.0001

    root = "./data/"
    train_dataset = YoloDataset(root=root, image_size=input_size, pattern='train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataset = YoloDataset(root=root, image_size=input_size, pattern='val')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = yolov1_net().cuda()
    yoloV1_loss = YoloV1Loss(lamda_coor=5.0, lamda_noobj=0.5)

    # model.children()里是按模块(Sequential)提取的子模块，而不是具体到每个层，具体可以参见pytorch帮助文档
    # 冻结resnet34特征提取层，特征提取层不参与参数更新
    for layer in model.children():
        layer.requires_grad = False
        break

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    for epoch in range(epochs):

        model.train()
        train_loss = 0.0
        # for images, ground_truth in tqdm(train_loader):
        for i, (images, ground_truth) in enumerate(train_loader):
            images = images.to(device)
            ground_truth = ground_truth.float().to(device)
            predict = model(images)
            loss = yoloV1_loss(predict, ground_truth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss
            if i % 100 == 0:
                print("\n[train] Epoch {}: {}/{} Loss: {}".format(epoch + 1, i, len(train_dataset), train_loss / i))
        print("\n[train] Epoch: {}/{} Loss: {}" .format(epoch+1, epochs, train_loss/len(train_dataset)))

        """
        # 验证
        model.eval()
        start_time = timeit.default_timer()
        bbox_count = 0
        val_corrects = 0.0
        for val_images_batch, val_ground_truth in tqdm(val_loader):
            val_images_batch = val_images_batch.to(device)
            val_ground_truth = val_ground_truth.to(device)
            with torch.no_grad():
                predicts = model(val_images_batch)

            for i in range(val_ground_truth.size(0)):
                for m in range(val_ground_truth.size(1)):  # num_grid_y代表行
                    for n in range(val_ground_truth.size(2)):  # num_grid_x代表列
                        if val_ground_truth[i, m, n, 4] == 1:
                            bbox_count += 1

                            val_labels = val_ground_truth[i, m, n, 10:]
                            # print("\n", val_labels)
                            val_prob_id = torch.max(val_labels, 0)[1]
                            # print("val_prob_id: {}".format(val_prob_id))

                            predict_labels = predicts[i, m, n, 10:]
                            predict_prob = nn.Softmax(dim=0)(predict_labels)
                            # print(predict_prob)
                            predict_prob_id = torch.max(predict_prob, 0)[1]
                            # print("predict_prob_id: {}".format(predict_prob_id))
                            val_corrects += torch.sum(predict_prob_id == val_prob_id)
                            # print(val_corrects)

        print(val_corrects, bbox_count)
        print("[val] Epoch: {}/{} Acc: {}".format(epoch + 1, epochs, val_corrects/bbox_count))
        stop_time = timeit.default_timer()
        print("Execution time: " + str(stop_time - start_time) + "\n")
        """
        if (epoch + 1) == epochs:
            torch.save(model, "./models/yolov1/yolov1_epoch" + str(epoch+1) + ".pth")
