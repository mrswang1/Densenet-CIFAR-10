import csv
from get_data_loader import get_data_loader
from model_utils import save_model,load_model
from densenet import Network
import torch.optim as optim
import torch
import torch.nn as nn
import random
import numpy as np
if __name__ == '__main__':
    seed = 1
    torch.manual_seed(seed)  # 为CPU设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    trainloader,valloader,testloader,classes = get_data_loader()
    net = Network()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1, momentum=0.9)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    net.to(device)
    best_acc = 0
    out = open("/data/ZM/wjdata/cifar10_1_result1.csv", "a")
    csv_writer = csv.writer(out, dialect="excel")
    train_loss = []
    val_acc = []
    for epoch in range(50):  # 多批次循环

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # 获取输入
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度置0
            optimizer.zero_grad()

            # 正向传播，反向传播，优化
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 打印状态信息
            running_loss += loss.item()
        train_loss.append(running_loss / 625)

        # 验证数据集
        correct = 0
        total = 0
        net.eval()
        with torch.no_grad():
            for data in valloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        val_acc.append(acc)
        print('Accuracy of the network on the 10000 val images: ', acc)
        if (acc > best_acc):
            save_model(net, "/data/ZM/wjdata/", "cifar_densenet_1.model")
    print(train_loss)
    print(val_acc)
    csv_writer.writerow(train_loss)
    csv_writer.writerow(val_acc)
    print('Finished Training')