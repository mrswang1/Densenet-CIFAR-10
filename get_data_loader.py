import torch
import torchvision
import torchvision.transforms as transforms

def get_data_loader():
    # 下载训练集和测试集，并将它们进行归一化处理
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='/data/ZM/wjdata', train=True,
                                            download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(root='/data/ZM/wjdata', train=False,
                                           download=True, transform=transform)

    print(len(trainset), len(testset))
    # 将训练集划分为训练集和验证集
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset=trainset,
        lengths=[40000, 10000],
        generator=torch.Generator().manual_seed(0)
    )
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader,valloader,testloader,classes