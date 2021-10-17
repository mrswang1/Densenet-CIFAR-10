# Densenet for CIFAR-10
This repository is about some implementations of Densenet Architecture for cifar10.  
  
I just use Pytorch to implementate all of these models.  
  
## Requirements
python        3.8.8  
torch         1.7.1  
torchvision   0.8.2  
numpy         1.20.1  
  
## Contents
- **get_data_loader.py** 数据处理文件（加载数据、将数据集划分为训练集和验证集（4:1)、将数据转换为DataLoader形式）
- **densenet.py** 模型文件
- **train.py** 训练文件
- **test.py** 测试文件
- **model_utils.py** 加载和保存模型文件
  
## Training
  `python train.py`  
    
## Testing
  `python test.py`  
  
## Result
本实验采用的优化器是SGD,通过使用不同的learning rate，得到如下的实验结果。  
- 首先得到了在不同learning rate的情况下，迭代步数与损失值之间的关系。  
  - 由图1我们可以发现当lr=1,发生了梯度爆炸，loss震动幅度非常大，模型无法收敛。  
  - 由图2我们可以发现随着learning rate的减小，损失函数收敛的速度变慢。虽然使用低学习率可以确保我们不会错过任何局部极小值，但也意味着我们将花费更长的时间来进行收敛。    
![image1](https://github.com/mrswang1/Densenet-CIFAR-10/blob/main/loss2.jpg)
![image](https://github.com/mrswang1/Densenet-CIFAR-10/blob/main/loss.jpg) 

- 本实验采用的评价指标为Accuracy，通过使用不同的learning rate，得到了验证集和测试集的Accuracy。  
  - 通过表格我们可以发现，当lr=1时，模型的效果非常差。
  - 随着lr的降低，验证集的准确率和测试集的准确率之间的差值越来越大。这说明当学习率过低时，模型容易产生过拟合的现象。  
|  lr     | val | test |
|  ---- | ----  | ----  |
| 0.001 | 71.39% | 66% |
| 0.01  | 80.12% | 77% |
| 0.1   |82.93% | 82% |
| 1.0   | 30.11%  |10%  |
