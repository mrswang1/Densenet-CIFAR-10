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
![](/loss.png) 
  
|  lr     | val | test |
|  ---- | ----  | ----  |
| 0.001 | 71.39% | 66% |
| 0.01  | 80.12% | 77% |
| 0.1   |82.93% | 82% |
| 1.0   | 30.11%  |10%  |
