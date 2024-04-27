### 从零开始构建三层神经网络分类器，实现图像分类

* 任务描述：
  手工搭建三层神经网络分类器，在数据集[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)上进行训练以实现图像分类

* 基本条件：

  （1）不使用pytorch，tensorflow等现成的支持自动微分的深度学习框架，使用了numpy
  （2）代码中包含**模型**、**训练**、**测试**和**参数查找**等部分，进行了模块化设计
  （3）其中模型部分允许自定义隐藏层大小、激活函数类型，支持通过反向传播计算给定损失的梯度；训练部分实现了SGD优化器、学习率下降、交叉熵损失和L2正则化，并能根据验证集指标自动保存最优的模型权重；参数查找环节可以调节学习率、隐藏层大小、正则化强度等超参数，观察并记录模型在不同超参数下的性能；测试部分支持导入训练好的模型，输出在测试集上的分类准确率（Accuracy）

  

### 下载数据

模型的训练和预测使用的是经典的Fashion-MNIST数据集，从[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)仓库可以下载，将四个文件名为 `.gz` 的文件移至本项目的 `/data` 目录下

### 模型训练

* 在根目录下运行 `python train.py`，可以自行指定模型训练中的参数

  ```powershell
  python train -e 50 -vs 1000 -bs 32 -lr 0.01 -l2 0.001 -dr 0.95 -ds 5000
  ```
  
* 参数含义：
  * -e：训练epoch数，默认50
  * -vs：valid_size 测试集大小，默认1000
  * -bs：batch_size，默认32
  * -lr：学习率，默认0.01
  * -l2：L2正则化参数，默认0.001
  * -dr：学习率衰减率，默认0.95
  * -ds：学习率衰减步数，默认5000（未使用衰减）

* 模型结构有关的超参数（如各层神经元的数量以及激活函数），则可以在 `train.py` 文件中进行修改


### 模型测试

* 运行了模型训练代码`train.py`后，如选择了`save_model=TRUE`，则会在`model/` 目录下保存训练出的最优模型参数

* 在根目录，运行以下代码即可读取最优模型参数进行预测：

  ```powershell
  python test.py
  ```

* 将会输出模型在测试集上的 $Loss$ 和 $Accuracy$

### 模型参数可视化

* 在模型训练完成后，在根目录运行以下代码可以对保存的最优模型进行参数可视化

  ```powershell
  python PlotParam.py
  ```

### 超参数网格搜索

* 运行 `param_select.py`，即可进行超参数的网格搜索

  ```powershell
  python param_select.py
  ```

* 在代码文件内可以自行设置搜索参数的网格值
* 最终各网格值的 `Loss` 和 `Accuracy` 会自动保存在 `paramsearch_results.json` 文件中

### 