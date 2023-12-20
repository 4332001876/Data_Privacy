# Data_Privacy_Lab1
**罗浩铭 PB21030838**

## DP-SGD

### 实验背景
本次实验运行在数据集breast cancer dataset上，该数据集包含569个样本，每个样本有30个特征，表示该肿瘤各特征，每个样本都有一个标签，标签为0或1，表示该样本是否为恶性肿瘤。训练时，将其中455个样本作为训练集，114个样本作为测试集。

本次实验的目的是使用DP-SGD算法训练一个二分类器，使得该二分类器能够对新的样本进行预测，判断该样本是否为恶性肿瘤。

### 代码实现
#### 实验框架debug
代码中的梯度下降公式有误，应改为如下形式（第一条公式可选择下面CrossEntropyLoss与L2Loss中的一种）：
```python
dz = -(y / (predictions + self.tau) - (1 - y) / (1 - predictions + self.tau)) # Cross entropy loss
# dz = predictions - y # L2_loss
dz = dz * (predictions * (1 - predictions)) # sigmoid derivative

dw = np.dot(X.T, dz) / num_samples
db = np.sum(dz) / num_samples
```

同时数据集的输入没有归一化，各输入特征的取值范围差异巨大（至少差4个数量级），导致训练过程中的动力学特性非常差，因此需要加入归一化。
```python
# normalize the data
X = (X - np.mean(X, axis=0)) / X.std(axis=0)
```

#### 使用Advanced Composition Theorem计算




