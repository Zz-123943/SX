# 深度学习代码学习笔记

## 一、代码 1：`train_alex.py`（基于 AlexNet 的图像分类训练代码）

### （一）代码功能
实现基于 AlexNet 架构的图像分类模型训练，使用自定义数据集。

### （二）代码结构与关键点

#### 数据集加载
- 使用自定义 `ImageTxtDataset` 类加载数据。
- 数据路径：`E:\dataset\image2\train`，标签存储在 `train.txt`。
- 数据预处理：
  - 调整大小：`transforms.Resize(224)`。
  - 数据增强：`transforms.RandomHorizontalFlip()`。
  - 转换为张量并归一化：`transforms.ToTensor()` 和 `transforms.Normalize`。

#### 模型定义
- 简化的 AlexNet 模型：
  - 5 个卷积层 + 3 个全连接层。
  - 输出层为 10 类分类任务。

#### 训练流程
- 批量大小：64。
- 损失函数：`CrossEntropyLoss`。
- 优化器：`SGD`，学习率 0.01，动量 0.9。
- 每 500 步记录训练损失，使用 TensorBoard 可视化。
- 每个 epoch 结束后评估测试集并保存模型。

#### 测试流程
- 禁用梯度计算：`torch.no_grad()`。
- 计算测试集损失和准确率，记录到 TensorBoard。

### （三）学习重点
- 自定义数据集的使用和预处理。
- AlexNet 架构的理解与调整。
- 训练与测试流程的实现。
- 数据增强技术的应用。

---

## 二、代码 2：`transformer.py`（基于 Transformer 的 Vision Transformer 模型代码）

### （一）代码功能
实现基于 Transformer 架构的 Vision Transformer（ViT）模型，处理序列化图像数据。

### （二）代码结构与关键点

#### 模块定义
- **FeedForward**：线性层 + GELU + Dropout。
- **Attention**：多头自注意力机制。
- **Transformer**：Transformer 层，包含注意力模块和前馈模块。
- **ViT**：将图像序列化为 patches，通过 Transformer 处理。

#### 模型结构
- 输入：序列化图像 `(batch_size, channels, seq_len)`。
- 输出：分类结果 `(batch_size, num_classes)`。

#### 测试代码
- 创建 ViT 模型实例，输入随机张量，验证模型输出。

### （三）学习重点
- Transformer 架构原理：多头自注意力、前馈网络、残差连接。
- Vision Transformer（ViT）的实现：图像序列化、位置嵌入、类别嵌入。
- `einops` 库的使用：简化张量操作。

---

## 三、总结
今日学习了两个深度学习模型：
1. **`train_alex.py`**：基于 AlexNet 的图像分类模型，重点在于自定义数据集的加载、预处理、训练与测试流程，以及数据增强技术。
2. **`transformer.py`**：基于 Transformer 的 Vision Transformer 模型，核心在于理解 Transformer 架构原理及其在图像数据中的应用，以及 `einops` 库的使用。

通过学习，加深了对 CNN 和 Transformer 架构的理解，掌握了自定义数据集处理和数据增强技术。

---

