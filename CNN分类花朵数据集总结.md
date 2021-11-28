### CNN分类花朵数据集总结

1. #### 数据集导入：

   这部分大概做了两三天的样子(就硬托)。拿到的数据集不是csv那种格式的，就是分好的图片。这种数据集的处理方法我试了两种。

   ##### 1.1 用ImageFolder做，用法如下：

   ```python
   from torchvision.datasets import ImageFolder
   from torchvision import transforms
   transform=transforms.compose([transforms.ToTensor])
   dataset=ImageFolder('data_dir',transform=transform)
   ```

   dataser是按文件顺序排列的数组，第一维是图片的张量，第二维是图片的标签。我没有用这种方法，因为我需要划分训练集和验证集，但是得到dataset后我不会划分。

   ##### 1.2 定义一个自己的数据集类：

   ```python
   from torch.utils.data import Dataset
   from PIL import Image
   class train_set(Dataset):
       def __init__(self):
           self.imgs=train_img
           self.label=train_label
       def __getitem__(self,index):
           fn=self.imgs[index]
           image=Image.open(fn)
           image_tensor=transform(image)
           label=self.label[index]
           return image_tensor,label
       def __len__(self):
           return len(self.imgs)
   ```

   这里最重要的是重写后两个方法，其中getitem是如何根据索引得到对应的图片和标签，方法任意，我的做法是先将所有图片的路径和标签保存到两个列表中，然后将其赋给数据集类的两个成员，在getitem中通过索引读取相应路径和标签，然后通过PIL的Image.open()读取图片，通过之前定义的transform将图片转为张量。

   ##### 1.3 使用DataLoader:

   DataLoader本质上是一个迭代器，每次迭代会返回一个Batch大小的数据

   ```python
   train_data=train_set()
   train_iter=data.DataLoader(train_data, 64, shuffle=True,
                               num_workers=4)
   test_data=test_set()
   test_iter=data.DataLoader(train_data, 64, shuffle=True,
                               num_workers=4)
   ```

   第二个参数batch_size大小很重要，我现在是拿80%的数据作训练集，一开始batch_size设为100，然后是128，256结果发现最后CNN的准确率到了70%就极限了，后来想到是不是数据太少，然后batch_size设的太大导致的，调到64后到了0.97，此时我的batch_size:train_set=0.013

2. #### CNN网络结构

   CNN的网络结构大概就是卷积层，池化层，卷积层，池化层，全链接层

   卷积层的目的就是通过卷积核识别出图片的某个特征，然后在神经网络中一层卷积层中通常有多个卷积核，提取多个特征。(当输入有3个通道，卷积核只有一个时，输出通道也只有一个，也就是说__卷积核数=输出通道数__)

   池化层有最大池化层和平均池化层，但目前来说常用的是最大池化层，最大池化层不仅可以降低feature map的分辨率，减少运算复杂度，增大感受野，还保留了相应的主要特征(去掉不必要的特征)。

   后面的全链接层就相当于多层感知机了。

   我定义的网络模型如下：

   ```python
   net = nn.Sequential(
       nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2), nn.ReLU(),
       nn.MaxPool2d(kernel_size=4, stride=2),
       nn.Conv2d(32, 64, kernel_size=3, stride=2,padding=2), nn.ReLU(),
       nn.MaxPool2d(kernel_size=2, stride=2),
       nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
       nn.MaxPool2d(kernel_size=2, stride=2),
       nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
       nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
       nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
       nn.Flatten(),
       # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过度拟合
       nn.Linear(1024, 1024), nn.ReLU(),
       nn.Dropout(p=0.2),
       nn.Linear(1024, 512), nn.ReLU(),
       nn.Dropout(p=0.1),
       nn.Linear(512, 4))
   ```

   大概就是AlexNet的网络结构

3. #### 训练模型

   学习率设为0.02，epoch设为100，调用d2l.train_ch6()训练，结果如下：

   ![cnn_flower_result](/home/li/图片/cnn_flower_result.png)

   d2l.train_ch6()是__动手学深度学习__附带的代码，目的是用GPU训练神经网络模型，代码如下：

   ```python
   def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
       """用GPU训练模型(在第六章定义)。"""
       def init_weights(m):
           if type(m) == nn.Linear or type(m) == nn.Conv2d:
               nn.init.xavier_uniform_(m.weight)
       net.apply(init_weights)
       print('training on', device)
       net.to(device)
       optimizer = torch.optim.SGD(net.parameters(), lr=lr)
       loss = nn.CrossEntropyLoss()
       animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                               legend=['train loss', 'train acc', 'test acc'])
       timer, num_batches = d2l.Timer(), len(train_iter)
       for epoch in range(num_epochs):
           # 训练损失之和，训练准确率之和，范例数
           metric = d2l.Accumulator(3)  
           net.train()
           for i, (X, y) in enumerate(train_iter):
               timer.start()
               optimizer.zero_grad()
               X, y = X.to(device), y.to(device)
               y_hat = net(X)
               l = loss(y_hat, y)
               l.backward()
               optimizer.step()
               with torch.no_grad():
                   metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
               timer.stop()
               train_l = metric[0] / metric[2]
               train_acc = metric[1] / metric[2]
               if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                   animator.add(epoch + (i + 1) / num_batches,
                                (train_l, train_acc, None))
           test_acc = evaluate_accuracy_gpu(net, test_iter)
           animator.add(epoch + 1, (None, None, test_acc))
       print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
             f'test acc {test_acc:.3f}')
       print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
             f'on {str(device)}')
   ```

   代码流程：

   对模型的卷积层和全链接层的权重参数初始化，其中用了xavier初始化的方法

   ```python
   nn.init.xacier_uniform_(weight)
   ```

   将模型参数（卷积层和全链接层的权重）移到GPU上

   ```python
   net.to(device)
   ```

   用小批量随即梯度下降法更新权重：

   ```python
   optimizer=torch.optim.SGD(net.parameters(),lr=lr)
   ```

   损失函数使用交叉熵损失函数（真实概率分布与预测概率分布之间的差异。交叉熵的值越小，模型预测效果就越好。）：

   ```python
   loss=nn.ClossEntropyLoss()
   ```

   每一批量的计算流程如下：

   清空优化器（SGD）的梯度

   计算预测值

   计算损失

   计算梯度

   更新参数

   在不计算梯度的情况下计算准确度（with torch.no_grad():）

