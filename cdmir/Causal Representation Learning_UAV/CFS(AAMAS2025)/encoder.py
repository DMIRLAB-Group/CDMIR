import torch
import torch.nn as nn


def tie_weights(src, trg):  #将两个模型的权重连接在一起，确保它们共享相同的权重和偏差
    assert type(src) == type(trg)   #检查源模型和目标模型的类型是否相同。如果它们的类型不同，会引发一个 AssertionError 异常，表示这两个模型无法连接。
    trg.weight = src.weight #将源模型的权重参数 src.weight 赋值给目标模型的权重参数 trg.weight
    trg.bias = src.bias #将源模型的偏差参数 src.bias 赋值给目标模型的偏差参数 trg.bias


OUT_DIM = {4: 35}   #指定了不同卷积层级别的输出维度。在这里，只定义了一个卷积层级别为4时的输出维度

class PixelEncoder(nn.Module):  #卷积神经网络（CNN）编码器，用于处理像素观察数据
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32):
        super().__init__()

        assert len(obs_shape) == 3  #检查输入的 obs_shape 是否具有三个维度，以确保输入是一个三通道的图像观测

        self.feature_dim = feature_dim  #将传递给构造函数的 feature_dim 参数赋值给类的 feature_dim 属性，表示编码器输出的特征维度
        self.num_layers = num_layers    #将传递给构造函数的 num_layers 参数赋值给类的 num_layers 属性，表示卷积层的数量

        #创建了一个卷积层，用于处理输入图像观测数据（通道数由 obs_shape[0] 决定），并将输出的特征图的深度设置为 num_filters。卷积操作将在输入图像上执行，卷积核的大小是 3x3，卷积核的滑动步幅为 2。这个卷积层通常用于深度学习模型中，以提取输入图像的特征信息
        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        #创建多个卷积层，每个新创建的卷积层的输入和输出通道数都是 num_filters，卷积核的大小是 3x3，卷积核的滑动步幅是 1。这些卷积层通常用于深度神经网络中，以逐渐提取输入图像的特征信息，并创建多个层次的特征表示。这种堆叠的卷积层结构有助于网络学习从原始图像中提取有用的抽象特征。
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = OUT_DIM[num_layers]   #根据 num_layers 参数选择适当的输出维度
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)#创建了一个全连接层（Fully Connected Layer），将卷积层提取的特征映射到一个更高级的表示，以便用于后续的任务，例如分类或回归
        self.ln = nn.LayerNorm(self.feature_dim)#创建了一个 Layer Normalization（层归一化）层，用于对神经网络的输出进行归一化处理。有助于缓解梯度消失问题，提高网络的训练速度和稳定性，并且通常在深度神经网络的各层之间使用。在这里，Layer Normalization 被应用于全连接层的输出，以确保输出在每个特征维度上保持稳定。这通常有助于网络更快地学习，并且对于一些任务来说，提高了泛化性能。
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.outputs = dict()   #存储各个网络层输出的字典。在前向传播过程中，每个层的输出将被存储在这个字典中，以便稍后检查或使用。

    def reparameterize(self, mu, logstd):   #从概率分布中采样一个值
        std = torch.exp(logstd) #计算了标准差 std,在正态分布中，标准差是对数标准差的指数形式，因此需要使用 torch.exp 来将 logstd 转换为标准差
        eps = torch.randn_like(std) #生成了一个与 std 相同形状的随机数 eps，其中每个元素都是从标准正态分布（均值为0，标准差为1）中独立抽取的。这个随机数向量 eps 通常被称为"噪声"或"随机样本"，它引入了模型的随机性
        return mu + eps * std   #执行了"重参数化技巧"，将随机样本 eps 缩放（乘以 std，标准差）并偏移（加上 mu，均值），从而生成了一个符合给定分布（在这里是高斯分布）的随机样本

    def forward_conv(self, obs):    #处理输入观察数据，通过卷积层和线性层生成特征表示
        obs = obs / 255.    #对输入的观察数据 obs 进行了归一化处理，将像素值的范围从 [0, 255] 缩放到 [0, 1]
        self.outputs['obs'] = obs   #将归一化后的观察数据存储在成员变量 outputs 中，以便之后可以查看和分析

        conv = torch.relu(self.convs[0](obs))   #将输入的图像数据通过卷积操作和激活函数生成一个新的特征图 conv
        self.outputs['conv1'] = conv    #将第一次卷积的结果 conv 存储在 outputs 中，以供查看和分析。

        #通过循环执行多次卷积操作，每次卷积都生成一个新的特征图，并将这些特征图存储在字典中，以供后续分析和使用。这是卷积神经网络中常见的特征提取过程。
        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))#获取模型的第 i 个卷积层 self.convs[i]，然后使用这个卷积层对输入的特征图 conv 进行卷积操作。这一步骤生成了一个新的特征图 conv，代表了经过当前卷积层的特征提取结果。
            self.outputs['conv%s' % (i + 1)] = conv #生成的特征图 conv 存储在一个字典 self.outputs 中
        # conv_avg = self.avgpool(conv)
        # print(conv_avg.shape)   #(8,32,1,1)
        #将卷积层产生的特征图 conv 转换成一个二维矩阵 h，其中每一行对应一个样本，每一列对应特征的某个维度。这种变形通常用于连接卷积层之后的全连接层，以便将卷积层的输出转化成全连接层可以处理的形式。这是深度学习中常见的操作，用于连接不同类型的神经网络层。
        h = conv.contiguous().view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):   #对输入数据进行前向传播，最终生成特征表示。然后通过一系列的线性变换、规范化和激活操作，最终生成一个模型的输出，该输出用于后续的强化学习任务
        h = self.forward_conv(obs)  #对输入数据进行卷积和激活操作，生成特征表示 h。这个特征表示是从输入数据中提取的有用信息，通常用于后续的学习和决策
        #用于控制是否应该从 h 中分离中间张量。在深度强化学习中，有时候需要保留梯度信息，有时候则需要断开梯度信息的传播，这取决于具体的训练策略。如果 detach 为 True，则 h 的梯度信息将被断开，不会影响后续的梯度计算
        if detach:
            h = h.detach()

        h_fc = self.fc(h)   #通过全连接层 (fc) 对特征 h 进行线性变换，这个操作可以理解为一个带有权重和偏置的线性变换
        self.outputs['fc'] = h_fc

        h_norm = self.ln(h_fc)  #应用了层归一化 (ln) 操作，对全连接层的输出进行规范化。层归一化有助于稳定神经网络的训练过程，确保不同样本的特征具有相似的分布
        self.outputs['ln'] = h_norm

        out = torch.tanh(h_norm)    #通过双曲正切函数 (tanh) 对规范化后的特征进行激活，将其缩放到 [-1, 1] 的范围内。这通常用于输出层，以产生模型的最终输出
        self.outputs['tanh'] = out

        return out  #将激活后的特征 out 作为网络的输出返回。这个输出通常代表了智能体对环境的某种估计或决策

    def copy_conv_weights_from(self, source):   #复制另一个编码器的卷积层的权重，以确保它们共享相同的权重
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):    #遍历了当前编码器中的卷积层
            tie_weights(src=source.convs[i], trg=self.convs[i]) #调用了之前定义的 tie_weights 函数，将卷积层 source.convs[i] 的权重连接到当前编码器的对应卷积层 self.convs[i] 上。这确保了这两个卷积层具有相同的权重和偏置，从而实现了权重共享。


class IdentityEncoder(nn.Module):   #这是一个恒等编码器，用于处理一维观察数据。它不执行任何特征提取，只返回输入数据本身。这对于处理非图像型观察数据可能很有用
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters):
        super().__init__()

        assert len(obs_shape) == 1  #用于确保观察数据的形状是一维的，如果不是一维的话，会触发一个断言错误
        self.feature_dim = obs_shape[0] #将特征维度 feature_dim 设置为输入观察数据的长度，因为在恒等编码器中，输出特征的维度与输入观察数据的维度相同

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):   #这是一个用于复制卷积层权重的方法
        pass



_AVAILABLE_ENCODERS = {'pixel': PixelEncoder, 'identity': IdentityEncoder}  #定义了可用的编码器类型，包括 'pixel' 和 'identity'

#根据给定的编码器类型创建并返回相应的编码器对象
def make_encoder(
    encoder_type, obs_shape, feature_dim, num_layers, num_filters
):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, num_layers, num_filters
    )
