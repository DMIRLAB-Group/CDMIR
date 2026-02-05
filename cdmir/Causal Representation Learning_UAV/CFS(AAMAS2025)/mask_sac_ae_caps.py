import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time
import utils
from encoder import make_encoder
from decoder import make_decoder
from torch.autograd import Variable
from Logger import Logger
import logging
from PIL import Image, ImageEnhance
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def action_clip(action):    #将输入的动作向量裁剪（限制）到特定的边界范围内
    action_bound = np.array([[0, 1.], [-1., 1.], [-1., 1.]])
    action_bound = action_bound 
    scaled_action = copy.deepcopy(action)
    scaled_action[:, 0] = np.clip(scaled_action[:, 0], a_min=action_bound[0][0], a_max=action_bound[0][1])
    scaled_action[:, 1] = np.clip(scaled_action[:, 1], a_min=action_bound[1][0], a_max=action_bound[1][1])
    scaled_action[:, 2] = np.clip(scaled_action[:, 2], a_min=action_bound[2][0], a_max=action_bound[2][1])
    return scaled_action

def states_handle(states): #处理输入的状态数据，将其分成状态、目标和速度，并将它们转换为PyTorch张量
    s_list, goal_list, speed_list = [], [], []

    for i in states:
        s_list.append(i[0])
        goal_list.append(i[1])
        speed_list.append(i[2])

    s_list = np.asarray(s_list)
    goal_list = np.asarray(goal_list)
    speed_list = np.asarray(speed_list)

    # s_array = np.asarray(s_list)
    # goal_array = np.asarray(goal_list)
    # speed_array = np.asarray(speed_list)

    state_tensor = Variable(torch.from_numpy(s_list)).float().to(device)
    goal_tensor = Variable(torch.from_numpy(goal_list)).float().to(device)
    speed_tensor = Variable(torch.from_numpy(speed_list)).float().to(device)

    # state_tensor = torch.from_numpy(s_array).float().to(device)  # 直接将数组转换为张量
    # goal_tensor = torch.from_numpy(goal_array).float().to(device)
    # speed_tensor = torch.from_numpy(speed_array).float().to(device)

    return state_tensor, goal_tensor, speed_tensor


def gaussian_logprob(noise, log_std):   #计算给定高斯分布下噪声值 noise 的对数概率密度,用于计算损失函数或优化目标
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):     #用于策略网络输出的处理，以确保策略输出在合适的范围内，并且可以用于计算概率分布的对数概率密度
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu_1 = mu[:, 0].unsqueeze(-1)
    mu_2 = mu[:, 1:]
    mu_1 = torch.sigmoid(mu_1)
    mu_2 = torch.tanh(mu_2)
    mu = torch.cat((mu_1, mu_2), dim=-1)
    if pi is not None:
        pi_1 = pi[:, 0].unsqueeze(-1)
        pi_2 = pi[:, 1:]
        pi_1 = torch.sigmoid(pi_1)
        pi_2 = torch.tanh(pi_2)
        pi = torch.cat((pi_1, pi_2), dim=-1)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m): #根据神经网络层的类型选择不同的权重初始化策略。对于线性层，它使用正交初始化方法来初始化权重矩阵，并将偏置设置为零。对于卷积层，它同样使用正交初始化方法来初始化权重矩阵，但只初始化卷积核中心位置的权重，并将偏置设置为零。
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):    #检查当前处理的神经网络层 m 是否是线性层（全连接层）类型
        nn.init.orthogonal_(m.weight.data)  #使用正交初始化方法初始化该层的权重矩阵 m.weight.data。正交初始化的目标是确保权重之间彼此正交，以帮助网络更好地训练。这个方法有助于避免权重矩阵的奇异性和梯度消失问题
        m.bias.data.fill_(0.0)  #对于线性层，这一行将该层的偏置（bias）初始化为零
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):     #检查当前处理的神经网络层 m 是否是二维卷积层（nn.Conv2d）或反卷积层（nn.ConvTranspose2d）类型
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)     #断言卷积层的权重矩阵是一个正方形，也就是高度（height）和宽度（width）的尺寸相等。这是因为这里使用的权重初始化方法要求权重矩阵是正方形的
        m.weight.data.fill_(0.0)       #对于卷积层，这一行将该层的权重矩阵初始化为零
        m.bias.data.fill_(0.0)      #将该层的偏置初始化为零
        mid = m.weight.size(2) // 2     #计算卷积核的中间位置，假定卷积核的尺寸是方形的，因此 m.weight.size(2) 和 m.weight.size(3) 都表示卷积核的尺寸。
        gain = nn.init.calculate_gain('relu')   #计算用于正交初始化的增益（gain）
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)    #使用正交初始化方法初始化卷积层的权重矩阵。在这里，仅初始化卷积核中心位置的权重，以确保它们正交，并使用之前计算的增益值。


class Generate_gate(nn.Module): #定义可微分掩码
    def __init__(self, dimension):
        super(Generate_gate, self).__init__()
        self.proj = nn.Sequential(nn.Linear(dimension, int(dimension/2)),
                                  nn.ReLU(),
                                  nn.Linear(int(dimension/2), dimension),
                                  nn.ReLU())
        self.epsilon = 1e-8

    def forward(self, x):

        alpha = self.proj(x)
        gate = (alpha**2) / (alpha**2 + self.epsilon)
        return gate


class Actor(nn.Module): #用于执行动作选择的，通常在强化学习中用作策略网络
    """MLP actor network."""
    def __init__(
        self, obs_shape, hidden_dim, encoder_type,
        encoder_feature_dim, log_std_min, log_std_max, num_layers, num_filters, feature_dim=32
    ):
        super().__init__()
        #创建了一个编码器（encoder）对象，并将其保存在 self.encoder 中。编码器的类型、输入状态形状、特征维度、层数和滤波器数量由构造函数的参数指定。这个编码器用于将输入状态转换为特征表示，以供后续的神经网络层使用
        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters
        )
        #将动作的对数标准差的最小值和最大值保存在模型的属性中，以便后续在动作选择过程中使用。通常，在确定动作的概率分布时，这些值用于约束对数标准差的范围
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        #修改代码，拆成三个MLP
        self.trunk_one = nn.Sequential(nn.Linear(self.encoder.feature_dim + 6, hidden_dim), nn.ReLU())

        self.trunk_two = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

        self.trunk_three = nn.Sequential(nn.Linear(hidden_dim, 6))

        #定义一个可以训练的权重，用来更新mask
        self.one_weight = nn.Parameter(torch.randn(50), requires_grad=True)
        self.two_weight = nn.Parameter(torch.randn(1024), requires_grad=True)
        self.three_weight = nn.Parameter(torch.randn(1024), requires_grad=True)
        # self.m_weight = nn.Parameter(torch.ones(50), requires_grad=True)
        #创建了一个名为Generate_gate的实例
        self.gate1 = Generate_gate(dimension=50)
        self.gate2 = Generate_gate(dimension=1024)
        self.gate3 = Generate_gate(dimension=1024)

        self.outputs = dict()   #创建了一个空字典，用于存储模型的输出
        self.apply(weight_init) #调用了一个名为 weight_init 的函数，用于自定义神经网络层的权重初始化。这个函数会对神经网络的权重进行初始化，以确保网络在训练过程中具有合适的初始状态。

    def forward(self, x, goal, speed, compute_pi=True, compute_log_pi=True,detach=False):
        a = self.encoder(x, detach=detach)  #将输入状态 x 通过编码器 self.encoder 进行处理，并将结果保存在变量 a 中

        #第一层加mask_one
        mask_one = self.gate1(self.one_weight)
        percentage_zeros_mask_one = (torch.sum(mask_one == 0).item() / mask_one.numel()) * 100
        feature_one = mask_one * a
        feature_one_cat = torch.cat((feature_one, goal, speed), dim=-1)  # 将编码器的输出 a、目标 goal 和速度 speed 沿着最后一个维度（dim=-1 表示最后一个维度）拼接在一起。这将把这些信息合并到一起，以供后续神经网络层使用
        mlp_one = self.trunk_one(feature_one_cat)

        #第二层加mask_two
        mask_two = self.gate2(self.two_weight)
        percentage_zeros_mask_two = (torch.sum(mask_two == 0).item() / mask_two.numel()) * 100
        feature_two = mask_two * mlp_one
        mlp_two = self.trunk_two(feature_two)

        mu, log_std = self.trunk_three(mlp_two).chunk(2, dim=-1)

        # 第三层加mask_three
        # mask_three = self.gate3(self.three_weight)
        # feature_three = mask_three * mlp_two
        # mu, log_std = self.trunk_three(feature_three).chunk(2, dim=-1)


        # mask = self.gate(self.m_weight)
        # mask_a = mask * a
        # mask_a = torch.cat((mask_a, goal, speed), dim=-1)  # 将编码器的输出 a、目标 goal 和速度 speed 沿着最后一个维度（dim=-1 表示最后一个维度）拼接在一起。这将把这些信息合并到一起，以供后续神经网络层使用
        #  mu, log_std = self.trunk(mask_a).chunk(2, dim=-1)    #将拼接后的输入 a 通过神经网络的主干部分 self.trunk 进行前向传播。self.trunk 会输出一个包含两个部分的张量 mu 和 log_std，这两部分分别表示动作的均值和对数标准差


        # mu = torch.clip(mu, -3)
        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)   #将对数标准差 log_std 应用了双曲正切函数 tanh，以确保它的值在 (-1, 1) 范围内
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)   #将 log_std 的值缩放到指定的范围 [self.log_std_min, self.log_std_max] 内。具体地，它将 log_std 的值映射到这个范围，并确保它在指定范围内
        if compute_pi:
            std = log_std.exp()     #计算动作的标准差 std，通过将 log_std 指数化得到
            noise = torch.randn_like(mu)    #生成一个与 mu 同样大小的随机噪声张量 noise，使用 torch.randn_like(mu) 创建
            pi = mu + noise * std
        else:
            pi = None   #表示不计算动作策略和策略的熵
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)   #计算高斯分布的对数概率。这个函数使用 noise 和经过处理的 log_std 作为输入参数
        else:
            log_pi = None   #表示不计算对数策略

        mu, pi, log_pi = squash(mu, pi, log_pi)     #应用了一些非线性变换，将均值 mu 和策略 pi 的每个分量限制在 [-1, 1] 范围内

        return mu, pi, log_pi, log_std, percentage_zeros_mask_one, percentage_zeros_mask_two, feature_one, feature_two


class QFunction(nn.Module): #用于估计在给定状态和动作的情况下的 Q 值
    """MLP for q-function."""
    def __init__(self, obs_dim, action_shape):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_shape, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 1)
        )

    def forward(self, feature, action): #这是神经网络模型的前向传播函数的定义。它接受两个输入参数：feature 表示状态特征，action 表示动作。前向传播函数的目标是估计 Q 值。
        assert feature.size(0) == action.size(0)

        obs_action = torch.cat([feature, action], dim=1)    #将状态特征 feature 和动作 action 沿着第一个维度（dim=1 表示按列拼接）拼接在一起，形成一个新的张量 obs_action。这是为了将状态和动作合并以输入到神经网络中
        return self.trunk(obs_action)   #前向传播函数将合并后的输入 obs_action 输入到神经网络的主干部分 self.trunk 中，并返回神经网络的输出，这个输出表示估计的 Q 值


class Critic(nn.Module):
    """Critic network, employes two q-functions."""
    def __init__(
        self, obs_shape, hidden_dim, action_shape, encoder_type,
        encoder_feature_dim, num_layers, num_filters, feature_dim=32
    ):
        super().__init__()


        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters
        )
        #创建了两个 Q-Function 模型，分别用于估计 Q 值。每个 Q-Function 模型都使用了 QFunction 类，它接受输入特征的维度（self.encoder.feature_dim + 6）和动作的形状（action_shape）作为参数。这两个 Q-Function 用于估计不同的 Q 值。
        self.q1 = QFunction(self.encoder.feature_dim + 6, action_shape)
        self.q2 = QFunction(self.encoder.feature_dim + 6, action_shape)

        self.outputs = dict()
        self.apply(weight_init) #应用了一个自定义的权重初始化函数 weight_init，用于初始化 Critic 模型的权重参数

    def forward(self, x, goal, speed, action, detach=False):
        # detach allows to stop gradient propogation to encoder
        v = self.encoder(x, detach=detach)  #将状态数据 x 输入到编码器 self.encoder 中，获得特征表示 v
        v = torch.cat((v, goal, speed), dim=-1) #将特征表示 v、目标数据 goal 和速度数据 speed 沿着最后一个维度（dim=-1 表示按列拼接）组合在一起，形成一个新的特征向量。这个特征向量包含了状态、目标和速度信息
         #  将特征表示 v 和动作数据 action 输入到两个不同的 Q-Function 模型中，分别是 self.q1 和 self.q2。它们用于估计两个不同的 Q 值
        q1 = self.q1(v, action)
        q2 = self.q2(v, action)

        return q1, q2
    


class SAC_Ae_Caps(object):
    """SAC+AE algorithm."""
    def __init__(
        self,
        env,
        num_env, 
        obs_shape,
        action_shape,
        batch_size=256,
        replayer_buffer=2e4,
        init_steps=100, 
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.1,
        alpha_beta=0.5,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        critic_beta=0.9,
        critic_target_update_freq=2,
        lr=1e-3,
        tau=0.005,
        encoder_type='pixel',
        encoder_feature_dim=50,
        decoder_type='pixel',
        decoder_update_freq=1,
        decoder_latent_lambda=1e-6,
        decoder_weight_lambda=1e-7,
        num_layers=4,
        num_filters=32,
        lam_a=-1.,
        lam_s=-1., 
        eps_s=1.,
        seed=0,
        mode='train'
    ):
        if env.index == 0:
            self.f_rec_loss = '../log/' + '/rec_loss.log'
            self.L = Logger(self.f_rec_loss, clevel=logging.INFO, Flevel=logging.INFO, CMD_render=False)
            self.batch_size = batch_size
            self.action_shape = action_shape
            self.actor_update_freq = actor_update_freq
            self.critic_target_update_freq = critic_target_update_freq
            self.discount = discount
            self.tau = tau
            self.decoder_update_freq = decoder_update_freq
            self.decoder_latent_lambda = decoder_latent_lambda
            self.lam_a = lam_a
            self.lam_s = lam_s
            self.eps_s = eps_s
            self.init_steps = init_steps
            self.update_flag = False
            self.mode = mode
            self.num_env = num_env

            np.random.seed(seed) 
            torch.cuda.manual_seed(seed)  
            torch.backends.cudnn.deterministic = True

            self.replayer_buffer = replayer_buffer
            self.replayer = utils.ReplayBuffer(self.replayer_buffer)
            self.action_bound = [[0 , 1.], [-1., 1.], [-1., 1.]]

            self.total_it = 0
            #创建了一个名为 actor 的对象
            self.actor = Actor(
            obs_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters
            ).to(device)
            #创建了一个 critic 对象
            self.critic = Critic(
                obs_shape, hidden_dim, action_shape, encoder_type,
                encoder_feature_dim, num_layers, num_filters
            ).to(device)
            #创建了一个 critic_target 对象
            self.critic_target = Critic(
                obs_shape, hidden_dim, action_shape, encoder_type,
                encoder_feature_dim, num_layers, num_filters
            ).to(device)

            self.critic_target.load_state_dict(self.critic.state_dict())        #将 critic 网络的权重（即状态字典，state_dict）加载到 critic_target 网络中

            # tie encoders between actor and critic
            self.actor.encoder.copy_conv_weights_from(self.critic.encoder)  #将 critic 网络的编码器（encoder）的卷积层权重复制给 actor 网络的编码器,以便更好地共享特征提取的能力。

            self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)  #创建了一个名为 log_alpha 的PyTorch张量,
            self.log_alpha.requires_grad = True    #log_alpha 通常用于调整策略的探索性, log_alpha 视为一个需要通过优化来学习的参数，PyTorch会自动计算关于 log_alpha 的损失函数的梯度，以便进行参数更新。
            # set target entropy to -|A|
            self.target_entropy = -np.prod(action_shape)    #计算了目标熵（target_entropy）的值，它的计算方式是将动作空间的维度（action_shape）相乘，然后取负数

            self.decoder = None
            if decoder_type != 'identity':
                # create decoder 创建一个解码器（decoder），并将其应用到神经网络的权重初始化函数（weight_init）上。
                self.decoder = make_decoder(
                    decoder_type, obs_shape, encoder_feature_dim, num_layers,
                    num_filters
                ).to(device)
                self.decoder.apply(weight_init)

                # optimizer for critic encoder for reconstruction loss创建一个Adam优化器，用于优化神经网络critic的编码器参数，以便在训练中最小化损失函数
                self.encoder_optimizer = torch.optim.Adam(
                    self.critic.encoder.parameters(), lr=lr
                )

                # optimizer for decoder创建了一个Adam优化器（optimizer）用于优化解码器（decoder）的参数
                self.decoder_optimizer = torch.optim.Adam(
                    self.decoder.parameters(),
                    lr=lr,
                    weight_decay=decoder_weight_lambda
                )

            # optimizers创建了一个Adam优化器（optimizer）用于优化Actor神经网络的参数
            self.actor_optimizer = torch.optim.Adam(
                self.actor.parameters(), lr=lr, betas=(actor_beta, 0.999)
            )
            #创建了一个Adam优化器（optimizer）用于优化Critic神经网络的参数。
            self.critic_optimizer = torch.optim.Adam(
                self.critic.parameters(), lr=lr, betas=(critic_beta, 0.999)
            )
            #创建了一个Adam优化器（optimizer）用于优化 self.log_alpha 这个参数。
            self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=lr/10, betas=(alpha_beta, 0.999)
            )

            self.train()
            self.critic_target.train()  #将SAC算法中的目标critic（target critic，即目标值函数网络）设置为训练模式
        else:
            pass

    def train(self, training=True): #设置SAC算法的训练模式
        self.training = training
        self.actor.train(training)  #将SAC算法中的actor（策略网络）设置为传入的训练模式
        self.critic.train(training) #将SAC算法中的critic（值函数网络）设置为传入的训练模式
        if self.decoder is not None:    #检查是否存在decoder（解码器网络），如果存在，也将其设置为传入的训练模
            self.decoder.train(training)
    
    @property
    def alpha(self):    #通过在训练过程中调整温度参数，可以在探索性和利用性之间找到平衡点。
        return self.log_alpha.exp()

    def generate_action(self, env, state_list): #根据输入的环境索引和状态信息生成相应的动作
        if env.index == 0:
            state_tensor, goal_tensor, speed_tensor = states_handle(state_list)
            action_bound = np.array(self.action_bound)
            if self.mode == 'train':
                mu, pi, _, _, percentage_zeros_mask_one, percentage_zeros_mask_two, _, _ = self.actor(state_tensor, goal_tensor, speed_tensor, compute_log_pi=False)
                pi = pi.cpu().data.numpy()
                scaled_action = copy.deepcopy(pi)
                scaled_action[:, 0] = np.clip(scaled_action[:, 0], a_min=action_bound[0][0], a_max=action_bound[0][1])  #对动作的第一个维度进行截断操作，将其限制在action_bound[0][0]和action_bound[0][1]之间
                scaled_action[:, 1] = np.clip(scaled_action[:, 1], a_min=action_bound[1][0], a_max=action_bound[1][1])
                scaled_action[:, 2] = np.clip(scaled_action[:, 2], a_min=action_bound[2][0], a_max=action_bound[2][1])
            elif self.mode == 'test':
                mu, _, _, _, _, _, _, _ = self.actor(state_tensor, goal_tensor, speed_tensor, compute_pi=False, compute_log_pi=False)
                mu = mu.cpu().data.numpy()
                scaled_action = copy.deepcopy(mu)
                scaled_action[:, 0] = np.clip(scaled_action[:, 0], a_min=action_bound[0][0], a_max=action_bound[0][1])
                scaled_action[:, 1] = np.clip(scaled_action[:, 1], a_min=action_bound[1][0], a_max=action_bound[1][1])
                scaled_action[:, 2] = np.clip(scaled_action[:, 2], a_min=action_bound[2][0], a_max=action_bound[2][1])
        else:
            scaled_action = None
            percentage_zeros_mask_one = None
            percentage_zeros_mask_two = None
        return scaled_action
        # return scaled_action, percentage_zeros_mask_one, percentage_zeros_mask_two

    def update_decoder(self, obs, target_obs, writer, epoch):  #更新解码器（Decoder）模型，实现自编码器的训练
        h = self.critic.encoder(obs)    #从 Critic 模型中获取观测数据 obs 的编码表示 h
        rec_obs = self.decoder(h)   #使用解码器（Decoder）模型，将编码表示 h 解码为重构观测数据 rec_obs
        rec_loss = F.mse_loss(target_obs, rec_obs)  #计算重构损失，它度量了重构观测数据 rec_obs 与目标观测数据 target_obs 之间的均方误差（Mean Squared Error，MSE）。这个损失函数用于衡量解码器的性能，目标是让重构数据尽可能地接近原始数据。
        # add L2 penalty on latent representation
        # see https://arxiv.org/pdf/1903.12436.pdf
        latent_loss = (0.5 * h.pow(2).sum(1)).mean()    #计算潜在表示 h 的 L2 正则化损失。这个损失函数鼓励编码器产生具有较小 L2 范数的潜在表示，以促进学到更有意义的表示。

        loss = rec_loss + self.decoder_latent_lambda * latent_loss

        if self.total_it % 400 == 0:
            writer.add_scalar("loss/rec_loss", rec_loss, epoch)
            writer.add_scalar("loss/latent_loss", latent_loss, epoch)
            writer.add_scalar("loss/decoder_loss", loss, epoch)

        self.encoder_optimizer.zero_grad()  #清零编码器和解码器的梯度，以准备进行反向传播计算梯度。
        self.decoder_optimizer.zero_grad()
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()   #使用优化器来更新编码器和解码器的参数，使它们朝着降低损失的方向迭代。

    def update_critic(self, state_tensor, goal_tensor, speed_tensor, action, reward,
                     n_state_tensor, n_goal_tensor, n_speed_tensor, not_done, writer, epoch):
        with torch.no_grad():
            _, policy_action, log_pi, _, _, _, _, _ = self.actor(n_state_tensor, n_goal_tensor, n_speed_tensor)
            target_Q1, target_Q2 = self.critic_target(n_state_tensor, n_goal_tensor, n_speed_tensor, policy_action) #）计算下一个状态的动作（policy_action）对应的Q值估计
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi  #下一个状态的值函数估计
            target_Q = reward + (not_done * self.discount * target_V)   #target_V 表示了在下一个状态下，智能体可以得到的预期奖励，考虑了两个Critic网络的估计以及策略的不确定性（熵）

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(state_tensor, goal_tensor, speed_tensor, action)   #获取当前状态下的Q值估计
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)   #将当前状态下的Q值估计调整到接近目标Q值的位置

        if self.total_it % 400 == 0:
            # print('critic loss :', critic_loss)
            # print("TQ", target_Q[0])
            # print("CQ1", current_Q1[0])
            # print("CQ2", current_Q2[0])
            writer.add_scalar("loss/critic", critic_loss, epoch)


        # Optimize the critic 通过critic_loss反向传播，优化Critic网络的参数，使用的优化器是self.critic_optimizer
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, state_tensor, goal_tensor, speed_tensor,
                                    n_state_tensor, n_goal_tensor, n_speed_tensor,
                                    b_state_tensor, b_goal_tensor, b_speed_tensor, O_z, writer, epoch
    ):
        # detach encoder, so we don't update it with the actor loss
        mu, pi, log_pi, _, _, _, feature_one_ori, feature_two_ori = self.actor(state_tensor, goal_tensor, speed_tensor, detach=True)    #从Actor网络中获取当前状态下的均值、采样的动作、以及动作的对数概率
        actor_Q1, actor_Q2 = self.critic(state_tensor, goal_tensor, speed_tensor, pi, detach=True)  #通过Actor网络获取当前状态下采取动作 pi 的Q值估计
        actor_Q = torch.min(actor_Q1, actor_Q2)

        #compute features which are masked loss
        img_noise = noise(O_z)        #加噪声
        img_motion = motion(img_noise)    #加模糊
        # img_blur = blur(img_noise)    #加模糊
        # img_contrast = adjust_contrast(img_blur)     #调整对比度
        # img_saturation = adjust_saturation(img_contrast)    #调整饱和度
        # img_brightness = adjust_brightness(img_saturation)     #调整亮度
        img_aug = torch.FloatTensor(img_motion).to(device)

        # timestamp = int(time.time())
        # # print("O_z:", O_z[0][0].shape)
        # cv2.imwrite(f'/home/robot/uav/ori_image_{timestamp}.png', O_z[0][0])
        # cv2.imwrite(f'/home/robot/uav/x1_image_{timestamp}.png', img_motion[0][0])

        mu_aug, pi_aug, _, _, _, _, feature_one_aug, feature_two_aug = self.actor(img_aug, goal_tensor, speed_tensor, detach=True)    #从Actor网络中获取当前状态下的均值、采样的动作、以及动作的对数概率
        actor_Q1_aug, actor_Q2_aug = self.critic(img_aug, goal_tensor, speed_tensor, pi_aug, detach=True)  # 通过Actor网络获取当前状态下采取动作 pi 的Q值估计
        actor_Q_aug = torch.min(actor_Q1_aug, actor_Q2_aug)

        loss_fac = F.mse_loss(feature_one_ori, feature_one_aug) + F.mse_loss(feature_two_ori, feature_two_aug)
        loss_mu = F.mse_loss(mu, mu_aug)
        loss_Q = F.mse_loss(actor_Q, actor_Q_aug)

        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean() + loss_fac + loss_mu + 0.001 * loss_Q #计算Actor的损失函数
        if self.lam_a > 0:  #检查正则化项的权重是否大于零。如果lam_a大于零，表示开启了正则化项
            mu_nxt, _, _, _, _, _, _, _ = self.actor(n_state_tensor, n_goal_tensor, n_speed_tensor, detach=True)    #到下一个状态的动作均值
            actor_loss += self.lam_a * torch.sum(((mu_nxt - mu)**2) / 2) / mu.shape[0]  #这个正则化项衡量了当前状态的动作均值（mu）与下一个状态的动作均值（mu_nxt）之间的差异
        if self.lam_s > 0:
            mu_bar, _, _, _, _, _, _, _ = self.actor(b_state_tensor, b_goal_tensor, b_speed_tensor, detach=True)
            actor_loss += self.lam_s * torch.sum(((mu_bar - mu)**2) / 2) / mu.shape[0]  #加入了一个熵正则化项，用于鼓励策略产生更多的探索性动作，而不是仅仅依赖于已知的最优动作

        if self.total_it % 400 == 0:
            writer.add_scalar("loss/actor", actor_loss, epoch)
            writer.add_scalar("loss/mask_feature", loss_fac, epoch)
            writer.add_scalar("loss/mask_mu", loss_mu, epoch)
            writer.add_scalar("loss/mask_Q", 0.001 * loss_Q, epoch)

        # optimize the actor,通过actor_loss反向传播，优化Actor网络的参数
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()  #计算α的损失函数
        alpha_loss.backward()   #通过alpha_loss反向传播，优化αα的值
        self.log_alpha_optimizer.step()

    def update(self, batch_size, writer, epoch):
        # Sample replay buffer 
        O_z, O_g, O_v, action, next_O_z, next_O_g, next_O_v, reward, not_done = self.replayer.sample(batch_size)    #从回放缓冲区中采样数据

        #将采样的数据转换为PyTorch张量
        state_tensor = torch.FloatTensor(O_z).to(device)
        goal_tensor = torch.FloatTensor(O_g).to(device)
        speed_tensor = torch.FloatTensor(O_v).to(device)
        n_state_tensor = torch.FloatTensor(next_O_z).to(device)
        n_goal_tensor = torch.FloatTensor(next_O_g).to(device)
        n_speed_tensor = torch.FloatTensor(next_O_v).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        not_done = torch.FloatTensor(not_done).unsqueeze(1).to(device)

        if self.lam_s > 0:  #如果self.lam_s大于0，表示需要处理随机状态
            b_state_tensor = self.decoder(self.actor.encoder(state_tensor)) #将当前状态 state_tensor 通过解码器（self.decoder）得到 b_state_tensor。
            b_state_tensor = torch.clip(b_state_tensor, 0.2, 20)    #对解码得到的状态进行了截断（clip）处理，限制在[0.2, 20]范围内
            b_goal_tensor = torch.FloatTensor(np.random.normal(O_g, self.eps_s)).to(device)
            b_speed_tensor = torch.FloatTensor(np.random.normal(O_v, self.eps_s)).to(device)
        else:
            b_state_tensor = None
            b_goal_tensor = None
            b_speed_tensor = None

        self.update_critic(state_tensor, goal_tensor, speed_tensor, action, reward, n_state_tensor, n_goal_tensor, n_speed_tensor, not_done, writer, epoch)#计算Critic网络的损失，并通过优化器进行参数更新
        if self.total_it % self.actor_update_freq == 0:
            #计算Actor的损失和αα的损失，并分别通过两个优化器进行参数更新
            self.update_actor_and_alpha(state_tensor, goal_tensor, speed_tensor,
                                        n_state_tensor, n_goal_tensor, n_speed_tensor,
                                        b_state_tensor, b_goal_tensor, b_speed_tensor, O_z, writer, epoch
                                        )
        #执行软更新（soft update）操作，将当前的Critic网络参数逐渐更新到目标Critic网络。这种软更新可以使得目标值更加平滑地逼近当前值，有助于算法的稳定性
        if self.total_it % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.q1, self.critic_target.q1, self.tau
            )
            utils.soft_update_params(
                self.critic.q2, self.critic_target.q2, self.tau
            )
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.tau * 5
            )

        if self.decoder is not None and self.total_it % self.decoder_update_freq == 0:  #传入当前状态的编码，更新解码器的参数
            self.update_decoder(state_tensor, state_tensor, writer, epoch)
    #每个经验元组都被存储到回放缓冲区（replay buffer）中，以便在后续的训练中使用

    def step(self, exp_list):
        for exp in exp_list:
            if exp is not None:
                [O_z, O_g, O_v, action, next_O_z, next_O_g, next_O_v, reward, not_done] = exp
                self.replayer.store(O_z, O_g, O_v, action, next_O_z, next_O_g, next_O_v, reward, not_done)
    
    def learn(self, writer, epoch):   #用于进行强化学习的学习过程
        # learn
        for _ in range(400):
            self.total_it += 1
            self.update(self.batch_size, writer, epoch)
    #函数用于保存训练过程中的模型参数和优化器状态

    def save(self, epoch, policy_path):
        actor_checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.actor.state_dict(),
            'optimizer_state_dict': self.actor_optimizer.state_dict(),
        }
        critic_checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.critic_optimizer.state_dict(),
        }
        decoder_checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.decoder_optimizer.state_dict(),
        }
        torch.save(actor_checkpoint, policy_path + '/actor{:03d}'.format(epoch))
        torch.save(critic_checkpoint, policy_path + '/critic{:03d}'.format(epoch))
        torch.save(decoder_checkpoint, policy_path + '/decoder{:03d}'.format(epoch))

    #加载已保存的模型参数和优化器状态，以及特定轮次的模型状态
    def load(self, model_file, mode):
        actor_file = model_file + '/actor'
        critic_file = model_file + '/critic'
        decoder_file = model_file + '/decoder'

        # 加载断点模型
        actor_state = torch.load(actor_file)
        critic_state = torch.load(critic_file)
        decoder_state = torch.load(decoder_file)
        # 加载断点的状态
        self.actor.load_state_dict(actor_state['model_state_dict'])
        self.actor_optimizer.load_state_dict(actor_state['optimizer_state_dict'])
        self.actor_target = copy.deepcopy(self.actor)

        self.critic.load_state_dict(critic_state['model_state_dict'])
        self.critic_optimizer.load_state_dict(critic_state['optimizer_state_dict'])
        self.critic_target = copy.deepcopy(self.critic)

        self.decoder.load_state_dict(decoder_state['model_state_dict'])
        self.decoder_optimizer.load_state_dict(decoder_state['optimizer_state_dict'])

        starting_epoch = actor_state['epoch'] + 1

        if mode == 'test':
            self.actor.eval()

        return starting_epoch


def colorful_spectrum_mix(img1, img2, alpha=1.0, ratio=1.0):
    """Input image size: ndarray of [H, W]， 因果干预傅里叶变换， 输入两个图，得到两个新图"""
    lam = np.random.uniform(0, alpha)
    # lam = 0.9
    assert img1.shape == img2.shape
    _, h, w = img1.shape
    h_crop = int(h * np.sqrt(ratio))
    w_crop = int(w * np.sqrt(ratio))
    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2

    img1_fft = np.fft.fft2(img1, axes=(0, 1))
    img2_fft = np.fft.fft2(img2, axes=(0, 1))
    img1_abs, img1_pha = np.abs(img1_fft), np.angle(img1_fft)
    img2_abs, img2_pha = np.abs(img2_fft), np.angle(img2_fft)

    img1_abs = np.fft.fftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.fftshift(img2_abs, axes=(0, 1))

    img1_abs_ = np.copy(img1_abs)
    img2_abs_ = np.copy(img2_abs)
    img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img2_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img1_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]
    img2_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img1_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img2_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]

    img1_abs = np.fft.ifftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.ifftshift(img2_abs, axes=(0, 1))

    img21 = img1_abs * (np.e ** (1j * img1_pha))
    img12 = img2_abs * (np.e ** (1j * img2_pha))
    img21 = np.real(np.fft.ifft2(img21, axes=(0, 1)))
    img12 = np.real(np.fft.ifft2(img12, axes=(0, 1)))
    img21 = np.uint8(np.clip(img21, 0, 255))
    img12 = np.uint8(np.clip(img12, 0, 255))

    return img21, img12


def fac(img1):
    img_hist = np.zeros_like(img1)
    for j in range(img1.shape[0]):

        current_image = img1[j]
        index = random.choice([index for index in range(img1.shape[0]) if index != j])
        paired_image = img1[index]
        current_image1, paired_image1 = colorful_spectrum_mix(current_image, paired_image)
        img_hist[j] = current_image1.reshape(1, 4, 84, 84)
    # print("fac_shape", img_hist.shape)
    img_hist = torch.FloatTensor(img_hist).to(device)
    return img_hist


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def factorization_loss(f_a, f_b):
    # empirical cross-correlation matrix
    f_a_norm = (f_a - f_a.mean(0)) / (f_a.std(0) + 1e-6)
    f_b_norm = (f_b - f_b.mean(0)) / (f_b.std(0) + 1e-6)  # 标准化
    c = torch.mm(f_a_norm.T, f_b_norm) / f_a_norm.size(0)  # 矩阵乘法

    on_diag = torch.diagonal(c).add_(-1).pow_(2).mean()
    # off_diag = off_diagonal(c).pow_(2).mean()
    loss = on_diag

    return loss


def kl(mu, log_var, z, epoch):
    """KL divergence"""
    normal_distribution = torch.distributions.MultivariateNormal(
        torch.zeros(56).cuda(), torch.eye(56).cuda())  # 先验分布，因为假设独立同分布
    q_dist = torch.distributions.normal.Normal(
        mu, torch.exp(torch.clamp(log_var, min=-40) / 2))  # 编码器输出的潜在空间表示

    log_qz = q_dist.log_prob(z)  # 计算q_dist对z的log概率
    log_pz = normal_distribution.log_prob(z)  # 计算先验分布对z的log概率
    kl = (log_qz.sum(dim=1) - log_pz).mean()
    log = torch.clamp(log_var, min=-10)
    # kl = 0.5 * torch.sum(torch.exp(log) + torch.pow(mu, 2) - 1. - log)
    C = torch.clamp(torch.tensor(20) / 1000 * epoch, 0, 20)

    loss_kl = (kl - C).abs()
    return loss_kl, log


def hist(img1):
    """直方图均衡"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    img_hist = np.zeros_like(img1)
    for j in range(img1.shape[0]):

        img2 = img1[j]
        img2 = img2.astype(np.uint8)
        # print("111", img2.shape) 4, 84, 84
        img_zero = np.zeros_like(img2)
        for i in range(4):
            k = img2[i]
            # print("22", k.shape) 84, 84
            channels = clahe.apply(k)  # ApplyCLAHE to each channel
            img_zero[i] = channels.reshape(1, 84, 84)
        img_hist[j] = img_zero.reshape(1, 4, 84, 84)
    img_hist = torch.FloatTensor(img_hist).to(device)
    return img_hist


def noise(img, std=0.1):
    """图片加噪声"""
    size = img.shape
    noise = np.random.normal(scale=std, size=size)

    # print("11", img.shape)
    img_noise = img + noise
    img_noise = np.clip(img_noise, a_min=0.2, a_max=20)
    # img_noise = torch.FloatTensor(img_noise).to(device)
    return img_noise


def blur(img):
    img_blur = np.zeros_like(img)
    for j in range(img.shape[0]):
        img2 = img[j]
        img2 = img2.astype(np.uint8)
        # print("111", img2.shape) 4, 84, 84
        img_zero = np.zeros_like(img2)
        for i in range(4):
            img3 = img2[i]
            # print("22", k.shape) 84, 84
            blur = cv2.blur(img3, (3, 3))
            img_zero[i] = blur.reshape(1, 84, 84)
        img_blur[j] = img_zero.reshape(1, 4, 84, 84)
    # img_blur = torch.FloatTensor(img_blur).to(device)

    return img_blur


def motion(img, degree=12, angle=45):
    img_motion = np.zeros_like(img)
    M = cv2.getRotationMatrix2D((degree/2, degree/2), angle, 1)
    motion_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_kernel, M, (degree, degree))
    motion_blur_kernel = motion_blur_kernel / degree

    for j in range(img.shape[0]):
        img2 = img[j]
        img2 = np.array(img2)
        img_zero = np.zeros_like(img2)
        for i in range(4):
            img3 = img2[i]
            motion = cv2.filter2D(img3, -1, motion_blur_kernel)
            img_zero[i] = motion.reshape(1, 84, 84)
        img_motion[j] = img_zero.reshape(1, 4, 84, 84)
    # img_motion = torch.FloatTensor(img_motion).to(device)
    return img_motion

def adjust_contrast(img, factor=1.1):
    """调整图像对比度"""
    img_contrast = np.zeros_like(img)
    for j in range(img.shape[0]):
        img2 = img[j].astype(np.uint8)
        img_zero = np.zeros_like(img2)
        for i in range(img2.shape[0]):
            img3 = img2[i]
            img_pil = Image.fromarray(img3)
            enhancer = ImageEnhance.Contrast(img_pil)
            img_contrasted = enhancer.enhance(factor)
            img_zero[i] = np.array(img_contrasted)  # 转换为 NumPy 数组
        img_contrast[j] = img_zero

    return img_contrast

def adjust_saturation(img, factor=1.1):
    """调整图像饱和度"""
    img_saturation = np.zeros_like(img)
    for j in range(img.shape[0]):
        img2 = img[j].astype(np.uint8)
        img_zero = np.zeros_like(img2)
        for i in range(img2.shape[0]):
            img3 = img2[i]
            img_pil = Image.fromarray(img3)
            enhancer = ImageEnhance.Color(img_pil)
            img_saturated = enhancer.enhance(factor)
            img_zero[i] = np.array(img_saturated)
        img_saturation[j] = img_zero

    return img_saturation

def adjust_brightness(img, factor=1.1):
    """调整图像亮度"""
    img_brightness = np.zeros_like(img)
    for j in range(img.shape[0]):
        img2 = img[j].astype(np.uint8)
        img_zero = np.zeros_like(img2)
        for i in range(img2.shape[0]):
            img3 = img2[i]
            img_pil = Image.fromarray(img3)
            enhancer = ImageEnhance.Brightness(img_pil)
            img_brightened = enhancer.enhance(factor)
            img_zero[i] = np.array(img_brightened)  # 转换为 NumPy 数组
        img_brightness[j] = img_zero

    return img_brightness
