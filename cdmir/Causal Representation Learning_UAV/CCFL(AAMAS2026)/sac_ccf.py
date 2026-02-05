import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.utils.tensorboard import SummaryWriter
import utils
from encoder_vae import make_encoder
from encoder_vae import PixelEncoder
from decoder import make_decoder
from torch.autograd import Variable
from Logger import Logger
import logging
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def l1_mask_regularization(masks, lambda_l1):
    l1_norm = sum(mask.abs().sum() for mask in masks)
    return lambda_l1 * l1_norm

def action_clip(action):
    action_bound = np.array([[0, 1.], [-1., 1.], [-1., 1.]])
    action_bound = action_bound
    scaled_action = copy.deepcopy(action)
    scaled_action[:, 0] = np.clip(scaled_action[:, 0], a_min=action_bound[0][0], a_max=action_bound[0][1])
    scaled_action[:, 1] = np.clip(scaled_action[:, 1], a_min=action_bound[1][0], a_max=action_bound[1][1])
    scaled_action[:, 2] = np.clip(scaled_action[:, 2], a_min=action_bound[2][0], a_max=action_bound[2][1])
    return scaled_action


def states_handle(states):
    s_list, goal_list, speed_list = [], [], []
    for i in states:
        s_list.append(i[0])
        goal_list.append(i[1])
        speed_list.append(i[2])

    s_list = np.asarray(s_list)
    goal_list = np.asarray(goal_list)
    speed_list = np.asarray(speed_list)

    state_tensor = Variable(torch.from_numpy(s_list)).float().to(device)
    goal_tensor = Variable(torch.from_numpy(goal_list)).float().to(device)
    speed_tensor = Variable(torch.from_numpy(speed_list)).float().to(device)

    return state_tensor, goal_tensor, speed_tensor


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu_1 = mu[:, 0].unsqueeze(-1)  # (8,)=>(8,1)
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


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

class Generate_gate(nn.Module):
    def __init__(self, dimension):
        super(Generate_gate, self).__init__()
        self.linear1 = nn.Linear(dimension, int(dimension/2))
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(int(dimension/2), dimension)
        self.epsilon = 1e-8

    def forward(self, x):
        gate_linear1 = self.linear1(x)
        gate_linear1_relu = self.relu(gate_linear1)
        gate_linear2 = self.linear2(gate_linear1_relu)
        gate_linear2_relu = self.relu(gate_linear2)

        gate = (gate_linear2_relu**2) / (gate_linear2_relu**2 + self.epsilon)
        return gate


class Actor(nn.Module):
    """MLP actor network."""

    def __init__(
            self, obs_shape, hidden_dim, encoder_type,
            encoder_feature_dim, log_std_min, log_std_max, num_layers, num_filters, feature_dim=32
    ):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers, num_filters)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # self.trunk = nn.Sequential(
        #     nn.Linear(self.encoder.feature_dim + 6 - 8, hidden_dim), nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        #     nn.Linear(hidden_dim, 6)
        # )

        # 修改代码，拆成三个MLP
        self.trunk_one = nn.Sequential(nn.Linear(self.encoder.feature_dim + 6 - 8, hidden_dim), nn.ReLU())

        self.trunk_two = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

        self.trunk_three = nn.Sequential(nn.Linear(hidden_dim, 6))

        # 定义一个可以训练的权重，用来更新mask
        self.one_weight = nn.Parameter(torch.randn(48), requires_grad=True)
        self.two_weight = nn.Parameter(torch.randn(1024), requires_grad=True)
        self.three_weight = nn.Parameter(torch.randn(1024), requires_grad=True)
        # self.m_weight = nn.Parameter(torch.ones(50), requires_grad=True)
        # 创建了一个名为Generate_gate的实例
        self.gate1 = Generate_gate(dimension=48)
        self.gate2 = Generate_gate(dimension=1024)
        self.gate3 = Generate_gate(dimension=1024)

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, x, goal, speed, compute_pi=True, compute_log_pi=True, detach=False):
        h, z, a, _, z2, z3, _, _ = self.encoder(x, detach=detach)

        # 第一层加mask_one
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


        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
                self.log_std_max - self.log_std_min
        ) * (log_std + 1)
        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std, percentage_zeros_mask_one, percentage_zeros_mask_two, mask_one, mask_two


class QFunction(nn.Module):
    """MLP for q-function."""

    def __init__(self, obs_dim, action_shape):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_shape, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 1)
        )

    def forward(self, feature, action):
        assert feature.size(0) == action.size(0)

        obs_action = torch.cat([feature, action], dim=1)
        return self.trunk(obs_action)


class Critic(nn.Module):
    """Critic network, employes two q-functions."""

    def __init__(
            self, obs_shape, hidden_dim, action_shape, encoder_type,
            encoder_feature_dim, num_layers, num_filters, feature_dim=32
    ):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers, num_filters)

        self.q1 = QFunction(self.encoder.feature_dim + 6 - 8, action_shape)
        self.q2 = QFunction(self.encoder.feature_dim + 6 - 8, action_shape)

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, x, goal, speed, action, detach=False):
        # detach allows to stop gradient propogation to encoder
        h, z, v, _, z2, _, _, _ = self.encoder(x, detach=detach)
        v = torch.cat((v, goal, speed), dim=-1)

        q1 = self.q1(v, action)
        q2 = self.q2(v, action)

        return q1, q2


class SAC_CCF_Caps(object):
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
            lr=1e-4,
            tau=0.005,
            encoder_type='pixel',
            encoder_feature_dim=56,
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
            self.f_fac_loss = '../log/' + '/fac_loss.log'
            self.f_actor_loss = '../log/' + '/actor_loss.log'
            self.Logger = Logger(self.f_rec_loss, clevel=logging.INFO, Flevel=logging.INFO, CMD_render=False)
            self.Logger1 = Logger(self.f_rec_loss, clevel=logging.INFO, Flevel=logging.INFO, CMD_render=False)
            self.Logger2 = Logger(self.f_rec_loss, clevel=logging.INFO, Flevel=logging.INFO, CMD_render=False)
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
            self.replayer_2 = utils.ReplayBuffer_2(self.replayer_buffer)
            self.action_bound = [[0, 1.], [-1., 1.], [-1., 1.]]
            self.temperature = nn.Parameter(torch.tensor(0.1))

            self.total_it = 0

            self.actor = Actor(
                obs_shape, hidden_dim, encoder_type,
                encoder_feature_dim, actor_log_std_min, actor_log_std_max,
                num_layers, num_filters
            ).to(device)

            self.critic = Critic(
                obs_shape, hidden_dim, action_shape, encoder_type,
                encoder_feature_dim, num_layers, num_filters
            ).to(device)

            self.critic_target = Critic(
                obs_shape, hidden_dim, action_shape, encoder_type,
                encoder_feature_dim, num_layers, num_filters
            ).to(device)

            self.critic_target.load_state_dict(self.critic.state_dict())

            # tie encoders between actor and critic
            self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

            self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
            self.log_alpha.requires_grad = True
            # set target entropy to -|A|
            self.target_entropy = -np.prod(action_shape)

            # create decoder rec
            self.decoder = make_decoder(decoder_type, obs_shape, encoder_feature_dim, num_layers,
                                            num_filters).to(device)
            self.decoder.apply(weight_init)

            # optimizer for critic encoder for reconstruction loss
            self.encoder_optimizer = torch.optim.Adam(
                self.critic.encoder.parameters(), lr=lr
            )

            # optimizer for decoder
            self.decoder_optimizer = torch.optim.Adam(
                self.decoder.parameters(),
                lr=lr,
                weight_decay=decoder_weight_lambda
            )

            # optimizers
            self.actor_optimizer = torch.optim.Adam(
                self.actor.parameters(), lr=lr, betas=(actor_beta, 0.999)
            )

            self.critic_optimizer = torch.optim.Adam(
                self.critic.parameters(), lr=lr, betas=(critic_beta, 0.999)
            )

            self.log_alpha_optimizer = torch.optim.Adam(
                [self.log_alpha], lr=lr / 10, betas=(alpha_beta, 0.999)
            )

            self.train()
            self.critic_target.train()
        else:
            pass

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if self.decoder is not None:
            self.decoder.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def generate_action(self, env, state_list):
        if env.index == 0:
            state_tensor, goal_tensor, speed_tensor = states_handle(state_list)
            action_bound = np.array(self.action_bound)
            if self.mode == 'train':
                mu, pi, _, _, percentage_zeros_mask_one, percentage_zeros_mask_two, _, _= self.actor(state_tensor, goal_tensor, speed_tensor, compute_log_pi=False)
                pi = pi.cpu().data.numpy()
                scaled_action = copy.deepcopy(pi)
                scaled_action[:, 0] = np.clip(scaled_action[:, 0], a_min=action_bound[0][0], a_max=action_bound[0][1])
                scaled_action[:, 1] = np.clip(scaled_action[:, 1], a_min=action_bound[1][0], a_max=action_bound[1][1])
                scaled_action[:, 2] = np.clip(scaled_action[:, 2], a_min=action_bound[2][0], a_max=action_bound[2][1])
            elif self.mode == 'test':
                mu, _, _, _, percentage_zeros_mask_one, percentage_zeros_mask_two, _, _ = self.actor(state_tensor, goal_tensor, speed_tensor, compute_pi=False,
                                         compute_log_pi=False)
                mu = mu.cpu().data.numpy()
                scaled_action = copy.deepcopy(mu)
                scaled_action[:, 0] = np.clip(scaled_action[:, 0], a_min=action_bound[0][0], a_max=action_bound[0][1])
                scaled_action[:, 1] = np.clip(scaled_action[:, 1], a_min=action_bound[1][0], a_max=action_bound[1][1])
                scaled_action[:, 2] = np.clip(scaled_action[:, 2], a_min=action_bound[2][0], a_max=action_bound[2][1])
        else:
            scaled_action = None
            percentage_zeros_mask_one = None
            percentage_zeros_mask_two = None
        return scaled_action, percentage_zeros_mask_one, percentage_zeros_mask_two

    def contrastive_loss(self, ball_representation_unchange, cube_representation_unchange,
                         orign_representation_unchange, temperature):  # 负样本对（多样性）损失
        # 归一化,out_1 (128,48)  128:batchsize   48:特征维度
        out_ball_pos = torch.nn.functional.normalize(ball_representation_unchange, p=2, dim=1)
        out_cube_pos = torch.nn.functional.normalize(cube_representation_unchange, p=2, dim=1)
        out_orign_pos = torch.nn.functional.normalize(orign_representation_unchange, p=2, dim=1)
        # 使用 unsqueeze 方法在第二维度处添加一个维度，变为 (128, 1, 48)
        ball_matrix = out_ball_pos.unsqueeze(1)
        # 使用 unsqueeze 方法在第三维度处添加一个维度，变为 (128, 48, 1)
        cube_matrix = out_cube_pos.unsqueeze(2)
        # pos score
        Pos = torch.exp(torch.bmm(ball_matrix,
                                  cube_matrix) / temperature)  # 计算‘out_1’与out_2的转置点积，得到一个相似度矩阵，除以‘temperature’后指数化，得到负分数矩阵

        # 使用 unsqueeze 方法在第二维度处添加一个维度，变为 (128, 1, 48)
        orign_matrix = out_orign_pos.unsqueeze(1)
        # 使用 unsqueeze 方法在第三维度处添加一个维度，变为 (128, 48, 1)
        ball_matrix = out_ball_pos.unsqueeze(2)
        cube_matrix = out_cube_pos.unsqueeze(2)
        # pos score
        neg_ball = torch.exp(torch.bmm(orign_matrix,
                                       ball_matrix) / temperature)  # 计算‘out_1’与out_2的转置点积，得到一个相似度矩阵，除以‘temperature’后指数化，得到负分数矩阵
        neg_cube = torch.exp(torch.bmm(orign_matrix, cube_matrix) / temperature)
        Neg = (neg_ball + neg_cube) * 0.5

        # contrastive loss
        loss = (- torch.log(Pos / (Pos + Neg))).mean()
        # print("neg_loss:", loss)
        return loss

    def update_decoder(self, obs, target_obs, O_z_ball, O_z_cube, O_z_orign, writer, epoch):
        # print(obs.shape)
        h, z, _, _, _, z_c1, mu, log_var = self.critic.encoder(obs)
        O_z_ball_tensor = torch.FloatTensor(O_z_ball).to(device)  # 圆球图像
        _, _, ball_representation_unchange, _, _, _, mu_ball, logvar_ball = self.critic.encoder(O_z_ball_tensor)  # 圆球表征

        O_z_cube_tensor = torch.FloatTensor(O_z_cube).to(device)  # 正方体图像
        _, _, cube_representation_unchange, _, _, _, mu_cube, logvar_cube = self.critic.encoder(O_z_cube_tensor)  # 正方体表征

        O_z_orign_tensor = torch.FloatTensor(O_z_orign).to(device)  # 原图图像
        _, _, orign_representation_unchange, _, _, _, mu_orign, logvar_orign = self.critic.encoder(O_z_orign_tensor)  # 原图表征

        # 重构损失
        rec_img, rec_h = self.decoder(z)
        rec_img_loss = F.mse_loss(rec_img, target_obs)
        rec_loss = F.mse_loss(h, rec_h)
        loss_kl, T = kl(mu, log_var, z, epoch)
        contrastive_loss = self.contrastive_loss(ball_representation_unchange, cube_representation_unchange, orign_representation_unchange, self.temperature)  # 对比学习损失
        T = T.detach().cpu().numpy()
        # print("kl", loss_kl)
        # add L2 penalty on latent representation
        # see https://arxiv.org/pdf/1903.12436.pdf
        # latent_loss = (0.5 * h.pow(2).sum(1)).mean()
        # loss_fac = self.img_intervention(O_z, z_c1)   128

        loss_vae = loss_kl * 0.0001 + rec_loss * 0.1
        loss = loss_vae + contrastive_loss * 0.001 + rec_img_loss


        if self.total_it % 400 == 0:
            print("rec_img", rec_img_loss)
            print("rec", rec_loss * 0.1)
            print("loss_kl", loss_kl*0.0001)
            print("contrastive_loss", contrastive_loss * 0.001)
            print("log_var", log_var.mean(), log_var.max(), log_var.min())
            print("mu", mu.mean())
            count = 0
            for i in range(T.shape[0]):
                for j in range(T.shape[1]):
                    if T[i, j] == -10:
                        count += 1
            # 输出计数器的值
            print("等于-10的元素个数是：", count)
            writer.add_scalar("loss/rec_img", rec_img_loss, epoch)
            writer.add_scalar("mu", mu.mean(), epoch)
            writer.add_scalar("log_var", log_var.mean(), epoch)
            writer.add_scalar("loss/rec", rec_loss, epoch)
            writer.add_scalar("loss/kl", loss_kl * 0.0001, epoch)
            writer.add_scalar("loss/contrastive_loss", contrastive_loss * 0.001, epoch)
        # loss = rec_loss + self.decoder_latent_lambda * latent_loss
        self.encoder_optimizer.zero_grad()
        # self.decoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        # rec_img_loss.backward()
        loss.backward()

        self.encoder_optimizer.step()
        # self.decoder_optimizer.step()
        self.decoder_optimizer.step()

    def update_critic(self, state_tensor, goal_tensor, speed_tensor, action, reward,
                      n_state_tensor, n_goal_tensor, n_speed_tensor, not_done, writer, epoch):
        with torch.no_grad():
            _, policy_action, log_pi, _, _, _, _, _ = self.actor(n_state_tensor, n_goal_tensor, n_speed_tensor)
            target_Q1, target_Q2 = self.critic_target(n_state_tensor, n_goal_tensor, n_speed_tensor, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(state_tensor, goal_tensor, speed_tensor, action, detach=True)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)

        if self.total_it % 400 == 0:
            writer.add_scalar("loss/critic", critic_loss, epoch)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, state_tensor, goal_tensor, speed_tensor,
                               n_state_tensor, n_goal_tensor, n_speed_tensor,
                               b_state_tensor, b_goal_tensor, b_speed_tensor, writer, epoch
                               ):
        # detach encoder, so we don't update it with the actor loss
        mu, pi, log_pi, _, _, _, mask_one, mask_two = self.actor(state_tensor, goal_tensor, speed_tensor, detach=True)
        actor_Q1, actor_Q2 = self.critic(state_tensor, goal_tensor, speed_tensor, pi, detach=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        lambda_l1 = 1e-2
        l1_loss = l1_mask_regularization([mask_one, mask_two], lambda_l1)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean() + l1_loss

        if self.lam_a > 0:
            mu_nxt, _, _, _, _, _, _, _ = self.actor(n_state_tensor, n_goal_tensor, n_speed_tensor, detach=True)
            actor_loss += self.lam_a * torch.sum(((mu_nxt - mu) ** 2) / 2) / mu.shape[0]
        if self.lam_s > 0:
            mu_bar, _, _, _, _, _, _, _ = self.actor(b_state_tensor, b_goal_tensor, b_speed_tensor, detach=True)
            actor_loss += self.lam_s * torch.sum(((mu_bar - mu) ** 2) / 2) / mu.shape[0]
        # self.Logger2.info(actor_loss)
        if self.total_it % 400 == 0:
            writer.add_scalar("loss/actor", actor_loss, epoch)
            writer.add_scalar("loss/L1", l1_loss, epoch)
        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update(self, batch_size, writer, epoch):
        # Sample replay buffer 
        O_z, O_g, O_v, action, next_O_z, next_O_g, next_O_v, reward, not_done = self.replayer.sample(batch_size)  # 128
        # print('oz形状', O_z.shape) (128, 4, 84, 84)
        O_z0, O_z1, O_z2 = self.replayer_2.sample(batch_size)
        O_z_ball, O_z_cube, O_z_orign = O_z0, O_z1, O_z2

        state_tensor = torch.FloatTensor(O_z).to(device)
        goal_tensor = torch.FloatTensor(O_g).to(device)
        speed_tensor = torch.FloatTensor(O_v).to(device)
        n_state_tensor = torch.FloatTensor(next_O_z).to(device)
        n_goal_tensor = torch.FloatTensor(next_O_g).to(device)
        n_speed_tensor = torch.FloatTensor(next_O_v).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        not_done = torch.FloatTensor(not_done).unsqueeze(1).to(device)


        if self.lam_s > 0:
            b_state_tensor, _ = self.decoder(self.actor.encoder(state_tensor))
            b_state_tensor = torch.clip(b_state_tensor, 0.2, 20)
            b_goal_tensor = torch.FloatTensor(np.random.normal(O_g, self.eps_s)).to(device)
            b_speed_tensor = torch.FloatTensor(np.random.normal(O_v, self.eps_s)).to(device)
        else:
            b_state_tensor = None
            b_goal_tensor = None
            b_speed_tensor = None

        self.update_critic(state_tensor, goal_tensor, speed_tensor, action, reward, n_state_tensor, n_goal_tensor,
                           n_speed_tensor, not_done, writer, epoch)
        if self.total_it % self.actor_update_freq == 0:
            self.update_actor_and_alpha(state_tensor, goal_tensor, speed_tensor,
                                        n_state_tensor, n_goal_tensor, n_speed_tensor,
                                        b_state_tensor, b_goal_tensor, b_speed_tensor, writer, epoch
                                        )

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

        if self.decoder is not None and self.total_it % self.decoder_update_freq == 0:
            self.update_decoder(state_tensor, state_tensor, O_z_ball, O_z_cube, O_z_orign, writer, epoch)

    def step(self, exp_list):
        for exp in exp_list:
            if exp is not None:
                [O_z, O_g, O_v, action, next_O_z, next_O_g, next_O_v, reward, not_done] = exp
                self.replayer.store(O_z, O_g, O_v, action, next_O_z, next_O_g, next_O_v, reward, not_done)

    def step_2(self, shape_list):
        for shape in shape_list:
            if shape is not None:
                # shape_0_z = shape
                [O_z0, O_z1, O_z2] = shape
                # self.replayer_2.store(shape_0_z)
                self.replayer_2.store(O_z0, O_z1, O_z2)


    def learn(self, writer, epoch):
        # learn
        for _ in range(400):
            self.total_it += 1
            self.update(self.batch_size, writer, epoch)

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


