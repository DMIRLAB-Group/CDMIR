import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

import utils
from vae import VAE
from torch.autograd import Variable
from Logger import Logger
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

class Actor(nn.Module):
    """MLP actor network."""
    def __init__(
        self, obs_shape, hidden_dim, autoencoder,
        encoder_feature_dim, log_std_min, log_std_max, num_layers, num_filters, feature_dim=32
    ):
        super().__init__()

        self.encoder = autoencoder

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(encoder_feature_dim+ 6, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 6)
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, x, goal, speed, compute_pi=True, compute_log_pi=True,detach=False):
        a = self.encoder(x)
        a = torch.cat((a, goal, speed), dim=-1)
        mu, log_std = self.trunk(a).chunk(2, dim=-1)

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

        return mu, pi, log_pi, log_std


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
        self, obs_shape, hidden_dim, action_shape, autoencoder,
        encoder_feature_dim, num_layers, num_filters, feature_dim=32
    ):
        super().__init__()


        self.encoder = autoencoder

        self.q1 = QFunction(encoder_feature_dim + 6, action_shape)
        self.q2 = QFunction(encoder_feature_dim + 6, action_shape)

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, x, goal, speed, action, detach=False):
        # detach allows to stop gradient propogation to encoder
        v = self.encoder(x, detach=detach)
        v = torch.cat((v, goal, speed), dim=-1)
           
        q1 = self.q1(v, action)
        q2 = self.q2(v, action)

        return q1, q2
    


class SAC_Vae(object):
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
        encoder_feature_dim=50,
        num_layers=4,
        num_filters=32,
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
            self.init_steps = init_steps
            self.update_flag = False
            self.mode = mode
            self.num_env = num_env
            self.vae = VAE()
            self.vae.load(model_file='../policy/vae',mode='test')
            self.encoder = self.vae.encoder

            np.random.seed(seed) 
            torch.cuda.manual_seed(seed)  
            torch.backends.cudnn.deterministic = True

            self.replayer_buffer = replayer_buffer
            self.replayer = utils.ReplayBuffer(self.replayer_buffer)
            self.action_bound = [[0 , 1.], [-1., 1.], [-1., 1.]]

            self.total_it = 0

            self.actor = Actor(
            obs_shape, hidden_dim, self.encoder,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters
            ).to(device)

            self.critic = Critic(
                obs_shape, hidden_dim, action_shape, self.encoder,
                encoder_feature_dim, num_layers, num_filters
            ).to(device)

            self.critic_target = Critic(
                obs_shape, hidden_dim, action_shape, self.encoder,
                encoder_feature_dim, num_layers, num_filters
            ).to(device)

            self.critic_target.load_state_dict(self.critic.state_dict())

            self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
            self.log_alpha.requires_grad = True
            # set target entropy to -|A|
            self.target_entropy = -np.prod(action_shape)

            self.decoder = None

            # optimizers
            self.actor_optimizer = torch.optim.Adam(
                self.actor.parameters(), lr=lr, betas=(actor_beta, 0.999)
            )

            self.critic_optimizer = torch.optim.Adam(
                self.critic.parameters(), lr=lr, betas=(critic_beta, 0.999)
            )

            self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=lr/10, betas=(alpha_beta, 0.999)
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
                mu, pi, _, _ = self.actor(state_tensor, goal_tensor, speed_tensor, compute_log_pi=False)
                pi = pi.cpu().data.numpy()
                scaled_action = copy.deepcopy(pi)
                scaled_action[:, 0] = np.clip(scaled_action[:, 0], a_min=action_bound[0][0], a_max=action_bound[0][1])
                scaled_action[:, 1] = np.clip(scaled_action[:, 1], a_min=action_bound[1][0], a_max=action_bound[1][1])
                scaled_action[:, 2] = np.clip(scaled_action[:, 2], a_min=action_bound[2][0], a_max=action_bound[2][1])
            elif self.mode == 'test':
                mu, _, _, _ = self.actor(state_tensor, goal_tensor, speed_tensor, compute_pi=False, compute_log_pi=False)
                mu = mu.cpu().data.numpy()
                scaled_action = copy.deepcopy(mu)
                scaled_action[:, 0] = np.clip(scaled_action[:, 0], a_min=action_bound[0][0], a_max=action_bound[0][1])
                scaled_action[:, 1] = np.clip(scaled_action[:, 1], a_min=action_bound[1][0], a_max=action_bound[1][1])
                scaled_action[:, 2] = np.clip(scaled_action[:, 2], a_min=action_bound[2][0], a_max=action_bound[2][1])
        else:
            scaled_action = None
        return scaled_action


    def update_critic(self, state_tensor, goal_tensor, speed_tensor, action, reward,
                     n_state_tensor, n_goal_tensor, n_speed_tensor, not_done):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(n_state_tensor, n_goal_tensor, n_speed_tensor)
            target_Q1, target_Q2 = self.critic_target(n_state_tensor, n_goal_tensor, n_speed_tensor, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(state_tensor, goal_tensor, speed_tensor, action)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)


        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, state_tensor, goal_tensor, speed_tensor,
                                    n_state_tensor, n_goal_tensor, n_speed_tensor,
                                    b_state_tensor, b_goal_tensor, b_speed_tensor
    ):
        # detach encoder, so we don't update it with the actor loss
        mu, pi, log_pi, _ = self.actor(state_tensor, goal_tensor, speed_tensor, detach=True)
        actor_Q1, actor_Q2 = self.critic(state_tensor, goal_tensor, speed_tensor, pi, detach=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update(self, batch_size):
        # Sample replay buffer 
        O_z, O_g, O_v, action, next_O_z, next_O_g, next_O_v, reward, not_done = self.replayer.sample(batch_size)


        state_tensor = torch.FloatTensor(O_z).to(device)
        goal_tensor = torch.FloatTensor(O_g).to(device)
        speed_tensor = torch.FloatTensor(O_v).to(device)
        n_state_tensor = torch.FloatTensor(next_O_z).to(device)
        n_goal_tensor = torch.FloatTensor(next_O_g).to(device)
        n_speed_tensor = torch.FloatTensor(next_O_v).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        not_done = torch.FloatTensor(not_done).unsqueeze(1).to(device)

        b_state_tensor = None
        b_goal_tensor = None
        b_speed_tensor = None

        self.update_critic(state_tensor, goal_tensor, speed_tensor, action, reward, n_state_tensor, n_goal_tensor, n_speed_tensor, not_done)
        if self.total_it % self.actor_update_freq == 0:
            self.update_actor_and_alpha(state_tensor, goal_tensor, speed_tensor,
                                        n_state_tensor, n_goal_tensor, n_speed_tensor,
                                        b_state_tensor, b_goal_tensor, b_speed_tensor
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


    def step(self, exp_list):
        for exp in exp_list:
            if exp is not None:
                [O_z, O_g, O_v, action, next_O_z, next_O_g, next_O_v, reward, not_done] = exp
                self.replayer.store(O_z, O_g, O_v, action, next_O_z, next_O_g, next_O_v, reward, not_done)
    
    def learn(self):
        # learn
        for _ in range(400):
            self.total_it += 1
            self.update(self.batch_size)

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
        torch.save(actor_checkpoint, policy_path + '/actor{:03d}'.format(epoch))
        torch.save(critic_checkpoint, policy_path + '/critic{:03d}'.format(epoch))


    def load(self, model_file, mode):
        actor_file = model_file + '/actor'
        critic_file = model_file + '/critic'

        # 加载断点模型
        actor_state = torch.load(actor_file)
        critic_state = torch.load(critic_file)
        # 加载断点的状态
        self.actor.load_state_dict(actor_state['model_state_dict'])
        self.actor_optimizer.load_state_dict(actor_state['optimizer_state_dict'])
        self.actor_target = copy.deepcopy(self.actor)

        self.critic.load_state_dict(critic_state['model_state_dict'])
        self.critic_optimizer.load_state_dict(critic_state['optimizer_state_dict'])
        self.critic_target = copy.deepcopy(self.critic)

        starting_epoch = actor_state['epoch'] + 1

        if mode == 'test':
            self.actor.eval()

        return starting_epoch
