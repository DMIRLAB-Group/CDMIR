import torch
import torch.nn as nn


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIM = {4: 35}


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""

    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32):
        super().__init__()

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.z_dim = 56

        self.s1_dim = 8
        self.s2_dim = 6
        self.s3_dim = 42
        # self.s4_dim = 4
        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = OUT_DIM[num_layers]
        '''self.fc = nn.Linear(self.feature_dim, self.feature_dim)  # 32*35*35,50'''

        self.encode1 = nn.Sequential(
            nn.Linear(num_filters * out_dim * out_dim, num_filters * out_dim),
            nn.BatchNorm1d(num_filters * out_dim), nn.ReLU(), nn.Dropout())
        self.fc_mu = nn.Sequential(nn.Linear(num_filters * out_dim, self.z_dim))
        self.fc_logvar = nn.Sequential(nn.Linear(num_filters * out_dim, self.z_dim))
        self.fc = nn.Linear(self.feature_dim - 8, self.feature_dim - 8)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)

        # eps = torch.normal(mean=0.0, std=0.5, size=std.shape)
        # eps = torch.FloatTensor(eps).to(device)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward_conv(self, obs):

        obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.contiguous().view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)
        f_rec = self.encode1(h)
        if detach:
            f_rec = f_rec.detach()

        '''h_fc = self.fc(h)
        self.outputs['fc'] = h_fc
        h_norm = self.ln(h_fc)  # 归一化
        self.outputs['ln'] = h_norm'''
        mu, log_var = self.fc_mu(f_rec), self.fc_logvar(f_rec)

        z = self.reparameterize(mu, log_var)


        # z = self.ln(z)
        # z = torch.tanh(z)
        z_idx = 0
        z_s1 = z[:, z_idx: z_idx + self.s1_dim]

        z_idx += self.s1_dim
        z_s2 = z[:, z_idx: z_idx + self.s2_dim]

        z_idx += self.s2_dim
        z_c1 = z[:, z_idx: z_idx + self.s3_dim]

        '''z_idx += self.s3_dim
        z_c2 = z[:, z_idx:]'''

        presentation = torch.hstack([z_s2, z_c1])
        presentation = self.fc(presentation)
        '''out = torch.tanh(h_norm)
        self.outputs['tanh'] = out'''
        # z, z_s1, z_s2, z_c1, z_c2, mu, log_var, pre_input
        return h, z, presentation, z_s1, z_s2, z_c1, mu, log_var

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])
            '''tie_weights(src=source.encode1[0], trg=self.encode1[0])
            tie_weights(src=source.fc_mu[0], trg=self.fc_mu[0])
            tie_weights(src=source.fc_logvar[0], trg=self.fc_logvar[0])'''



class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass


_AVAILABLE_ENCODERS = {'pixel': PixelEncoder, 'identity': IdentityEncoder}


def make_encoder(
        encoder_type, obs_shape, feature_dim, num_layers, num_filters
):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, num_layers, num_filters
    )
