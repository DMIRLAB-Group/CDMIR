import torch
import numpy as np
import torch.utils.data
import pandas as pd
from torch import nn, optim
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias

class Dataset:
    def __init__(self, capacity):
        self.memory = pd.DataFrame(index=range(capacity), columns=['O_z'])
        self.i = 0
        self.count = 0
        self.capacity = capacity

    def store(self, *args):
        self.memory.loc[self.i] = args
        self.i = (self.i + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample(self, size):
        indices = np.random.choice(self.count, size=size)
        return np.stack(self.memory.loc[indices, 'O_z'])

class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=4, num_filters=32):
        super().__init__()

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        self.fc = nn.Linear(39200, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()

    def forward_conv(self, obs):
        obs = obs / 255.

        conv = torch.relu(self.convs[0](obs))

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))

        h = conv.contiguous().view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)

        h_norm = self.ln(h_fc)

        out = torch.tanh(h_norm)

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

class PixelDecoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers=4, num_filters=32):
        super().__init__()

        self.num_layers = num_layers
        self.num_filters = num_filters
        self.out_dim = 35

        self.fc = nn.Linear(
            feature_dim, num_filters * self.out_dim * self.out_dim
        )

        self.deconvs = nn.ModuleList()

        for i in range(self.num_layers - 1):
            self.deconvs.append(
                nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1)
            )
        self.deconvs.append(
            nn.ConvTranspose2d(
                num_filters, obs_shape[0], 3, stride=2, output_padding=1
            )
        )

        self.outputs = dict()

    def forward(self, h):
        h = torch.relu(self.fc(h))
        self.outputs['fc'] = h

        deconv = h.view(-1, self.num_filters, self.out_dim , self.out_dim)
        self.outputs['deconv1'] = deconv

        for i in range(0, self.num_layers - 1):
            deconv = torch.relu(self.deconvs[i](deconv))
            self.outputs['deconv%s' % (i + 1)] = deconv

        obs = self.deconvs[-1](deconv)
        self.outputs['obs'] = obs

        return obs

class VAE(nn.Module):
    def __init__(self, learning_rate = 1e-3, z_dim=50, batchsize = 128):
        super(VAE, self).__init__()

        self.total_it = 0
        self.decoder_latent_lambda=1e-6
        self.decoder_weight_lambda=1e-7
        self.dataset = Dataset(20000)
        self.encoder = PixelEncoder([4, 84, 84], feature_dim=z_dim)
        self.decoder = PixelDecoder([4, 84, 84], feature_dim=z_dim)

         # optimizer for critic encoder for reconstruction loss
        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(), lr=learning_rate
        )

        # optimizer for decoder
        self.decoder_optimizer = torch.optim.Adam(
            self.decoder.parameters(),
            lr=learning_rate,
            weight_decay=1e-7
        )

        self.batch_size = batchsize

    
    
    def store(self, exp_list):
        for exp in exp_list:
            if exp is not None:
                O_z = exp
                self.dataset.store(O_z)
    
    def learn(self):
        # learn
        for _ in range(400):
            self.total_it += 1
            self.update(self.batch_size)

    def update(self, batch_size):
        O_z = self.dataset.sample(batch_size)
        state_tensor = torch.FloatTensor(O_z).to(device)
        self.update_decoder(state_tensor, state_tensor)

    def update_decoder(self, obs, target_obs):
        h = self.encoder(obs)

        rec_obs = self.decoder(h)
        rec_loss = F.mse_loss(target_obs, rec_obs)
        if self.total_it % 100 == 0:
            print(rec_loss)

        latent_loss = (0.5 * h.pow(2).sum(1)).mean()

        loss = rec_loss + self.decoder_latent_lambda * latent_loss
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

    
    def save(self, epoch, path):
        vae_checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.encoder_optimizer.state_dict()
        }
        torch.save(vae_checkpoint, path + '/vae_{:03d}'.format(epoch))
    
    def load(self, model_file, mode):
        file = model_file + '/vae'
        state = torch.load(file)
        self.load_state_dict(state['model_state_dict'])
        self.encoder_optimizer.load_state_dict(state['optimizer_state_dict'])
        starting_epoch = state['epoch'] + 1
        if mode == 'test':
            self.eval()

        return starting_epoch

