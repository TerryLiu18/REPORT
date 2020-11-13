import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps

class VAE(nn.Module):
    """docstring for ClassName"""
    def __init__(self, feat_input_size, z_dim=100):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.loss = nn.MSELoss()
        self.encoder = nn.Sequential(
                    nn.Linear(feat_input_size, 100),
                    nn.ReLU(),
                    nn.Dropout(p=0.2),
                    nn.Linear(100, 128),
                    nn.ReLU(),
                    nn.Dropout(p=0.2),
                    nn.Linear(128, 2*z_dim),
                    )

        self.decoder = nn.Sequential(
                    nn.Linear(z_dim, 128),
                    nn.ReLU(),
                    nn.Dropout(p=0.2),
                    nn.Linear(128, 100),
                    nn.ReLU(),
                    nn.Dropout(p=0.2),
                    nn.Linear(100, feat_input_size),
                    nn.Sigmoid(),
                    )
        
    def _kl_loss(self, mu, logvar):
        klds = -0.5*(1 + logvar - mu.pow(2) - torch.exp(logvar))
        total_kld = klds.sum(1).mean(0, True)
        return total_kld

    def _recon_loss(self, x_recon, x):
        batch_size = x.size(0)
        rec_loss = self.loss(x_recon, x) ##average over the loss
        return rec_loss

    def forward(self, x):
        latent_z = self.encoder(x)
        mu = latent_z[:,:self.z_dim]
        logvar = latent_z[:,self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self.decoder(z)
        kl_loss = self._kl_loss(mu, logvar)
        rec_loss = self._recon_loss(x_recon, x)
        
        return z, kl_loss, rec_loss
        




        
        