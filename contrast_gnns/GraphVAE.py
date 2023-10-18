import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

class GVAE(nn.Module):
    def __init__(self, in_dim, hid_dim, z_dim):
        super(GVAE, self).__init__()
        self.encoder1 = dgl.nn.pytorch.GraphConv(in_dim, hid_dim)
        self.encoder2 = dgl.nn.pytorch.conv.GraphConv(hid_dim, z_dim * 2)
        self.decoder = dgl.nn.pytorch.conv.GraphConv(z_dim, in_dim)

    def forward(self, g, features):
        h = F.relu(self.encoder1(g, features))
        mean_logvar = self.encoder2(g, h)
        mean, log_var = torch.chunk(mean_logvar, 2, dim=1)
        std = torch.exp(0.5 * log_var)
        z = (torch.randn(mean.shape).to(std.device)) * std + mean  
        return self.decoder(g, z), mean, log_var

    def loss(self, pred, features, mean, log_var):
        reconstruction_loss = F.mse_loss(pred, features)  
        kl_divergence = -0.5 * torch.mean(torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1))  
        return reconstruction_loss + kl_divergence
