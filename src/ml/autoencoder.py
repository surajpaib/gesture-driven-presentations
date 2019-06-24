import os
import sys

import torch
from torch import nn

from config import CONFIG

sys.path.append('./')
if not os.path.exists(CONFIG["autoencoder_img_path"]):
    os.mkdir(CONFIG["autoencoder_img_path"])


class Autoencoder(nn.Module):
    def __init__(self, train_data_shape, latent_space_dim=CONFIG["latent_space_dim"]):
        super(Autoencoder, self).__init__()
        self.path = os.path.dirname(os.path.realpath(__file__)).split("src")[0].replace("\\", "/") + \
                    'autoencoder_' + str(train_data_shape) + '.pth'
        self.encoder = nn.Sequential(
            nn.Linear(train_data_shape, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, latent_space_dim))
        self.decoder = nn.Sequential(
            nn.Linear(latent_space_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, train_data_shape),
            nn.Tanh())

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_latent_space(self, x):
        return self.encoder(x)

    def load_state(self):
        self.load_state_dict(torch.load(self.path))

    def save_state(self):
        torch.save(self.state_dict(), self.path)
