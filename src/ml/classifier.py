import os

import torch
from torch import nn

from config import CONFIG


class Classifier(nn.Module):
    def __init__(self, n_classes):
        super(Classifier, self).__init__()
        # Input channel 1 because its gray scale
        self.layer = nn.Sequential(
            nn.Linear(CONFIG["latent_space_dim"], n_classes),
            nn.ReLU(),
            nn.Softmax())
        self.path = os.path.dirname(os.path.realpath(__file__)).split("src")[0].replace("\\", "/") + \
                    'classifier_' + str(n_classes) + '.pth'

    def forward(self, x):
        return self.layer(x)

    def save_state(self):
        torch.save(self.state_dict(), self.path)

    def load_state(self):
        self.load_state_dict(torch.load(self.path))
