from common import *
import random
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, N_GPU):
        super(Generator, self).__init__()
        self.ngpu = N_GPU

        self.model = nn.Sequential(
            
            nn.ConvTranspose2d(N_NOISE, N_FEATURES_GEN * 8, 4, 1, 0, bias=True),
            nn.BatchNorm2d(N_FEATURES_GEN * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(N_FEATURES_GEN * 8, N_FEATURES_GEN * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(N_FEATURES_GEN * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(N_FEATURES_GEN * 4, N_FEATURES_GEN * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(N_FEATURES_GEN * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(N_FEATURES_GEN * 2, N_FEATURES_GEN, 4, 2, 1, bias=True),
            nn.BatchNorm2d(N_FEATURES_GEN),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(N_FEATURES_GEN, 3, 4, 2, 1, bias=True),
            nn.Tanh()
        )

    def forward(self, input):
        return self.model(input)


class Discriminator(nn.Module):
    def __init__(self, N_GPU):
        super(Discriminator, self).__init__()
        self.ngpu = N_GPU

        self.model = nn.Sequential(
            
            nn.Conv2d(3, N_FEATURES_DIS, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(N_FEATURES_DIS, N_FEATURES_DIS * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(N_FEATURES_DIS * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(N_FEATURES_DIS * 2, N_FEATURES_DIS * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(N_FEATURES_DIS * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(N_FEATURES_DIS * 4, N_FEATURES_DIS * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(N_FEATURES_DIS * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(N_FEATURES_DIS * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)
