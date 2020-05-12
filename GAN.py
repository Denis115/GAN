from common import *
import random
import torch
import torch.nn as nn

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        x= self.conv(x)
        x= self.bn(x)
        return self.relu(x)

class BasicConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(BasicConvTranspose, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)
    
    def forward(self, x):
        x= self.conv(x)
        x= self.bn(x)
        return self.relu(x)

class Generator(nn.Module):
    def __init__(self, N_GPU):
        super(Generator, self).__init__()
        self.ngpu = N_GPU

        self.conv1 = BasicConvTranspose(N_NOISE, N_FEATURES_GEN * 8, kernel_size=4, stride=1, padding=0, bias=True)
        self.conv2 = BasicConvTranspose(N_FEATURES_GEN * 8, N_FEATURES_GEN * 4, 4, 2, 1, True)
        self.conv3 = BasicConvTranspose(N_FEATURES_GEN * 4, N_FEATURES_GEN * 2, 4, 2, 1, True)
        self.conv4 = BasicConvTranspose(N_FEATURES_GEN * 2, N_FEATURES_GEN * 1, 4, 2, 1, True)
        
        self.conv = nn.ConvTranspose2d(N_FEATURES_GEN, 3, 4, 2, 1, bias=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        
        return self.tanh(self.conv(self.conv4(self.conv3(self.conv2(self.conv1(x))))))
       
class Discriminator(nn.Module):
    def __init__(self, N_GPU):
        super(Discriminator, self).__init__()
        self.ngpu = N_GPU

        self.model = nn.Sequential(
            
            nn.Conv2d(3, N_FEATURES_DIS, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            BasicConv(N_FEATURES_DIS, N_FEATURES_DIS * 2, 4, 2, 1, bias=False),
            BasicConv(N_FEATURES_DIS * 2, N_FEATURES_DIS * 4, 4, 2, 1, bias=False), 
            BasicConv(N_FEATURES_DIS * 4, N_FEATURES_DIS * 8, 4, 2, 1, bias=False),
            nn.Conv2d(N_FEATURES_DIS * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)
