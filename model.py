import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class Discriminator(nn.Module):
  def __init__(self, img_channels, disc_features):
    super().__init__()
    self.disc = nn.Sequential(
        nn.Conv2d(img_channels, disc_features, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        self.block(disc_features, disc_features*2, 4, 2, 1),
        self.block(disc_features*2, disc_features*4, 4, 2, 1),
        self.block(disc_features*4, disc_features*8, 4, 2, 1),
        nn.Conv2d(disc_features*8, 1, kernel_size=4, stride=2, padding=0),
        nn.Sigmoid(),
    )

  def block(self, in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2)
    )

  def forward(self, x):
    return self.disc(x)

class Generator(nn.Module):
  def __init__(self, z_dim, img_channels, gen_features):
    super().__init__()
    self.gen = nn.Sequential(
        self.block(z_dim, gen_features*16, 4, 2, 0),
        self.block(gen_features*16, gen_features*8, 4, 2, 1),
        self.block(gen_features*8, gen_features*4, 4, 2, 1),
        self.block(gen_features*4, gen_features*2, 4, 2, 1),
        nn.ConvTranspose2d(gen_features*2, img_channels, kernel_size=4, stride=2, padding=1),
        nn.Tanh(),
    )

  def block(self, in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias= False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

  def forward(self, x):
    return self.gen(x)


