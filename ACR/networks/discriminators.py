import numpy as np
import torch.nn as nn


class NLayerDiscriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        input_nc = config['input_nc']

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_nc, out_channels=64, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kw, stride=2, padding=padw)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=kw, stride=2, padding=padw)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=kw, stride=2, padding=padw)
        self.bn4 = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=kw, stride=1, padding=padw)
        self.bn5 = nn.BatchNorm2d(512)

        self.conv6 = nn.Conv2d(512, 1, kernel_size=kw, stride=1, padding=padw)

    def forward(self, x):
        conv1 = self.conv1(x)

        conv2 = self.conv2(conv1)
        conv2 = self.bn2(conv2)
        conv2 = self.act(conv2)

        conv3 = self.conv3(conv2)
        conv3 = self.bn3(conv3)
        conv3 = self.act(conv3)

        conv4 = self.conv4(conv3)
        conv4 = self.bn4(conv4)
        conv4 = self.act(conv4)

        conv5 = self.conv5(conv4)
        conv5 = self.bn5(conv5)
        conv5 = self.act(conv5)

        conv6 = self.conv6(conv5)

        outputs = conv6

        return outputs, [conv1, conv2, conv3, conv4, conv5]