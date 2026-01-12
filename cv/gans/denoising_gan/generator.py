import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.LeakyReLU(),
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5),
            nn.LeakyReLU(),
        )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5),
            nn.LeakyReLU(),
        )

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=5),
            nn.LeakyReLU(),
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=5),
            nn.LeakyReLU(),
        )

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=5),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        d1 = self.decoder1(e3)
        d1 = torch.concat([e2, d1], dim=1)
        d2 = self.decoder2(d1)
        d2 = torch.concat([e1, d2], dim=1)
        d3 = self.decoder3(d2)
        return d3