import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU()
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU()
        )

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), 
            nn.ReLU()
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, 3, stride = 2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        d1 = self.decoder1(e2)
        d1 = torch.concat([d1, e1], dim=1)
        d2 = self.decoder2(d1)
        return d2