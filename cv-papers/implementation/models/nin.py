import torch
import torch.nn as nn
from torchsummary import summary


class NIN(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.mlpblock1 = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(192, 160, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(160, 96, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.mlpblock2 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.mlpblock3 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(192, num_classes, kernel_size=1),
            nn.ReLU()
        )

        self.dropout = nn.Dropout(p=0.5)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.mlpblock1(x)
        x = self.mlpblock2(x)
        x = self.dropout(x)
        x = self.mlpblock3(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return x

if __name__ == '__main__':
    nin = NIN()
    summary(nin, input_size=(3, 224, 224))