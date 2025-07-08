import torch
import torch.nn as nn
from torchsummary import summary


# Stem (입력 초기 처리 계층)
class Stem(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.lrn = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)

        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.lrn(x)
        x = self.maxpool1(x)
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.lrn(x)
        x = self.maxpool2(x)
        return x
    
# Inception Module (모듈화된 병렬 합성곱 블록)
class InceptionModule(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3_reduce, ch3x3, ch5x5_reduce, ch5x5, pool_proj):
        super().__init__()

        # 1X1
        self.branch1 = nn.Conv2d(in_channels, ch1x1, kernel_size=1)

        # 1x1 -> 3x3
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3_reduce, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch3x3_reduce, ch3x3, kernel_size=3, padding=1)
            )

        # 1x1 -> 5x5
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5_reduce, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch5x5_reduce, ch5x5, kernel_size=5, padding=2)
        )

        # maxpool -> 1x1
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )
    
    def forward(self, x):
        x = torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x),
        ], dim=1)

        return x

# Auxiliary Classifier (중간 출력 분류기) 4b, 6a
class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, dropout):
        super().__init__()

        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = nn.Conv2d(in_channels, 128, kernel_size=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.relu(self.conv(x))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Classifier Head (avg pool + dropout + fc)
class ClassifierHead(nn.Module):
    def __init__(self, num_classes, dropout=0.4):
        super().__init__()
        
        self.avg = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.avg(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# InceptionV1
class InceptionV1(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True):
        super().__init__()
        self.aux_logits = aux_logits

        self.stem = Stem()

        # Inception blocks
        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        
        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)
        
        if aux_logits:
            self.aux1 = AuxiliaryClassifier(512, num_classes, 0.7)
            self.aux2 = AuxiliaryClassifier(528, num_classes, 0.7)
        
        self.head = ClassifierHead(num_classes, 0.4)

    def forward(self, x):
        x = self.stem(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool(x)
        
        x = self.inception4a(x)
        x = self.inception4b(x)
        
        aux1 = self.aux1(x) if self.aux_logits and self.training else None
        
        x = self.inception4c(x)
        x = self.inception4d(x)
        
        aux2 = self.aux2(x) if self.aux_logits and self.training else None
        
        x = self.inception4e(x)
        x = self.maxpool(x)
        
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.head(x)

        if self.aux_logits and self.training:
            return x, aux1, aux2
        else:
            return x


if __name__ == '__main__':
    inceptionv1 = InceptionV1()
    summary(inceptionv1, input_size=(3, 224, 224))
