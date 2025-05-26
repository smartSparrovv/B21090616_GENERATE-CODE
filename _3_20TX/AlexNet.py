import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),  # Conv1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),  # Conv2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),  # Conv3
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),  # Conv4
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),  # Conv5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))  # 适应不同输入尺寸

        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),  # FC6
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),  # FC7
            nn.ReLU(inplace=True),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(4096, 4),  # FC8 (最终分类层)
        )

    def forward(self, x, mode='source'):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # 展平
        features = x
        x = self.fc1(x)
        x = self.fc2(x)

        if mode == 'target':
            return features
        x = self.fc3(x)
        return features, x





