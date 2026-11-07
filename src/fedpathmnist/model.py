import torch.nn as nn


class SmallPathCNN(nn.Module):
    """A compact RGB CNN suited to 28 x 28 PathMNIST images."""

    def __init__(self, num_classes: int = 9) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, inputs):
        features = self.features(inputs).flatten(1)
        return self.classifier(features)
