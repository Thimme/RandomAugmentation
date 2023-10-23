from torchvision import datasets, models, transforms
import torch.nn as nn

# retrain simple classifier with 
class SimpleClassifier(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)

    def forward(self, x):
        x = self.resnet(x)
        return x