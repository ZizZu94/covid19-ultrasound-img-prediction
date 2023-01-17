import torch
from torch import nn
import torchvision
from torchvision import models

class CustomResNet18(nn.Module):
    def __init__(self, class_num, dropout = 0.4):
        super(CustomResNet18, self).__init__()

        self.class_num = class_num
        self.dropout = dropout

        backbone = models.resnet18(pretrained=True)
        self.fc_inputs = backbone.fc.in_features

        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2]) # we take layers before the classifier and the avgpool

        self.avgpool = backbone.avgpool

        # classifier
        self.out = nn.Sequential(
            nn.Linear(self.fc_inputs, 1024, bias=True), # If set to False, the layer will not learn an additive bias.
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(1024, self.class_num),
        )
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)  # reshape the tensor
        x = self.out(x)
        return x

class CustomResNet50(nn.Module):
    def __init__(self, class_num, dropout = 0.4):
        super(CustomResNet50, self).__init__()

        self.class_num = class_num
        self.dropout = dropout

        backbone = models.resnet50(pretrained=True)
        self.fc_inputs = backbone.fc.in_features

        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2]) # we take layers before the classifier and the avgpool

        self.avgpool = backbone.avgpool

        # classifier
        self.out = nn.Sequential(
            nn.Linear(self.fc_inputs, 1024, bias=True), # If set to False, the layer will not learn an additive bias.
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(1024, self.class_num),
        )
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)  # reshape the tensor
        x = self.out(x)
        return x

