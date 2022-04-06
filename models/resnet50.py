import torch
from torch import nn
import torchvision
from torchvision import models

class CustomResNet50(nn.Module):
    def __init__(self, img_w, img_h, class_num, dropout = 0.4):
        
        super(CustomResNet50, self).__init__()

        self.img_w = img_w
        self.img_h = img_h
        self.class_n = class_num
        self.dropout = dropout

        backbone = models.resnet50(pretrained=True)
        self.fc_inputs = backbone.fc.in_features

        base_layers = list(backbone.children())[:-1]

        self.feature_extractor = nn.Sequential(*base_layers)

        # classifier
        self.out = nn.Sequential(
            nn.Linear(self.fc_inputs, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, self.class_num),
            #nn.LogSoftmax(dim=1) # For using NLLLoss()
        )
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.shape[0], -1)  # reshape the tensor
        x = self.out(x)
        return x


