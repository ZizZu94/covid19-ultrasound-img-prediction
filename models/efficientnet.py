import torch
from torch import nn
import torchvision
from torchvision import models

class CustomEfficientNetB0(nn.Module):
    def __init__(self, img_w, img_h, class_num, dropout = 0.4):
        super(CustomEfficientNetB0, self).__init__()

        self.img_w = img_w
        self.img_h = img_h
        self.class_num = class_num
        self.dropout = dropout

        backbone = models.efficientnet_b0(pretrained=True)
        self.fc_inputs = backbone.classifier[1].in_features

        base_layers = list(backbone.children())[:-1]

        self.feature_extractor = nn.Sequential(*base_layers)

        # classifier
        self.out = nn.Sequential(
            nn.Linear(self.fc_inputs, 1024, bias=True), # If set to False, the layer will not learn an additive bias.
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(1024, self.class_num),
            #nn.LogSoftmax(dim=1) # For using NLLLoss()
        )
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.shape[0], -1)  # reshape the tensor
        x = self.out(x)
        return x

class CustomEfficientNetB4(nn.Module):
    def __init__(self, img_w, img_h, class_num, dropout = 0.4):
        super(CustomEfficientNetB4, self).__init__()

        self.img_w = img_w
        self.img_h = img_h
        self.class_num = class_num
        self.dropout = dropout

        backbone = models.efficientnet_b4(pretrained=True)
        self.fc_inputs = backbone.classifier[1].in_features

        base_layers = list(backbone.children())[:-1]

        self.feature_extractor = nn.Sequential(*base_layers)

        # classifier
        self.out = nn.Sequential(
            nn.Linear(self.fc_inputs, 1024, bias=True), # If set to False, the layer will not learn an additive bias.
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(1024, self.class_num),
            #nn.LogSoftmax(dim=1) # For using NLLLoss()
        )
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.shape[0], -1)  # reshape the tensor
        x = self.out(x)
        return x


