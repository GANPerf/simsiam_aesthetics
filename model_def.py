# -*- coding: utf-8 -*-
from __future__ import print_function, division
import torch
import torch.nn as nn

from torchvision import models






class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        
        pretrained_resnet = models.resnet50( pretrained=True )
        
        self.conv1 = pretrained_resnet.conv1
        self.bn1 = pretrained_resnet.bn1
        self.relu = pretrained_resnet.relu
        self.maxpool = pretrained_resnet.maxpool

        self.layer1 = pretrained_resnet.layer1
        self.layer2 = pretrained_resnet.layer2
        self.layer3 = pretrained_resnet.layer3
        self.layer4 = pretrained_resnet.layer4

        self.avgpool = pretrained_resnet.avgpool


        self.classifier = nn.Sequential(
            nn.Linear( 2048, 512 ),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear( 512, 5 ),
        )

        initialize_weights( self.classifier )


        # freezing layers
        for layer in [self.conv1, self.bn1]:
            for param in layer.parameters():
               param.requires_grad = False

        # freezing Sequential layers
        for seq in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for layer in seq.children():
                for param in layer.parameters():
                    param.requires_grad = False

        

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view( x.size(0), -1 )

        x = self.classifier(x)

        return x









class Model_D(nn.Module):

    def __init__(self):
        super(Model_D, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(12, 8),
            nn.ReLU(True),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

        initialize_weights( self.model )
        

    def forward(self, x):
        return self.model(x)








def initialize_weights( model ):

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()





