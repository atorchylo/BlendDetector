from torch import nn
import torchvision


class ResNet18(nn.Module):
    def __init__(self, in_ch, num_cls):
        super(ResNet18, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(in_ch, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(512, num_cls)
        self.name = "Multi-class_ResNet18"
        self.num_cls = num_cls

    def forward(self, x):
        x = self.resnet(x)
        return x