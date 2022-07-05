from torch import nn
import torchvision


class ResNet(nn.Module):
    def __init__(self, in_ch, num_cls, depth):
        super(ResNet, self).__init__()
        allowed_depth = [18, 34, 50, 101, 152]
        if depth == 18:
            self.resnet = torchvision.models.resnet18(pretrained=False)
        elif depth == 34:
            self.resnet = torchvision.models.resnet34(pretrained=False)
        elif depth == 50:
            self.resnet = torchvision.models.resnet50(pretrained=False)
        elif depth == 101:
            self.resnet = torchvision.models.resnet101(pretrained=False)
        elif depth == 152:
            self.resnet = torchvision.models.resnet152(pretrained=False)
        else:
            raise NameError(f"Should be one of {allowed_depth}")

        self.resnet.conv1 = nn.Conv2d(in_ch, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(512, num_cls)
        self.name = "Multi-class_ResNet18"
        self.num_cls = num_cls

    def forward(self, x):
        x = self.resnet(x)
        return x