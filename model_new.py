import copy

import timm
import torch
import torch.nn as nn

def create_model(num_classes=7, drop_rate=0):
    model = ResNet18(num_classes=num_classes, drop_rate=drop_rate)

    return model

class ResNet18(nn.Module):
    def __init__(self, num_classes=7, drop_rate=0):
        super(ResNet18, self).__init__()
        self.drop_rate = drop_rate

        model = timm.create_model('resnet18', pretrained=False)
        checkpoint = torch.load('./pretrain/resnet18_msceleb.pth')
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.feature1 = nn.Sequential(*list(model.children())[:-1])

        self.feature2 = copy.deepcopy(self.feature1)
        self.classifier1 = nn.Linear(512, num_classes)
        self.classifier2 = nn.Linear(512, num_classes)

    def forward(self, image, branch = True):
        #if branch is True for computing branch 1, else computing branch2
        if branch:
            feature = self.feature1(image)
            output = self.classifier1(feature)
        else:
            feature = self.feature2(image)
            output = self.classifier2(feature)
        return output, feature

if __name__=="__main__":
    model = create_model()
