import torch
import torch.nn as nn
from network.backbone.resnet import deformable_resnet18,deformable_resnet50
import torch.utils.model_zoo as model_zoo
from cfglib.config import config as cfg

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',

}


class ResNet_DCN(nn.Module):
    def __init__(self, name="deformable_resnet18", pretrain=False):
        super().__init__()

        if name == "deformable_resnet18":
            self.base_net = deformable_resnet18(pretrained=False)
            if pretrain:
                print("load the {} weight from ./cache".format(name))
                self.base_net.load_state_dict(
                    model_zoo.load_url(model_urls["resnet18"], model_dir="./cache",
                                       map_location=torch.device(cfg.device)), strict=False)

        elif name == "deformable_resnet50":
            self.base_net = deformable_resnet50(pretrained=False)
            if pretrain:
                print("load the {} weight from ./cache".format(name))
                self.base_net.load_state_dict(
                    model_zoo.load_url(model_urls["resnet50"], model_dir="./cache",
                                       map_location=torch.device(cfg.device)), strict=False)
        else:
            print(" base model is not support !")

        # print(base_net)
        self.up2 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        C1, C2, C3, C4, C5 = self.base_net(x)
        # up2 --> 1/2
        C1 = self.up2(C1)

        return C1, C2, C3, C4, C5


if __name__ == '__main__':
    import torch
    input = torch.randn((4, 3, 512, 512))
    net = ResNet_DCN()
    C1, C2, C3, C4, C5 = net(input)
    print(C1.size())
    print(C2.size())
    print(C3.size())
    print(C4.size())
    print(C5.size())
