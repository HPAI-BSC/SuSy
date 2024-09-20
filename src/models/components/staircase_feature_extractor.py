import torch
import torchvision.models as models
from torch import nn


class FeatureExtractorStaircase(nn.Module):
    def __init__(
            self, 
            model_path: str, 
            freeze: bool) -> None:    
                         
        super().__init__()

        self.freeze = freeze

        self.resnet_model = models.resnet18(weights=None)
        
        if model_path is not None:
            self.resnet_model.load_state_dict(torch.load(model_path))
        else:
            print("No model path provided, initializing with random weights")
            
        if self.freeze:
            self.resnet_model.eval()

        # self.resnet_model.cuda()
        
        self.stages = nn.ModuleList([
            nn.Sequential(self.resnet_model.conv1, self.resnet_model.bn1, self.resnet_model.relu, self.resnet_model.maxpool),
            self.resnet_model.layer1,
            self.resnet_model.layer2,
            self.resnet_model.layer3,
            self.resnet_model.layer4
        ])

        self.f11 = self.create_bottleneck_same_size(self.resnet_model.layer1[0].conv1.in_channels)

        self.f12 = self.create_bottleneck(self.resnet_model.layer2[0].conv1.in_channels)
        self.f22 = self.create_bottleneck(self.resnet_model.layer2[0].conv1.in_channels)

        self.f13 = self.create_bottleneck(self.resnet_model.layer3[0].conv1.in_channels)
        self.f23 = self.create_bottleneck(self.resnet_model.layer3[0].conv1.in_channels)
        self.f33 = self.create_bottleneck(self.resnet_model.layer3[0].conv1.in_channels)

        self.f14 = self.create_bottleneck(self.resnet_model.layer4[0].conv1.in_channels)
        self.f24 = self.create_bottleneck(self.resnet_model.layer4[0].conv1.in_channels)
        self.f34 = self.create_bottleneck(self.resnet_model.layer4[0].conv1.in_channels)
        self.f44 = self.create_bottleneck(self.resnet_model.layer4[0].conv1.in_channels)

        self.bottlenecks = nn.ModuleList([
            self.f11,
            self.f12, self.f22,
            self.f13, self.f23, self.f33,
            self.f14, self.f24, self.f34, self.f44
        ])


        if freeze:
            for stage in self.stages:
                for param in stage.parameters():
                    param.requires_grad = False

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    #The feature map Fi is first reduced to the number of channels to a quarter through the 1×1 convolution layer, which is used to decrease the computation complexities of the following procedures. Then the feature map is reduced to its resolution to half through the 3×3 convolution layer with a stride of 2. Finally, the feature map is passed through the 1×1 convolution layer to increase the number of channels eight times.
    def create_bottleneck(self, in_channels, bias=True):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.Conv2d(in_channels // 4, in_channels // 2, kernel_size=3, stride=2, padding=1, bias=bias),
            nn.Conv2d(in_channels // 2, in_channels * 2, kernel_size=1, stride=1, padding=0, bias=bias)
        )

    def create_bottleneck_same_size(self, in_channels, bias=True):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.Conv2d(in_channels // 4, in_channels // 2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        )


    def forward(self, x):
        features = []
        for i, stage in enumerate(self.stages):
            x = stage(x)  
            features.append(x)
        
        f1  = self.f11(features[0]) + features[1] 
        f1 = self.f12(f1) + features[2]
        f1 = self.f13(f1) + features[3]
        f1 = self.f14(f1)

        f2  = self.f22(features[1]) + features[2]
        f2 = self.f23(f2) + features[3]
        f2 = self.f24(f2)

        f3  = self.f33(features[2]) + features[3]
        f3 = self.f34(f3)

        f4  = self.f44(features[3])
        
        features = f1 + f2 + f3 + f4 + features[4]
            
        feature_vectors = [self.gap(f).squeeze() for f in features]
        #feature_vectors = torch.concat(feature_vectors, dim=1)
        feature_vectors = torch.stack(feature_vectors, dim=1)

        return feature_vectors.T
    
if __name__ == "__main__":
    _ = FeatureExtractorStaircase(None, False)