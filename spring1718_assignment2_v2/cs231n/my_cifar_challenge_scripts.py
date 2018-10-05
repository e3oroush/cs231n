import torch
from torch import nn
from torchvision.models.resnet import BasicBlock

def conv_maxpool_layer(in_channels=3, out_channels=64, conv_kernel_size=3, conv_stride=1, pool_kernel_size=3,pool_stride=2):
    padding=(conv_kernel_size-conv_stride)/2
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, conv_kernel_size, conv_stride, padding),
                  nn.ReLU(),
                  nn.MaxPool2d(pool_kernel_size,pool_stride))
def bn_conv_maxpool_layer(in_channels=3, out_channels=64, conv_kernel_size=3, conv_stride=1, pool_kernel_size=3,pool_stride=2):
    padding=(conv_kernel_size-conv_stride)/2
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, conv_kernel_size, conv_stride, padding),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(),
                  nn.MaxPool2d(pool_kernel_size,pool_stride))

class SimpleConvNet(nn.Module):

    def __init__(self, num_classes=10):
        super(SimpleConvNet, self).__init__()
        self.features = nn.Sequential(
            conv_maxpool_layer(in_channels=3, out_channels=64, conv_kernel_size=3, conv_stride=1, pool_kernel_size=2, pool_stride=2), # 16 x 16
            conv_maxpool_layer(in_channels=64, out_channels=128, conv_kernel_size=3, conv_stride=1, pool_kernel_size=2, pool_stride=2), # 8 x 8
            conv_maxpool_layer(in_channels=128, out_channels=512, conv_kernel_size=3, conv_stride=1, pool_kernel_size=2, pool_stride=2)) # 4 x 4
        self.classifier = nn.Sequential(nn.Linear(4*4*512,4096),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(4096,num_classes))
        self.__initialize_params()
        
    def __initialize_params(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        f=self.features(x)
        f=f.view(f.shape[0], -1)
        o = self.classifier(f)
        return o


class SimpleRegularizedConvNet(SimpleConvNet):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            bn_conv_maxpool_layer(in_channels=3, out_channels=64, conv_kernel_size=3, conv_stride=1, pool_kernel_size=2, pool_stride=2), # 16 x 16
            bn_conv_maxpool_layer(in_channels=64, out_channels=128, conv_kernel_size=3, conv_stride=1, pool_kernel_size=2, pool_stride=2), # 8 x 8
            bn_conv_maxpool_layer(in_channels=128, out_channels=512, conv_kernel_size=3, conv_stride=1, pool_kernel_size=2, pool_stride=2)) # 4 x 4
        self.classifier = nn.Sequential(nn.Dropout(p=0.5),
                                        nn.Linear(4*4*512,4096),
                                        nn.Dropout(p=0.5),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(4096,num_classes))

class ConvNet(SimpleRegularizedConvNet):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(64),
                                      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2,stride=2), # 16 x 16
                                      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(128),
                                      nn.Conv2d(in_channels=128,out_channels=128, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2), # 8 x 8
                                    #   nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, padding=1),
                                    #   nn.BatchNorm2d(512),
                                    #   nn.Conv2d(in_channels=512,out_channels=512, kernel_size=3, padding=1),
                                    #   nn.BatchNorm2d(512),
                                    #   nn.ReLU(inplace=True),
                                    #   nn.MaxPool2d(kernel_size=2, stride=2) # 4 x 4
                                      )



class SimpleReseNet(SimpleConvNet):
    def __init__(self, num_classes=10):
        super().__init__()
        # BasicBlock.expansion=2
        self.features = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(64),
                                      BasicBlock(64, 64),
                                      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=2), # 16 x 16
                                      nn.BatchNorm2d(64),
                                      BasicBlock(64,64),
                                      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=2), # 8 x 8
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True)
                                      )
        self.classifier = nn.Sequential(nn.Dropout(p=0.5),
                                        nn.Linear(8*8*64,4096),
                                        nn.Dropout(p=0.5),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(4096,num_classes))


if __name__ == '__main__':
    m=SimpleConvNet(num_classes=10)