from adjoint import AdjointFunc,flat_parameters
import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, padding=1,stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ODEFunc(nn.Module):
    def __init__(self,inplanes):
        super(ODEFunc, self).__init__()
        self.conv1 = conv3x3(inplanes+1,inplanes)
        self.norm1 = nn.BatchNorm2d(inplanes)
        self.conv2 = conv3x3(inplanes+1,inplanes)
        self.norm2 = nn.BatchNorm2d(inplanes)

    def forward(self,x,t):
        t_tensor1 = torch.ones((x.size()[0],1,x.size()[2],x.size()[3]),device="cuda")*t
        t_tensor1.requires_grad = False
        x = torch.cat((x,t_tensor1),dim=1)
        x = self.conv1(x)
        x = self.norm1(torch.relu(x))

        t_tensor2 = torch.ones_like(t_tensor1)*t
        t_tensor2.requires_grad = False
        x = torch.cat((x,t_tensor2),dim=1)
        x = self.conv2(x)
        x = self.norm2(torch.relu(x))
        return x

class ODEBlock(nn.Module):
    def __init__(self,inplanes):
        super(ODEBlock,self).__init__()
        self.func = ODEFunc(inplanes)

    def forward(self,x):
        x = AdjointFunc.apply(x,self.func,torch.tensor([0.0],device="cuda"),torch.tensor([1.0],device="cuda"),flat_parameters(self.func.parameters()))
        return x


class Model(nn.Module):
    def __init__(self,num_classes=10):
        super(Model,self).__init__()
        dim = 64
        self.downsampling = nn.Sequential(
            nn.Conv2d(1,dim,5,2,0),nn.BatchNorm2d(dim),nn.ReLU(inplace=True),
            nn.Conv2d(dim,dim,4,2),nn.BatchNorm2d(dim),nn.ReLU(inplace=True),
            conv1x1(dim,dim)#to avoid error
        )

        self.neuralODE = ODEBlock(dim)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self,x):
        x = self.downsampling(x)
        x = self.neuralODE(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x