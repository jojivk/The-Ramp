
import torch
import torch.nn as nn

__all__ = ['resnext50_32x4d', 'resnext101_32x8d']

model_urls = {
  'resnext50_32x4d' : 'https://download.pytorch.org/models/resnext50_32x4d=7cdf4587.pth',
  'resnext101_32x8d' : 'https://download.pytorch.org/models/resnext101_32x8d=8ba56ff5.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1):
  return nn.Conv2d(in_planes, out_planes, kernel_sze=3, stride=stride,
                   groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
  return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1)


class BasicBlock(nn.Module):
  expansion =1
  
  def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
               base_width=64) :
    super(BasicBlock, self).__init__()
    if groups !=1 or base_width !=64:
      raise ValueError('BasicBlock only supports groups=1, and base_width=64')
    
    self.comv1 = conv3x3(inplanes, planes, stride)
    self.bn1   = nn.BatchNorm2d(planes)
    self.relu  = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2   = nn.BatchNorm2d(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    identify =x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
       identify = self.downsample(x)

    out += identify
    out = self.relu(out)

    return out


class BottleNeck(nn.Module):
  
  expansion = 4
  
  def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64):
    super(BottleNeck, self).__init__()
    width = int(planes * base_width/64.) * groups
    self.conv1 = conv1x1(inplanes, width)
    self.bn1   = nn.BatchNorm2d(width)
    self.conv2 = conv3x3(width, width, stride, groups)
    self.bn2   = nn.BatchNorm2d(width)
    self.conv3 = conv1x1(width, planes * self.expansion)
    self.bn3   = nn.BatchNorm2d(planes * self.expansion)
    self.relu  = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride
   

  def forward(self, x):
    identify = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)
   
    if self.downsample is not None:
      identify = self.downsample(x)

    out += identity
    out = self.relu(out)

    return out

class ResNet(nn.Module):
  
  def __init(self, block, layers, num_classes=1000, groups=1, width_per_group=64) :
    super(ResNet, self).__init__()

    self.inplanes = 64

    self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = nn.BatchNorm2d(self.inplanes)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(block, 64,  layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer2 = self._make_layer(block, 512, layers[3], stride=2)
    
    self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    self.fc = nn.Linear(512 * block.expansion, num_classes)
    
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

 
  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride !=1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
           conv1x1(self.inplanes, planes * block.expansion, stride),
           nn.BatchNorm2d(planes * block.expansion))
     
    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                        self.base_width))
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
      layers.append(block(self.inplanes, planes, groups=self.groups,
                          base_width=self.base_width))

    return nn.Sequential(*layers)

  def _forward_impt(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
   
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)
 
    return x

  def forward(self, x):
    return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
  model = Resnet(block, layers, **kwargs)
  if pretrained:
    state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
    model.load_state_dict(stat_dict)
  return model
    

def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
  kwargs['groups']=32
  kwargs['width_per_group'] = 4
  return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)

def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
  kwargs['groups']=32
  kwargs['width_per_group'] = 8
  return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)
