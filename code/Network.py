### YOUR CODE HERE
# import tensorflow as tf
import torch
from torch import Tensor
import torch.nn as nn
from collections import OrderedDict  # pylint: disable=g-importing-member

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""This script defines the network.
"""

class MyNetwork(nn.Module):

    def __init__(self, configs):
        super(MyNetwork,self).__init__()
        self.configs = configs
        self.network = ResNetV2(configs['width_multiplier'],configs['dropout_rate'],self.configs['layers_per_block'])

    def __call__(self, inputs, training):
        '''
        Args:
            inputs: A Tensor representing a batch of input images.
            training: A boolean. Used by operations that work differently
                in training and testing phases such as batch normalization.
        Return:
            The output Tensor of the network.
        '''
        return self.build_network(inputs, training)

    def build_network(self, inputs, training):
        if training:
            # set it to train mode
            self.network.train()
        else:
            # set it to evaluation mode
            self.network.eval()
        return self.network.forward(inputs)


"""PreAct ResNet with GroupNorm and Weight Standardization."""

class StdConv2d(nn.Conv2d):

  def forward(self, x):
    w = self.weight
    v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
    w = (w - m) / torch.sqrt(v + 1e-10)
    return F.conv2d(x, w, self.bias, self.stride, self.padding,
                    self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
  return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                   padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
  return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                   padding=0, bias=bias)


class PreActBlock(nn.Module):

  def __init__(self, cin, cout, stride, dropout_rate=0.):
    super().__init__()

    self.gn1 = nn.GroupNorm(32, cin) if cin>=32 else nn.GroupNorm(16,cin)
    self.conv1 = conv3x3(cin, cout, stride)
    self.gn2 = nn.GroupNorm(32, cout)
    self.conv2 = conv3x3(cout, cout, 1)
    self.relu = nn.ReLU(inplace=True)
    self.dropout = nn.Dropout(p=dropout_rate)

    if (cin != cout):
      self.downsample = conv1x1(cin, cout, stride)

  def forward(self, x):
    if hasattr(self, 'downsample'):
      # change x to add shortcut after pre-activation
      x = self.relu(self.gn1(x))
      out = self.conv1(x)
      residual = self.downsample(x)
    else:
      # don't change x
      out = self.relu(self.gn1(x))
      out = self.conv1(out)
      residual = x      

    out = self.dropout(self.relu(self.gn2(out)))
    out = self.conv2(out)

    # print('out.shape: ',out.shape,' residual.shape: ', residual.shape)
    return out + residual

class stack_layer(nn.Module):
  def __init__(self, num_layers, cin, cout, stride, dropout_rate=0.):
    super(stack_layer, self).__init__()

    self.stack = nn.ModuleList()
    for i in range(num_layers):
      if i==0:
        self.stack.append(PreActBlock(cin,cout,stride,dropout_rate))
      else:
        self.stack.append(PreActBlock(cout,cout,1,dropout_rate))
    
  def forward(self,x):
    for i in range(len(self.stack)):
      x = self.stack[i](x)
    return x



class ResNetV2(nn.Module):

  def __init__(self, width_factor, dropout_rate, num_layers_per_block):
    super().__init__()
    wf = width_factor
    self.wf = wf

    self.root = StdConv2d(3,16,kernel_size=3,stride=1,padding=1,bias=False)
    self.layer1 = stack_layer(num_layers_per_block, 16, 16*wf,1,dropout_rate)
    self.layer2 = stack_layer(num_layers_per_block, 16*wf, 32*wf,2,dropout_rate)
    self.layer3 = stack_layer(num_layers_per_block, 32*wf, 64*wf,2,dropout_rate)

    self.gn = nn.GroupNorm(32,64*wf)
    self.relu = nn.ReLU(inplace=True)
    self.fc = nn.Linear(64*wf,10)

  def forward(self, x):
    out = self.layer3(self.layer2(self.layer1(self.root(x))))
    out = self.relu(self.gn(out))
    out = F.avg_pool2d(out,8)
    out = out.view(-1,64*self.wf)
    out = self.fc(out)
    return out

### END CODE HERE