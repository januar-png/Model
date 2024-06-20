!pip install jcopdl
import torch
from torch import nn
from jcopdl.layers import linear_block, conv_block

class CNNpenyakitPadi(nn.Module):
    def __init__(self, output_size):
        super(CNNpenyakitPadi, self).__init__()
        self.feature = nn.Sequential(
            
            conv_block(3, 16, kernel=3, stride=1, pad=1, 
                      batch_norm=True, activation='relu',
                      pool_type='maxpool', pool_kernel=2, pool_stride=2),
            
            conv_block(16, 32, kernel=3, stride=1, pad=1, 
                      batch_norm=True, activation='relu',
                      pool_type='maxpool', pool_kernel=2, pool_stride=2),
            
            conv_block(32, 64, kernel=3, stride=1, pad=1, 
                      batch_norm=True, activation='relu',
                      pool_type='maxpool', pool_kernel=2, pool_stride=2),
                         
            conv_block(64, 128, kernel=3, stride=1, pad=1, 
                      batch_norm=True, activation='relu',
                      pool_type='maxpool', pool_kernel=2, pool_stride=2),
            
            conv_block(128, 256, kernel=3, stride=1, pad=1, 
                      batch_norm=True, activation='relu',
                      pool_type='maxpool', pool_kernel=2, pool_stride=2),
            
            conv_block(256, 512, kernel=3, stride=1, pad=1, 
                      batch_norm=True, activation='relu',
                      pool_type='maxpool', pool_kernel=2, pool_stride=2),
            
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            linear_block(512*3*3, 256, activation='relu'),
            linear_block(256, output_size, activation='softmax')
        )
    def forward(self, x):
        x = self.feature(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
