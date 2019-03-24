
import time
import torch.nn as nn
import torch

class Conv3d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        
        super(Conv3d, self).__init__()
        
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)
        
        #
        self.bn = nn.BatchNorm3d(out_channels, eps=1e-3, momentum=0.001, affine=True)
        self.relu = nn.ReLU(inplace=True)
        
    
    def forward(self, input):
        
        x = self.conv(input)
        x = self.bn(x)
        x = self.relu(x)
        
        return x
        

class SpatialTemporalConv(nn.Module):
    
    # Separate conv3d into
    # conv2d spatial-convolution with kernel-size 1*K*K
    # conv1d temporal-convolution with kernel-size Ktem*1*1
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        
        super(SpatialTemporalConv, self).__init__()
        
        # TODO kernel-size might should take the depth of inputs
        self.spatial_conv = nn.Conv3d(in_channels, out_channels,
                                      kernel_size=(1, kernel_size, kernel_size),
                                      stride=(1, stride, stride),
                                      padding=(0, padding, padding))
        
        self.temporal_conv = nn.Conv3d(out_channels, out_channels,
                                       kernel_size=(kernel_size, 1, 1),
                                       stride=(stride, 1, 1),
                                       padding=(padding, 0, 0))
        
        self.bn = nn.BatchNorm3d(out_channels, eps=1e-3, momentum=0.001, affine=True)
        self.relu = nn.ReLU(inplace=True)
        
        self.bn2 = nn.BatchNorm3d(out_channels, eps=1e-3, momentum=0.001, affine=True)
        self.relu2 = nn.ReLU(inplace=True)
        
        nn.init.normal(self.temporal_conv.weight, mean=0, std=0.01)
        nn.init.constant(self.temporal_conv.bias, 0)
        
    
    def forward(self, x):
        
        x = self.spatial_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.temporal_conv(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        return x
        

class SeparableIncBlock1(nn.Module):
    
    def __init__(self, ):
        
        super(SeparableIncBlock1, self).__init__()
        
        self.branch0 = nn.Sequential(
            Conv3d(192, 64, kernel_size=1, stride=1)
        )
        
        self.branch1 = nn.Sequential(
            Conv3d(192, 96, kernel_size=1, stride=1),
            SpatialTemporalConv(96, 128, kernel_size=3, stride=1, padding=1)
        )
        
        self.branch2 = nn.Sequential(
            Conv3d(192, 16, kernel_size=1, stride=1),
            SpatialTemporalConv(16, 32, kernel_size=3, stride=1, padding=1)
        )
        
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            Conv3d(192, 32, kernel_size=1, stride=1)
        )
    
    
    def forward(self, x):
        
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        
        result = torch.cat((x0, x1, x2, x3), 1)
        
        return result

class SeparableIncBlock2(nn.Module):
    
    def __init__(self):
        
        super(SeparableIncBlock2, self).__init__()
        
        self.branch0 = nn.Sequential(
            Conv3d(256, 64, kernel_size=1, stride=1)
        )

        self.branch1 = nn.Sequential(
            Conv3d(256, 128, kernel_size=1, stride=1),
            SpatialTemporalConv(128, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            Conv3d(256, 32, kernel_size=1, stride=1),
            SpatialTemporalConv(32, 48, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            Conv3d(256, 32, kernel_size=1, stride=1)
        )
        
    
    def forward(self, x):
        
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
    
        result = torch.cat((x0, x1, x2, x3), 1)
    
        return result


class SeparableIncBlock3(nn.Module):
    
    def __init__(self):
        
        super(SeparableIncBlock3, self).__init__()
        
        self.branch0 = nn.Sequential(
            Conv3d(240, 96, kernel_size=1, stride=1)
        )

        self.branch1 = nn.Sequential(
            Conv3d(240, 96, kernel_size=1, stride=1),
            SpatialTemporalConv(96, 104, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            Conv3d(240, 16, kernel_size=1, stride=1),
            SpatialTemporalConv(16, 24, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            Conv3d(240, 32, kernel_size=1, stride=1)
        )
        
    
    def forward(self, x):
    
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
    
        result = torch.cat((x0, x1, x2, x3), 1)
    
        return result
    
    
class SeparableIncBlock4(nn.Module):
    
    def __init__(self):
        
        super(SeparableIncBlock4, self).__init__()
        
        self.branch0 = nn.Sequential(
            Conv3d(256, 112, kernel_size=1, stride=1)
        )

        self.branch1 = nn.Sequential(
            Conv3d(256, 144, kernel_size=1, stride=1),
            SpatialTemporalConv(144, 288, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            Conv3d(256, 32, kernel_size=1, stride=1),
            SpatialTemporalConv(32, 64, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            Conv3d(256, 64, kernel_size=1, stride=1)
        )
    
    
    def forward(self, x):
    
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
    
        result = torch.cat((x0, x1, x2, x3), 1)
    
        return result
    
class S3D(nn.Module):
    
    def __init__(self, num_classes=400, dropout_keep_prob=1, in_channels=3, spatial_squeeze=True):
        
        super(S3D, self).__init__()
        
        self.features = nn.Sequential(
            
            SpatialTemporalConv(in_channels, 64, kernel_size=7, stride=2, padding=3), # (64, 16, 112, 112)
            nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), # (64, 16, 56, 56),
            Conv3d(64, 64, kernel_size=1, stride=1), #(64, 16, 56, 56)
            SpatialTemporalConv(64, 192, kernel_size=3, stride=1, padding=1), #(192, 16, 56, 56),
            nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), #(192, 16, 28, 28)
            SeparableIncBlock1(), # (126, 16, 28, 28)
            SeparableIncBlock2(), # (240, 16, 28, 28)
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)), # (480, 8, 14, 14)
            SeparableIncBlock3(), # (256, 16, 14, 14)
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)), # (256, 4, 7, 7)
            SeparableIncBlock4(), # (528, 4, 7, 7)
            
            nn.AvgPool3d(kernel_size=(1, 7, 7), stride=1), # (528, 4, 1, 1)
            nn.Dropout3d(dropout_keep_prob),
            nn.Conv3d(528, num_classes, kernel_size=1, stride=1, bias=True)  # (400, 4, 1, 1)
            
        )
        
        self.softmax = nn.Softmax()
        
    
    def forward(self, x):
        
        logits = self.features(x)
        
        averaged_logit = torch.mean(logits, 2)
        
        predictions = self.softmax(averaged_logit)
        
        
        return averaged_logit, predictions

if __name__ == '__main__':
    
    d = torch.autograd.Variable(torch.randn([2, 3, 64, 224, 224])).cuda()
    m = S3D().cuda()
    start = time.time()
    averaged_logit, predictions = m(d)
    end = time.time()
    print(averaged_logit.size())
    print(predictions.size())
    print(f'cost: {end - start} seconds')
