######################## 2023A2_network_ (camera 2 + LiDAR) ###########################
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class DownsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)
    
    
class conv3x3 (nn.Module):
    def __init__(self, chann, dropprob, dilated):        
        super().__init__()
        
        self.conv1 = nn.Conv2d(chann, chann, (3,3), stride=1, padding = (1, 1), bias=True, dilation = (1, 1))
        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)
        self.conv2 = nn.Conv2d(chann, chann, (3,3), stride=1, padding = (dilated, dilated), bias=True, dilation = (dilated, dilated))
        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)
        

    def forward(self, input):

        output = self.conv1(input)        
        output = self.bn1(output)
        output = F.relu(output)
        
        output = self.conv2(output)        
        output = self.bn2(output)
                

        if (self.dropout.p != 0):
            output = self.dropout(output)
        
        return F.relu(output+input)    

class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)

class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class EnDecoder (nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        
        self.progress1 = nn.ModuleList()
        self.progress1.append(DownsamplerBlock(3,16))
        self.progress1.append(DownsamplerBlock(16, 64))

#         self.output_miniCat = nn.Conv2d(64, num_classesCat, 1, stride=1, padding=0, bias=True)

        self.progress2 = nn.ModuleList()        
        self.progress2.append(DownsamplerBlock(3,16)) 
        self.progress2.append(DownsamplerBlock(16,32))

        self.progress3 = nn.ModuleList()
        self.progress3.append(DownsamplerBlock(2,8))
        self.progress3.append(DownsamplerBlock(8, 32))
        
        
        self.progress4 = nn.ModuleList()

        for x in range(0, 5):
            self.progress4.append(conv3x3(128, 0.03, 1))   
        self.progress4.append(DownsamplerBlock(128,256))         
        
        for x in range(0, 2):    #2 times
            self.progress4.append(conv3x3(256, 0.3, 2))
            self.progress4.append(conv3x3(256, 0.3, 4))
            self.progress4.append(conv3x3(256, 0.3, 8))
            self.progress4.append(conv3x3(256, 0.3, 16))
            
        self.progress4.append(UpsamplerBlock(256,64))
        self.progress4.append(conv3x3(64, 0, 1))
        self.progress4.append(conv3x3(64, 0, 1))        

        self.progress4.append(UpsamplerBlock(64,16))
        self.progress4.append(conv3x3(16, 0, 1))
        self.progress4.append(conv3x3(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d( 16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input1, input2, input3):
        x0_1 = input1        
        for layer in self.progress1:
            x0_1 = layer(x0_1)
            
        x0_2=input2
        for layer in self.progress2:
            x0_2 = layer(x0_2)
            
        x0_3=input3
        for layer in self.progress3:
            x0_3 = layer(x0_3)            
            
#         output_mini = self.output_miniCat(x0_1)
        
        x1= torch.cat((x0_1, x0_2, x0_3), 1)
        for layer in self.progress4:
            x1 = layer(x1)  

            
            
        output = self.output_conv(x1)

        return output

    
#ERFNet
class DDRNet(nn.Module):
    def __init__(self, num_classes):  #use encoder to pass pretrained encoder
        super().__init__()
        self.endecoder = EnDecoder(num_classes)

    def forward(self, input1, input2, input3):        
        return self.endecoder.forward(input1, input2, input3)