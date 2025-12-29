import torch
import torch.nn as nn
import torch.functional as F


class CNNNetwork(nn.Module):
    def __init__(self, inputW, inputH, outputSize, device, stackSize=4):
        super().__init__()
        self.stackSize = stackSize
        self.device = device
        self.act = nn.ReLU()
        
        inputChannels = 3 * self.stackSize
        
        w = (inputW-8)/4
        h = (inputH-8)/4
        self.conv1 = nn.Conv2d(inputChannels, 32, kernel_size=8, stride=4)
        
        w = (w-4)/2
        h = (h-4)/2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        
        w = (w-3)
        h = (h-3)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        self.hidden = nn.Linear(h*w*64, 512)
        self.output = nn.Linear(512, outputSize)
        

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        x = x.to(device=self.device)
        x = x.permute(0,3,1,2)
        x = x/255.0
        
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        
        x = x.reshape(x.size(0), -1)
        
        x = self.act(self.hidden(x))
        x = self.output(x)
        
        return x