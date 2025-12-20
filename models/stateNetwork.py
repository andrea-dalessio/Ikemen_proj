import torch
import torch.nn as nn
from sklearn import preprocessing as p

class StateNetwork(nn.Module):
    def __init__(self, inputSize, outputSize, device) -> None:
        super().__init__()
        self.inputSize = inputSize
        self.outputSize = outputSize
        
        self.act_fn = nn.ReLU()
        
        self.lin1 = nn.Linear(self.inputSize, 256)
        self.lin2 = nn.Linear(256, 128)
        self.lin3 = nn.Linear(128, 64)
        self.output = nn.Linear(64, self.outputSize)
        
        self.scaler = p.StandardScaler()
        
    def forward(self, x):
        
        x = self.scaler.transform(x)
        
        x = torch.tensor(x)
        
        x = self.act_fn(self.lin1(x))
        x = self.act_fn(self.lin2(x))
        x = self.act_fn(self.lin3(x))
        x = self.output(x)
        
        
    def prepareScaler(self, dataset):
        self.scaler.fit(dataset)