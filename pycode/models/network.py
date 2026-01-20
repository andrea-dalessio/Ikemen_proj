import torch as tc
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.fc = nn.Linear(size, size)
        self.ln = nn.LayerNorm(size)
    
    def forward(self, x):
        return F.relu(self.ln(self.fc(x))) + x
    
class DecisionNetwork(nn.Module):
    def __init__(self, size, moves_n, hits_n):
        super().__init__()
        self.input_layer = nn.Linear(size, 512)
        
        # Residual Blocks
        self.res_block1 = ResidualBlock(512)
        self.res_block2 = ResidualBlock(512)
        self.res_block3 = ResidualBlock(512)
        
        self.feature_head = nn.Linear(512, 256)
        
        # Policy Heads (actor-critic)
        self.movePolicy = nn.Linear(256, moves_n)
        self.hitPolicy = nn.Linear(256, hits_n)
        self.valueEstimator = nn.Linear(256, 1)
        
    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        hidden = F.relu(self.feature_head(x))
        logits_move = self.movePolicy(hidden)
        logits_attack = self.hitPolicy(hidden)
        value = self.valueEstimator(hidden)
        
        return logits_move, logits_attack, value
        
    def actionOnly(self,x):
        x = F.relu(self.input_layer(x))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        hidden = F.relu(self.feature_head(x))
        logits_move = self.movePolicy(hidden)
        logits_attack = self.hitPolicy(hidden)
        
        return logits_move, logits_attack
    

# class VisualNetwork(DecisionNetwork):
#     def __init__(self, h, w, ch, moves_n, hits_n):
#         # window_width: 640
#         # window_height: 480
        
        
        
#         self.analysisNetwirk = DecisionNetwork(0, moves_n, hits_n)