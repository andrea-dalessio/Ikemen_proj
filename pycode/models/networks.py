import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.fc = nn.Linear(size, size)
        self.ln = nn.LayerNorm(size)
    
    def forward(self, x):
        return F.relu(self.ln(self.fc(x))) + x
    
class DecisionBlock(nn.Module):
    def __init__(self, input_size, move_size, hit_size):
        super().__init__()
        self.input = nn.Linear(input_size, 512)
        self.res1 = ResidualBlock(512)
        self.res2 = ResidualBlock(512)
        self.res3 = ResidualBlock(512)
        
        self.featureHead = nn.Linear(512, 256)
        
        self.moveHead = nn.Linear(256, move_size)
        self.hitHead = nn.Linear(256, hit_size)
        self.valueHead = nn.Linear(256,1)
        
        self.i_size = input_size
        self.m_size = move_size
        self.h_size = hit_size
        
    def makeCopy(self)->"DecisionBlock":
        res = DecisionBlock(self.i_size, self.m_size, self.h_size)
        res.load_state_dict(self.state_dict())
        return res
    
    def getFeatures(self, x):
        x = F.relu(self.input(x))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = F.relu(self.featureHead(x))
        return x
    
    def getMoveAndAttack(self, x):
        x = self.getFeatures(x)
        logits_move = self.moveHead(x)
        logits_attack = self.hitHead(x)
        
        return logits_move, logits_attack

    def getMoveAndAttackAndValue(self, x):
        x = self.getFeatures(x)
        logits_move = self.moveHead(x)
        logits_attack = self.hitHead(x)
        value = self.valueHead(x)
        return logits_move, logits_attack, value
    
class VisualDecisor(nn.Module):
    def __init__(self, move_size, hit_size, freeze_backbone=True):
        super().__init__()
        
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.decisor = DecisionBlock(2048, move_size, hit_size)
        
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        
    def makeCopy(self)->"VisualDecisor":
        res = VisualDecisor(self.m_size, self.h_size)
        res.load_state_dict(self.state_dict())
        return res
    
    def getFeatures(self, x):
        if x.dim()==3:
            x = x.unsqueeze(0)
            
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = x.mean(dim=(2,3))
        x = self.decisor.getFeatures(x)
        return x
    
    def getMoveAndAttack(self, x):
        x = self.getFeatures(x)
        logits_move = self.decisor.moveHead(x)
        logits_attack = self.decisor.hitHead(x)
        
        return logits_move, logits_attack

    def getMoveAndAttackAndValue(self, x):
        x = self.getFeatures(x)
        logits_move = self.decisor.moveHead(x)
        logits_attack = self.decisor.hitHead(x)
        value = self.decisor.valueHead(x)
        return logits_move, logits_attack, value
    
    def setPhase2(self):
        for p in self.backbone.layer4.parameters():
            p.requires_grad = True