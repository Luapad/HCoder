import torch
import torch.nn as nn
import torch.nn.functional as F

# Correlation Fusion (패턴 상호작용 학습)
class CorrelationFusion(nn.Module):
    def __init__(self, feat_dim=128, hidden_dim=128):
        super().__init__()
        
        # Video(128) + Audio(128) + Interaction(128) = 384
        input_dim = feat_dim * 3 
        
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, v_feat, a_feat):
        # 안정화 & 정규화
        v_feat = torch.nan_to_num(v_feat).clamp(-3000, 3000)
        a_feat = torch.nan_to_num(a_feat).clamp(-3000, 3000)
        v_feat = F.normalize(v_feat + 1e-8, dim=1)
        a_feat = F.normalize(a_feat + 1e-8, dim=1)

        # 상호작용 
        interaction = v_feat * a_feat  
        
        # 결합 
        combined = torch.cat([v_feat, a_feat, interaction], dim=1)

        # 판별
        out = self.fc_layers(combined)
        
        #  CorrelationFusion은 가중치를 반환 X (호환성을 위해 None 반환)
        return out, None, None

