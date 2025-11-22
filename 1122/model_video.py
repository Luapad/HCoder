import torch
import torch.nn as nn
import torch.nn.functional as F

# TinyRes3D
class TinyRes3D(nn.Module):
    def __init__(self):
        super().__init__()

        # R(2+1)D 블록 정의
        def r2p1d_block(in_c, out_c):
            return nn.Sequential(
                # 시간 컨볼루션
                nn.Conv3d(in_c, out_c, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
                # 공간 컨볼루션
                nn.Conv3d(out_c, out_c, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True)
            )

        # 블록 정의 (채널 수: 3 -> 32 -> 64 -> 128)
        self.block1 = r2p1d_block(3, 32)
        self.block2 = r2p1d_block(32, 64)
        self.block3 = r2p1d_block(64, 128)

        # 잔차 연결을 위한 채널 맞춤 (Projection)
        self.res_proj1 = nn.Conv3d(32, 64, kernel_size=1)
        self.res_proj2 = nn.Conv3d(64, 128, kernel_size=1)

        self.pool = nn.MaxPool3d((2, 2, 2))
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # 분류 헤드 (특징 추출 시 사용하지 않으나, 전체 구조 보존을 위해 포함)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, 1)

    # 서버 추론 및 Fusion Model에서 사용할 특징 추출 함수 (핵심)
    def forward_features(self, x):
        # Block 1
        x = self.block1(x)
        res1 = x
        x = self.pool(x)

        # Block 2 (잔차 연결)
        x = self.block2(x)
        res1_down = F.interpolate(res1, size=x.shape[2:], mode="trilinear", align_corners=False)
        res1_down = self.res_proj1(res1_down)
        x = x + res1_down
        res2 = x
        x = self.pool(x)

        # Block 3 (잔차 연결)
        x = self.block3(x)
        res2_down = F.interpolate(res2, size=x.shape[2:], mode="trilinear", align_corners=False)
        res2_down = self.res_proj2(res2_down)
        x = x + res2_down

        # Global pooling: (B, 128, 1, 1, 1)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # (B, 128)

        # 안정화 로직
        x = torch.nan_to_num(x).clamp(-3000, 3000)

        # 128차원 특징 벡터 반환
        return x

    # 일반 forward (이 부분은 사용되지 않음)
    def forward(self, x):
        feat = self.forward_features(x)
        feat = self.dropout(feat)
        return torch.sigmoid(self.fc(feat))
