import torch
import torch.nn as nn

# AudioCNN_TConv (음성 특징 추출기)
class AudioCNN_TConv(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=128):
        super().__init__()

        # 특징 이미지 (Frequency x Time) 처리를 위한 2D 컨볼루션 블록
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # 주파수 축 다운샘플링
            nn.Dropout(0.1),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Dropout(0.1),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # 시간 축의 문맥 정보를 포착하기 위한 1D 컨볼루션 및 시간 평균 풀링
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # 시간축을 1로 줄여 글로벌 특징 추출
        )

        # 최종 128차원 특징 벡터를 생성하는 FC 레이어
        self.fc_feat = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # 분류 헤드 (특징 추출 시 사용하지 않으나, 전체 구조 보존을 위해 포함)
        self.fc_out = nn.Linear(hidden_dim, 1)

    # 서버 추론 및 Fusion Model에서 사용할 특징 추출 함수 (핵심)
    def forward_features(self, x):
        # 1. 2D Conv
        x = self.conv2d(x)  # (B, 128, F', T)
        x = torch.mean(x, dim=2)  # 주파수 축(F') 평균: (B, 128, T)

        # 2. 1D Temporal Conv + Global Pooling
        x = self.temporal_conv(x).squeeze(-1)  # (B, 128)

        # 3. Final Feature Projection
        feat = self.fc_feat(x)

        # 안정화 로직
        feat = torch.nan_to_num(feat).clamp(-3000, 3000)

        # 128차원 특징 벡터 반환
        return feat

    # 일반 forward (이 부분은 사용되지 않음)
    def forward(self, x):
        feat = self.forward_features(x)
        out = self.fc_out(feat)
        # BCEWithLogitsLoss를 사용했으므로 Sigmoid를 여기서 적용하지 않고 로짓(out) 반환
        return out, feat
