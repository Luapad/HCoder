import os
import csv
import warnings
import numpy as np
from tqdm import tqdm

# 경고 메시지 억제
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ==========================================
# 1. TinyRes3D (Video Feature Extractor) - Frozen
# ==========================================
class TinyRes3D(nn.Module):
    def __init__(self):
        super().__init__()

        def r2p1d_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=(3,1,1), padding=(1,0,0)),
                nn.Conv3d(out_c, out_c, kernel_size=(1,3,3), padding=(0,1,1)),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True)
            )

        self.block1 = r2p1d_block(3, 32)
        self.block2 = r2p1d_block(32, 64)
        self.block3 = r2p1d_block(64, 128)

        self.res_proj1 = nn.Conv3d(32, 64, kernel_size=1)
        self.res_proj2 = nn.Conv3d(64, 128, kernel_size=1)

        self.pool = nn.MaxPool3d((2,2,2))
        self.global_pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, 1) 

    def forward_features(self, x):
        x = self.block1(x)
        res1 = x
        x = self.pool(x)

        x = self.block2(x)
        res1_down = F.interpolate(res1, size=x.shape[2:], mode="trilinear", align_corners=False)
        res1_down = self.res_proj1(res1_down)
        x = x + res1_down
        res2 = x
        x = self.pool(x)

        x = self.block3(x)
        res2_down = F.interpolate(res2, size=x.shape[2:], mode="trilinear", align_corners=False)
        res2_down = self.res_proj2(res2_down)
        x = x + res2_down

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # (B,128)
        x = torch.nan_to_num(x).clamp(-3000, 3000) # 안정화
        return x

    def forward(self, x):
        feat = self.forward_features(x)
        feat = self.dropout(feat)
        return torch.sigmoid(self.fc(feat))


# ==========================================
# 2. AudioCNN_TConv (Audio Feature Extractor) - Frozen
# ==========================================
class AudioCNN_TConv(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=128):
        super().__init__()

        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2,1)),
            nn.Dropout(0.1),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2,1)),
            nn.Dropout(0.1),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc_feat = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.conv2d(x) 
        x = torch.mean(x, dim=2) 
        x = self.temporal_conv(x).squeeze(-1)
        feat = self.fc_feat(x)
        feat = torch.nan_to_num(feat).clamp(-3000, 3000) 
        out = self.fc_out(feat)
        return out, feat


# ==========================================
# 3. Correlation Fusion (패턴 상호작용 학습)
# ==========================================
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

        # 상호작용 (Interaction)
        interaction = v_feat * a_feat  
        
        # 결합 (Concatenate)
        combined = torch.cat([v_feat, a_feat, interaction], dim=1)

        # 판별
        out = self.fc_layers(combined)
        return out


# ==========================================
# 4. Fusion Model Wrapper (모델 로드 성공 로그 추가)
# ==========================================
class FusionModel(nn.Module):
    def __init__(self, video_weight_path, audio_weight_path, device):
        super().__init__()

        # Pre-trained Load & Freeze
        self.video_encoder = TinyRes3D().to(device)
        try:
            self.video_encoder.load_state_dict(torch.load(video_weight_path, map_location=device))
            # 🚨 모델 로드 성공 로그 추가
            print(f"[SUCCESS] 🎬 Video Encoder 로드 성공: ({video_weight_path})")
            self.video_encoder.eval()
            for p in self.video_encoder.parameters():
                p.requires_grad = False
        except Exception as e:
            print(f"[FATAL] Video Encoder 로드 실패: {video_weight_path}, 오류: {e}")
            raise

        self.audio_encoder = AudioCNN_TConv().to(device)
        try:
            self.audio_encoder.load_state_dict(torch.load(audio_weight_path, map_location=device))
            # 🚨 모델 로드 성공 로그 추가
            print(f"[SUCCESS] 🎧 Audio Encoder 로드 성공: ({audio_weight_path})")
            self.audio_encoder.eval()
            for p in self.audio_encoder.parameters():
                p.requires_grad = False
        except Exception as e:
            print(f"[FATAL] Audio Encoder 로드 실패: {audio_weight_path}, 오류: {e}")
            raise

        # Trainable Fusion Head
        self.fusion = CorrelationFusion(feat_dim=128, hidden_dim=128)

    def forward(self, video, audio):
        with torch.no_grad():
            v_feat = self.video_encoder.forward_features(video)
            _, a_feat = self.audio_encoder(audio)

        out = self.fusion(v_feat, a_feat)
        return out, None, None


# ==========================================
# 5. Dataset (특징 로드 성공 로그 추가)
# ==========================================
class FusionDataset(Dataset):
    def __init__(self, video_root, audio_root, label_csv):
        self.video_root = video_root
        self.audio_root = audio_root

        # CSV 로드
        if not os.path.exists(label_csv):
            raise FileNotFoundError(f"CSV file not found: {label_csv}")

        with open(label_csv, newline="") as f:
            reader = csv.DictReader(f)
            self.samples = []
            all_samples = list(reader)
            
            # 💡 tqdm 적용: 특징 목록 검증
            for row in tqdm(all_samples, desc=f"특징 목록 검증 ({os.path.basename(label_csv)})"):
                fname = row["filename"]
                v_path = os.path.join(self.video_root, fname)
                a_path = os.path.join(self.audio_root, fname)
                
                if os.path.exists(v_path) and os.path.exists(a_path):
                    self.samples.append((fname, int(row["label"])))

        # 🚨 특징 목록 로드 성공 로그 추가
        print(f"[SUCCESS] 특징 목록 로드 성공: {label_csv}에서 유효 샘플 {len(self.samples)}개 확인.")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]

        try:
            # Video load
            v_path = os.path.join(self.video_root, fname)
            video = np.load(v_path)
            video = np.transpose(video, (3,0,1,2)) / 255.0 # (T,H,W,C) -> (C,T,H,W)
            video = torch.tensor(video, dtype=torch.float32)

            # Audio load
            a_path = os.path.join(self.audio_root, fname)
            audio = np.load(a_path)
            audio = torch.tensor(audio, dtype=torch.float32)
            audio = audio.unsqueeze(0) # (1,F,T)
            
            return video, audio, torch.tensor(label, dtype=torch.float32)
            
        except Exception as e:
            warnings.warn(f"Error loading {fname}: {e}", UserWarning)
            # Audio 특징 개수 182, Video 프레임 수 16, 크기 112x112 가정
            return torch.zeros((3, 16, 112, 112)), torch.zeros((1, 182, 100)), torch.tensor(0.0)


# ==========================================
# 6. Train / Eval Logic (tqdm 적용)
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    model.video_encoder.eval()
    model.audio_encoder.eval()
    
    total_loss = 0
    # tqdm 적용: 훈련 에포크
    with tqdm(total=len(loader), desc=f"Train {epoch}", leave=False) as pbar:
        for video, audio, label in loader:
            video, audio = video.to(device), audio.to(device)
            label = label.to(device).unsqueeze(1)

            optimizer.zero_grad()
            preds, _, _ = model(video, audio)
            loss = criterion(preds, label)
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * video.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            pbar.update(1)

    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device, desc="Valid"):
    model.eval()
    total_loss = 0
    y_true, y_pred, y_score = [], [], []

    with torch.no_grad():
        # 🚨 tqdm 적용: 검증/테스트 평가
        for video, audio, label in tqdm(loader, desc=desc, leave=False):
            video, audio = video.to(device), audio.to(device)
            label = label.to(device).unsqueeze(1)

            preds, _, _ = model(video, audio)
            loss = criterion(preds, label)
            total_loss += loss.item() * video.size(0)

            prob = torch.sigmoid(preds).cpu().numpy().flatten()
            y_true.extend(label.cpu().numpy().flatten())
            y_pred.extend((prob > 0.5).astype(int))
            y_score.extend(prob)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_score)
    except:
        auc = 0.5
    return total_loss / len(loader.dataset), acc, f1, auc


# ==========================================
# 7. Main Execution (최종 경로 및 로그 적용)
# ==========================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    
    # --- [경로 설정] ---
    video_root_base = "liar3_dataset" 
    audio_root_base = "liar3_audio_dataset_final"
    csv_train = "train_labels.csv"
    csv_val   = "val_labels.csv"
    csv_test  = "test_labels.csv"
    
    BATCH_SIZE = 4
    LR = 1e-4      
    EPOCHS = 30
    
    print(f"[INFO] Video Root Base: {video_root_base}")
    print(f"[INFO] Audio Root Base: {audio_root_base}")

    # 데이터셋 생성
    train_set = FusionDataset(
        video_root=os.path.join(video_root_base, "train"), 
        audio_root=os.path.join(audio_root_base, "train"), 
        label_csv=csv_train
    )
    val_set = FusionDataset(
        video_root=os.path.join(video_root_base, "val"), 
        audio_root=os.path.join(audio_root_base, "val"), 
        label_csv=csv_val
    )
    test_set = FusionDataset(
        video_root=os.path.join(video_root_base, "test"), 
        audio_root=os.path.join(audio_root_base, "test"), 
        label_csv=csv_test
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # 모델 로드 (로그는 FusionModel.__init__에서 처리됨)
    model = FusionModel(
        "video_model_11_21.pt",  
        "audio_best_model_final.pt", 
        device
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.fusion.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, verbose=True)

    best_auc = 0
    best_path = "fusion_interaction_best.pt"

    # 학습 시작
    print("\n[START] Training with Correlation Fusion...")
    for epoch in range(1, EPOCHS + 1):
        print(f"\n[Epoch {epoch}/{EPOCHS}]")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc, val_f1, val_auc = evaluate(model, val_loader, criterion, device, desc="Validation")

        scheduler.step(val_auc)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f}")
        print(f"ACC={val_acc:.3f} | F1={val_f1:.3f} | AUC={val_auc:.3f} | LR={current_lr:.1e}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.fusion.state_dict(), best_path) # Fusion Head만 저장
            print(f"★ Best Model (Fusion Head) Saved (AUC: {best_auc:.3f})")

    # 최종 테스트
    print("\n[STEP] Final Test Evaluation...")
    if os.path.exists(best_path):
        model.fusion.load_state_dict(torch.load(best_path))
    
    test_loss, test_acc, test_f1, test_auc = evaluate(model, test_loader, criterion, device, desc="Test")
    print(f"\n[FINAL RESULT] ACC={test_acc:.3f} | F1={test_f1:.3f} | AUC={test_auc:.3f}")

if __name__ == "__main__":
    main()