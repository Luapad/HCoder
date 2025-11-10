import os, torch, csv, time, random, warnings
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

# ==========================================
#  기존 모델 import
# ==========================================
from liar3_video import Tiny3DConvNet
from liar3_voice import AudioCNN_TConv

warnings.filterwarnings("ignore", category=UserWarning)


# ==========================================
# 1. Fusion Head 정의 (Adaptive Gating)
# ==========================================
class AdaptiveFusion(nn.Module):
    def __init__(self, feat_dim=128, hidden_dim=64):
        super().__init__()
        self.gate_v = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.gate_a = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.fc = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, v_feat, a_feat):
        v_feat = nn.functional.normalize(v_feat, dim=1)
        a_feat = nn.functional.normalize(a_feat, dim=1)

        w_v = self.gate_v(v_feat)
        w_a = self.gate_a(a_feat)

        w_sum = w_v + w_a + 1e-6
        w_v = w_v / w_sum
        w_a = w_a / w_sum

        fused = w_v * v_feat + w_a * a_feat
        out = self.fc(fused)
        return out, w_v, w_a


# ==========================================
# 2. Fusion 모델 전체 구조
# ==========================================
class FusionModel(nn.Module):
    def __init__(self, video_weight_path, audio_weight_path, device):
        super().__init__()
        # Video encoder (freeze)
        self.video_encoder = Tiny3DConvNet()
        self.video_encoder.load_state_dict(torch.load(video_weight_path, map_location=device))
        self.video_encoder.eval()
        for p in self.video_encoder.parameters():
            p.requires_grad = False

        # Audio encoder (freeze)
        self.audio_encoder = AudioCNN_TConv(time_steps=157)
        self.audio_encoder.load_state_dict(torch.load(audio_weight_path, map_location=device))
        self.audio_encoder.eval()
        for p in self.audio_encoder.parameters():
            p.requires_grad = False

        # Fusion Head (trainable)
        self.fusion = AdaptiveFusion(feat_dim=128, hidden_dim=64)

    def forward(self, video, audio):
        with torch.no_grad():
            # 영상 feature 추출
            v_out = self.video_encoder.features(video)
            v_feat = v_out.view(v_out.size(0), -1)
            # 음성 feature 추출
            _, a_feat = self.audio_encoder(audio)
        # Fusion Head 통과
        out, w_v, w_a = self.fusion(v_feat, a_feat)
        return out, w_v, w_a


# ==========================================
# 3. Dataset 클래스
# ==========================================
class FusionDataset(Dataset):
    def __init__(self, video_root, audio_root, label_csv):
        self.video_root = video_root
        self.audio_root = audio_root
        with open(label_csv, newline='') as f:
            reader = csv.DictReader(f)
            self.samples = [(row['filename'], int(row['label'])) for row in reader]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]

        v_path = os.path.join(self.video_root, fname)
        a_path = os.path.join(self.audio_root, fname)

        # 비디오 feature
        v = np.load(v_path)  # (T, H, W, C)
        v = np.transpose(v, (3, 0, 1, 2)) / 255.0
        v = torch.tensor(v, dtype=torch.float32)

        # 오디오 feature
        a = np.load(a_path)
        a = (a - np.mean(a)) / (np.std(a) + 1e-6)
        a = torch.tensor(a, dtype=torch.float32).unsqueeze(0)

        return v, a, torch.tensor(label, dtype=torch.float32)


# ==========================================
# 4. 학습 루프
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    with tqdm(total=len(loader), desc=f"Train {epoch}") as pbar:
        for video, audio, label in loader:
            video, audio, label = video.to(device), audio.to(device), label.to(device).unsqueeze(1)
            optimizer.zero_grad()
            preds, _, _ = model(video, audio)
            loss = criterion(preds, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * video.size(0)
            pbar.update(1)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device, desc="Valid"):
    model.eval()
    y_true, y_pred, y_score = [], [], []
    total_loss = 0
    with torch.no_grad():
        for video, audio, label in tqdm(loader, desc=desc):
            video, audio, label = video.to(device), audio.to(device), label.to(device).unsqueeze(1)
            preds, _, _ = model(video, audio)
            loss = criterion(preds, label)
            total_loss += loss.item() * video.size(0)

            probs = torch.sigmoid(preds).cpu().numpy().flatten()
            y_true.extend(label.cpu().numpy().flatten())
            y_pred.extend((probs > 0.5).astype(int))
            y_score.extend(probs)

    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = 0.5
    return total_loss / len(loader.dataset), acc, f1, auc


# ==========================================
# 5. Main
# ==========================================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")

    # 데이터 경로
    video_root = "liar3_dataset/test"        # 영상 numpy
    audio_root = "liar3_audio_dataset/test"  # 음성 numpy
    label_csv = "liar3_dataset/test_labels.csv"

    # Encoder weight
    video_weight = "liar3_best_model_32f.pt"
    audio_weight = "audio_best_model.pt"

    # Dataset
    train_set = FusionDataset("liar3_dataset/train", "liar3_audio_dataset/train", "liar3_dataset/train_labels.csv")
    val_set = FusionDataset("liar3_dataset/val", "liar3_audio_dataset/val", "liar3_dataset/val_labels.csv")
    test_set = FusionDataset(video_root, audio_root, label_csv)

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=8, shuffle=False)

    # 모델 초기화
    model = FusionModel(video_weight, audio_weight, device).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.fusion.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # 학습
    best_auc = 0
    model_path = "fusion_best_model.pt"
    for epoch in range(1, 31):
        print(f"\n[Epoch {epoch}/30]")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc, val_f1, val_auc = evaluate(model, val_loader, criterion, device, desc="Validation")
        scheduler.step(val_auc)
        print(f"Train={train_loss:.4f} | Val={val_loss:.4f} | Acc={val_acc:.3f} | "
              f"F1={val_f1:.3f} | AUC={val_auc:.3f}")
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), model_path)
            print(f"[INFO] ✅ Best model saved (AUC={best_auc:.3f})")

    # 테스트 평가
    print("\n[STEP 5] 테스트 세트 평가 중...")
    model.load_state_dict(torch.load(model_path))
    test_loss, test_acc, test_f1, test_auc = evaluate(model, test_loader, criterion, device, desc="Test")
    print(f"\n[Test Result] Acc={test_acc:.3f}, F1={test_f1:.3f}, AUC={test_auc:.3f}")


if __name__ == "__main__":
    main()
