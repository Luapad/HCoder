import os
import csv
import warnings
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ==========================================
# 1. TinyRes3D (Feature 안정화 포함)
# ==========================================
import torch.nn.functional as F


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

    # Feature extractor used in fusion
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

        # 안정화
        x = torch.nan_to_num(x).clamp(-3000, 3000)

        return x

    def forward(self, x):
        feat = self.forward_features(x)
        feat = self.dropout(feat)
        return torch.sigmoid(self.fc(feat))


# ==========================================
# 2. AudioCNN_TConv (Feature 안정화 포함)
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
        # Conv2D: (B,1,F,T)
        x = self.conv2d(x)         # (B,128,F',T)
        x = torch.mean(x, dim=2)   # (B,128,T)

        x = self.temporal_conv(x).squeeze(-1)  # (B,128)

        feat = self.fc_feat(x)

        # 안정화
        feat = torch.nan_to_num(feat).clamp(-3000, 3000)

        out = self.fc_out(feat)
        return out, feat


# ==========================================
# 3. Adaptive Fusion Head (정규화 안정화 포함)
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

        # 안정화
        v_feat = torch.nan_to_num(v_feat).clamp(-3000, 3000)
        a_feat = torch.nan_to_num(a_feat).clamp(-3000, 3000)

        # Normalize
        v_feat = F.normalize(v_feat + 1e-8, dim=1)
        a_feat = F.normalize(a_feat + 1e-8, dim=1)

        w_v = self.gate_v(v_feat)
        w_a = self.gate_a(a_feat)

        w_sum = w_v + w_a + 1e-6
        w_v = w_v / w_sum
        w_a = w_a / w_sum

        fused = w_v * v_feat + w_a * a_feat

        out = self.fc(fused)
        return out, w_v, w_a


# ==========================================
# 4. Fusion Model
# ==========================================
class FusionModel(nn.Module):
    def __init__(self, video_weight_path, audio_weight_path, device):
        super().__init__()

        # video encoder
        self.video_encoder = TinyRes3D().to(device)
        self.video_encoder.load_state_dict(torch.load(video_weight_path, map_location=device))
        self.video_encoder.eval()
        for p in self.video_encoder.parameters():
            p.requires_grad = False

        # audio encoder
        self.audio_encoder = AudioCNN_TConv().to(device)
        self.audio_encoder.load_state_dict(torch.load(audio_weight_path, map_location=device))
        self.audio_encoder.eval()
        for p in self.audio_encoder.parameters():
            p.requires_grad = False

        # fusion head
        self.fusion = AdaptiveFusion(feat_dim=128, hidden_dim=64)

    def forward(self, video, audio):
        with torch.no_grad():
            v_feat = self.video_encoder.forward_features(video)

        with torch.no_grad():
            _, a_feat = self.audio_encoder(audio)

        out, w_v, w_a = self.fusion(v_feat, a_feat)
        return out, w_v, w_a


# ==========================================
# 5. Dataset
# ==========================================
class FusionDataset(Dataset):
    def __init__(self, video_root, audio_root, label_csv):
        self.video_root = video_root
        self.audio_root = audio_root

        with open(label_csv, newline="") as f:
            reader = csv.DictReader(f)
            self.samples = [(row["filename"], int(row["label"])) for row in reader]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]

        # video npy: (T,H,W,C) → (C,T,H,W)
        video = np.load(os.path.join(self.video_root, fname))
        video = np.transpose(video, (3,0,1,2)) / 255.0
        video = torch.tensor(video, dtype=torch.float32)

        # audio npy: (F,T) → (1,F,T)
        audio = np.load(os.path.join(self.audio_root, fname))
        audio = torch.tensor(audio, dtype=torch.float32)
        audio = audio.unsqueeze(0)

        return video, audio, torch.tensor(label, dtype=torch.float32)


# ==========================================
# 6. Train / Eval
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0

    with tqdm(total=len(loader), desc=f"Train {epoch}") as pbar:
        for video, audio, label in loader:
            video, audio = video.to(device), audio.to(device)
            label = label.to(device).unsqueeze(1)

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
    total_loss = 0
    y_true, y_pred, y_score = [], [], []

    with torch.no_grad():
        for video, audio, label in tqdm(loader, desc=desc):
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
# 7. Main
# ==========================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] Using device:", device)

    train_set = FusionDataset(
        "liar3_dataset/train",
        "liar3_audio_dataset_try1/train",
        "liar3_dataset/train_labels.csv"
    )

    val_set = FusionDataset(
        "liar3_dataset/val",
        "liar3_audio_dataset_try1/val",
        "liar3_dataset/val_labels.csv"
    )

    test_set = FusionDataset(
        "liar3_dataset/test",
        "liar3_audio_dataset_try1/test",
        "liar3_dataset/test_labels.csv"
    )

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False, pin_memory=False)
    test_loader = DataLoader(test_set, batch_size=4, shuffle=False, pin_memory=False)

    model = FusionModel(
        "liar3_best_model_try3.pt",
        "audio_best_model_try1.pt",
        device
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.fusion.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_auc = 0
    best_path = "fusion_best_model_fixed.pt"

    for epoch in range(1, 31):
        print(f"\n[Epoch {epoch}/30]")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc, val_f1, val_auc = evaluate(model, val_loader, criterion, device, desc="Validation")

        scheduler.step(val_auc)

        print(f"Train={train_loss:.4f} | Val={val_loss:.4f} | "
              f"ACC={val_acc:.3f} | F1={val_f1:.3f} | AUC={val_auc:.3f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_path)
            print(f"[INFO] Best Saved (AUC={best_auc:.3f})")

    print("\n[STEP] Test Evaluation...")
    model.load_state_dict(torch.load(best_path))

    test_loss, test_acc, test_f1, test_auc = evaluate(
        model, test_loader, criterion, device, desc="Test"
    )

    print(f"[TEST] ACC={test_acc:.3f} | F1={test_f1:.3f} | AUC={test_auc:.3f}")


if __name__ == "__main__":
    main()
