import os, csv, time, random, warnings, tempfile
import numpy as np
import librosa
from moviepy.editor import VideoFileClip
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)


# =========================================================
# 0. mp4 → wav 변환 (Windows-safe)
# =========================================================
def extract_audio_from_mp4(mp4_path, wav_path):
    try:
        clip = VideoFileClip(mp4_path)
        clip.audio.write_audiofile(wav_path, verbose=False, logger=None)
        clip.close()
        return True
    except Exception as e:
        print(f"[WARN] {mp4_path} 오디오 추출 실패: {e}")
        return False


# =========================================================
# 1. Global mean/std 계산
# =========================================================
def compute_global_mfcc_stats(video_dir, csv_path, sr=16000, n_mfcc=60, duration=5.0):
    print("\n[STEP 0] Global MFCC 통계 계산 중...")
    vals = []

    # DictReader → 리스트로 감싸서 tqdm이 전체 개수 인식하도록 수정
    with open(csv_path, newline="") as f:
        reader = list(csv.DictReader(f))
        total = len(reader)

        for row in tqdm(reader, total=total, desc="Global MFCC 추출 진행률"):
            mp4_name = row["filename"].replace(".npy", ".mp4")
            mp4_path = os.path.join(video_dir, mp4_name)
            if not os.path.exists(mp4_path):
                continue

            temp_wav = os.path.join(tempfile.gettempdir(), f"temp_{random.randint(0,999999)}.wav")
            if not extract_audio_from_mp4(mp4_path, temp_wav):
                continue

            try:
                y, _ = librosa.load(temp_wav, sr=sr)
                max_len = int(duration * sr)
                if len(y) > max_len:
                    start = random.randint(0, len(y) - max_len)
                    y = y[start:start + max_len]
                else:
                    y = np.pad(y, (0, max_len - len(y)))
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
                vals.append(mfcc)
            except Exception:
                continue
            finally:
                if os.path.exists(temp_wav):
                    os.remove(temp_wav)

    if not vals:
        raise RuntimeError("MFCC 통계 계산 실패 — 데이터 확인 필요")

    all_mfcc = np.concatenate(vals, axis=1)
    global_mean = np.mean(all_mfcc)
    global_std = np.std(all_mfcc)
    print(f"[INFO] Global mean={global_mean:.4f}, std={global_std:.4f}")
    return global_mean, global_std



# =========================================================
# 2. mp4 → feature npy 변환 (Δ, ΔΔ, RMS, Pitch 포함)
# =========================================================
def create_audio_dataset(video_dir="data", csv_root="liar3_dataset",
                         output_root="liar3_audio_dataset",
                         sr=16000, n_mfcc=60, duration=5.0,
                         global_mean=None, global_std=None):
    os.makedirs(output_root, exist_ok=True)
    splits = ["train", "val", "test"]

    for split in splits:
        csv_in = os.path.join(csv_root, f"{split}_labels.csv")
        if not os.path.exists(csv_in):
            print(f"[WARN] {csv_in} 없음. 스킵.")
            continue

        split_dir = os.path.join(output_root, split)
        os.makedirs(split_dir, exist_ok=True)
        csv_out = os.path.join(output_root, f"{split}_labels.csv")

        with open(csv_in, newline="") as fin:
            total = sum(1 for _ in fin) - 1

        print(f"\n[INFO] {split} set 변환 시작 — 총 {total}개 파일")
        with open(csv_in, newline="") as fin, open(csv_out, "w", newline="") as fout:
            reader = csv.DictReader(fin)
            writer = csv.DictWriter(fout, fieldnames=["filename", "label"])
            writer.writeheader()

            for idx, row in enumerate(reader, start=1):
                percent = (idx / total) * 100
                print(f"\r[{split}] 변환 진행률: {percent:5.1f}% ({idx}/{total})", end="")

                base = row["filename"].replace(".npy", ".mp4")
                label = int(row["label"])
                video_path = os.path.join(video_dir, base)
                if not os.path.exists(video_path):
                    continue

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav_file:
                    temp_wav_path = tmp_wav_file.name

                if not extract_audio_from_mp4(video_path, temp_wav_path):
                    if os.path.exists(temp_wav_path):
                        os.remove(temp_wav_path)
                    continue

                try:
                    y, _ = librosa.load(temp_wav_path, sr=sr)
                    max_len = int(duration * sr)
                    if len(y) > max_len:
                        start = random.randint(0, len(y) - max_len)
                        y = y[start:start + max_len]
                    else:
                        y = np.pad(y, (0, max_len - len(y)))

                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
                    delta = librosa.feature.delta(mfcc)
                    delta2 = librosa.feature.delta(mfcc, order=2)

                    rms = librosa.feature.rms(y=y)
                    rms = np.tile(rms, (1, mfcc.shape[1] // rms.shape[1] + 1))[:, :mfcc.shape[1]]

                    f0, _, _ = librosa.pyin(y, fmin=50, fmax=300, sr=sr)
                    f0 = np.nan_to_num(f0).reshape(1, -1)
                    f0 = np.tile(f0, (1, mfcc.shape[1] // f0.shape[1] + 1))[:, :mfcc.shape[1]]

                    feat = np.concatenate([mfcc, delta, delta2, rms, f0], axis=0)

                    if global_mean is not None and global_std is not None:
                        feat = (feat - global_mean) / (global_std + 1e-6)

                    np.save(os.path.join(split_dir, base.replace(".mp4", ".npy")), feat)
                    writer.writerow({"filename": base.replace(".mp4", ".npy"), "label": label})

                except Exception as e:
                    print(f"[WARN] {video_path} 변환 실패: {e}")
                finally:
                    if os.path.exists(temp_wav_path):
                        try:
                            os.remove(temp_wav_path)
                        except PermissionError:
                            time.sleep(0.2)
                            try:
                                os.remove(temp_wav_path)
                            except:
                                pass
        print(f"\n[INFO] {split} set 변환 완료.")
    print("\n[INFO] liar3_audio_dataset 생성 완료.")


# =========================================================
# 3. Dataset (Per-sample 정규화)
# =========================================================
class AudioLieDataset(Dataset):
    def __init__(self, data_dir, label_csv):
        self.data_dir = data_dir
        with open(label_csv, newline="") as f:
            reader = csv.DictReader(f)
            self.samples = [(row["filename"], int(row["label"])) for row in reader]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        arr = np.load(os.path.join(self.data_dir, fname))
        arr = (arr - np.mean(arr)) / (np.std(arr) + 1e-6)
        arr = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
        return arr, torch.tensor(label, dtype=torch.float32)


# =========================================================
# 4. 모델 (Pool stride 수정)
# =========================================================
class AudioCNN_TConv(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=128, time_steps=157):
        super().__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2, 1)), nn.Dropout(0.3),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((2, 1)), nn.Dropout(0.3),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU()
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(128 * 45, 128, kernel_size=3, padding=1),
            nn.LayerNorm([128, time_steps]),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc_feat = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.conv2d(x)
        B, C, H, T = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, -1, T)
        x = self.temporal(x).squeeze(-1)
        feat = self.fc_feat(x)
        out = self.fc_out(feat)
        return out, feat


# =========================================================
# 5. 학습/평가 루프 (진행률 % 표시 추가)
# =========================================================
def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    total_batches = len(loader)
    with tqdm(total=total_batches, desc=f"Train {epoch}") as pbar:
        for batch_idx, (X, y) in enumerate(loader, start=1):
            X, y = X.to(device), y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            preds, _ = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.size(0)

            percent = (batch_idx / total_batches) * 100
            pbar.set_postfix_str(f"{percent:5.1f}% 완료")
            pbar.update(1)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device, desc="Valid"):
    model.eval()
    y_true, y_pred, y_score = [], [], []
    total_loss = 0
    total_batches = len(loader)
    with tqdm(total=total_batches, desc=desc) as pbar:
        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(loader, start=1):
                X, y = X.to(device), y.to(device).unsqueeze(1)
                preds, _ = model(X)
                loss = criterion(preds, y)
                total_loss += loss.item() * X.size(0)
                probs = torch.sigmoid(preds).cpu().numpy().flatten()
                y_true.extend(y.cpu().numpy().flatten())
                y_pred.extend((probs > 0.5).astype(int))
                y_score.extend(probs)

                percent = (batch_idx / total_batches) * 100
                pbar.set_postfix_str(f"{percent:5.1f}% 완료")
                pbar.update(1)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = 0.5
    return total_loss / len(loader.dataset), acc, f1, auc


# =========================================================
# 6. main()
# =========================================================
def main():
    video_dir = "data"
    csv_root = "liar3_dataset"
    dataset_root = "liar3_audio_dataset"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================
    # STEP 1. 오디오 데이터셋 준비 (존재하면 생략)
    # =========================================================
    if (
        os.path.exists(dataset_root)
        and all(os.path.exists(os.path.join(dataset_root, f"{s}_labels.csv")) for s in ["train", "val", "test"])
    ):
        print(f"[STEP 1] 기존 liar3_audio_dataset 발견 → 변환 건너뜀")
    else:
        print(f"[STEP 1] liar3_audio_dataset 생성 중...")
        global_mean, global_std = compute_global_mfcc_stats(video_dir, os.path.join(csv_root, "train_labels.csv"))
        create_audio_dataset(video_dir, csv_root, dataset_root,
                             global_mean=global_mean, global_std=global_std)

    # =========================================================
    # STEP 2. Dataset 로드
    # =========================================================
    train_set = AudioLieDataset(f"{dataset_root}/train", f"{dataset_root}/train_labels.csv")
    val_set = AudioLieDataset(f"{dataset_root}/val", f"{dataset_root}/val_labels.csv")
    test_set = AudioLieDataset(f"{dataset_root}/test", f"{dataset_root}/test_labels.csv")

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

    # =========================================================
    # STEP 3. 모델 설정
    # =========================================================
    model = AudioCNN_TConv(time_steps=157).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_auc = 0
    model_path = "audio_best_model.pt"

    # =========================================================
    # STEP 4. 학습 루프
    # =========================================================
    for epoch in range(1, 81):
        print(f"\n[Epoch {epoch}/80]")
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc, val_f1, val_auc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_auc)
        print(f"Train={train_loss:.4f} | Val={val_loss:.4f} | Acc={val_acc:.3f} | "
              f"F1={val_f1:.3f} | AUC={val_auc:.3f} | time={time.time()-t0:.1f}s")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), model_path)
            print(f"  -> [INFO] Best AUC={best_auc:.4f} saved.")

    # =========================================================
    # STEP 5. 테스트 세트 평가
    # =========================================================
    print("\n[STEP 5] 테스트 세트 평가 중...")
    model.load_state_dict(torch.load(model_path))
    test_loss, test_acc, test_f1, test_auc = evaluate(model, test_loader, criterion, device, desc="Test")
    print(f"\n[Test Result] Acc={test_acc:.3f}, F1={test_f1:.3f}, AUC={test_auc:.3f}")



if __name__ == "__main__":
    main()