import os, csv, time, random, warnings, tempfile
import numpy as np
import librosa
from moviepy.editor import VideoFileClip
from tqdm import tqdm # tqdm 적용
from scipy.interpolate import interp1d
import gc 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)


# ----------------------------------------------------------
# 0. mp4 → wav & 길이 맞추기 (파일 액세스 오류 방지 로직 포함)
# ----------------------------------------------------------
def extract_audio_from_mp4(mp4_path, wav_path):
    """MP4 파일에서 WAV 파일로 오디오 추출 (파일 핸들 해제 포함)"""
    try:
        clip = VideoFileClip(mp4_path)
        clip.audio.write_audiofile(wav_path, verbose=False, logger=None, fps=16000) 
        clip.close()
        
        # 파일 핸들 강제 해제 (WinError 32 방지)
        del clip
        gc.collect() 
        
        return True
    except Exception as e:
        print(f"[WARN] {mp4_path} 오디오 추출 실패: {e}")
        return False

def match_length(arr, target_len):
    """특징의 시간 축 길이를 target_len에 맞춤 (선형 보간)"""
    old_len = arr.shape[1]
    if old_len == target_len:
        return arr

    x_old = np.linspace(0, 1, old_len)
    x_new = np.linspace(0, 1, target_len)
    
    f = interp1d(x_old, arr.flatten(), kind="linear")
    new = f(x_new)
    
    return new.reshape(1, -1)


# ----------------------------------------------------------
# 1. 특징별 (Feature-wise) Global Stats 계산 (MP4 추출)
# ----------------------------------------------------------
def compute_global_stats(video_dir, csv_path, sr=16000, n_mels=60, duration=5.0):
    """훈련 데이터셋 전체의 특징별 평균 및 표준편차 계산 (MP4 파일에서 추출)"""
    print("\n[STEP 0] Global Feature 통계 계산 중... (MP4 파일에서 추출)")
    vals = []

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
        total = len(rows)

    os.makedirs('temp_wavs', exist_ok=True) 

    # 💡 tqdm 적용: Global 통계 추출
    for row in tqdm(rows, total=total, desc="Global 통계 추출"):
        mp4_name = row["filename"].replace(".npy", ".mp4")
        mp4_path = os.path.join(video_dir, mp4_name)
        if not os.path.exists(mp4_path):
            continue

        temp_wav = os.path.join('temp_wavs', f"temp_{random.randint(0, 999999)}.wav")

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

            # ----- NEW FEATURE PIPELINE (182 Features) -----
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
            mel_spec = librosa.power_to_db(S, ref=np.max) 
            
            delta = librosa.feature.delta(mel_spec)
            delta2 = librosa.feature.delta(mel_spec, order=2)
            T = mel_spec.shape[1] 

            # Spectral Flux 대체: onset_strength 사용 (librosa 버전 문제 회피)
            flux_feat = librosa.onset.onset_strength(y=y, sr=sr, S=mel_spec)
            flux_feat = flux_feat.reshape(1, -1) 
            flux_feat = match_length(flux_feat, T)

            # Spectral Centroid Delta
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            delta_centroid = librosa.feature.delta(centroid, order=1)
            delta_centroid = np.nan_to_num(delta_centroid)
            delta_centroid = match_length(delta_centroid, T)
            
            feat = np.concatenate([mel_spec, delta, delta2, flux_feat, delta_centroid], axis=0)
            
            vals.append(feat.T)

        except Exception as e:
            print(f"\n[ERROR] 특징 계산 실패 ({mp4_name}): {e}") 
            continue

        finally:
            time.sleep(0.5) # 파일 핸들 해제를 위한 짧은 지연
            if os.path.exists(temp_wav):
                os.remove(temp_wav)

    if not vals:
        raise RuntimeError("global feature 계산 실패 (MP4 추출)")

    all_feat_time_first = np.concatenate(vals, axis=0) 
    g_mean = np.mean(all_feat_time_first, axis=0) 
    g_std = np.std(all_feat_time_first, axis=0)   

    print(f"[INFO] Global stats computed for {g_mean.shape[0]} features.") 
    return g_mean, g_std 


# ----------------------------------------------------------
# 1-A. 특징 파일에서 Global Stats 계산 (NPY 파일 로드)
# ----------------------------------------------------------
def recompute_stats_from_npy(dataset_root):
    """기존 .npy 파일이 있을 때, npy 파일을 로드하여 통계 계산"""
    train_dir = os.path.join(dataset_root, "train")
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"훈련 특징 폴더 ({train_dir})를 찾을 수 없습니다.")
        
    print("\n[STEP 0] Global Feature 통계 계산 중... (기존 .npy 파일 로드)")
    
    vals = []
    npy_files = [f for f in os.listdir(train_dir) if f.endswith(".npy")]
    total = len(npy_files)

    # 💡 tqdm 적용: NPY 통계 계산
    for fname in tqdm(npy_files, total=total, desc="NPY 통계 계산"):
        fpath = os.path.join(train_dir, fname)
        try:
            feat = np.load(fpath)
            vals.append(feat.T) # (Time, Features)로 변환
        except Exception as e:
            print(f"\n[WARN] {fpath} 로드 실패: {e}")
            continue

    if not vals:
        raise RuntimeError("global feature 계산 실패 (NPY 로드)")

    all_feat_time_first = np.concatenate(vals, axis=0) 
    g_mean = np.mean(all_feat_time_first, axis=0) 
    g_std = np.std(all_feat_time_first, axis=0)   

    print(f"[INFO] Global stats computed from {total} NPY files for {g_mean.shape[0]} features.") 
    return g_mean, g_std 


# ----------------------------------------------------------
# 0-A. 통계치 저장/로드 도우미 함수 (Robust Load 포함)
# ----------------------------------------------------------
def save_global_stats(g_mean, g_std, output_root):
    """계산된 통계치를 .npy 파일로 저장 (g_mean/g_std 분리 저장)"""
    os.makedirs(output_root, exist_ok=True)
    np.save(os.path.join(output_root, "g_mean.npy"), g_mean)
    np.save(os.path.join(output_root, "g_std.npy"), g_std)
    print(f"[INFO] Global stats saved to {output_root}/.")

def load_global_stats_robust(output_root):
    """
    저장된 통계치를 로드합니다. (이전 버전의 통합 파일도 처리)
    """
    g_mean_path = os.path.join(output_root, "g_mean.npy")
    g_std_path = os.path.join(output_root, "g_std.npy")
    single_path = os.path.join(output_root, "global_stats.npy")

    # 1. 1차: 분리된 파일 확인 (최신 포맷)
    if os.path.exists(g_mean_path) and os.path.exists(g_std_path):
        g_mean = np.load(g_mean_path)
        g_std = np.load(g_std_path)
        print(f"[INFO] 통계 파일 로드 완료: g_mean.npy, g_std.npy")
        return g_mean, g_std
    
    # 2. 2차: 통합 파일 확인 (이전 버전 호환)
    if os.path.exists(single_path):
        try:
            combined = np.load(single_path)
            N_FEATS = combined.shape[0] // 2
            g_mean = combined[:N_FEATS]
            g_std = combined[N_FEATS:]
            
            # 통합 파일 로드 후, 다음 실행을 위해 분리 파일로 저장 (마이그레이션)
            save_global_stats(g_mean, g_std, output_root) 
            os.remove(single_path) # 통합 파일은 삭제
            print(f"[INFO] 통합 통계 파일 (global_stats.npy) 로드 및 분리/저장 완료.")
            return g_mean, g_std
        except Exception as e:
            print(f"[WARN] 통합 통계 파일 로드 실패: {e}")

    return None, None


# ----------------------------------------------------------
# 2. Dataset 생성 (tqdm, npy 건너뛰기 로직 포함)
# ----------------------------------------------------------
def create_audio_dataset(video_dir="data", csv_root="",
                         output_root="liar3_audio_dataset",
                         sr=16000, n_mels=60, duration=5.0,
                         global_mean=None, global_std=None):
    """MP4에서 오디오 특징을 추출하고 정규화하여 .npy 파일로 저장"""
    os.makedirs(output_root, exist_ok=True)
    os.makedirs('temp_wavs', exist_ok=True)

    for split in ["train", "val", "test"]:
        csv_in = os.path.join(csv_root, f"{split}_labels.csv")
        if not os.path.exists(csv_in):
            print(f"\n[WARN] {csv_in} 파일을 찾을 수 없어 {split} 데이터셋 생성을 건너뜜.")
            continue

        split_dir = os.path.join(output_root, split)
        os.makedirs(split_dir, exist_ok=True)
        csv_out = os.path.join(output_root, f"{split}_labels.csv")

        rows = list(csv.DictReader(open(csv_in)))
        total = len(rows)

        print(f"\n[INFO] {split} 변환 시작 — {total}개")

        with open(csv_out, "w", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=["filename", "label"])
            writer.writeheader()

            # 💡 tqdm 적용: 각 분할 파일 변환
            for row in tqdm(rows, total=total, desc=f"변환 중 ({split})"):
                mp4 = row["filename"].replace(".npy", ".mp4")
                label = int(row["label"])
                npy_filename = mp4.replace(".mp4", ".npy")
                npy_path = os.path.join(split_dir, npy_filename)

                # .npy 파일이 이미 존재하면 건너뛰기
                if os.path.exists(npy_path):
                    writer.writerow({"filename": npy_filename, "label": label})
                    continue 
                
                mp4_path = os.path.join(video_dir, mp4)

                if not os.path.exists(mp4_path):
                    continue
                
                wav_path = os.path.join('temp_wavs', f"temp_{random.randint(0, 999999)}.wav")

                if not extract_audio_from_mp4(mp4_path, wav_path):
                    os.remove(wav_path)
                    continue

                try:
                    y, _ = librosa.load(wav_path, sr=sr)
                    max_len = int(duration * sr)
                    if len(y) > max_len:
                        start = random.randint(0, len(y) - max_len)
                        y = y[start:start + max_len]
                    else:
                        y = np.pad(y, (0, max_len - len(y)))

                    # 특징 추출 로직 (182개)
                    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
                    mel_spec = librosa.power_to_db(S, ref=np.max) 
                    delta = librosa.feature.delta(mel_spec)
                    delta2 = librosa.feature.delta(mel_spec, order=2)
                    T = mel_spec.shape[1] 

                    flux_feat = match_length(librosa.onset.onset_strength(y=y, sr=sr, S=mel_spec).reshape(1, -1), T)
                    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
                    delta_centroid = match_length(np.nan_to_num(librosa.feature.delta(centroid, order=1)), T)
                    
                    feat = np.concatenate([mel_spec, delta, delta2, flux_feat, delta_centroid], axis=0)

                    # 정규화 적용
                    if global_mean is not None:
                        mean_col = global_mean.reshape(-1, 1)
                        std_col = global_std.reshape(-1, 1)
                        feat = (feat - mean_col) / (std_col + 1e-6)

                    np.save(npy_path, feat)

                    writer.writerow({"filename": npy_filename, "label": label})
                except Exception as e:
                    print(f"\n[WARN] {mp4_path} 처리 실패: {e}")
                finally:
                    time.sleep(0.5) 
                    if os.path.exists(wav_path):
                        os.remove(wav_path)

        print(f"\n[INFO] {split} 완료.")


# ----------------------------------------------------------
# 3. Dataset (이하 동일)
# ----------------------------------------------------------
class AudioLieDataset(Dataset):
    def __init__(self, data_dir, label_csv):
        self.data_dir = data_dir
        self.samples = list(csv.DictReader(open(label_csv)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname = self.samples[idx]["filename"]
        label = int(self.samples[idx]["label"])
        arr = np.load(os.path.join(self.data_dir, fname))
        arr = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
        return arr, torch.tensor(label, dtype=torch.float32)


# ----------------------------------------------------------
# 4. Model (이하 동일)
# ----------------------------------------------------------
class AudioCNN_TConv(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=128):
        super().__init__()

        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
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
        x = self.conv2d(x)  # (B, 128, H, T)
        x = torch.mean(x, dim=2)  # (B, 128, T)

        x = self.temporal_conv(x).squeeze(-1)  # (B, 128)
        feat = self.fc_feat(x)
        out = self.fc_out(feat)
        return out, feat


# ----------------------------------------------------------
# 5. Train / Eval 함수 (이하 동일)
# ----------------------------------------------------------
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

            pbar.set_postfix_str(f"{(batch_idx / total_batches) * 100:5.1f}% 완료")
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

                pbar.set_postfix_str(f"{(batch_idx / total_batches) * 100:5.1f}% 완료")
                pbar.update(1)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_score)
    except:
        auc = 0.5

    return total_loss / len(loader.dataset), acc, f1, auc


# ----------------------------------------------------------
# 6. main() - 통계 로직 최종 수정 반영
# ----------------------------------------------------------
def main():
    video_dir = "data_cut" 
    csv_root = "" 
    dataset_root = "liar3_audio_dataset_final" 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    N_MELS = 60 

    train_label_path_in = os.path.join(csv_root, "train_labels.csv") 
    
    global_mean = None
    global_std = None
    
    # 💡 STEP 1: Global Stats 계산/로드
    
    # 1.1. 저장된 통계 파일 로드 시도 (분리 파일 또는 통합 파일)
    global_mean, global_std = load_global_stats_robust(dataset_root)

    if global_mean is not None:
        # 통계 파일 로드 성공
        print(f"[STEP 1] 통계 계산 건너뛰고 로드 완료. ({global_mean.shape[0]} features)")
    else:
        # 통계 파일이 없을 경우
        
        # 1.2. 특징 파일(.npy)이 이미 생성되어 있는지 먼저 확인
        train_npy_dir = os.path.join(dataset_root, "train")
        npy_exists = os.path.exists(train_npy_dir) and any(f.endswith(".npy") for f in os.listdir(train_npy_dir))
        
        if npy_exists:
            # 1.2.1. npy 파일이 있다면, npy 파일을 로드해서 통계 계산 (MP4 추출 건너뛰기)
            print(f"[STEP 1] Global Stats 생성 중... (기존 NPY 파일에서 통계 재계산)")
            try:
                global_mean, global_std = recompute_stats_from_npy(dataset_root)
                save_global_stats(global_mean, global_std, dataset_root)
            except Exception as e:
                print(f"[WARN] NPY 기반 통계 계산 실패 ({e}). MP4에서 재추출 시작.")
                global_mean, global_std = compute_global_stats(
                    video_dir, train_label_path_in, n_mels=N_MELS
                )
                save_global_stats(global_mean, global_std, dataset_root)

        else:
            # 1.2.2. npy 파일도 없다면, MP4에서 추출 및 통계 계산 (최종 fallback)
            print(f"[STEP 1] Global Stats 생성 중... (MP4에서 추출)")
            global_mean, global_std = compute_global_stats(
                video_dir, train_label_path_in, n_mels=N_MELS
            )
            save_global_stats(global_mean, global_std, dataset_root)
        
    # STEP 2: Dataset 생성/확인
    print(f"\n[STEP 2] 특징 파일 (.npy) 생성/확인 중...")
    create_audio_dataset(
        video_dir, csv_root, dataset_root,
        n_mels=N_MELS,
        global_mean=global_mean,
        global_std=global_std
    )
    
    # STEP 3: dataset 로드 (생성된 CSV 파일 사용)
    train_set = AudioLieDataset(
        f"{dataset_root}/train",
        f"{dataset_root}/train_labels.csv"
    )
    val_set = AudioLieDataset(
        f"{dataset_root}/val",
        f"{dataset_root}/val_labels.csv"
    )
    test_set = AudioLieDataset(
        f"{dataset_root}/test",
        f"{dataset_root}/test_labels.csv"
    )

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

    # STEP 4: 모델 설정 및 학습
    model = AudioCNN_TConv().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    best_auc = 0
    model_path = "audio_best_model_final.pt"

    for epoch in range(1, 81):
        print(f"\n[Epoch {epoch}/80]")
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc, val_f1, val_auc = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_auc)

        print(
            f"Train={train_loss:.4f} | "
            f"Val={val_loss:.4f} | Acc={val_acc:.3f} | "
            f"F1={val_f1:.3f} | AUC={val_auc:.3f} | "
            f"time={time.time() - t0:.1f}s"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), model_path)
            print(f"  -> [INFO] Best AUC={best_auc:.4f} saved.")

    # STEP 5: 테스트 평가
    print("\n[STEP 5] 테스트 세트 평가 중...")
    model.load_state_dict(torch.load(model_path))

    test_loss, test_acc, test_f1, test_auc = evaluate(
        model, test_loader, criterion, device, desc="Test"
    )

    print(
        f"\n[Test Result] "
        f"Acc={test_acc:.3f}, F1={test_f1:.3f}, AUC={test_auc:.3f}"
    )


if __name__ == "__main__":
    main()