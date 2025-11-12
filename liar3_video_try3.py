import os, csv, json, cv2, random
import numpy as np
from tqdm import tqdm
from deepface import DeepFace
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import torch.nn.functional as F


# =========================================================
# 0. DeepFace + DBSCAN 기반 인물 클러스터링
# =========================================================
def generate_person_ids_from_videos(video_dir, cache_path="liar3_face_cache.json"):
    # 이 함수는 DeepFace의 'represent' (임베딩) 기능을 사용하므로 수정 없음
    if os.path.exists(cache_path):
        print(f"[INFO] 캐시 파일 {cache_path} 발견 → 인물 식별 단계 건너뜀")
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        return cache["person_id_map"], cache["n_persons"]

    embeddings, video_files = [], []
    video_list = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]

    print(f"[INFO] 총 {len(video_list)}개 영상에서 얼굴 임베딩 추출 중...")

    for vf in tqdm(video_list, desc="DeepFace Embedding"):
        path = os.path.join(video_dir, vf)
        try:
            cap = cv2.VideoCapture(path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release()
                continue

            idxs = np.linspace(0, total - 1, 5).astype(int)
            for idx in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                # ★★★ 참고: 'represent'는 'extract_faces'와 다름 ★★★
                # 임베딩 추출은 detector_backend='opencv'가 기본이며 잘 동작함
                result = DeepFace.represent(frame, model_name="Facenet", enforce_detection=False)

                if isinstance(result, list) and len(result) > 0:
                    emb = result[0]["embedding"]
                    embeddings.append(emb)
                    video_files.append(vf)
                    break
            cap.release()
        except Exception as e:
            print(f"[WARN] {vf} 처리 중 오류: {e}")

    print(f"[INFO] 임베딩 추출 성공한 영상 수: {len(embeddings)} / {len(video_list)}")

    if len(embeddings) == 0:
        raise RuntimeError("❌ DeepFace가 어떤 영상에서도 얼굴을 찾지 못했습니다.")

    print("[INFO] DBSCAN으로 인물 클러스터링 중...")
    cluster = DBSCAN(eps=0.3, min_samples=2, metric='cosine').fit(embeddings)
    labels = cluster.labels_

    person_id_map = {}
    noise_id_counter = max(labels) + 1 if len(labels) > 0 else 0
    for vf, pid in zip(video_files, labels):
        if pid == -1:
            pid = noise_id_counter
            noise_id_counter += 1
        person_id_map[vf] = int(pid)

    n_persons = noise_id_counter
    clean_map = {k: int(v) for k, v in person_id_map.items()}
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"person_id_map": clean_map, "n_persons": int(n_persons)}, f, indent=2)

    print(f"[INFO] 인물 {n_persons}명 식별 완료 → {cache_path}")
    return clean_map, n_persons


# =========================================================
# 1. 영상 → 얼굴 crop (★★★ CV2 직접 탐지 ★★★)
# =========================================================
def video_to_npy(video_path, save_path, face_cascade, num_frames=32, size=112):
    """
    개선사항 (데이터 다양화 없음):
      - scaleFactor/minNeighbors 완화 → 검출 민감도 증가
      - 얼굴 탐지 실패 시 대비: 최근 성공 얼굴 + CLAHE 밝기 보정
      - 순차적 프레임 샘플링 (랜덤 없음)
    """
    if os.path.exists(save_path):
        return True

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return False

    # 프레임을 균등 분포로 샘플링
    idxs = np.linspace(0, total - 1, num_frames).astype(int)

    frames = []
    last_good_face = None
    fail_count = 0

    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            fail_count += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 밝기/조명 불균형 보정 (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 검출 민감도 약간 완화
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(30, 30)
        )

        if len(faces) > 0:
            x, y, w, h = faces[0]
            crop = rgb[y:y + h, x:x + w]
            face_img = cv2.resize(crop, (size, size))
            last_good_face = face_img
        else:
            fail_count += 1
            if last_good_face is not None:
                face_img = last_good_face.copy()
            else:
                face_img = cv2.resize(rgb, (size, size))

        frames.append(face_img)

    cap.release()

    if len(frames) == 0:
        return False

    while len(frames) < num_frames:
        frames.append(frames[-1].copy())

    np.save(save_path, np.stack(frames)[:num_frames])

    if 0 < fail_count < len(idxs):
        print(f"[INFO] 일부 얼굴 탐지 실패: {fail_count}/{len(idxs)} ({os.path.basename(video_path)})")
    elif fail_count == len(idxs):
        print(f"[WARN] 모든 프레임 탐지 실패 (전체 프레임 fallback): {os.path.basename(video_path)})")

    return True



# =========================================================
# 2. Person-based Split + CSV 생성 (★★★ 수정됨 ★★★)
# =========================================================
# ★★★ face_cascade 인자 추가 ★★★
def create_person_split(video_dir, person_id_map, face_cascade, output_root="liar3_dataset", seed=42):
    random.seed(seed)
    os.makedirs(output_root, exist_ok=True)

    person_to_files = {}
    for vf, pid in person_id_map.items():
        person_to_files.setdefault(pid, []).append(vf)

    person_ids = list(person_to_files.keys())
    train_ids, temp_ids = train_test_split(person_ids, test_size=0.3, random_state=seed)
    val_ids, test_ids = train_test_split(temp_ids, test_size=1 / 3, random_state=seed)
    splits = {"train": train_ids, "val": val_ids, "test": test_ids}

    for split, pids in splits.items():
        split_dir = os.path.join(output_root, split)
        os.makedirs(split_dir, exist_ok=True)
        csv_path = os.path.join(output_root, f"{split}_labels.csv")

        all_files = []
        for pid in pids:
            all_files.extend(person_to_files[pid])

        print(f"[INFO] {split} 세트 변환 중 ({len(all_files)}개 영상)")

        failed_conversions = 0

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "label"])
            writer.writeheader()

            for vf in tqdm(all_files, desc=f"{split} set"):
                npy_name = vf.replace(".mp4", ".npy")
                src_path = os.path.join(video_dir, vf)
                dst_path = os.path.join(split_dir, npy_name)

                # ★★★ face_cascade를 인자로 전달 ★★★
                ok = video_to_npy(src_path, dst_path, face_cascade, num_frames=32, size=112)

                if ok:
                    label = 1 if "lie" in vf.lower() else 0
                    writer.writerow({"filename": npy_name, "label": label})
                else:
                    failed_conversions += 1  # 이 로직은 거의 실행되지 않음 (캡처 실패 등)

        print(f"[INFO] {split} set 생성 완료 → {len(pids)}명, CSV: {csv_path}")
        if failed_conversions > 0:
            print(f"[WARN] {split} 세트에서 영상 파일 자체를 읽지 못해 건너뛴 영상: {failed_conversions}개")


# =========================================================
# 3. Tiny3DConvNet 정의
# =========================================================


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

        # 기본 블록
        self.block1 = r2p1d_block(3, 32)
        self.block2 = r2p1d_block(32, 64)
        self.block3 = r2p1d_block(64, 128)

        # residual 연결용 1x1x1 conv (채널 맞추기)
        self.res_proj1 = nn.Conv3d(32, 64, kernel_size=1)
        self.res_proj2 = nn.Conv3d(64, 128, kernel_size=1)

        self.pool = nn.MaxPool3d((2,2,2))
        self.global_pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        # Block 1
        x = self.block1(x)
        res1 = x
        x = self.pool(x)

        # Block 2 + residual(1)
        x = self.block2(x)
        res1_down = F.interpolate(res1, size=x.shape[2:], mode='trilinear', align_corners=False)
        res1_down = self.res_proj1(res1_down)
        x = x + res1_down
        res2 = x
        x = self.pool(x)

        # Block 3 + residual(2)
        x = self.block3(x)
        res2_down = F.interpolate(res2, size=x.shape[2:], mode='trilinear', align_corners=False)
        res2_down = self.res_proj2(res2_down)
        x = x + res2_down

        # Global pooling + FC
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return torch.sigmoid(self.fc(x))


# =========================================================
# 4. Dataset + 학습 루프 (tqdm 포함)
# =========================================================
class LieDataset(Dataset):
    def __init__(self, data_dir, label_csv):
        self.data_dir = data_dir
        with open(label_csv, newline='') as f:
            reader = csv.DictReader(f)
            self.samples = [(row['filename'], int(row['label'])) for row in reader]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        try:
            arr = np.load(os.path.join(self.data_dir, fname))
        except Exception as e:
            print(f"Error loading {fname}: {e}")
            # 에러 발생 시 검은 화면 반환
            arr = np.zeros((32, 112, 112, 3), dtype=np.uint8)

        arr = np.transpose(arr, (3, 0, 1, 2))  # (C, T, H, W)
        arr = torch.tensor(arr, dtype=torch.float32) / 255.0
        return arr, torch.tensor(label, dtype=torch.float32)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X, y in tqdm(loader, desc="Training", leave=False):
        X, y = X.to(device), y.to(device).unsqueeze(1)
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device, desc="Validating"):
    model.eval()
    y_true, y_pred, y_score = [], [], []
    total_loss = 0
    with torch.no_grad():
        for X, y in tqdm(loader, desc=desc, leave=False):
            X, y = X.to(device), y.to(device).unsqueeze(1)
            preds = model(X)
            loss = criterion(preds, y)
            total_loss += loss.item() * X.size(0)
            y_true.extend(y.cpu().numpy().flatten())
            y_pred.extend((preds.cpu().numpy() > 0.5).astype(int).flatten())
            y_score.extend(preds.cpu().numpy().flatten())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = 0.5
    return total_loss / len(loader.dataset), acc, f1, auc


# =========================================================
# 5. Main 실행 루프 (★★★ 수정됨 ★★★)
# =========================================================
def main():
    video_dir = "data"
    cache_path = "liar3_face_cache.json"
    dataset_root = "liar3_dataset"

    # STEP 1: 인물 식별 (캐시 재사용)
    print("[STEP 1] 인물 식별 중...")
    person_id_map, n_persons = generate_person_ids_from_videos(video_dir, cache_path)

    # ★★★ CV2 탐지기 로드 ★★★
    print("[INFO] CV2 Haar Cascade 탐지기 로드 중...")
    # cv2.data.haarcascades는 OpenCV가 설치된 경로의 xml 파일들을 가리킴
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
        print(f"❌ CV2 캐스케이드 파일 없음: {cascade_path}")
        print("OpenCV 설치가 잘못되었을 수 있습니다. (예: pip install opencv-python-headless)")
        return
    face_cascade = cv2.CascadeClassifier(cascade_path)
    print("[INFO] 탐지기 로드 완료.")

    # STEP 2: 데이터셋 생성 (캐시 재사용)
    print("[STEP 2] 데이터셋 생성 중...")
    if os.path.exists(os.path.join(dataset_root, "train_labels.csv")):
        print(f"[INFO] {dataset_root} 이미 존재 → 변환 생략")
    else:
        # ★★★ face_cascade를 인자로 전달 ★★★
        create_person_split(video_dir, person_id_map, face_cascade, dataset_root)

    # STEP 3: 학습
    print("[STEP 3] Tiny3D 학습 시작...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_set = LieDataset(f"{dataset_root}/train", f"{dataset_root}/train_labels.csv")
    val_set = LieDataset(f"{dataset_root}/val", f"{dataset_root}/val_labels.csv")
    test_set = LieDataset(f"{dataset_root}/test", f"{dataset_root}/test_labels.csv")

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=8, shuffle=False)

    model = TinyRes3D().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = None
    best_auc = 0
    log_file = open("liar3_training_log.txt", "w")

    # 모델 파일명
    model_save_path = "liar3_best_model_try3.pt"

    for epoch in range(1, 81):
        print(f"\n[Epoch {epoch:02d}/80]")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, val_auc = evaluate(model, val_loader, criterion, device)


        print(
            f"Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Acc={val_acc:.3f} | F1={val_f1:.3f} | AUC={val_auc:.3f}")
        log_file.write(f"{epoch},{train_loss:.4f},{val_loss:.4f},{val_acc:.3f},{val_f1:.3f},{val_auc:.3f}\n")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), model_save_path)

    log_file.close()

    print("\n[STEP 4] 테스트 세트 평가 중...")
    try:
        model.load_state_dict(torch.load(model_save_path))
    except FileNotFoundError:
        print(f"❌ 저장된 모델({model_save_path})을 찾을 수 없습니다. 학습이 실패했을 수 있습니다.")
        return

    test_loss, test_acc, test_f1, test_auc = evaluate(model, test_loader, criterion, device, desc="Testing")

    print(f"\n[Test Result] Acc={test_acc:.3f}, F1={test_f1:.3f}, AUC={test_auc:.3f}")
    with open("liar3_result_log.txt", "w") as f:
        f.write(f"Acc={test_acc:.3f}, F1={test_f1:.3f}, AUC={test_auc:.3f}\n")


if __name__ == "__main__":
    main()