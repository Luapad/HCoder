import torch
import torch.nn as nn
import cv2
import numpy as np

class LieModel(nn.Module):

    def __init__(self):
        super(LieModel, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(3, 32, 3, 1, 1), nn.BatchNorm3d(32), nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(32, 64, 3, 1, 1), nn.BatchNorm3d(64), nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(64, 128, 3, 1, 1), nn.BatchNorm3d(128), nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return torch.sigmoid(x)

    def analyze_video_segment(self, segment_path):
        print(f"모델 분석 시도: {segment_path}")

        # 영상 로딩
        NUM_FRAMES = 32
        SIZE = 112
        cap = cv2.VideoCapture(segment_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            raise ValueError("비디오 프레임을 읽을 수 없습니다.")

        idxs = np.linspace(0, total - 1, NUM_FRAMES).astype(int)
        frames = []
        last_good_face = None

        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 얼굴 탐지
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 0:
                x, y, w, h = faces[0]
                crop = rgb[y:y + h, x:x + w]
                face_img = cv2.resize(crop, (SIZE, SIZE))
                last_good_face = face_img
            else:
                if last_good_face is not None:
                    face_img = last_good_face.copy()
                else:
                    face_img = cv2.resize(rgb, (SIZE, SIZE))

            frames.append(face_img)

        cap.release()

        if len(frames) == 0:
            raise ValueError("비디오에서 프레임을 추출하지 못했습니다.")

        # 프레임 개수 부족 시 마지막 프레임으로 채움
        while len(frames) < NUM_FRAMES:
            frames.append(frames[-1].copy())

        # 텐서 변환
        arr = np.stack(frames)[:NUM_FRAMES]
        arr = np.transpose(arr, (3, 0, 1, 2))
        tensor = torch.tensor(arr, dtype=torch.float32) / 255.0
        input_tensor = tensor.unsqueeze(0)

        with torch.no_grad():
            prob = self(input_tensor).item()

        if prob > 0.5:
            result_text = "거짓 (Lie)"
            confidence = prob * 100
        else:
            result_text = "진실 (Truth)"
            confidence = (1 - prob) * 100

        print(f"모델 추론 완료: {result_text} (Raw: {prob:.4f}, Conf: {confidence:.2f}%)")

        return {"result": result_text, "accuracy": confidence}
