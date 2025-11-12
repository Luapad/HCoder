import torch
import torch.nn as nn
import cv2
import numpy as np
import ffmpeg
import sys

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

    def analyze_video_segment(self, video_path, start_time, end_time):

        cv2.setNumThreads(0)
        print(f"모델 분석 시도 (ffmpeg - 정확한 탐색): {video_path} @ {start_time:.2f}s - {end_time:.2f}s")

        NUM_FRAMES = 32
        SIZE = 112

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if fps <= 0: fps = 30

        duration = end_time - start_time
        if duration <= 0:
            raise ValueError(f"유효한 비디오 구간이 아닙니다 (duration 0).")

        try:
            in_stream = ffmpeg.input(video_path)

            out_bytes, err = (
                in_stream
                .output(
                    'pipe:',
                    ss=start_time,
                    t=duration,
                    format='rawvideo',
                    pix_fmt='bgr24',
                    loglevel='error'
                )
                .run(capture_stdout=True, capture_stderr=True)
            )
            if not out_bytes:
                raise ValueError(f"ffmpeg이 프레임을 추출하지 못했습니다: {err.decode()}")

        except ffmpeg.Error as e:
            print('stderr:', e.stderr.decode(), file=sys.stderr)
            raise e

        all_frames_in_segment = np.frombuffer(out_bytes, np.uint8).reshape([-1, height, width, 3])

        if not all_frames_in_segment.any():
            raise ValueError("구간 프레임을 읽을 수 없습니다.")

        num_read_frames = len(all_frames_in_segment)
        idxs = np.linspace(0, num_read_frames - 1, NUM_FRAMES).astype(int)

        sampled_frames = []
        for i in idxs:
            sampled_frames.append(all_frames_in_segment[i])

        frames = []
        last_good_face = None
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        for frame in sampled_frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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

        if len(frames) == 0:
            raise ValueError("비디오에서 프레임을 추출하지 못했습니다.")

        while len(frames) < NUM_FRAMES:
            frames.append(frames[-1].copy())

        arr = np.stack(frames)[:NUM_FRAMES]
        arr = np.transpose(arr, (3, 0, 1, 2))
        tensor = torch.tensor(arr, dtype=torch.float32) / 255.0
        input_tensor = tensor.unsqueeze(0)

        torch.use_deterministic_algorithms(True)

        with torch.no_grad():
            prob = self(input_tensor).item()

        if prob > 0.5:
            result_text = "Lie"
            confidence = prob * 100
        else:
            result_text = "Truth"
            confidence = (1 - prob) * 100

        print(f"모델 추론 완료 (ffmpeg-정확): {result_text} (Raw: {prob:.4f}, Conf: {confidence:.2f}%)")

        return {"result": result_text, "accuracy": confidence}