import torch
import torch.nn as nn
import numpy as np
import cv2
import ffmpeg
import librosa
from scipy.interpolate import interp1d
import os

from model_video import TinyRes3D
from model_audio import AudioCNN_TConv
from model_fusion import CorrelationFusion as AdaptiveFusion 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
FACE_CASCADE = cv2.CascadeClassifier(CASCADE_PATH)

# 특징 추출 파라미터
NUM_FRAMES = 32
FACE_SIZE = 112
AUDIO_SR = 16000
N_MFCC = 60 

# 오디오 특징 길이 정규화를 위한 함수
def _match_length(arr, target_len):
    old_len = arr.shape[1]
    if old_len == target_len:
        return arr
    x_old = np.linspace(0, 1, old_len)
    x_new = np.linspace(0, 1, target_len)
    f = interp1d(x_old, arr.flatten(), kind="linear")
    new = f(x_new)
    return new.reshape(1, -1)

# 얼굴 영상 특징 텐서 생성
def _get_face_features_tensor(clip_path, start_t, duration, num_frames=NUM_FRAMES, size=FACE_SIZE):
    cv2.setNumThreads(0)
    # FFmpeg으로 프레임 추출
    try:
        probe = ffmpeg.probe(clip_path)
        video_info = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        width = int(video_info['width'])
        height = int(video_info['height'])

        out_bytes, err = (
            ffmpeg.input(clip_path, ss=start_t, t=duration)
            .output('pipe:', format='rawvideo', pix_fmt='bgr24', loglevel='error')
            .run(capture_stdout=True, capture_stderr=True)
        )
        if not out_bytes:
            raise ValueError(f"ffmpeg이 프레임을 추출하지 못했습니다: {err.decode()}")

    except Exception as e:
        print(f"Video Extraction Error: {e}")
        raise ValueError("비디오 프레임 추출에 실패했습니다.")

    all_frames_in_segment = np.frombuffer(out_bytes, np.uint8).reshape([-1, height, width, 3])
    num_read_frames = len(all_frames_in_segment)

    if num_read_frames == 0:
        raise ValueError("구간 프레임을 읽을 수 없습니다.")

    # 32프레임 샘플링
    idxs = np.linspace(0, num_read_frames - 1, num_frames).astype(int)
    sampled_frames = [all_frames_in_segment[i] for i in idxs]

    # 얼굴 탐지 및 크롭 (Haar Cascade + Fallback)
    frames = []
    last_good_face = None

    for frame in sampled_frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        face_img = None

        if len(faces) > 0:
            x, y, w, h = faces[0]
            crop = rgb[y:y + h, x:x + w]
            face_img = cv2.resize(crop, (size, size))
            last_good_face = face_img
        else:
            if last_good_face is not None:
                face_img = last_good_face.copy()
            else:
                face_img = cv2.resize(rgb, (size, size))

        frames.append(face_img)

    while len(frames) < num_frames:
        frames.append(frames[-1].copy())

    arr = np.stack(frames)[:num_frames]
    arr = np.transpose(arr, (3, 0, 1, 2))

    tensor = torch.tensor(arr, dtype=torch.float32) / 255.0
    return tensor.unsqueeze(0)


# 음성 음향 특징 텐서 생성 (새로운 Mel-Spectrogram 기반 로직)
def _get_audio_features_tensor(clip_path, start_t, end_t, sr=AUDIO_SR, n_mels=N_MFCC, g_mean=None, g_std=None):
    duration = end_t - start_t
    max_len_samples = int(duration * sr)

    try:
        # 오디오 로드 및 길이 맞추기
        y, _ = librosa.load(clip_path, sr=sr, offset=start_t, duration=duration)
    except Exception as e:
        print(f"[ERROR] Librosa load failed: {e}")
        feat_shape = (n_mels * 3 + 2, 100) 
        return torch.zeros((1, 1, feat_shape[0], feat_shape[1]), dtype=torch.float32)

    if len(y) < max_len_samples:
        y = np.pad(y, (0, max_len_samples - len(y)))
    elif len(y) > max_len_samples:
        y = y[:max_len_samples]

    # 특징 추출 (Mel-Spec, Delta, Delta2, Onset, Centroid-Delta)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spec = librosa.power_to_db(S, ref=np.max) 
    
    delta = librosa.feature.delta(mel_spec)
    delta2 = librosa.feature.delta(mel_spec, order=2)
    T = mel_spec.shape[1] 

    # Onset Strength
    flux_feat = librosa.onset.onset_strength(y=y, sr=sr, S=mel_spec)
    flux_feat = flux_feat.reshape(1, -1) 
    flux_feat = _match_length(flux_feat, T)

    # Spectral Centroid Delta
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    delta_centroid = librosa.feature.delta(centroid, order=1)
    delta_centroid = np.nan_to_num(delta_centroid)
    delta_centroid = _match_length(delta_centroid, T)
    
    feat = np.concatenate([mel_spec, delta, delta2, flux_feat, delta_centroid], axis=0)

    # 3. 정규화
    if g_mean is not None and g_std is not None:
        mean_col = g_mean.reshape(-1, 1)
        std_col = g_std.reshape(-1, 1)
        feat = (feat - mean_col) / (std_col + 1e-6)
    
    tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor

class LieModel(nn.Module):
    def __init__(self, device,
                 video_path="video_model_11_21.pt", 
                 audio_path="audio_best_model_final.pt",
                 fusion_path="fusion_interaction_best.pt",
                 stats_root="."):
        super(LieModel, self).__init__()

        self.device = device
        self.g_mean = None
        self.g_std = None

        # 오디오 통계 로드
        g_mean_path = os.path.join(stats_root, "g_mean.npy")
        g_std_path = os.path.join(stats_root, "g_std.npy")

        try:
            if os.path.exists(g_mean_path) and os.path.exists(g_std_path):
                self.g_mean = np.load(g_mean_path)
                self.g_std = np.load(g_std_path)
                print(f"[INFO] 통계 자료 (g_mean/g_std) 로드 완료")
            else:
                raise FileNotFoundError("통계 파일 없음")
        except Exception as e:
            print(f"[WARN] 오디오 통계 파일 로드 실패: {e}")
            print("[WARN] 오디오 특징 정규화 없이 진행합니다.")
        
        try:
            self.video_encoder = TinyRes3D().to(device)
            self.audio_encoder = AudioCNN_TConv().to(device)
            # CorrelationFusion 인스턴스화
            self.fusion = AdaptiveFusion().to(device) 

            # Video Encoder 로드
            # print(f"[INFO] Video 가중치 로드 중: {video_path}")
            v_state_dict = torch.load(video_path, map_location=device)
            self.video_encoder.load_state_dict(v_state_dict)
            print(f"[INFO] video 모델 로드 완료")

            # Audio Encoder 로드
            # print(f"[INFO] Audio 가중치 로드 중: {audio_path}")
            a_state_dict = torch.load(audio_path, map_location=device)
            self.audio_encoder.load_state_dict(a_state_dict)
            print(f"[INFO] audio 모델 로드 완료")

            # C. Fusion Head 로드
            # print(f"[INFO] Fusion 가중치 로드 중: {fusion_path}")
            f_state_dict = torch.load(fusion_path, map_location=device)
            self.fusion.load_state_dict(f_state_dict)
            print(f"[INFO] Fusion 모델 로드 완료")

            # 4. 모델 설정 및 로그 출력
            self.video_encoder.eval()
            for p in self.video_encoder.parameters(): p.requires_grad = False
            self.audio_encoder.eval()
            for p in self.audio_encoder.parameters(): p.requires_grad = False
            self.fusion.eval()

        except FileNotFoundError as e:
            print(f"[ERROR] 모델 파일을 찾을 수 없습니다: {e}")
            raise RuntimeError(f"모델 로드 실패: {e}")
        except Exception as e:
            print(f"[ERROR] 모델 로드 중 오류 발생: {e}")
            raise RuntimeError(f"모델 로드 중 오류: {e}")

    # NEW METHOD: 단일 영상 모델 예측 (로그 출력용 헬퍼)
    def analyze_video_only(self, video_tensor):
        if video_tensor.dtype == torch.float16:
            video_tensor = video_tensor.float()
            
        with torch.no_grad():
            prob = self.video_encoder(video_tensor).item() 

        if prob > 0.5:
            result_text = "Lie"
        else:
            result_text = "Truth"
            
        return {"result": result_text, "prob": prob}

    # NEW METHOD: 단일 음성 모델 예측 (로그 출력용 헬퍼)
    def analyze_audio_only(self, audio_tensor):
        if audio_tensor.dtype == torch.float16:
            audio_tensor = audio_tensor.float()
            
        with torch.no_grad():
            logits, _ = self.audio_encoder(audio_tensor)
            prob = torch.sigmoid(logits).item()

        if prob > 0.5:
            result_text = "Lie"
        else:
            result_text = "Truth"
            
        return {"result": result_text, "prob": prob}

    def analyze_video_segment(self, temp_clip_path, start_in_clip, end_in_clip):

        # 클래스 불균형 보정 설정
        CLASS_IMBALANCE_RATIO = 1.10882 
        ADJUSTMENT_TERM = np.log(CLASS_IMBALANCE_RATIO) 
        CLASSIFICATION_THRESHOLD = 0.5 

        # 특징 추출 및 텐서 이동
        video_tensor = _get_face_features_tensor(
            temp_clip_path, start_in_clip, end_in_clip - start_in_clip
        )
        audio_tensor = _get_audio_features_tensor(
            temp_clip_path, start_in_clip, end_in_clip,
            g_mean=self.g_mean, g_std=self.g_std 
        )

        video_tensor = video_tensor.to(self.device)
        audio_tensor = audio_tensor.to(self.device)
        
        if self.device.type == 'cuda' and next(self.parameters()).dtype == torch.float16:
            video_tensor = video_tensor.half()
            audio_tensor = audio_tensor.half()


        # 특징 벡터 생성 및 통합 추론
        with torch.no_grad():
            
            # 단일 모델 예측을 위한 데이터 준비 (로깅용)
            video_result = self.analyze_video_only(video_tensor) 
            audio_result = self.analyze_audio_only(audio_tensor)

            # 퓨전 모델 특징 추출 및 로짓 계산
            v_feat = self.video_encoder.forward_features(video_tensor) 
            _, a_feat = self.audio_encoder(audio_tensor) 

            logits, w_v, w_a = self.fusion(v_feat, a_feat) 
            
            # Logit 조정
            logits_adjusted = logits - ADJUSTMENT_TERM
            
            # 보정된 Logit으로 최종 확률 계산
            prob = torch.sigmoid(logits_adjusted).item() 

        # 최종 결과 판단
        if prob > CLASSIFICATION_THRESHOLD:
            result_text = "Lie"
            confidence = prob * 100
        else:
            result_text = "Truth"
            confidence = (1 - prob) * 100

        log_w_v = w_v.item() if w_v is not None else 1.0 
        log_w_a = w_a.item() if w_a is not None else 0.0 
        
        print("\n--- 예측 결과 ---")
        print(f"[V] 영상 예측: {video_result['result']} (Prob: {video_result['prob']:.4f})")
        print(f"[A] 음성 예측: {audio_result['result']} (Prob: {audio_result['prob']:.4f})")
        print(f"[F] 퓨전 예측: {result_text} (Prob: {prob:.4f})")
        print("----------------------------")
        
        return {
            "result": result_text,
            "accuracy": confidence,
            "visual_weight": log_w_v,
            "audio_weight": log_w_a
        }