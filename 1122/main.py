import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid
import whisper
from moviepy.editor import VideoFileClip

from model import LieModel
from parsing import extract_audio, transcribe_with_segments

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch 사용 디바이스: {DEVICE}")

# Flask 설정
app = Flask(__name__)
CORS(app)

# 모델 로드 위한 변수
model = None
whisper_model = None
WHISPER_MODEL_NAME = "small"

try:
    # LieModel 로드 및 GPU 이동
    model = LieModel(
        DEVICE,
        video_path="video_model_11_21.pt", 
        audio_path="audio_best_model_final.pt",
        fusion_path="fusion_interaction_best.pt"
    )

    model.to(DEVICE)
    model.eval()

    model_device = next(model.parameters()).device
    model_dtype = str(next(model.parameters()).dtype).replace('torch.', '')
    print(f"\nLieModel")
    print(f"모델 위치 : {model_device}, dtype: {model_dtype}")

    # Whisper(parsing model) 로드 및 GPU 이동
    whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
    whisper_model.to(DEVICE)

    whisper_device = next(whisper_model.parameters()).device
    whisper_dtype = str(next(whisper_model.parameters()).dtype).replace('torch.', '')
    print(f"\nWhisper")
    print(f"모델 위치 : {whisper_device}, dtype: {whisper_dtype}")

except Exception as e:
    print(f"모델 로드 중 오류 발생: {e}")
    model = None
    whisper_model = None

# API : detection / analyze
@app.route('/analyze', methods=['POST'])
def analyze_video():
    temp_dir = "temp_uploads"
    # 변수 초기화 (finally 블록에서 접근할 수 있도록)
    original_video_path = None
    temp_audio_path = None
    temp_clip_path = None

    analysis_results = []

    if model is None or whisper_model is None:
        return jsonify({"error행
if __name__ == '__main__':
    print("\nFlask Running http://127.0.0.1:5000")
    app.run(debug=True, port=5000, use_reloader=False)

