import os
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from moviepy import VideoFileClip
import uuid

from model import LieModel
from parsing import extract_audio, transcribe_with_segments

# Flask 설정
app = Flask(__name__)
CORS(app)

# 모델 로드
try:
    model = LieModel()
    state_dict = torch.load("model.pt", map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    print("PyTorch 모델 로드 성공 ('model.pt')")

except Exception as e:
    print(f"⚠️ 모델 로드 실패: {e}")
    model = None

# 거짓말 탐지 API
@app.route('/analyze', methods=['POST'])
def analyze_video():
    temp_dir = "temp_uploads"
    temp_video_path = None
    segment_path = None

    try:
        if 'video' not in request.files:
            return jsonify({"error": "비디오 파일이 없습니다."}), 400

        video_file = request.files['video']
        start_time = float(request.form['startTime'])
        end_time = float(request.form['endTime'])

        os.makedirs(temp_dir, exist_ok=True)
        safe_filename = str(uuid.uuid4()) + "_" + video_file.filename
        temp_video_path = os.path.join(temp_dir, safe_filename)
        video_file.save(temp_video_path)

        print(f"파일 수신: {temp_video_path}, 구간: {start_time}s - {end_time}s")

        segment_path = os.path.join(temp_dir, "segment_" + str(uuid.uuid4()) + ".mp4")
        with VideoFileClip(temp_video_path) as video:
            segment = video.subclip(start_time, end_time)
            segment.write_videofile(segment_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)

        # 모델 분석
        if model is not None:
            analysis_result = model.analyze_video_segment(segment_path)
        else:
            analysis_result = {"error": "모델이 로드되지 않았습니다."}

        return jsonify(analysis_result)

    except Exception as e:
        print(f"오류 발생: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        try:
            if segment_path and os.path.exists(segment_path):
                os.remove(segment_path)
                print(f"[INFO] 임시 구간 영상 삭제: {segment_path}")
            if temp_video_path and os.path.exists(temp_video_path):
                os.remove(temp_video_path)
                print(f"[INFO] 임시 원본 영상 삭제: {temp_video_path}")
        except Exception as e:
            print(f"[ERROR] 임시 파일 정리 실패: {e}")


# parsing API
@app.route('/analyze_full', methods=['POST'])
def analyze_full():
    video_path = None  # 임시 파일 경로를 미리 선언
    audio_path = None
    temp_dir = "temp_uploads"

    try:
        if 'video' not in request.files:
            return jsonify({"error": "비디오 파일이 없습니다."}), 400

        video_file = request.files['video']
        os.makedirs(temp_dir, exist_ok=True)

        # 파일 이름 중복 방지를 위해 uuid 사용
        safe_filename = str(uuid.uuid4()) + "_" + video_file.filename
        video_path = os.path.join(temp_dir, safe_filename)
        video_file.save(video_path)

        # Whisper로 문장별 구간 분석
        print("[INFO] Whisper로 문장 구간 분석 시작")
        audio_path = extract_audio(video_path)  # 'temp_audio.wav' 생성
        segments = transcribe_with_segments(audio_path)

        # Flask 응답으로 JSON 반환
        return jsonify(segments)

    except Exception as e:
        print(f"[ERROR] Whisper 분석 실패: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        # 작업이 끝나면 임시 파일 삭제
        try:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
                print(f"[INFO] 임시 오디오 삭제: {audio_path}")
            if video_path and os.path.exists(video_path):
                os.remove(video_path)
                print(f"[INFO] 임시 비디오 삭제: {video_path}")
        except Exception as e:
            print(f"[ERROR] 임시 파일 정리 실패: {e}")

if __name__ == '__main__':
    print("Flask start : http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
