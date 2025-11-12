import os
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid
import json


from moviepy.editor import VideoFileClip

from model import LieModel
from parsing import extract_audio, transcribe_with_segments

# Flask settings
app = Flask(__name__)
CORS(app)

# model load
try:
    model = LieModel()
    state_dict = torch.load("model.pt", map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    print("모델 로드 성공 ('model.pt')")

except Exception as e:
    print(f"모델 로드 실패: {e}")
    model = None

# API : detection
# main.py의 /analyze 함수를 이걸로 교체

@app.route('/analyze', methods=['POST'])
def analyze_video():
    temp_dir = "temp_uploads"
    temp_video_path = None
    temp_audio_path = None

    analysis_results = []

    try:
        if 'video' not in request.files:
            return jsonify({"error": "비디오 파일이 없습니다."}), 400

        # [수정] 프론트에서 startTime과 endTime을 받습니다.
        start_time = float(request.form['startTime'])
        end_time = float(request.form['endTime'])

        video_file = request.files['video']

        os.makedirs(temp_dir, exist_ok=True)
        safe_filename = str(uuid.uuid4()) + "_" + video_file.filename
        temp_video_path = os.path.join(temp_dir, safe_filename)
        video_file.save(temp_video_path)

        print(f"[INFO] 파일 수신: {temp_video_path}, 요청 구간: {start_time:.2f}s - {end_time:.2f}s")

        # 1. [B 방식] 요청된 구간의 오디오를 추출합니다.
        temp_audio_path = os.path.join(temp_dir, str(uuid.uuid4()) + ".wav")
        with VideoFileClip(temp_video_path) as video:
            # subclip으로 오디오 추출
            subclip_audio = video.subclip(start_time, end_time).audio

            if subclip_audio:
                # 임시 파일로 저장
                subclip_audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
            else:
                return jsonify([]), 200  # 오디오 없으면 빈 리스트 반환

        # 2. [B 방식] 추출된 오디오를 Whisper로 파싱합니다. (매번 실행됨)
        print("[INFO] Whisper 파싱 재실행...")
        # parsing.py의 transcribe_with_segments 함수를 사용
        segments = transcribe_with_segments(temp_audio_path)

        if not segments:
            return jsonify([]), 200

        # 3. 파싱된 문장들을 각각 모델로 분석합니다.
        for seg in segments:
            # 파싱된 시간은 0초부터의 상대적 시간입니다. 원본 시간으로 변환합니다.
            new_start = seg['start'] + start_time
            new_end = seg['end'] + start_time

            # [수정] model.py는 이제 원본 비디오 경로를 사용해 프레임을 정확히 읽습니다.
            if model is not None:
                result = model.analyze_video_segment(
                    temp_video_path,
                    new_start,
                    new_end
                )

                result['text'] = seg.get('text', '')
                result['start'] = new_start
                result['end'] = new_end
                analysis_results.append(result)
            else:
                analysis_results.append({"error": "모델이 로드되지 않았습니다."})

        return jsonify(analysis_results)

    except Exception as e:
        print(f"오류 발생: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        # 임시 파일 정리
        try:
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                print(f"[INFO] 임시 오디오 삭제 완료: {temp_audio_path}")

            if temp_video_path and os.path.exists(temp_video_path):
                # 원본 파일은 분석 후 삭제
                os.remove(temp_video_path)
                print(f"[INFO] 임시 원본 영상 삭제 완료: {temp_video_path}")
        except Exception as e:
            print(f"[ERROR] 임시 파일 정리 실패: {e}")

# API : parsing
@app.route('/analyze_full', methods=['POST'])
def analyze_full():
    video_path = None  # 임시 파일 경로
    audio_path = None
    temp_dir = "temp_uploads"

    try:
        if 'video' not in request.files:
            return jsonify({"error": "비디오 파일이 없습니다."}), 400

        video_file = request.files['video']
        os.makedirs(temp_dir, exist_ok=True)

        safe_filename = str(uuid.uuid4()) + "_" + video_file.filename
        video_path = os.path.join(temp_dir, safe_filename)
        video_file.save(video_path)

        # Whisper로 문장별 구간 분석
        print("[INFO] Whisper로 문장 구간 분석 시작")
        audio_path = extract_audio(video_path)
        segments = transcribe_with_segments(audio_path)

        # Flask 응답으로 JSON 반환
        return jsonify(segments)

    except Exception as e:
        print(f"[ERROR] Whisper 분석 실패: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        try:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
                print(f"[INFO] 임시 오디오 삭제: {audio_path}")
            if video_path and os.path.exists(video_path):
                os.remove(video_path)
                print(f"[INFO] 임시 비디오 삭제: {video_path}")
        except Exception as e:
            print(f"[ERROR] 임시 파일 정리 실패: {e}")

# Flask Run
if __name__ == '__main__':
    print("Flask Running http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
