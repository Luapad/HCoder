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

# 모델 로드 위한 변수 초기화
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
    # 로컬 변수 초기화 (finally 블록에서 접근할 수 있도록)
    original_video_path = None
    temp_audio_path = None
    temp_clip_path = None

    analysis_results = []

    if model is None or whisper_model is None:
        return jsonify({"error": "모델이 서버에 로드되지 않았습니다."}), 500

    try:
        if 'video' not in request.files:
            return jsonify({"error": "비디오 파일이 없습니다."}), 400

        # 원본 파일 저장
        start_time = float(request.form['startTime'])
        end_time = float(request.form['endTime'])
        video_file = request.files['video']
        os.makedirs(temp_dir, exist_ok=True)
        safe_filename = str(uuid.uuid4()) + "_" + video_file.filename
        original_video_path = os.path.join(temp_dir, safe_filename)
        video_file.save(original_video_path)
        print(f"\n[INFO] 원본 영상 저장 완료, 요청 구간: {start_time:.2f}s - {end_time:.2f}s\n")

        # 임시 클립 생성
        temp_clip_path = os.path.join(temp_dir, f"clip_{uuid.uuid4()}.mp4")
        try:
            # moviepy로 원본 영상에서 구간을 잘라 새 파일 생성
            with VideoFileClip(original_video_path) as video:
                subclip = video.subclip(start_time, end_time)

                subclip.write_videofile(
                    temp_clip_path,
                    codec="libx264",
                    audio_codec="aac",
                    verbose=False,
                    logger=None
                )
            print(f"[INFO] 임시 클립 생성 완료\n")

        except Exception as e:
            print(f"[ERROR] 임시 클립 생성 실패: {e}")
            raise

        # 임시 클립에서 오디오 추출
        temp_audio_path = os.path.join(temp_dir, str(uuid.uuid4()) + ".wav")
        with VideoFileClip(temp_clip_path) as video:
            subclip_audio = video.audio
            if subclip_audio:
                subclip_audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
            else:
                return jsonify([]), 200

        # Whisper 파싱
        print("[INFO] Whisper 문장 분석 시작...\n")
        segments = transcribe_with_segments(temp_audio_path, whisper_model)

        if not segments:
            return jsonify([]), 200

        # LieModel 분석
        for seg in segments:
            start_in_clip = seg['start']
            end_in_clip = seg['end']

            # 임시 클립 경로와 클립 내 시간(start/end_in_clip)을 전달
            result = model.analyze_video_segment(
                temp_clip_path,
                start_in_clip,
                end_in_clip
            )

            result['text'] = seg.get('text', '')
            # 원본 영상 기준 시간으로 반환
            result['start'] = start_time + start_in_clip
            result['end'] = start_time + end_in_clip
            analysis_results.append(result)

        return jsonify(analysis_results)

    except Exception as e:
        print(f"오류 발생: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        # 임시 파일 정리 (원본 영상, 임시 클립, 오디오 파일 모두 삭제)
        try:
            # 임시 클립 삭제
            if temp_clip_path and os.path.exists(temp_clip_path):
                os.remove(temp_clip_path)
                # print(f"[INFO] 임시 영상 삭제 완료\n")

            # 임시 오디오 파일 삭제
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                # print(f"[INFO] 임시 오디오 삭제 완료\n")

            # 원본 영상 파일 삭제
            if original_video_path and os.path.exists(original_video_path):
                os.remove(original_video_path)
                # print(f"[INFO] 원본 영상 삭제 완료\n")
            
            print(f"[INFO] 임시 파일 삭제 완료\n")

        except Exception as e:
            print(f"[ERROR] 임시 파일 정리 실패: {e}")

# API : parsing / analyze_full
@app.route('/analyze_full', methods=['POST'])
def analyze_full():
    video_path = None
    audio_path = None
    temp_dir = "temp_uploads"

    if whisper_model is None:
        return jsonify({"error": "Whisper 모델이 서버에 로드되지 않았습니다."}), 500

    try:
        if 'video' not in request.files:
            return jsonify({"error": "비디오 파일이 없습니다."}), 400

        video_file = request.files['video']
        os.makedirs(temp_dir, exist_ok=True)

        safe_filename = str(uuid.uuid4()) + "_" + video_file.filename
        video_path = os.path.join(temp_dir, safe_filename)
        video_file.save(video_path)

        # 전체 오디오 추출 및 파싱
        print("\n[INFO] Whisper : 문장 분석 시작...\n")
        audio_path = extract_audio(video_path)

        segments = transcribe_with_segments(audio_path, whisper_model)

        # Flask 응답으로 JSON 반환
        return jsonify(segments)

    except Exception as e:
        print(f"[ERROR] Whisper 분석 실패: {e}\n")
        return jsonify({"error": str(e)}), 500

    finally:
        # 임시 파일 정리 (영상 파일 및 오디오 파일은 삭제)
        try:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
                # print(f"[INFO] 오디오 파일 삭제 완료\n")
            if video_path and os.path.exists(video_path):
                os.remove(video_path)
                # print(f"[INFO] 영상 파일 삭제 완료\n")
            print(f"[INFO] 임시 파일 삭제 완료\n")
        except Exception as e:
            print(f"[ERROR] 임시 파일 정리 실패: {e}\n")


# Flask Run
if __name__ == '__main__':
    print("\nFlask Running http://127.0.0.1:5000")
    app.run(debug=True, port=5000, use_reloader=False)