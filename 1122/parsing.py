import torch
from moviepy.editor import VideoFileClip

# DEVICE 변수를 정의하여 GPU 사용 가능 여부를 확인하고 저장합니다.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 영상에서 오디오 추출
def extract_audio(video_path, audio_path="temp_audio.wav"):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
    clip.close()
    return audio_path


#  Whisper로 문장별 구간 추출
def transcribe_with_segments(audio_path, model):
    # 모델 추론 시작 전, 모델의 dtype을 확인하여 로그에 표시합니다.
    model_device = next(model.parameters()).device
    model_dtype = str(next(model.parameters()).dtype).replace('torch.', '')
    print(f"[WHISPER] 사용 디바이스: {model_device}, dtype: {model_dtype}\n")

    # fp16=False를 명시적으로 전달하여 float16 모델에 float32 입력을 처리하도록 합니다.
    result = model.transcribe(audio_path, fp16=False)

    print(f"[WHISPER] {len(result['segments'])}개 세그먼트 추출됨.\n")

    segments = []
    for seg in result["segments"]:
        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip()
        })
    return segments
