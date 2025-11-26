import torch
from moviepy.editor import VideoFileClip

# GPU 사용 가능 여부
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 영상에서 오디오 추출
def extract_audio(video_path, audio_path="temp_audio.wav"):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
    clip.close()
    return audio_path


#  Whisper로 문장별 구간 추출
def transcribe_with_segments(audio_path, model):
    model_device = next(model.parameters()).device
    model_dtype = str(next(model.parameters()).dtype).replace('torch.', '')
    print(f"[WHISPER] 사용 디바이스: {model_device}, dtype: {model_dtype}\n")

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

