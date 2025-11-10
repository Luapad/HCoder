import os
import whisper
from moviepy import VideoFileClip

# 영상에서 오디오 추출
def extract_audio(video_path, audio_path="temp_audio.wav"):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path)
    clip.close()
    return audio_path

#  Whisper로 문장별 구간 추출
def transcribe_with_segments(audio_path, model_name="small"):
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path)

    segments = []
    for seg in result["segments"]:
        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip()
        })
    return segments

# 구간별 영상 저장 : parsing.py만 단독 실행 시 결과 확인 위한 코드임, main.py에서는 사용 안 함.
def split_video_by_segments(video_path, segments, output_dir="clips"):
    os.makedirs(output_dir, exist_ok=True)
    clip = VideoFileClip(video_path)

    for i, seg in enumerate(segments):
        start, end = seg["start"], seg["end"]
        subclip = clip.subclipped(start, end)
        output_path = os.path.join(output_dir, f"seg_{i+1:03d}.mp4")
        subclip.write_videofile(output_path, codec="libx264", audio_codec="aac")

        print(f"[INFO] : {output_path} ({start:.2f}s → {end:.2f}s)")
        print(f"       Text: {seg['text']}\n")

    clip.close()

if __name__ == "__main__":
    video_path = "input.mp4"  # parsing 할 영상 경로
    print("[STEP 1] Extracting audio...")
    audio_path = extract_audio(video_path)

    print("[STEP 2] Transcribing with Whisper...")
    segments = transcribe_with_segments(audio_path, model_name="small")

    print(f"[STEP 3] Splitting video into {len(segments)} sentence clips...")
    split_video_by_segments(video_path, segments)

    print("Done! Each sentence clip is in 'clips/' folder.")