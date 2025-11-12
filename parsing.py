import os
import whisper
from moviepy.editor import VideoFileClip

# 영상에서 오디오 추출
def extract_audio(video_path, audio_path="temp_audio.wav"):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
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

# 문장 구간별로 영상 자르기
def split_video_by_segments(video_path, segments, output_dir="clips"):
    os.makedirs(output_dir, exist_ok=True)
    clip = VideoFileClip(video_path)

    for i, seg in enumerate(segments):
        start, end = seg["start"], seg["end"]
        subclip = clip.subclip(start, end)
        output_path = os.path.join(output_dir, f"seg_{i+1:03d}.mp4")
        subclip.write_videofile(output_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)

        print(f"[INFO] Saved: {output_path} ({start:.2f}s → {end:.2f}s)")
        print(f"       Text: {seg['text']}\n")

    clip.close()

# 전체 파이프라인
if __name__ == "__main__":
    video_path = "input.mp4"
    print("[STEP 1] Extracting audio...")
    audio_path = extract_audio(video_path)

    print("[STEP 2] Transcribing with Whisper...")
    segments = transcribe_with_segments(audio_path, model_name="small")

    print(f"[STEP 3] Splitting video into {len(segments)} sentence clips...")
    split_video_by_segments(video_path, segments)

    print("✅ Done! Each sentence clip is in 'clips/' folder.")
