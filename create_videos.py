import os
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy.video.compositing.CompositeVideoClip import concatenate_videoclips
from moviepy.video.VideoClip import ImageClip

def make_clip(wav_path, img_path):
    audioclip = AudioFileClip(wav_path)
    imgclip = ImageClip(img_path).set_duration(audioclip.duration)

    # red vertical line moving with time
    def line_effect(get_frame, t):
        frame = get_frame(t).copy()
        x = int(frame.shape[1] * t / audioclip.duration)
        frame[:, x:x+2, :] = [255, 0, 0]
        return frame

    videoclip = imgclip.fl(line_effect).set_audio(audioclip)
    return videoclip

# === main ===
spectrograms_folder=snippets_folder = r"F:\Linnea\2024_high\dataset\clustering_test\training set\spectrograms\3- One_Shot_Clap Classification 2"
out_file = os.path.join(spectrograms_folder, "all_snippets.mp4") #r"F:\Linnea\2024_high\dataset\clustering_test\all_snippets.mp4"

clips = []
for fname in sorted(os.listdir(snippets_folder)):
    if fname.lower().endswith(".wav"):
        base = os.path.splitext(fname)[0]
        wav_path = os.path.join(snippets_folder, fname)
        img_path = os.path.join(spectrograms_folder, base + ".png")
        if os.path.exists(img_path):
            clips.append(make_clip(wav_path, img_path))
        else:
            print("Missing spectrogram for:", fname)

# concatenate all
final = concatenate_videoclips(clips, method="compose")
final.write_videofile(out_file, fps=24, codec="libx264", audio_codec="aac")
