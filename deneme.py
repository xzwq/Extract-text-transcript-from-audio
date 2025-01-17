import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
import whisper


audio_path = ""


output_path = ""


cleaned_audio_path = "cleaned_audio.wav"
audio = AudioSegment.from_wav(audio_path)
audio = audio.set_frame_rate(16000).set_channels(1)
audio.export(cleaned_audio_path, format="wav")


chunks = split_on_silence(
    audio,
    min_silence_len=500,  
    silence_thresh=-30    
)


chunk_folder = "audio_chunks"
os.makedirs(chunk_folder, exist_ok=True)

chunk_paths = []
for i, chunk in enumerate(chunks):
    chunk_path = os.path.join(chunk_folder, f"chunk_{i}.wav")
    chunk.export(chunk_path, format="wav")
    chunk_paths.append(chunk_path)


model = whisper.load_model("base")
transcript = ""

for chunk_path in chunk_paths:
    try:
        result = model.transcribe(chunk_path, language="tr")
        transcript += result["text"] + "\n"
    except Exception as e:
        transcript += "[Hata: Anlaşılamayan kısım]\n"


with open(output_path, "w", encoding="utf-8") as file:
    file.write(transcript)

print(f"Metin başarıyla {output_path} konumuna kaydedildi.")


for chunk_path in chunk_paths:
    os.remove(chunk_path)
os.rmdir(chunk_folder)
os.remove(cleaned_audio_path)
