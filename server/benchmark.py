import os
import sys
import time
import gc
import jiwer
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download

venv_base = sys.prefix
site_packages = os.path.join(venv_base, "Lib", "site-packages")
cublas_bin = os.path.join(site_packages, "nvidia", "cublas", "bin")
cudnn_bin = os.path.join(site_packages, "nvidia", "cudnn", "bin")

if os.path.exists(cublas_bin):
    os.environ["PATH"] = cublas_bin + os.pathsep + os.environ["PATH"]
    os.add_dll_directory(cublas_bin)

if os.path.exists(cudnn_bin):
    os.environ["PATH"] = cudnn_bin + os.pathsep + os.environ["PATH"]
    os.add_dll_directory(cudnn_bin)

base_folder = "Sample-Audio"
file_pairs = [
    {"audio": os.path.join(base_folder, f"D{i}.mp3"), "text_file": os.path.join(base_folder, f"S{i}.txt")}
    for i in range(1, 11)
]

print("Starting Whisper Large-v3-Turbo GPU Evaluation...")
print("=" * 60)

repo_id = "deepdml/faster-whisper-large-v3-turbo-ct2"
model_path = snapshot_download(repo_id=repo_id, max_workers=1)

model = WhisperModel(model_path, device="cuda", compute_type="int8_float16")

if os.path.exists(file_pairs[0]["audio"]):
    try:
        _ = model.transcribe(file_pairs[0]["audio"], beam_size=5)
    except Exception:
        pass

for pair in file_pairs:
    audio_path = pair["audio"]
    text_path = pair["text_file"]
    
    if not os.path.exists(audio_path) or not os.path.exists(text_path):
        continue
        
    with open(text_path, "r", encoding="utf-8") as f:
        reference_text = f.read().strip().lower()
        
    print(f"  -> Transcribing: {audio_path}")
    start_time = time.perf_counter()
    
    segments, _ = model.transcribe(audio_path, beam_size=5)
    transcribed_text = " ".join([segment.text for segment in segments]).strip().lower()
    
    end_time = time.perf_counter()
    execution_ms = (end_time - start_time) * 1000
    
    wer_score = jiwer.wer(reference_text, transcribed_text)
    
    print(f"  | Time: {execution_ms:.2f} ms | WER: {wer_score:.4f}")
    print(f"  | Ref: {reference_text}")
    print(f"  | Hyp: {transcribed_text}")
    print("  " + "-" * 30)

del model
gc.collect()

print("\nEvaluation process finished.")