import os
import warnings
import site
import re
import time
import sys
import wave
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from transformers import pipeline
from huggingface_hub import snapshot_download

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

venv_base = sys.prefix
site_packages_path = os.path.join(venv_base, "Lib", "site-packages")
cublas_bin = os.path.join(site_packages_path, "nvidia", "cublas", "bin")
cudnn_bin = os.path.join(site_packages_path, "nvidia", "cudnn", "bin")

if os.path.exists(cublas_bin):
    os.environ["PATH"] = cublas_bin + os.pathsep + os.environ["PATH"]
    os.add_dll_directory(cublas_bin)

if os.path.exists(cudnn_bin):
    os.environ["PATH"] = cudnn_bin + os.pathsep + os.environ["PATH"]
    os.add_dll_directory(cudnn_bin)

app = FastAPI(title="Emotion Analysis Microservice")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

whisper_model = None
whisper_model_large = None
roberta_model = None
vad_model = None

def categorize_emotion(emotion: str, score: float) -> str:
    positive_emotions = {"admiration", "amusement", "approval", "caring", "desire", "excitement", "gratitude", "joy", "love", "optimism", "pride", "relief"}
    negative_emotions = {"anger", "annoyance", "disappointment", "disapproval", "disgust", "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness"}

    if emotion in positive_emotions:
        return "positive" if score >= 0.60 else "neutral"
    elif emotion in negative_emotions:
        return "negative" if score >= 0.70 else "neutral"
    else:
        return "positive" if emotion == "surprise" and score >= 0.80 else "neutral"

@app.on_event("startup")
def load_models():
    global whisper_model, whisper_model_large, roberta_model, vad_model
    
    print("\n--- Initializing Models ---")
    
    print("[1/4] Loading Whisper 'small' model from cache...")
    whisper_model = WhisperModel("small", device="cuda", compute_type="int8_float16")
    
    print("[2/4] Downloading/Loading Whisper 'large-v3-turbo' model...")
    large_model_path = snapshot_download(repo_id="deepdml/faster-whisper-large-v3-turbo-ct2")
    whisper_model_large = WhisperModel(large_model_path, device="cuda", compute_type="int8_float16")
    
    print("[3/4] Loading RoBERTa emotion classifier...")
    roberta_model = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions")
    
    print("[4/4] Loading Silero VAD model...")
    vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True, verbose=True)
    
    vad_model.eval()
    if torch.cuda.is_available():
        vad_model.to('cuda')
        
    print("--- All Models Loaded Successfully ---\n")

def predict_emotion(text: str) -> dict:
    result = roberta_model(text, truncation=True, max_length=512)[0]
    return {"label": result["label"], "score": round(result["score"], 4)}

@app.websocket("/live-stream")
async def live_stream(websocket: WebSocket):
    await websocket.accept()
    analyzed_sentences = set()
    raw_audio_buffer = bytearray()
    
    temp_file = f"temp_{id(websocket)}.wav" 
    
    chunk_timer = time.time() 
    last_speech_time = time.time()
    last_text = ""

    try:
        while True:
            chunk = await websocket.receive_bytes()
            raw_audio_buffer.extend(chunk)
            
            if time.time() - chunk_timer > 2.5:
                if len(raw_audio_buffer) == 0:
                    continue
                    
                with wave.open(temp_file, 'wb') as wav_file:
                    wav_file.setnchannels(1)      
                    wav_file.setsampwidth(2)      
                    wav_file.setframerate(16000)  
                    wav_file.writeframes(raw_audio_buffer)

                try:
                    segments, _ = whisper_model.transcribe(temp_file, beam_size=5, vad_filter=True)
                    full_transcript = " ".join([segment.text for segment in segments]).strip()

                    if full_transcript:
                        if full_transcript != last_text:
                            last_speech_time = time.time()
                            last_text = full_transcript

                        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', full_transcript) if s.strip()]
                        
                        partial_text = ""
                        force_clear_buffer = False
                        has_trailing_partial = False
                        
                        for sentence in sentences:
                            has_punctuation = bool(re.search(r'[.!?]$', sentence))
                            is_hanging = not has_punctuation and (time.time() - last_speech_time > 2.0)
                            is_too_long = not has_punctuation and len(sentence.split()) >= 20

                            if has_punctuation or is_hanging or is_too_long:
                                clean_sentence = sentence if has_punctuation else sentence + "."
                                
                                if clean_sentence not in analyzed_sentences:
                                    emotion_data = predict_emotion(clean_sentence)
                                    sentiment = categorize_emotion(emotion_data["label"], emotion_data["score"])
                                    
                                    await websocket.send_json({
                                        "type": "analyzed",
                                        "text": clean_sentence,
                                        "emotion": emotion_data["label"],
                                        "sentiment_category": sentiment,
                                        "confidence": emotion_data["score"]
                                    })
                                    
                                    analyzed_sentences.add(clean_sentence)
                                    analyzed_sentences.add(sentence) 
                                    force_clear_buffer = True 
                            else:
                                partial_text = sentence
                                has_trailing_partial = True

                        await websocket.send_json({
                            "type": "partial",
                            "text": partial_text
                        })

                        if force_clear_buffer and not has_trailing_partial:
                            raw_audio_buffer.clear()
                            last_text = ""
                            last_speech_time = time.time()
                            
                    else:
                        await websocket.send_json({
                            "type": "partial",
                            "text": ""
                        })

                except Exception:
                    pass 
                
                chunk_timer = time.time()
                
                if len(raw_audio_buffer) > (16000 * 2 * 10): 
                    raw_audio_buffer.clear()
                    last_text = ""
                    last_speech_time = time.time()

    except WebSocketDisconnect:
        pass
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

@app.websocket("/live-stream-vad")
async def live_stream_vad(websocket: WebSocket):
    await websocket.accept()
    analyzed_sentences = set()
    speech_buffer = []
    is_speaking = False
    silence_frames = 0
    SILENCE_THRESHOLD = 0.8 
    WINDOW_SIZE = 512

    try:
        while True:
            chunk = await websocket.receive_bytes()
            audio_chunk = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            
            for offset in range(0, len(audio_chunk), WINDOW_SIZE):
                vad_block = audio_chunk[offset:offset + WINDOW_SIZE]
                if len(vad_block) < WINDOW_SIZE:
                    continue
                    
                tensor_chunk = torch.from_numpy(vad_block)
                if torch.cuda.is_available():
                    tensor_chunk = tensor_chunk.to('cuda')

                try:
                    speech_prob = vad_model(tensor_chunk, 16000).item()
                except Exception as vad_err:
                    print(f"VAD Model Error: {vad_err}")
                    continue

                if speech_prob > 0.5:
                    is_speaking = True
                    silence_frames = 0
                    speech_buffer.append(vad_block)
                    
                    if len(speech_buffer) % 12 == 0:
                        temp_audio = np.concatenate(speech_buffer)
                        segments, _ = whisper_model_large.transcribe(temp_audio, beam_size=1) 
                        partial_transcript = " ".join([s.text for s in segments]).strip()
                        await websocket.send_json({"type": "partial", "text": partial_transcript})
                        
                else:
                    if is_speaking:
                        silence_frames += (WINDOW_SIZE / 16000.0)
                        speech_buffer.append(vad_block)

                        if silence_frames > SILENCE_THRESHOLD:
                            final_audio = np.concatenate(speech_buffer)
                            segments, _ = whisper_model_large.transcribe(final_audio, beam_size=5, vad_filter=True)
                            transcript = " ".join([s.text for s in segments]).strip()

                            if transcript and transcript not in analyzed_sentences:
                                emotion_data = predict_emotion(transcript)
                                sentiment = categorize_emotion(emotion_data["label"], emotion_data["score"])

                                await websocket.send_json({
                                    "type": "analyzed",
                                    "text": transcript,
                                    "emotion": emotion_data["label"],
                                    "sentiment_category": sentiment,
                                    "confidence": emotion_data["score"]
                                })
                                
                                analyzed_sentences.add(transcript)

                            speech_buffer = []
                            is_speaking = False
                            silence_frames = 0
                            
                            await websocket.send_json({"type": "partial", "text": ""})

    except WebSocketDisconnect:
        print("Live Stream VAD: Client disconnected gracefully.")
    except Exception as e:
        print(f"Live Stream VAD Error: {e}")