import os
import shutil
import warnings
import site
import re
import time
import sys
import wave
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from transformers import pipeline

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
roberta_model = None

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
    global whisper_model, roberta_model
    whisper_model = WhisperModel("small", device="cuda", compute_type="int8_float16")
    roberta_model = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions")

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

    try:
        while True:
            chunk = await websocket.receive_bytes()
            raw_audio_buffer.extend(chunk)
            
            # Increased from 2.0 to 2.5 seconds to give Whisper slightly more context per loop
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
                        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', full_transcript) if s.strip()]
                        
                        partial_text = ""
                        sentence_completed = False # Track if we actually finished a sentence
                        
                        for sentence in sentences:
                            # CRITICAL FIX: Ensure the sentence actually ends with punctuation
                            if re.search(r'[.!?]$', sentence):
                                if sentence not in analyzed_sentences:
                                    emotion_data = predict_emotion(sentence)
                                    sentiment = categorize_emotion(emotion_data["label"], emotion_data["score"])
                                    
                                    await websocket.send_json({
                                        "type": "analyzed",
                                        "text": sentence,
                                        "emotion": emotion_data["label"],
                                        "sentiment_category": sentiment,
                                        "confidence": emotion_data["score"]
                                    })
                                    analyzed_sentences.add(sentence)
                                    sentence_completed = True
                            else:
                                partial_text = sentence

                        await websocket.send_json({
                            "type": "partial",
                            "text": partial_text
                        })

                        # ONLY clear the buffer if a complete sentence was found.
                        # Otherwise, let the buffer keep growing so the partial word isn't cut off.
                        if sentence_completed:
                            raw_audio_buffer.clear()

                except Exception as e:
                    pass 
                
                chunk_timer = time.time()
                
                # Fail-safe: If buffer gets over 10 seconds with no punctuation, clear it anyway to prevent lag
                if len(raw_audio_buffer) > (16000 * 2 * 10): 
                    raw_audio_buffer.clear()

    except WebSocketDisconnect:
        pass
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)