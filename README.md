# Live Emotion Matrix: Whisper-RoBERTa Pipeline

A high-performance, dual-modal microservice architecture that captures live audio streams, transcribes them using OpenAI's **Whisper (CUDA-accelerated)**, and performs real-time sentence-boundary sentiment analysis using a fine-tuned **RoBERTa** model.

## Features

- **Real-Time WebSocket Streaming:** Bi-directional communication capturing 16kHz raw PCM audio from the browser.
- **Sliding Window Buffer:** Advanced audio chunking logic to ensure sentences are never cut off mid-speech.
- **Dual-State UI:** Sleek React frontend displaying live partial text alongside finalized, color-coded sentiment blocks.
- **CUDA Optimization:** Dynamic DLL injection for Windows environments to guarantee GPU utilization for inference speeds under 200ms per chunk.
- **Sentence Boundary Detection:** Intelligent parsing that waits for complete thoughts before triggering the RoBERTa classification model.

---

## Architecture Overview

The project is split into two distinct environments:

1. **`/server` (Backend):** A FastAPI microservice running Uvicorn and managing the WebSocket tunnels and PyTorch models.
2. **`/ui` (Frontend):** A Next.js / React application handling microphone permissions, raw audio extraction (AudioContext API), and UI rendering.

---

## Setup & Installation

### Prerequisites

- **NVIDIA GPU** (Highly Recommended)
- **CUDA Toolkit 12.x** installed on your system.
- **Python 3.10+**
- **Node.js 18+**

### 1. Backend Setup (`/server`)

Navigate to the server directory:

```bash
cd server
```

Create and activate a virtual environment:

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

Install the required Python dependencies:

```bash
pip install fastapi uvicorn websockets faster-whisper transformers soundfile pydub
```

> **Note:** If you encounter CUDA pathing issues on Windows, the `emotion_router.py` script automatically attempts to inject the `cublas` and `cudnn` DLL paths from your virtual environment's `site-packages`.

### 2. Frontend Setup (`/ui`)

Open a new terminal window and navigate to the UI directory:

```bash
cd ui
```

Install the Node modules:

```bash
npm install
```

---

## Running the Application

You must run both the backend and frontend simultaneously in separate terminals.

### 1. Start the FastAPI Engine

In your `/server` terminal (ensure your virtual environment is activated):

```bash
python -m uvicorn emotion_router:app --port 8001 --reload
```

Wait until you see:

```text
Models Loaded. Ready for Live Stream.
```

### 2. Start the React UI

In your `/ui` terminal:

```bash
npm run dev
```

### 3. Initialize

Open your browser and navigate to:

```text
http://localhost:3000/model
```

Click **"Initialize Live Microphone"**, grant browser permissions, and begin speaking.

---

## Troubleshooting

### Error: `Library cublas64_12.dll is not found`

Ensure you are running the FastAPI server from inside the activated virtual environment. The script uses `sys.prefix` to dynamically locate the CUDA libraries installed via `pip`.

### Error: `Invalid data found when processing input`

This occurs if the frontend sends WebM headers instead of raw PCM data. Ensure your frontend is utilizing the **AudioContext API** (`Int16Array` conversion) as defined in `page.tsx`, rather than standard `MediaRecorder` blobs.

### Text hangs in the **"Listening..."** indicator indefinitely

Whisper relies on detecting pauses or inflections to generate punctuation (`.`, `!`, `?`). If you trail off, the **3.0-second VAD (Voice Activity Detection)** timeout in `emotion_router.py` will force the text through RoBERTa automatically.
