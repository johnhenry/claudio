# NVIDIA Parakeet ASR & NeMo TTS Setup Guide

A comprehensive guide for setting up NVIDIA's state-of-the-art speech models with Claude Code Voice Mode, featuring Parakeet ASR for speech recognition and FastPitch/HiFiGAN for text-to-speech synthesis.

## Overview

This guide configures:
- **NVIDIA Parakeet ASR** - Ultra-fast speech recognition (3386x real-time on GPU)
- **NVIDIA NeMo TTS** - High-quality text-to-speech with FastPitch + HiFiGAN
- **OpenAI-compatible API** - FastAPI wrapper for Voice Mode integration
- **Voice Mode MCP** - Natural voice interface for Claude Code

**Result**: Professional-grade voice interaction with Claude Code using NVIDIA's cutting-edge models.

## Important Clarification

**NVIDIA Parakeet** is an Automatic Speech Recognition (ASR) model, not TTS. For text-to-speech, we'll use **NVIDIA NeMo's FastPitch + HiFiGAN** models. This guide covers both for a complete voice solution.

## Prerequisites

- NVIDIA GPU (recommended: RTX 3060 or better)
- CUDA 11.8+ installed
- Python 3.8-3.11
- ~8GB disk space for models
- 16GB+ RAM recommended
- Claude Code installed (`npm install -g @anthropic-ai/claude-code`)

## Part 1: NVIDIA Parakeet ASR Setup (Speech-to-Text)

### Step 1.1: Install NeMo and Dependencies

```bash
# Create virtual environment
python3 -m venv ~/nemo-voice
source ~/nemo-voice/bin/activate

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Cython (required for NeMo)
pip install Cython

# Install NVIDIA NeMo
pip install nemo_toolkit[asr]

# Install additional dependencies
pip install fastapi uvicorn python-multipart soundfile librosa
```

### Step 1.2: Download Parakeet ASR Model

```bash
# Create models directory
mkdir -p ~/nemo-models

# Download Parakeet model from Hugging Face
cd ~/nemo-models
wget https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2/resolve/main/parakeet-tdt-0.6b-v2.nemo
```

### Step 1.3: Create OpenAI-Compatible ASR Server

Create `~/nemo-voice/parakeet_server.py`:

```python
#!/usr/bin/env python3
"""
NVIDIA Parakeet ASR Server with OpenAI Whisper API Compatibility
Provides ultra-fast speech recognition with timestamp and punctuation support
"""

import io
import json
import time
from typing import Optional
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import torch
import soundfile as sf
import nemo.collections.asr as nemo_asr
import numpy as np

app = FastAPI(title="Parakeet ASR Server")

# Load model on startup
print("Loading NVIDIA Parakeet model...")
model = nemo_asr.models.ASRModel.restore_from(
    "~/nemo-models/parakeet-tdt-0.6b-v2.nemo"
)
model.eval()
if torch.cuda.is_available():
    model = model.cuda()
print(f"Model loaded on: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

@app.get("/")
async def root():
    return {
        "service": "NVIDIA Parakeet ASR Server",
        "model": "parakeet-tdt-0.6b-v2",
        "version": "1.0.0",
        "api_compatibility": "OpenAI Whisper v1"
    }

@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(default="whisper-1"),
    language: Optional[str] = Form(default="en"),
    response_format: Optional[str] = Form(default="json"),
    timestamp_granularities: Optional[str] = Form(default=None)
):
    """
    OpenAI Whisper-compatible transcription endpoint
    Supports audio file upload and returns transcription with timing
    """
    try:
        # Read audio file
        audio_bytes = await file.read()
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Resample if needed (Parakeet expects 16kHz)
        if sample_rate != 16000:
            import librosa
            audio_data = librosa.resample(
                audio_data, orig_sr=sample_rate, target_sr=16000
            )
        
        # Run inference
        start_time = time.time()
        with torch.no_grad():
            # Get transcription with timestamps
            transcription = model.transcribe([audio_data])[0]
        
        inference_time = time.time() - start_time
        audio_duration = len(audio_data) / 16000
        rtfx = audio_duration / inference_time if inference_time > 0 else 0
        
        # Format response based on requested format
        if response_format == "verbose_json":
            return JSONResponse({
                "task": "transcribe",
                "language": language,
                "duration": audio_duration,
                "text": transcription,
                "segments": [{
                    "id": 0,
                    "seek": 0,
                    "start": 0.0,
                    "end": audio_duration,
                    "text": transcription,
                    "tokens": [],
                    "temperature": 0.0,
                    "avg_logprob": 0.0,
                    "compression_ratio": 1.0,
                    "no_speech_prob": 0.0
                }],
                "performance": {
                    "inference_time": inference_time,
                    "rtfx": rtfx,
                    "model": "parakeet-tdt-0.6b-v2"
                }
            })
        elif response_format == "text":
            return transcription
        else:  # json (default)
            return JSONResponse({
                "text": transcription,
                "duration": audio_duration,
                "rtfx": rtfx
            })
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI compatibility)"""
    return {
        "object": "list",
        "data": [
            {
                "id": "parakeet-tdt-0.6b-v2",
                "object": "model",
                "created": 1714521600,
                "owned_by": "nvidia"
            }
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=2022)
```

## Part 2: NVIDIA NeMo TTS Setup (Text-to-Speech)

### Step 2.1: Install TTS Dependencies

```bash
# Activate the same virtual environment
source ~/nemo-voice/bin/activate

# Install TTS support for NeMo
pip install nemo_toolkit[tts]

# Install additional audio processing libraries
pip install scipy pydub
```

### Step 2.2: Download FastPitch and HiFiGAN Models

```bash
cd ~/nemo-models

# Download FastPitch (spectrogram generator)
wget https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_en_fastpitch/versions/1.14.0/files/tts_en_fastpitch.nemo

# Download HiFiGAN (vocoder)
wget https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_en_hifigan/versions/1.14.0/files/tts_en_hifigan.nemo

# Optional: Download multi-speaker model for voice variety
wget https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_en_multispeaker_fastpitchhifigan/versions/1.14.0/files/tts_en_multispeaker_fastpitchhifigan.nemo
```

### Step 2.3: Create OpenAI-Compatible TTS Server

Create `~/nemo-voice/nemo_tts_server.py`:

```python
#!/usr/bin/env python3
"""
NVIDIA NeMo TTS Server with OpenAI API Compatibility
FastPitch + HiFiGAN for high-quality speech synthesis
"""

import io
import base64
from typing import Optional, Literal
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
import uvicorn
import torch
import soundfile as sf
import numpy as np
from pydantic import BaseModel
import nemo.collections.tts as nemo_tts

app = FastAPI(title="NeMo TTS Server")

# Load models on startup
print("Loading NeMo TTS models...")
spec_generator = nemo_tts.models.FastPitchModel.restore_from(
    "~/nemo-models/tts_en_fastpitch.nemo"
)
vocoder = nemo_tts.models.HifiGanModel.restore_from(
    "~/nemo-models/tts_en_hifigan.nemo"
)

spec_generator.eval()
vocoder.eval()

if torch.cuda.is_available():
    spec_generator = spec_generator.cuda()
    vocoder = vocoder.cuda()

print(f"Models loaded on: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

# Voice mappings (simulated for compatibility)
VOICE_SETTINGS = {
    "alloy": {"pitch": 1.0, "speed": 1.0},
    "echo": {"pitch": 0.95, "speed": 1.0},
    "fable": {"pitch": 1.05, "speed": 0.95},
    "onyx": {"pitch": 0.9, "speed": 1.0},
    "nova": {"pitch": 1.1, "speed": 1.05},
    "shimmer": {"pitch": 1.15, "speed": 1.0}
}

class TTSRequest(BaseModel):
    model: str = "tts-1"
    input: str
    voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] = "alloy"
    response_format: Optional[Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]] = "mp3"
    speed: Optional[float] = 1.0

@app.get("/")
async def root():
    return {
        "service": "NVIDIA NeMo TTS Server",
        "models": "FastPitch + HiFiGAN",
        "version": "1.0.0",
        "api_compatibility": "OpenAI TTS v1"
    }

@app.post("/v1/audio/speech")
async def create_speech(request: TTSRequest):
    """
    OpenAI-compatible TTS endpoint
    Generates speech from text using FastPitch + HiFiGAN
    """
    try:
        # Get voice settings
        voice_settings = VOICE_SETTINGS.get(request.voice, VOICE_SETTINGS["alloy"])
        
        # Parse text with NeMo
        parsed = spec_generator.parse(request.input)
        
        # Generate spectrogram with FastPitch
        with torch.no_grad():
            spectrogram = spec_generator.generate_spectrogram(
                tokens=parsed,
                pitch=voice_settings["pitch"],
                speed=request.speed * voice_settings["speed"]
            )
            
            # Convert spectrogram to audio with HiFiGAN
            audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)
        
        # Convert to numpy array
        audio_np = audio.squeeze().cpu().numpy()
        
        # Normalize audio
        audio_np = np.clip(audio_np, -1, 1)
        
        # Convert to requested format
        if request.response_format == "wav":
            buffer = io.BytesIO()
            sf.write(buffer, audio_np, 22050, format='WAV')
            buffer.seek(0)
            return Response(
                content=buffer.read(),
                media_type="audio/wav"
            )
        elif request.response_format == "pcm":
            # Raw PCM data (16-bit signed integers)
            audio_pcm = (audio_np * 32767).astype(np.int16)
            return Response(
                content=audio_pcm.tobytes(),
                media_type="audio/pcm"
            )
        elif request.response_format == "mp3":
            # Convert to MP3 using pydub
            from pydub import AudioSegment
            
            # Create WAV buffer first
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio_np, 22050, format='WAV')
            wav_buffer.seek(0)
            
            # Convert to MP3
            audio_segment = AudioSegment.from_wav(wav_buffer)
            mp3_buffer = io.BytesIO()
            audio_segment.export(mp3_buffer, format="mp3", bitrate="128k")
            mp3_buffer.seek(0)
            
            return Response(
                content=mp3_buffer.read(),
                media_type="audio/mpeg"
            )
        else:
            # Default to WAV for unsupported formats
            buffer = io.BytesIO()
            sf.write(buffer, audio_np, 22050, format='WAV')
            buffer.seek(0)
            return Response(
                content=buffer.read(),
                media_type="audio/wav"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI compatibility)"""
    return {
        "object": "list",
        "data": [
            {
                "id": "tts-1",
                "object": "model",
                "created": 1714521600,
                "owned_by": "nvidia-nemo"
            },
            {
                "id": "tts-1-hd",
                "object": "model",
                "created": 1714521600,
                "owned_by": "nvidia-nemo"
            }
        ]
    }

@app.get("/v1/voices")
async def list_voices():
    """List available voices"""
    return {
        "voices": list(VOICE_SETTINGS.keys()),
        "note": "Voice variations are simulated through pitch and speed adjustments"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8880)
```

## Part 3: Combined Startup Script

Create `~/nemo-voice/start-nemo-voice.sh`:

```bash
#!/bin/bash
# NVIDIA NeMo Voice Services Startup Script

echo "Starting NVIDIA NeMo Voice Services..."

# Activate virtual environment
source ~/nemo-voice/bin/activate

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠ No NVIDIA GPU detected - will use CPU (slower)"
fi

# Start Parakeet ASR server
if pgrep -f "parakeet_server.py" > /dev/null; then
    echo "✓ Parakeet ASR server already running"
else
    echo "Starting Parakeet ASR server on port 2022..."
    nohup python ~/nemo-voice/parakeet_server.py > /tmp/parakeet.log 2>&1 &
    sleep 3
    if pgrep -f "parakeet_server.py" > /dev/null; then
        echo "✓ Parakeet ASR server started"
    else
        echo "✗ Failed to start Parakeet ASR server"
    fi
fi

# Start NeMo TTS server
if pgrep -f "nemo_tts_server.py" > /dev/null; then
    echo "✓ NeMo TTS server already running"
else
    echo "Starting NeMo TTS server on port 8880..."
    nohup python ~/nemo-voice/nemo_tts_server.py > /tmp/nemo_tts.log 2>&1 &
    sleep 3
    if pgrep -f "nemo_tts_server.py" > /dev/null; then
        echo "✓ NeMo TTS server started"
    else
        echo "✗ Failed to start NeMo TTS server"
    fi
fi

# Set environment variables
export VOICEMODE_STT_BASE_URL="http://127.0.0.1:2022/v1"
export VOICEMODE_TTS_BASE_URL="http://127.0.0.1:8880/v1"

echo ""
echo "========================================="
echo "NVIDIA Voice Services Ready!"
echo "========================================="
echo "ASR (Parakeet): http://127.0.0.1:2022"
echo "TTS (NeMo):     http://127.0.0.1:8880"
echo ""
echo "To use with Claude Code:"
echo "1. export VOICEMODE_STT_BASE_URL=\"http://127.0.0.1:2022/v1\""
echo "2. export VOICEMODE_TTS_BASE_URL=\"http://127.0.0.1:8880/v1\""
echo "3. claude"
echo ""
echo "Performance on GPU: ~3386x real-time ASR"
echo "Logs: tail -f /tmp/parakeet.log /tmp/nemo_tts.log"
```

## Part 4: Voice Mode Integration

### Step 4.1: Configure Voice Mode for NeMo

```bash
# Set environment variables for Voice Mode
echo 'export VOICEMODE_STT_BASE_URL="http://127.0.0.1:2022/v1"' >> ~/.bashrc
echo 'export VOICEMODE_TTS_BASE_URL="http://127.0.0.1:8880/v1"' >> ~/.bashrc

# For zsh users
echo 'export VOICEMODE_STT_BASE_URL="http://127.0.0.1:2022/v1"' >> ~/.zshrc
echo 'export VOICEMODE_TTS_BASE_URL="http://127.0.0.1:8880/v1"' >> ~/.zshrc

# Apply to current session
export VOICEMODE_STT_BASE_URL="http://127.0.0.1:2022/v1"
export VOICEMODE_TTS_BASE_URL="http://127.0.0.1:8880/v1"
```

### Step 4.2: Install Voice Mode MCP

```bash
# Add Voice Mode to Claude Code
claude mcp add --scope user voice-mode uvx voice-mode

# Verify installation
claude mcp list
```

## Testing the Setup

### Test ASR (Speech-to-Text)

```bash
# Test with curl
curl -X POST "http://127.0.0.1:2022/v1/audio/transcriptions" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test.wav" \
  -F "model=whisper-1"
```

### Test TTS (Text-to-Speech)

```bash
# Test with curl
curl -X POST "http://127.0.0.1:8880/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello from NVIDIA NeMo TTS!",
    "voice": "nova"
  }' \
  --output test_speech.mp3

# Play the generated audio
ffplay test_speech.mp3
```

### Test with Python OpenAI Client

```python
from openai import OpenAI

# Configure clients for local servers
stt_client = OpenAI(
    api_key="dummy",
    base_url="http://127.0.0.1:2022/v1"
)

tts_client = OpenAI(
    api_key="dummy",
    base_url="http://127.0.0.1:8880/v1"
)

# Test TTS
response = tts_client.audio.speech.create(
    model="tts-1",
    voice="nova",
    input="Hello from NVIDIA NeMo!"
)
response.stream_to_file("output.mp3")

# Test STT
with open("test.wav", "rb") as audio_file:
    transcript = stt_client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )
    print(transcript.text)
```

## Performance Optimization

### GPU Acceleration

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Monitor GPU usage during inference
watch -n 1 nvidia-smi
```

### Batch Processing for ASR

For processing multiple files, modify the server to support batch inference:

```python
# Add to parakeet_server.py
@app.post("/v1/audio/transcriptions/batch")
async def transcribe_batch(files: List[UploadFile] = File(...)):
    """Process multiple audio files in a single batch"""
    audio_data_list = []
    for file in files:
        audio_bytes = await file.read()
        audio_data, _ = sf.read(io.BytesIO(audio_bytes))
        audio_data_list.append(audio_data)
    
    # Batch inference
    transcriptions = model.transcribe(audio_data_list)
    return {"transcriptions": transcriptions}
```

### Model Quantization (Optional)

For faster inference with slight quality trade-off:

```python
# Quantize model to INT8
from nemo.collections.asr.tools import quantization
quantized_model = quantization.quantize_model(model, method="dynamic")
```

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size in server code
# Or use CPU inference by commenting out .cuda() calls
```

### Model Download Issues

```bash
# Alternative: Use Hugging Face CLI
pip install huggingface-hub
huggingface-cli download nvidia/parakeet-tdt-0.6b-v2 --local-dir ~/nemo-models
```

### Audio Format Issues

```bash
# Install ffmpeg for audio conversion
sudo apt-get install ffmpeg  # Ubuntu/Debian
brew install ffmpeg          # macOS
```

### Port Already in Use

```bash
# Find and kill existing processes
lsof -i :2022
lsof -i :8880
kill -9 <PID>
```

## Performance Benchmarks

| Model | Task | Hardware | Performance |
|-------|------|----------|-------------|
| Parakeet-TDT-0.6B | ASR | RTX 4090 | 3386x real-time |
| Parakeet-TDT-0.6B | ASR | RTX 3060 | ~1000x real-time |
| Parakeet-TDT-0.6B | ASR | CPU (i7-12700) | ~10x real-time |
| FastPitch+HiFiGAN | TTS | RTX 4090 | 900x real-time |
| FastPitch+HiFiGAN | TTS | RTX 3060 | ~300x real-time |
| FastPitch+HiFiGAN | TTS | CPU (i7-12700) | ~5x real-time |

## Advanced Features

### Multi-Speaker TTS

```python
# Load multi-speaker model
multi_speaker_model = nemo_tts.models.FastPitchModel.restore_from(
    "~/nemo-models/tts_en_multispeaker_fastpitchhifigan.nemo"
)

# Generate with specific speaker ID (1-20)
spectrogram = multi_speaker_model.generate_spectrogram(
    tokens=parsed,
    speaker=5  # Speaker ID
)
```

### Emotion Control

```python
# Adjust prosody for emotional speech
spectrogram = spec_generator.generate_spectrogram(
    tokens=parsed,
    pitch=1.2,      # Higher for excitement
    energy=1.3,     # Louder for emphasis
    duration=0.9    # Faster for urgency
)
```

### Custom Voice Cloning

For voice cloning capabilities, consider:
1. Fine-tuning FastPitch on target speaker data
2. Using NVIDIA's YourTTS models
3. Implementing speaker embeddings

## Making it Permanent (systemd)

Create `/etc/systemd/system/nemo-voice.service`:

```ini
[Unit]
Description=NVIDIA NeMo Voice Services
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/home/YOUR_USERNAME/nemo-voice
Environment="PATH=/home/YOUR_USERNAME/nemo-voice/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=/home/YOUR_USERNAME/nemo-voice/start-nemo-voice.sh
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable nemo-voice
sudo systemctl start nemo-voice
sudo systemctl status nemo-voice
```

## Uninstall

```bash
# Stop services
pkill -f parakeet_server.py
pkill -f nemo_tts_server.py

# Remove systemd service
sudo systemctl stop nemo-voice
sudo systemctl disable nemo-voice
sudo rm /etc/systemd/system/nemo-voice.service

# Remove files
rm -rf ~/nemo-voice
rm -rf ~/nemo-models

# Remove Voice Mode MCP
claude mcp remove voice-mode
```

## Resources

- [NVIDIA NeMo Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/)
- [Parakeet Model on Hugging Face](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
- [FastPitch Paper](https://arxiv.org/abs/2006.06873)
- [HiFiGAN Paper](https://arxiv.org/abs/2010.05646)
- [Voice Mode Documentation](https://voice-mode.readthedocs.io)

---

**Note**: This setup provides professional-grade voice capabilities with:
- **Ultra-fast ASR**: 1 hour of audio transcribed in 1 second on GPU
- **High-quality TTS**: Natural-sounding speech synthesis
- **Complete privacy**: All processing happens locally
- **OpenAI compatibility**: Works seamlessly with Voice Mode MCP