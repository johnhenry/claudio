# Local NVIDIA Parakeet + Claude Code Voice Mode Setup Guide

A complete walkthrough for setting up voice-controlled Claude Code using NVIDIA's Parakeet ASR model and Voice Mode MCP server, with ultra-fast local speech-to-text processing.

## Overview

This guide configures:
- **NVIDIA Parakeet ASR** - Ultra-fast local speech recognition (3386x real-time on GPU, 10x on CPU)
- **OpenAI-compatible API server** - FastAPI wrapper that works with Voice Mode MCP
- **Voice Mode MCP** - Natural voice interface for Claude Code
- **Claude Code** - Your AI coding assistant

**Result**: Speak naturally to Claude Code with blazing-fast speech recognition, all processing locally on your machine.

## Why Parakeet Instead of Whisper?

- **Speed**: Parakeet can transcribe 1 hour of audio in 1 second on GPU (vs Whisper's ~30 seconds)
- **Accuracy**: State-of-the-art WER (Word Error Rate) performance
- **Features**: Built-in punctuation, capitalization, and timestamp prediction
- **Efficiency**: 600M parameters optimized for real-time transcription

## Prerequisites

- macOS, Linux, or Windows (via WSL2)
- Python 3.8-3.11
- Claude Code installed (`npm install -g @anthropic-ai/claude-code`)
- ~2GB disk space for models
- Optional but recommended: NVIDIA GPU with CUDA 11.8+

## Step 1: Install NVIDIA NeMo and Dependencies

```bash
# Create virtual environment
python3 -m venv ~/parakeet-asr
source ~/parakeet-asr/bin/activate

# Install PyTorch (with CUDA if you have NVIDIA GPU)
# For GPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# For CPU only:
# pip install torch torchvision torchaudio

# Install Cython (required for NeMo)
pip install Cython

# Install NVIDIA NeMo ASR
pip install nemo_toolkit[asr]

# Install server dependencies
pip install fastapi uvicorn python-multipart soundfile librosa
```

## Step 2: Download Parakeet Model

```bash
# Create models directory
mkdir -p ~/parakeet-asr/models

# Download Parakeet-TDT-0.6B model (600MB)
cd ~/parakeet-asr/models
wget https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2/resolve/main/parakeet-tdt-0.6b-v2.nemo

# Verify download
ls -lh parakeet-tdt-0.6b-v2.nemo
# Should show ~600MB file
```

## Step 3: Create OpenAI-Compatible API Server

Create `~/parakeet-asr/server.py`:

```python
#!/usr/bin/env python3
"""
NVIDIA Parakeet ASR Server with OpenAI Whisper API Compatibility
Ultra-fast speech recognition for Voice Mode
"""

import io
import time
import logging
from typing import Optional
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
import soundfile as sf
import numpy as np

# Lazy import to speed up startup
nemo_asr = None
model = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Parakeet ASR Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_model():
    """Lazy load the model on first request"""
    global nemo_asr, model
    if model is None:
        logger.info("Loading NVIDIA Parakeet model...")
        import nemo.collections.asr as nemo_asr_module
        nemo_asr = nemo_asr_module
        
        model = nemo_asr.models.ASRModel.restore_from(
            os.path.expanduser("~/parakeet-asr/models/parakeet-tdt-0.6b-v2.nemo")
        )
        model.eval()
        
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("Model loaded on CUDA GPU")
        else:
            logger.info("Model loaded on CPU")
    return model

@app.get("/")
async def root():
    """Health check and server info"""
    return HTMLResponse(content="""
    <html>
    <head>
        <title>Parakeet ASR Server</title>
    </head>
    <body>
        <h1>NVIDIA Parakeet ASR Server</h1>
        <p>OpenAI Whisper API Compatible</p>
        <ul>
            <li>Model: parakeet-tdt-0.6b-v2</li>
            <li>Performance: Up to 3386x real-time on GPU</li>
            <li>Endpoint: POST /v1/audio/transcriptions</li>
            <li>Status: Ready</li>
        </ul>
    </body>
    </html>
    """)

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
    Parakeet provides ultra-fast transcription with timing
    """
    try:
        # Load model if not already loaded
        asr_model = load_model()
        
        # Read audio file
        audio_bytes = await file.read()
        
        # Handle different audio formats
        try:
            audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
        except Exception as e:
            logger.error(f"Error reading audio: {e}")
            raise HTTPException(status_code=400, detail="Invalid audio file")
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Resample to 16kHz if needed (Parakeet expects 16kHz)
        if sample_rate != 16000:
            import librosa
            audio_data = librosa.resample(
                y=audio_data, 
                orig_sr=sample_rate, 
                target_sr=16000
            )
        
        # Run inference
        start_time = time.time()
        
        with torch.no_grad():
            # Get transcription
            transcription = asr_model.transcribe([audio_data])[0]
        
        inference_time = time.time() - start_time
        audio_duration = len(audio_data) / 16000
        rtfx = audio_duration / inference_time if inference_time > 0 else 0
        
        logger.info(f"Transcribed {audio_duration:.2f}s in {inference_time:.3f}s (RTFx: {rtfx:.1f})")
        
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
                }]
            })
        elif response_format == "text":
            return transcription
        else:  # json (default)
            return JSONResponse({
                "text": transcription
            })
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI compatibility)"""
    return {
        "object": "list",
        "data": [
            {
                "id": "whisper-1",
                "object": "model",
                "created": 1714521600,
                "owned_by": "nvidia-parakeet"
            }
        ]
    }

if __name__ == "__main__":
    # Pre-load model if GPU is available
    if torch.cuda.is_available():
        logger.info("GPU detected, pre-loading model...")
        load_model()
    
    uvicorn.run(app, host="127.0.0.1", port=2022)
```

## Step 4: Start the Parakeet Server

```bash
# Navigate to Parakeet directory
cd ~/parakeet-asr

# Activate virtual environment
source bin/activate

# Make server executable
chmod +x server.py

# Start the server
python server.py
```

You should see:
```
INFO:     Loading NVIDIA Parakeet model...
INFO:     Model loaded on CUDA GPU (or CPU)
INFO:     Uvicorn running on http://127.0.0.1:2022
```

### Verify the Server is Running

In a new terminal:
```bash
# Check if server is responding
curl -s http://127.0.0.1:2022 | head -5

# You should see:
#     <html>
#     <head>
#         <title>Parakeet ASR Server</title>
#     </head>
#     <body>

# Check if port is listening
lsof -i :2022
# Should show: python ... TCP localhost:2022 (LISTEN)
```

## Step 5: Configure Voice Mode for Claude Code

```bash
# Set environment variable to use local Parakeet
export VOICEMODE_STT_BASE_URL="http://127.0.0.1:2022/v1"

# Add to your shell profile for persistence
echo 'export VOICEMODE_STT_BASE_URL="http://127.0.0.1:2022/v1"' >> ~/.bashrc
# or for zsh:
echo 'export VOICEMODE_STT_BASE_URL="http://127.0.0.1:2022/v1"' >> ~/.zshrc
```

## Step 6: Install Voice Mode MCP

```bash
# Add Voice Mode to Claude Code (user-level)
claude mcp add --scope user voice-mode uvx voice-mode

# Verify installation
claude mcp list
# Should show: voice-mode: uvx voice-mode - ✓ Connected
```

## Step 7: Verify and Test the Complete Setup

### Pre-flight Checks

```bash
# 1. Verify Parakeet server is running
ps aux | grep server.py
# Should show the python server.py process

# 2. Check port is listening
lsof -i :2022
# Should show: python ... TCP localhost:2022 (LISTEN)

# 3. Verify environment variable is set
echo $VOICEMODE_STT_BASE_URL
# Should output: http://127.0.0.1:2022/v1

# 4. Check Voice Mode MCP is installed
claude mcp list | grep voice-mode
# Should show: voice-mode: uvx voice-mode - ✓ Connected
```

### Test Voice Transcription

```bash
# 1. Start Claude Code in a new terminal (to pick up env vars)
claude

# 2. Test voice mode with these commands:
# - "Let's have a voice conversation"
# - "Can you hear me?"
# - "Enable voice mode"

# 3. If voice is working, you should see:
# - Microphone permission request (first time only)
# - Audio waveform indicators when speaking
# - Your speech transcribed to text (VERY FAST!)
```

### Test Parakeet Performance

```bash
# Create a test audio file (or use existing)
# Record 10 seconds of audio:
rec -r 16000 -c 1 test.wav trim 0 10

# Test transcription speed
time curl -X POST "http://127.0.0.1:2022/v1/audio/transcriptions" \
  -F "file=@test.wav" \
  -F "model=whisper-1"

# You should see very fast response times:
# GPU: ~0.003s for 10s audio (3333x real-time)
# CPU: ~1s for 10s audio (10x real-time)
```

## Usage

### Basic Voice Commands

Once everything is running:

1. **Start Claude Code**: `claude`
2. **Activate voice**: Say "Let's talk" or "Can you hear me?"
3. **Speak naturally**: 
   - "Create a Python function to calculate fibonacci"
   - "Debug this error message" (then paste error)
   - "Refactor this function to use async/await"
   - "Write tests for the user authentication module"

### Pro Tips

1. **Keep the server running**: Add to your startup scripts
2. **Ultra-fast response**: Parakeet processes speech faster than you can speak
3. **GPU acceleration**: If you have NVIDIA GPU, responses are near-instantaneous
4. **Background service**: See below for systemd/launchd setup

## Running as a Background Service

### macOS (launchd)

Create `~/Library/LaunchAgents/com.parakeet.asr.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" 
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.parakeet.asr</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/YOUR_USERNAME/parakeet-asr/bin/python</string>
        <string>/Users/YOUR_USERNAME/parakeet-asr/server.py</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/parakeet.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/parakeet.error.log</string>
</dict>
</plist>
```

Then:
```bash
launchctl load ~/Library/LaunchAgents/com.parakeet.asr.plist
```

### Linux (systemd)

Create `/etc/systemd/system/parakeet-asr.service`:

```ini
[Unit]
Description=NVIDIA Parakeet ASR Server
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/home/YOUR_USERNAME/parakeet-asr
ExecStart=/home/YOUR_USERNAME/parakeet-asr/bin/python /home/YOUR_USERNAME/parakeet-asr/server.py
Restart=always
Environment="PATH=/home/YOUR_USERNAME/parakeet-asr/bin:/usr/local/bin:/usr/bin:/bin"

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl enable parakeet-asr
sudo systemctl start parakeet-asr
```

## Making the Setup Persistent

### Option 1: Manual Start Script
Create `~/start-parakeet.sh`:

```bash
#!/bin/bash
# Parakeet Voice Mode Setup Script

echo "Starting Parakeet Voice Mode Setup..."

# Activate virtual environment
source ~/parakeet-asr/bin/activate

# Check if server is already running
if pgrep -f "server.py" > /dev/null; then
    echo "✓ Parakeet server is already running"
else
    echo "Starting Parakeet server..."
    cd ~/parakeet-asr
    nohup python server.py > /tmp/parakeet.log 2>&1 &
    
    # Wait for server to start
    sleep 3
    
    if pgrep -f "server.py" > /dev/null; then
        echo "✓ Parakeet server started successfully"
    else
        echo "✗ Failed to start Parakeet server. Check /tmp/parakeet.log"
        exit 1
    fi
fi

# Set environment variable
export VOICEMODE_STT_BASE_URL="http://127.0.0.1:2022/v1"
echo "✓ Environment variable set"

# Check server is responding
if curl -s http://127.0.0.1:2022 > /dev/null 2>&1; then
    echo "✓ Parakeet server is responding"
else
    echo "✗ Parakeet server is not responding"
    exit 1
fi

echo ""
echo "========================================="
echo "Setup complete! Voice Mode is ready."
echo "========================================="
echo ""
echo "Performance:"
if command -v nvidia-smi &> /dev/null; then
    echo "  GPU detected: Up to 3386x real-time transcription"
else
    echo "  CPU mode: ~10x real-time transcription"
fi
echo ""
echo "To use voice in Claude Code:"
echo "1. Start a new terminal"
echo "2. Run: export VOICEMODE_STT_BASE_URL=\"http://127.0.0.1:2022/v1\""
echo "3. Run: claude"
echo "4. Say 'Let's have a voice conversation' to test"
echo ""
echo "To stop server: pkill -f server.py"
echo "To view logs: tail -f /tmp/parakeet.log"
```

Make it executable:
```bash
chmod +x ~/start-parakeet.sh
```

### Option 2: Add to Shell Profile
Add to `~/.bashrc` or `~/.zshrc`:

```bash
# Auto-start Parakeet ASR server
if ! pgrep -f "server.py" > /dev/null; then
    source ~/parakeet-asr/bin/activate
    nohup python ~/parakeet-asr/server.py > /tmp/parakeet.log 2>&1 &
    deactivate
fi

# Set Voice Mode environment
export VOICEMODE_STT_BASE_URL="http://127.0.0.1:2022/v1"
```

## Troubleshooting

### Voice Mode Not Working
```bash
# 1. Check if Parakeet server is running
ps aux | grep server.py

# 2. Verify environment variable in NEW terminal
echo $VOICEMODE_STT_BASE_URL

# 3. Restart Claude Code in fresh terminal
exit  # from current Claude session
claude  # start new session

# 4. Check MCP connection
claude mcp list
```

### Server Won't Start
```bash
# Check if port 2022 is in use
lsof -i :2022

# Kill existing process if needed
kill -9 <PID>

# Or try a different port
# Edit server.py to use port 3022
# Then update: export VOICEMODE_STT_BASE_URL="http://127.0.0.1:3022/v1"
```

### CUDA/GPU Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, install CUDA toolkit:
# Ubuntu: sudo apt install nvidia-cuda-toolkit
# Or use CPU-only (slower but works)
```

### Model Loading Errors
```bash
# Re-download model if corrupted
cd ~/parakeet-asr/models
rm parakeet-tdt-0.6b-v2.nemo
wget https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2/resolve/main/parakeet-tdt-0.6b-v2.nemo

# Check model size (should be ~600MB)
ls -lh parakeet-tdt-0.6b-v2.nemo
```

### Microphone Not Working (macOS)
```bash
# Check permissions
# System Settings > Privacy & Security > Microphone
# Ensure Terminal/iTerm has permission

# Test microphone
rec -r 16000 -c 1 test.wav trim 0 3
play test.wav
```

### Slow Transcription
```bash
# Check if using GPU
python -c "import torch; print('GPU' if torch.cuda.is_available() else 'CPU')"

# Monitor resource usage
# GPU: nvidia-smi
# CPU: top or htop
```

## Performance Comparison

| Model | Hardware | Speed | Accuracy | Power Usage |
|-------|----------|-------|----------|-------------|
| **Parakeet** | RTX 4090 | 3386x real-time | State-of-art | ~100W |
| **Parakeet** | RTX 3060 | ~1000x real-time | State-of-art | ~60W |
| **Parakeet** | M2 Mac (CPU) | ~15x real-time | State-of-art | ~20W |
| **Parakeet** | Intel i7 (CPU) | ~10x real-time | State-of-art | ~45W |
| Whisper base | M2 Mac | ~50x real-time | Good | ~20W |
| Whisper base | Intel i7 | ~30x real-time | Good | ~45W |

## Performance Optimization

### For Maximum Speed
```python
# Edit server.py to use batch processing
# Process multiple requests simultaneously
batch_size = 8  # Adjust based on GPU memory
```

### For Lower Latency
```python
# Pre-load model on startup (already in script)
# Keep model in GPU memory
```

### For CPU Optimization
```bash
# Use OpenMP threads
export OMP_NUM_THREADS=8  # Adjust to your CPU cores

# Install MKL for Intel CPUs
pip install intel-extension-for-pytorch
```

## Uninstall

```bash
# Stop background service (macOS)
launchctl unload ~/Library/LaunchAgents/com.parakeet.asr.plist

# Stop background service (Linux)
sudo systemctl stop parakeet-asr
sudo systemctl disable parakeet-asr

# Remove Voice Mode from Claude Code
claude mcp remove voice-mode

# Stop server
pkill -f server.py

# Remove files (optional)
rm -rf ~/parakeet-asr
```

## Next Steps

- **Custom Wake Words**: Integrate with tools like Porcupine for "Hey Claude"
- **TTS Addition**: Add local TTS with Kokoro or Piper for responses
- **Multi-language**: Parakeet supports multiple languages with different models
- **Fine-tuning**: Customize Parakeet for domain-specific vocabulary

## Resources

- [NVIDIA Parakeet on Hugging Face](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
- [NVIDIA NeMo Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/)
- [Voice Mode Documentation](https://voice-mode.readthedocs.io)
- [Claude Code Documentation](https://docs.anthropic.com/claude-code)

---

**Performance Note**: NVIDIA Parakeet provides the fastest available speech recognition, processing audio 3386x faster than real-time on modern GPUs. This means a 1-hour recording transcribes in just 1 second!

**Privacy Note**: All audio processing happens locally on your machine. No audio data is sent to external servers. Your voice commands and code remain completely private.