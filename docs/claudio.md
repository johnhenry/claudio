# Claudio: Complete Voice Mode Setup Guide for Claude Code

A comprehensive guide for setting up voice-controlled Claude Code with multiple speech recognition and text-to-speech options, from simple cloud-based solutions to fully local, privacy-focused configurations.

## 🎯 Quick Decision Guide

Choose your setup based on your priorities:

| Setup | Speech-to-Text | Text-to-Speech | Best For |
|-------|---------------|----------------|----------|
| **[Quick Start](#quick-start-cloud-based)** | OpenAI Whisper API | OpenAI TTS API | Fastest setup, pay-per-use |
| **[Whisper Local](#option-1-whisper-local-stt)** | whisper.cpp (local) | OpenAI TTS API | Privacy for input, easy setup |
| **[Parakeet Pro](#option-2-nvidia-parakeet-stt)** | NVIDIA Parakeet (local) | OpenAI TTS API | Fastest STT, NVIDIA GPU users |
| **[Full Local](#option-3-full-local-stack)** | Whisper/Parakeet | Kokoro/NeMo TTS | Complete privacy, no API costs |
| **[Full NVIDIA](#option-4-full-nvidia-stack)** | Parakeet ASR | NeMo TTS | Professional grade, GPU optimized |

## 📋 Prerequisites

### Required for All Setups
- Claude Code installed (`npm install -g @anthropic-ai/claude-code`)
- Voice Mode MCP installed (covered in setup)
- Microphone access

### Platform-Specific
- **macOS**: Xcode Command Line Tools
- **Linux**: build-essential, cmake
- **Windows**: WSL2 recommended

### Hardware Recommendations
- **Minimum**: 8GB RAM, 4GB disk space
- **Recommended**: 16GB RAM, 10GB disk space
- **GPU Users**: NVIDIA GPU with CUDA 11.8+ for maximum performance

## 🚀 Quick Start (Cloud-Based)

The simplest setup using OpenAI's APIs for both STT and TTS.

```bash
# 1. Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# 2. Install Voice Mode MCP
claude mcp add --scope user voice-mode uvx voice-mode

# 3. Start Claude Code
claude

# 4. Test voice
# Say "Let's have a voice conversation"
```

**Pros**: Zero setup, high quality
**Cons**: Requires API key, costs ~$0.15/minute, data goes to cloud

## 🎤 Speech-to-Text Options

### Option 1: Whisper Local STT

Local speech recognition using whisper.cpp - good balance of speed and accuracy.

<details>
<summary><b>Click to expand Whisper setup instructions</b></summary>

#### Step 1.1: Install whisper.cpp

```bash
# Clone and build
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
make

# Download model (choose one)
bash ./models/download-ggml-model.sh base.en    # 140MB, recommended
# bash ./models/download-ggml-model.sh small.en  # 466MB, more accurate
# bash ./models/download-ggml-model.sh tiny.en   # 39MB, fastest
```

#### Step 1.2: Start Whisper Server

```bash
cd ~/whisper.cpp
./build/bin/whisper-server \
  -m models/ggml-base.en.bin \
  --host 127.0.0.1 \
  --port 2022
```

#### Step 1.3: Configure Voice Mode

```bash
export VOICEMODE_STT_BASE_URL="http://127.0.0.1:2022/v1"
```

**Performance**: ~50x real-time on Apple Silicon, ~30x on Intel

</details>

### Option 2: NVIDIA Parakeet STT

Ultra-fast speech recognition using NVIDIA's state-of-the-art ASR model.

<details>
<summary><b>Click to expand Parakeet setup instructions</b></summary>

#### Step 2.1: Create Python Environment

```bash
python3 -m venv ~/parakeet-asr
source ~/parakeet-asr/bin/activate
```

#### Step 2.2: Install Dependencies

```bash
# PyTorch (choose based on your system)
pip install torch torchvision torchaudio  # CPU/Mac
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # NVIDIA GPU

# NeMo and server deps
pip install Cython
pip install "nemo_toolkit[asr]"
pip install fastapi uvicorn python-multipart soundfile librosa
```

#### Step 2.3: Download Parakeet Model

```bash
mkdir -p ~/parakeet-asr/models
cd ~/parakeet-asr/models
wget https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2/resolve/main/parakeet-tdt-0.6b-v2.nemo
```

#### Step 2.4: Create Server

Create `~/parakeet-asr/server.py`:

```python
#!/usr/bin/env python3
import io, os, time, logging
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn, torch, soundfile as sf, numpy as np

nemo_asr = None
model = None
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Parakeet ASR Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, 
                   allow_methods=["*"], allow_headers=["*"])

def load_model():
    global nemo_asr, model
    if model is None:
        logger.info("Loading NVIDIA Parakeet model...")
        import nemo.collections.asr as nemo_asr_module
        nemo_asr = nemo_asr_module
        model_path = os.path.expanduser("~/parakeet-asr/models/parakeet-tdt-0.6b-v2.nemo")
        model = nemo_asr.models.ASRModel.restore_from(model_path)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("Model loaded on CUDA GPU")
        else:
            logger.info("Model loaded on CPU")
    return model

@app.get("/")
async def root():
    return HTMLResponse("""<h1>Parakeet ASR Server</h1><p>Ready</p>""")

@app.post("/v1/audio/transcriptions")
async def transcribe(file: UploadFile = File(...), model: str = Form(default="whisper-1"),
                     language: str = Form(default="en"), response_format: str = Form(default="json")):
    try:
        asr_model = load_model()
        audio_bytes = await file.read()
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
        
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        if sample_rate != 16000:
            import librosa
            audio_data = librosa.resample(y=audio_data, orig_sr=sample_rate, target_sr=16000)
        
        start_time = time.time()
        with torch.no_grad():
            transcription = asr_model.transcribe([audio_data])[0]
        
        inference_time = time.time() - start_time
        audio_duration = len(audio_data) / 16000
        rtfx = audio_duration / inference_time if inference_time > 0 else 0
        
        logger.info(f"Transcribed {audio_duration:.2f}s in {inference_time:.3f}s (RTFx: {rtfx:.1f})")
        
        if response_format == "text":
            return transcription
        else:
            return JSONResponse({"text": transcription})
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    if torch.cuda.is_available():
        load_model()
    uvicorn.run(app, host="127.0.0.1", port=2022)
```

#### Step 2.5: Start Server

```bash
cd ~/parakeet-asr
source bin/activate
python server.py
```

#### Step 2.6: Configure Voice Mode

```bash
export VOICEMODE_STT_BASE_URL="http://127.0.0.1:2022/v1"
```

**Performance**: 
- **GPU**: Up to 3386x real-time
- **CPU**: ~15x real-time on M2, ~10x on Intel

</details>

## 🔊 Text-to-Speech Options

### Option 1: OpenAI TTS (Cloud)

Default option, high quality, requires API key.

```bash
# Just set your API key
export OPENAI_API_KEY="your-api-key-here"
```

### Option 2: Kokoro TTS (Local)

Fast, lightweight local TTS with multiple voices.

<details>
<summary><b>Click to expand Kokoro setup instructions</b></summary>

#### Install Kokoro

```bash
# Clone Kokoro FastAPI
git clone https://github.com/remsky/Kokoro-FastAPI.git ~/kokoro
cd ~/kokoro

# Using Docker (recommended)
docker compose up -d

# Or manual Python setup
pip install -r requirements.txt
python app.py
```

#### Configure Voice Mode

```bash
export VOICEMODE_TTS_BASE_URL="http://127.0.0.1:8880/v1"
```

**Voices**: Multiple languages and styles included

</details>

### Option 3: NVIDIA NeMo TTS (Local)

Professional-grade TTS using FastPitch + HiFiGAN.

<details>
<summary><b>Click to expand NeMo TTS setup instructions</b></summary>

#### Step 3.1: Install Dependencies

```bash
source ~/parakeet-asr/bin/activate  # Use same env if you have Parakeet
pip install "nemo_toolkit[tts]"
pip install scipy pydub
```

#### Step 3.2: Download Models

```bash
mkdir -p ~/nemo-models
cd ~/nemo-models

# FastPitch (spectrogram generator)
wget https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_en_fastpitch/versions/1.14.0/files/tts_en_fastpitch.nemo

# HiFiGAN (vocoder)
wget https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_en_hifigan/versions/1.14.0/files/tts_en_hifigan.nemo
```

#### Step 3.3: Create TTS Server

Create `~/nemo-tts/server.py` with FastAPI wrapper (see full code in readme.parakeet.md).

#### Step 3.4: Configure Voice Mode

```bash
export VOICEMODE_TTS_BASE_URL="http://127.0.0.1:8880/v1"
```

**Performance**: 900x real-time on GPU, 5x on CPU

</details>

## 🎭 Complete Setup Examples

### Option 3: Full Local Stack

Complete privacy with Whisper + Kokoro.

```bash
# 1. Start Whisper (follow Option 1 STT above)
cd ~/whisper.cpp
./build/bin/whisper-server -m models/ggml-base.en.bin --host 127.0.0.1 --port 2022 &

# 2. Start Kokoro (follow Option 2 TTS above)
cd ~/kokoro
docker compose up -d

# 3. Configure Voice Mode
export VOICEMODE_STT_BASE_URL="http://127.0.0.1:2022/v1"
export VOICEMODE_TTS_BASE_URL="http://127.0.0.1:8880/v1"

# 4. Install Voice Mode MCP
claude mcp add --scope user voice-mode uvx voice-mode

# 5. Start Claude
claude
```

### Option 4: Full NVIDIA Stack

Maximum performance with Parakeet + NeMo TTS.

```bash
# 1. Start Parakeet (follow Option 2 STT above)
cd ~/parakeet-asr && source bin/activate
python server.py &

# 2. Start NeMo TTS (follow Option 3 TTS above)
cd ~/nemo-tts && source bin/activate
python server.py &

# 3. Configure Voice Mode
export VOICEMODE_STT_BASE_URL="http://127.0.0.1:2022/v1"
export VOICEMODE_TTS_BASE_URL="http://127.0.0.1:8880/v1"

# 4. Start Claude
claude
```

## 🛠️ Helper Scripts

### Universal Startup Script

Create `~/start-voice.sh`:

```bash
#!/bin/bash
# Claudio Voice Mode Startup Script

echo "Starting Claudio Voice Mode..."

# Detect and start STT service
if [ -d ~/whisper.cpp ]; then
    echo "Starting Whisper STT..."
    cd ~/whisper.cpp
    nohup ./build/bin/whisper-server -m models/ggml-base.en.bin \
          --host 127.0.0.1 --port 2022 > /tmp/stt.log 2>&1 &
    export VOICEMODE_STT_BASE_URL="http://127.0.0.1:2022/v1"
elif [ -f ~/parakeet-asr/server.py ]; then
    echo "Starting Parakeet STT..."
    cd ~/parakeet-asr
    source bin/activate
    nohup python server.py > /tmp/stt.log 2>&1 &
    export VOICEMODE_STT_BASE_URL="http://127.0.0.1:2022/v1"
fi

# Detect and start TTS service
if [ -d ~/kokoro ]; then
    echo "Starting Kokoro TTS..."
    cd ~/kokoro
    docker compose up -d
    export VOICEMODE_TTS_BASE_URL="http://127.0.0.1:8880/v1"
elif [ -f ~/nemo-tts/server.py ]; then
    echo "Starting NeMo TTS..."
    cd ~/nemo-tts
    source bin/activate
    nohup python server.py > /tmp/tts.log 2>&1 &
    export VOICEMODE_TTS_BASE_URL="http://127.0.0.1:8880/v1"
fi

echo "Voice Mode Ready!"
echo "STT: $VOICEMODE_STT_BASE_URL"
echo "TTS: $VOICEMODE_TTS_BASE_URL"
echo ""
echo "Start Claude with: claude"
```

## 📊 Performance Comparison

### Speech-to-Text Performance

| Model | Hardware | Speed | Accuracy | Size |
|-------|----------|-------|----------|------|
| **OpenAI API** | Cloud | N/A | Excellent | 0 |
| **Whisper tiny** | M2 Mac | 100x | Good | 39MB |
| **Whisper base** | M2 Mac | 50x | Better | 140MB |
| **Whisper small** | M2 Mac | 30x | Best | 466MB |
| **Parakeet** | M2 Mac (CPU) | 15x | Excellent | 2.3GB |
| **Parakeet** | RTX 3060 | 1000x | Excellent | 2.3GB |
| **Parakeet** | RTX 4090 | 3386x | Excellent | 2.3GB |

### Text-to-Speech Performance

| Model | Hardware | Speed | Quality | Voices |
|-------|----------|-------|---------|--------|
| **OpenAI API** | Cloud | N/A | Excellent | 6 |
| **Kokoro** | CPU | 10x | Good | 20+ |
| **Kokoro** | GPU | 50x | Good | 20+ |
| **NeMo TTS** | CPU | 5x | Excellent | Customizable |
| **NeMo TTS** | GPU | 900x | Excellent | Customizable |

## 🔧 Troubleshooting

### Common Issues

<details>
<summary><b>Voice Mode Not Working</b></summary>

```bash
# 1. Check services are running
ps aux | grep -E "whisper|parakeet|server.py"

# 2. Verify environment variables (in new terminal)
echo $VOICEMODE_STT_BASE_URL
echo $VOICEMODE_TTS_BASE_URL

# 3. Restart Claude in fresh terminal
exit
export VOICEMODE_STT_BASE_URL="http://127.0.0.1:2022/v1"
claude

# 4. Check MCP connection
claude mcp list
```

</details>

<details>
<summary><b>Port Already in Use</b></summary>

```bash
# Find what's using the port
lsof -i :2022  # STT port
lsof -i :8880  # TTS port

# Kill the process
kill -9 <PID>

# Or use different ports
# STT: Change --port 3022 in server command
# TTS: Change port in TTS server config
```

</details>

<details>
<summary><b>Microphone Not Working (macOS)</b></summary>

1. System Settings > Privacy & Security > Microphone
2. Ensure Terminal/iTerm has permission
3. Test with: `rec -r 16000 test.wav trim 0 3`

</details>

<details>
<summary><b>GPU Not Detected</b></summary>

```bash
# Check CUDA (NVIDIA)
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA toolkit if needed
# Ubuntu: sudo apt install nvidia-cuda-toolkit
# Or download from NVIDIA website
```

</details>

## 🚀 Making it Permanent

### macOS (launchd)

<details>
<summary><b>Click for launchd setup</b></summary>

Create `~/Library/LaunchAgents/com.claudio.voice.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" 
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.claudio.voice</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/YOUR_USERNAME/start-voice.sh</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/claudio.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/claudio.error.log</string>
</dict>
</plist>
```

Load with: `launchctl load ~/Library/LaunchAgents/com.claudio.voice.plist`

</details>

### Linux (systemd)

<details>
<summary><b>Click for systemd setup</b></summary>

Create `/etc/systemd/system/claudio-voice.service`:

```ini
[Unit]
Description=Claudio Voice Services
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
ExecStart=/home/YOUR_USERNAME/start-voice.sh
Restart=always
Environment="PATH=/usr/local/bin:/usr/bin:/bin"

[Install]
WantedBy=multi-user.target
```

Enable with:
```bash
sudo systemctl enable claudio-voice
sudo systemctl start claudio-voice
```

</details>

## 📈 Optimization Tips

### For Speed
- Use smaller models (tiny/base for Whisper)
- Enable GPU acceleration where possible
- Pre-load models on startup
- Use batch processing for multiple requests

### For Quality
- Use larger models (small/medium for Whisper)
- Quality microphone with noise cancellation
- Adjust VAD aggressiveness for your environment
- Fine-tune models on domain-specific data

### For Privacy
- Use local models exclusively
- Disable telemetry in configurations
- Run services on localhost only
- Use firewall rules to block external access

## 🗑️ Uninstall

```bash
# Stop all services
pkill -f whisper-server
pkill -f server.py
docker compose down  # If using Docker

# Remove Voice Mode MCP
claude mcp remove voice-mode

# Remove installations (optional)
rm -rf ~/whisper.cpp
rm -rf ~/parakeet-asr
rm -rf ~/kokoro
rm -rf ~/nemo-models
rm -rf ~/nemo-tts

# Remove from shell profile
# Remove any export VOICEMODE_* lines from ~/.bashrc or ~/.zshrc
```

## 📚 Resources

### Documentation
- [Voice Mode Documentation](https://voice-mode.readthedocs.io)
- [Claude Code Documentation](https://docs.anthropic.com/claude-code)
- [MCP Protocol Spec](https://modelcontextprotocol.io)

### Model Sources
- [OpenAI Whisper](https://github.com/openai/whisper)
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp)
- [NVIDIA Parakeet](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
- [Kokoro TTS](https://github.com/remsky/Kokoro-FastAPI)

### Community
- [GitHub Issues](https://github.com/anthropics/claude-code/issues)
- [Discord Community](https://discord.gg/anthropic)

---

**Privacy Note**: Local setups process all audio on your machine. No data is sent to external servers unless you explicitly use cloud APIs.

**Performance Note**: GPU acceleration can provide 10-100x speedup for both STT and TTS. Consider NVIDIA GPUs for professional use.

**Cost Note**: 
- Cloud APIs: ~$0.15/minute of conversation
- Local setups: One-time setup, no ongoing costs

---

*Claudio: Your voice, your code, your choice of privacy and performance.*