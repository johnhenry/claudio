# Whisper.cpp Local Speech Recognition Setup Guide

A complete walkthrough for setting up voice-controlled Claude Code using whisper.cpp for speech-to-text and Voice Mode MCP server, with fully local speech recognition and system text-to-speech.

## Overview

This guide configures:
- **whisper.cpp** - Fast local speech recognition (no API costs, complete privacy for input)
- **whisper-server** - Built-in server from whisper.cpp (OpenAI Whisper API compatible)
- **Voice Mode MCP** - Natural voice interface for Claude Code
- **Claude Code** - Your AI coding assistant
- **System TTS** - Built-in text-to-speech using your OS's native voice synthesis

**Result**: Speak naturally to Claude Code with fully local speech processing and voice responses.

## Architecture

| Component | Function | Location | Privacy |
|-----------|----------|----------|---------|
| **Speech-to-Text** | whisper.cpp | Local (CPU optimized) | ✅ Fully private |
| **Text-to-Speech** | System TTS (say/espeak) | Local | ✅ Fully private |
| **Processing** | Claude Code | Local/Cloud | Depends on Claude config |

## Performance Characteristics

| Model | Size | Speed (M2 Mac) | Accuracy | Use Case |
|-------|------|----------------|----------|----------|
| tiny.en | 39MB | ~100x real-time | Lower | Quick commands |
| base.en | 140MB | ~50x real-time | Good | **Best balance** ⭐ |
| small.en | 466MB | ~30x real-time | Better | Longer dictation |
| medium.en | 1.5GB | ~15x real-time | Great | Professional use |

## Compatibility Notes

⚠️ **Important**: This setup uses whisper.cpp's built-in server which has been tested to work with Voice Mode MCP, though it may not be fully OpenAI-compatible. 

- **Current Setup**: Uses whisper.cpp's built-in `whisper-server` - simple and working
- **If you need full OpenAI compatibility**, consider alternatives like:
  - matatonic/openedai-whisper (Python-based, fully OpenAI-compatible)
  - litongjava/whisper-cpp-server (Docker-based, easy deployment)
  - carloscdias/whisper-cpp-python (Python bindings with server mode)

## Prerequisites

### Hardware Requirements
- macOS (Apple Silicon or Intel), Linux, or Windows (via WSL2)
- 4GB+ RAM recommended
- ~2GB disk space for models
- Microphone access

### Software Requirements
- Claude Code installed (`npm install -g @anthropic-ai/claude-code`)
- Git for cloning repositories
- C/C++ compiler:
  - macOS: Xcode Command Line Tools (`xcode-select --install`)
  - Linux: build-essential (`sudo apt install build-essential`)
  - Windows: Use WSL2 with Linux instructions
- CMake for building whisper.cpp:
  - macOS: `brew install cmake`
  - Linux: `sudo apt install cmake`

### Optional Requirements
- None - everything runs locally

## Step 1: Install whisper.cpp

```bash
# Clone whisper.cpp
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp

# Build whisper.cpp
make

# Download the base English model (140MB)
# For better accuracy, use 'small.en' (466MB) or 'medium.en' (1.5GB)
bash ./models/download-ggml-model.sh base.en

# Test that it works
./main -m models/ggml-base.en.bin -f samples/jfk.wav
```

### Choosing Your Model

| Model | Size | Speed | Accuracy | Recommendation |
|-------|------|-------|----------|----------------|
| tiny.en | 39MB | Fastest | Lower | Quick commands |
| base.en | 140MB | Fast | Good | **Best balance** ⭐ |
| small.en | 466MB | Medium | Better | Longer dictation |
| medium.en | 1.5GB | Slower | Great | Professional use |

### Build Options for Optimization

```bash
# For Apple Silicon Macs (M1/M2/M3) with Metal acceleration
make clean && make GGML_METAL=1

# For Intel Macs with AVX support
make clean && make GGML_OPENBLAS=1

# For NVIDIA GPUs (Linux)
make clean && make GGML_CUDA=1
```

## Step 2: Download Whisper Model

```bash
cd ~/whisper.cpp
bash ./models/download-ggml-model.sh base.en
```

## Step 3: Start the Whisper Server

We'll use whisper.cpp's built-in server which has proven compatibility with Voice Mode MCP:

```bash
# Navigate to your whisper.cpp directory
cd ~/whisper.cpp

# Start the whisper server on port 2022
./build/bin/whisper-server \
  -m models/ggml-base.en.bin \
  --host 127.0.0.1 \
  --port 2022
```

You should see the model loading with Metal acceleration (on macOS):
```
whisper_model_load: loading model
whisper_model_load: n_vocab       = 51864
whisper_backend_init_gpu: using Metal backend
ggml_metal_init: GPU name:   Apple M2
```

### Verify the Server is Running

In a new terminal:
```bash
# Check if server is responding
curl -s http://127.0.0.1:2022 | head -3

# You should see:
#     <html>
#     <head>
#         <title>Whisper.cpp Server</title>

# Check if port is listening
lsof -i :2022
# Should show: whisper-se ... TCP localhost:2022 (LISTEN)
```

### API Endpoints

The whisper-server provides these OpenAI-compatible endpoints:
- `GET /` - Health check and server info
- `POST /v1/audio/transcriptions` - Transcribe audio (OpenAI Whisper API format)
- `GET /v1/models` - List available models

## Step 4: Configure Voice Mode for Claude Code

```bash
# Set environment variable to use local Whisper
export VOICEMODE_STT_BASE_URL="http://127.0.0.1:2022/v1"

# System TTS is used by default for voice responses
# No additional configuration needed

# Add to your shell profile for persistence
echo 'export VOICEMODE_STT_BASE_URL="http://127.0.0.1:2022/v1"' >> ~/.bashrc
# or for zsh:
echo 'export VOICEMODE_STT_BASE_URL="http://127.0.0.1:2022/v1"' >> ~/.zshrc
```

### Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `VOICEMODE_STT_BASE_URL` | Speech-to-text endpoint | `http://127.0.0.1:2022/v1` |
| `VOICEMODE_TTS_BASE_URL` | Text-to-speech endpoint | System TTS (default) |

## Step 5: Install Voice Mode MCP

```bash
# Add Voice Mode to Claude Code (user-level)
claude mcp add --scope user voice-mode uvx voice-mode

# Verify installation
claude mcp list
```

## Step 6: Verify and Test the Complete Setup

### Pre-flight Checks

```bash
# 1. Verify whisper-server is running
ps aux | grep whisper-server
# Should show the whisper-server process

# 2. Check port is listening
lsof -i :2022
# Should show: whisper-se ... TCP localhost:2022 (LISTEN)

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
# - Your speech transcribed to text
```

### Test API Directly

```bash
# Record a test audio file
rec -r 16000 -c 1 test.wav trim 0 5

# Test transcription via API
curl -X POST "http://127.0.0.1:2022/v1/audio/transcriptions" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test.wav" \
  -F "model=whisper-1"
```

### Verify Whisper Processing

```bash
# Monitor whisper-server logs (in the terminal where it's running)
# You should see activity when speaking:
# - Audio processing messages
# - Transcription results
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

### Voice Interaction Tips

- **Clear speech**: Speak clearly and at a moderate pace
- **Pause between commands**: Allow a moment for processing
- **Background noise**: Minimize for best results
- **Microphone position**: Keep consistent distance from mic

### Pro Tips

1. **Keep the API server running**: Add to your startup scripts
2. **Use push-to-talk**: Configure a hotkey for cleaner recordings
3. **Adjust models**: Use larger models for complex technical discussions
4. **Background service**: See below for systemd/launchd setup

## Running as a Background Service

### macOS (launchd)

Create `~/Library/LaunchAgents/com.whisper.server.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" 
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.whisper.server</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/YOUR_USERNAME/whisper.cpp/build/bin/whisper-server</string>
        <string>-m</string>
        <string>/Users/YOUR_USERNAME/whisper.cpp/models/ggml-base.en.bin</string>
        <string>--host</string>
        <string>127.0.0.1</string>
        <string>--port</string>
        <string>2022</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/whisper-server.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/whisper-server.error.log</string>
</dict>
</plist>
```

Then:
```bash
# Load the service
launchctl load ~/Library/LaunchAgents/com.whisper.server.plist

# Check status
launchctl list | grep whisper

# View logs
tail -f /tmp/whisper-server.log
```

### Linux (systemd)

Create `/etc/systemd/system/whisper-server.service`:

```ini
[Unit]
Description=Whisper.cpp Server
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
ExecStart=/home/YOUR_USERNAME/whisper.cpp/build/bin/whisper-server \
  -m /home/YOUR_USERNAME/whisper.cpp/models/ggml-base.en.bin \
  --host 127.0.0.1 \
  --port 2022
Restart=always

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl enable whisper-server
sudo systemctl start whisper-server
sudo systemctl status whisper-server
```

## Making the Setup Persistent

### Option 1: Manual Start Script

Add to your shell profile (~/.zshrc or ~/.bashrc):
```bash
# Start whisper server in background on shell start
alias whisper-start='cd ~/whisper.cpp && nohup ./build/bin/whisper-server -m models/ggml-base.en.bin --host 127.0.0.1 --port 2022 > /tmp/whisper.log 2>&1 &'
alias whisper-stop='pkill -f whisper-server'
alias whisper-logs='tail -f /tmp/whisper.log'

# Set Voice Mode environment
export VOICEMODE_STT_BASE_URL="http://127.0.0.1:2022/v1"
```

### Option 2: Automatic Start (Recommended)

Use the launchd service created earlier:
```bash
# Load the service (starts automatically on login)
launchctl load ~/Library/LaunchAgents/com.whisper.server.plist

# Check status
launchctl list | grep whisper

# View logs
tail -f /tmp/whisper-server.log
tail -f /tmp/whisper-server.error.log

# Stop/start manually if needed
launchctl stop com.whisper.server
launchctl start com.whisper.server
```

### Helper Script

Create `~/start-whisper.sh`:
```bash
#!/bin/bash
# Whisper Voice Mode Setup Script

echo "Starting Whisper Voice Mode Setup..."

# Check if whisper-server is already running
if pgrep -x "whisper-server" > /dev/null; then
    echo "✓ Whisper server is already running"
else
    echo "Starting whisper server..."
    cd ~/whisper.cpp
    nohup ./build/bin/whisper-server \
        -m models/ggml-base.en.bin \
        --host 127.0.0.1 \
        --port 2022 > /tmp/whisper.log 2>&1 &
    
    sleep 2
    
    if pgrep -x "whisper-server" > /dev/null; then
        echo "✓ Whisper server started successfully"
    else
        echo "✗ Failed to start whisper server"
        exit 1
    fi
fi

# Set environment variable
export VOICEMODE_STT_BASE_URL="http://127.0.0.1:2022/v1"
echo "✓ Environment variable set"

echo ""
echo "Ready! Start Claude Code with: claude"
```

Make it executable: `chmod +x ~/start-whisper.sh`

## Troubleshooting

### Voice Mode Not Working
```bash
# 1. Check if whisper-server is actually running
ps aux | grep whisper-server

# 2. Verify environment variable is set in NEW terminal
echo $VOICEMODE_STT_BASE_URL

# 3. Restart Claude Code in a fresh terminal
exit  # from current Claude session
claude  # start new session

# 4. Check MCP connection
claude mcp list
```

### Whisper Server Won't Start
```bash
# Check if port 2022 is in use
lsof -i :2022

# Kill existing process if needed
kill -9 <PID>

# Or try a different port
./build/bin/whisper-server -m models/ggml-base.en.bin --host 127.0.0.1 --port 3022
# Then update: export VOICEMODE_STT_BASE_URL="http://127.0.0.1:3022/v1"
```

### API Compatibility Issues
```bash
# Note: whisper.cpp's built-in server may not be fully OpenAI-compatible
# But it has been tested to work with Voice Mode MCP
# 
# If you experience issues, alternative solutions include:
# 1. matatonic/openedai-whisper (Python-based, fully OpenAI-compatible)
# 2. litongjava/whisper-cpp-server (Docker-based with OpenAI API)
# 3. carloscdias/whisper-cpp-python (Python bindings with server mode)
```

### Microphone Not Working (macOS)
```bash
# Check permissions
# System Settings > Privacy & Security > Microphone
# Ensure Terminal/iTerm has permission

# Test microphone
rec -r 16000 -c 1 test.wav trim 0 3
play test.wav

# Check default input device
system_profiler SPAudioDataType | grep "Default Input"
```

### Build Errors
```bash
# Missing cmake
brew install cmake  # macOS
apt install cmake   # Linux

# whisper.cpp build fails
make clean
cmake -B build
cmake --build build --config Release

# Or try simple make
make clean
make
```

### Slow Transcription
```bash
# Use smaller model for faster response
cd ~/whisper.cpp
bash ./models/download-ggml-model.sh tiny.en

# Restart with tiny model
./build/bin/whisper-server -m models/ggml-tiny.en.bin --host 127.0.0.1 --port 2022
```

### TTS Not Working
```bash
# Check if system TTS is available
# macOS:
which say

# Linux:
which espeak || which festival

# Test system TTS
# macOS:
say "Hello from system text to speech"

# Linux:
espeak "Hello from system text to speech"
```

## Performance Optimization

### For Faster Response
- Use `tiny.en` or `base.en` models
- Run on SSD not HDD
- Close unnecessary applications
- Consider GPU acceleration (if supported)

### For Better Accuracy
- Use `small.en` or `medium.en` models
- Speak clearly and at moderate pace
- Use a quality microphone
- Reduce background noise

### CPU Optimization
```bash
# Check CPU features
sysctl -a | grep cpu.features  # macOS
lscpu | grep Flags              # Linux

# Build with optimizations
make clean
make GGML_OPENBLAS=1  # For BLAS acceleration
make GGML_METAL=1      # For Apple Silicon
```

### Memory Usage
- tiny.en: ~390MB RAM
- base.en: ~500MB RAM
- small.en: ~1GB RAM
- medium.en: ~2.6GB RAM

## Uninstall

```bash
# Remove Voice Mode from Claude Code
claude mcp remove voice-mode

# Stop background service (macOS)
launchctl unload ~/Library/LaunchAgents/com.whisper.server.plist
rm ~/Library/LaunchAgents/com.whisper.server.plist

# Stop background service (Linux)
sudo systemctl stop whisper-server
sudo systemctl disable whisper-server
sudo rm /etc/systemd/system/whisper-server.service

# Remove whisper.cpp (optional)
rm -rf ~/whisper.cpp

# Remove environment variables
# Edit ~/.bashrc or ~/.zshrc and remove the VOICEMODE_STT_BASE_URL line
```

## Next Steps

### Add Local Text-to-Speech
- **Piper**: Fast, lightweight, multilingual (`pip install piper-tts`)
- **Coqui TTS**: High quality, voice cloning capable
- **bark**: Realistic speech with emotion
- **Kokoro**: Small, fast, good quality

### Enhance Speech Recognition
- **Custom Wake Words**: Integrate with tools like Porcupine for "Hey Claude"
- **Noise Cancellation**: Add RNNoise for cleaner input
- **Multi-language**: Download non-English whisper models
- **Voice Activity Detection**: Add silero-vad for better segmentation

### Alternative Implementations
- **Faster Whisper**: 4x faster with same accuracy (Python)
- **WhisperX**: With word-level timestamps and speaker diarization
- **Whisper JAX**: For TPU acceleration

## Resources

- [whisper.cpp Documentation](https://github.com/ggml-org/whisper.cpp)
- [OpenAI Whisper Paper](https://arxiv.org/abs/2212.04356)
- [Voice Mode Documentation](https://voice-mode.readthedocs.io)
- [Claude Code Documentation](https://docs.anthropic.com/claude-code)
- [MCP Protocol Spec](https://modelcontextprotocol.io)

---

**Privacy Note**: All voice processing happens locally on your machine. Speech-to-text uses whisper.cpp and text-to-speech uses your system's built-in voice synthesis. No audio data or text is sent to external servers. Your voice commands and responses remain completely private.