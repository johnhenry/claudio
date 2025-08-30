# Local Whisper + Claude Code Voice Mode Setup Guide

A complete walkthrough for setting up voice-controlled Claude Code using whisper.cpp and Voice Mode MCP server, with fully local speech-to-text processing.

## Overview

This guide configures:
- **whisper.cpp** - Fast local speech recognition (no API costs, complete privacy)
- **whisper-server** - Built-in server from whisper.cpp (works with Voice Mode MCP)
- **Voice Mode MCP** - Natural voice interface for Claude Code
- **Claude Code** - Your AI coding assistant

**Result**: Speak naturally to Claude Code, with all processing happening locally on your machine.

## Compatibility Notes

⚠️ **Important**: This setup uses whisper.cpp's built-in server which has been tested to work with Voice Mode MCP, though it may not be fully OpenAI-compatible. 

- **Current Setup**: Uses whisper.cpp's built-in `whisper-server` - simple and working
- **If you need full OpenAI compatibility**, consider alternatives like:
  - matatonic/openedai-whisper (Python-based, fully OpenAI-compatible)
  - litongjava/whisper-cpp-server (Docker-based, easy deployment)
  - carloscdias/whisper-cpp-python (Python bindings with server mode)

## Prerequisites

- macOS or Linux (Windows via WSL2)
- Claude Code installed (`npm install -g @anthropic-ai/claude-code`)
- Git for cloning repositories
- C/C++ compiler (Xcode Command Line Tools on macOS, build-essential on Linux)
- CMake for building whisper.cpp (`brew install cmake` on macOS, `apt install cmake` on Linux)
- ~2GB disk space for models
- Microphone access

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


## Step 4: Configure Voice Mode for Claude Code

```bash
# Set environment variable to use local Whisper
export VOICEMODE_STT_BASE_URL="http://127.0.0.1:2022/v1"

# Optional: Set local TTS if you have it
# export VOICEMODE_TTS_BASE_URL="http://127.0.0.1:8880/v1"

# Add to your shell profile for persistence
echo 'export VOICEMODE_STT_BASE_URL="http://127.0.0.1:2022/v1"' >> ~/.bashrc
# or for zsh:
echo 'export VOICEMODE_STT_BASE_URL="http://127.0.0.1:2022/v1"' >> ~/.zshrc
```

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

### Pro Tips

1. **Keep the API server running**: Add to your startup scripts
2. **Use push-to-talk**: Configure a hotkey for cleaner recordings
3. **Adjust models**: Use larger models for complex technical discussions
4. **Background service**: See below for systemd/launchd setup

## Running as a Background Service

### macOS (launchd)

Create `~/Library/LaunchAgents/com.whisper.api.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" 
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.whisper.api</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/YOUR_USERNAME/whisper-api-server/target/release/whisper-api-server</string>
        <string>--model-path</string>
        <string>/Users/YOUR_USERNAME/whisper.cpp/models/ggml-base.en.bin</string>
        <string>--port</string>
        <string>2022</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
```

Then:
```bash
launchctl load ~/Library/LaunchAgents/com.whisper.api.plist
```

### Linux (systemd)

Create `/etc/systemd/system/whisper-api.service`:

```ini
[Unit]
Description=Whisper API Server
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
ExecStart=/home/YOUR_USERNAME/whisper-api-server/target/release/whisper-api-server \
  --model-path /home/YOUR_USERNAME/whisper.cpp/models/ggml-base.en.bin \
  --port 2022
Restart=always

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl enable whisper-api
sudo systemctl start whisper-api
```

## Making the Setup Persistent

### Option 1: Manual Start
Add to your shell profile (~/.zshrc or ~/.bashrc):
```bash
# Start whisper server in background on shell start
alias whisper-start='cd ~/whisper.cpp && nohup ./build/bin/whisper-server -m models/ggml-base.en.bin --host 127.0.0.1 --port 2022 > /tmp/whisper.log 2>&1 &'
alias whisper-stop='pkill -f whisper-server'
alias whisper-logs='tail -f /tmp/whisper.log'
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

## Uninstall

```bash
# Remove Voice Mode from Claude Code
claude mcp remove voice-mode

# Stop background service (macOS)
launchctl unload ~/Library/LaunchAgents/com.whisper.api.plist

# Stop background service (Linux)
sudo systemctl stop whisper-api
sudo systemctl disable whisper-api

# Remove files (optional)
rm -rf ~/whisper.cpp
rm -rf ~/whisper-api-server
```

## Next Steps

- **Custom Wake Words**: Integrate with tools like Porcupine for "Hey Claude"
- **Better TTS**: Set up local Kokoro or Piper for responses
- **Noise Cancellation**: Add RNNoise for cleaner input
- **Multi-language**: Download non-English whisper models

## Resources

- [whisper.cpp Documentation](https://github.com/ggml-org/whisper.cpp)
- [Voice Mode Documentation](https://voice-mode.readthedocs.io)
- [Claude Code Documentation](https://docs.anthropic.com/claude-code)
- [MCP Protocol Spec](https://modelcontextprotocol.io)

---

**Privacy Note**: This setup processes all audio locally on your machine. No audio data is sent to external servers. Your voice commands and code remain completely private.