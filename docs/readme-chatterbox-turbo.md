# Chatterbox-turbo TTS Setup Guide

A complete guide for setting up Chatterbox-turbo as a local text-to-speech solution for voice-controlled Claude Code.

## Overview

Chatterbox-turbo is a fast, lightweight local TTS (Text-to-Speech) engine that provides an OpenAI-compatible API. It's ideal for users who want:
- **Privacy**: All audio processing happens locally
- **Speed**: Fast inference without cloud latency
- **Cost**: No API fees or usage limits
- **Quality**: Natural-sounding speech synthesis

## Why Chatterbox-turbo?

| Feature | Chatterbox-turbo | OpenAI TTS | Kokoro | NeMo TTS |
|---------|------------------|------------|--------|----------|
| **Privacy** | ✅ Fully local | ❌ Cloud | ✅ Local | ✅ Local |
| **Setup** | ⭐⭐⭐⭐ Easy | ⭐⭐⭐⭐⭐ Easiest | ⭐⭐⭐ Medium | ⭐⭐ Complex |
| **Cost** | Free | Pay per use | Free | Free |
| **Quality** | Good | Excellent | Good | Excellent |
| **Speed** | Fast | Fast | Fast | Very Fast |
| **Voices** | Default | 6 voices | 20+ voices | Customizable |

## Prerequisites

- Python 3.8+ (3.11 recommended)
- pip package manager
- ~200MB disk space for dependencies
- Microphone (for full voice mode)

## Quick Start

The easiest way to get started is using the provided script:

```bash
# Clone the repository if you haven't already
git clone https://github.com/johnhenry/claudio.git
cd claudio

# Run the start script
./scripts/start-chatterbox.sh
```

This script will:
1. Create a Python virtual environment at `~/chatterbox-turbo`
2. Install all required dependencies
3. Copy the fixed server.py file
4. Start the TTS server on port 8004
5. Set the environment variable for Voice Mode

## Manual Installation

If you prefer to install manually or customize the setup:

### Step 1: Create Virtual Environment

```bash
python3 -m venv ~/chatterbox-turbo
source ~/chatterbox-turbo/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install --upgrade pip
pip install fastapi uvicorn[standard] pydantic chatterbox-tts
```

### Step 3: Copy Server Files

```bash
mkdir -p ~/chatterbox-turbo/server
cp providers/chatterbox-turbo/chatterbox-tts/server.py ~/chatterbox-turbo/server/
```

### Step 4: Start the Server

```bash
cd ~/chatterbox-turbo/server
python server.py
```

The server will start on `http://127.0.0.1:8004`

## Configuration

### With Voice Mode MCP

To use Chatterbox-turbo with Voice Mode:

```bash
# Set the TTS endpoint
export VOICEMODE_TTS_BASE_URL="http://127.0.0.1:8004/v1"

# If you also have STT configured (e.g., Whisper)
export VOICEMODE_STT_BASE_URL="http://127.0.0.1:2022/v1"

# Start Claude Code
claude
```

### Complete Voice Setup Example

Combine Chatterbox-turbo with Whisper for a fully local voice solution:

```bash
# Terminal 1: Start Whisper STT
./scripts/start-whisper.sh

# Terminal 2: Start Chatterbox-turbo TTS
./scripts/start-chatterbox.sh

# Terminal 3: Configure and start Claude
export VOICEMODE_STT_BASE_URL="http://127.0.0.1:2022/v1"
export VOICEMODE_TTS_BASE_URL="http://127.0.0.1:8004/v1"
claude
```

Then say "Let's have a voice conversation" to start using voice mode.

## API Endpoints

Chatterbox-turbo provides an OpenAI-compatible API:

### POST /v1/audio/speech

Generate speech from text.

**Request Body** (JSON):
```json
{
  "input": "Hello, this is a test.",
  "model": "tts-1",
  "voice": "default",
  "response_format": "mp3"
}
```

**Response**: Audio file (WAV format)

### GET /health

Check server health status.

**Response**:
```json
{
  "status": "healthy",
  "model": "chatterbox-turbo"
}
```

## Testing the Server

### Test with curl

```bash
# Health check
curl http://127.0.0.1:8004/health

# Generate speech and save to file
curl -X POST http://127.0.0.1:8004/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello from Chatterbox-turbo!", "model": "tts-1"}' \
  --output test.wav

# Play the audio (macOS)
afplay test.wav

# Play the audio (Linux)
aplay test.wav
```

### Test with Python

```python
import requests

url = "http://127.0.0.1:8004/v1/audio/speech"
data = {
    "input": "This is a test of Chatterbox-turbo",
    "model": "tts-1",
    "voice": "default"
}

response = requests.post(url, json=data)

if response.status_code == 200:
    with open("output.wav", "wb") as f:
        f.write(response.content)
    print("Audio saved to output.wav")
else:
    print(f"Error: {response.status_code}")
```

## The Server Fix

This repository includes a **fixed version** of the Chatterbox-turbo server that addresses a critical bug in the original implementation.

### What was the bug?

The original `server.py` expected TTS parameters as **query parameters** instead of in the **POST request body**:

```python
# Original (buggy) code
@app.post("/v1/audio/speech")
async def generate_speech(
    input: str,           # Expected as query param
    model_name: str = "tts-1",
    voice: str = "default",
    ...
):
```

This didn't work with Voice Mode MCP, which sends parameters as JSON in the request body (following OpenAI's API standard).

### The Fix

Our fixed version uses **Pydantic models** to properly parse the JSON request body:

```python
# Fixed code
class TTSRequest(BaseModel):
    input: str
    model: str = "tts-1"
    voice: str = "default"
    response_format: str = "mp3"

@app.post("/v1/audio/speech")
async def generate_speech(request: TTSRequest):
    if not request.input:
        raise HTTPException(status_code=400, detail="No input text provided")
    
    wav = model.generate(request.input)
    ...
```

This makes the server truly OpenAI-compatible and works seamlessly with Voice Mode MCP.

## Troubleshooting

### Server Won't Start

**Check Python version:**
```bash
python3 --version  # Should be 3.8 or higher
```

**Check if port 8004 is in use:**
```bash
lsof -i :8004
# If something is using it, kill it:
lsof -ti :8004 | xargs kill -9
```

**Check logs:**
```bash
tail -f /tmp/chatterbox-turbo.log
```

### Import Error: chatterbox-tts

If you get import errors, ensure you've installed the package:

```bash
source ~/chatterbox-turbo/bin/activate
pip install chatterbox-tts
```

### Voice Mode Not Using Chatterbox-turbo

Make sure environment variables are set in the same terminal where you run Claude:

```bash
# Check current settings
echo $VOICEMODE_TTS_BASE_URL

# Should output: http://127.0.0.1:8004/v1
# If not, export it:
export VOICEMODE_TTS_BASE_URL="http://127.0.0.1:8004/v1"
```

### Audio Quality Issues

The Chatterbox-turbo default voice is optimized for speed. For higher quality:

1. Consider using NeMo TTS or OpenAI TTS instead
2. Ensure your audio output device is properly configured
3. Adjust system volume levels

## Stopping the Server

Use the provided stop script:

```bash
./scripts/stop-chatterbox.sh
```

Or manually:

```bash
# Find and kill the process
pkill -f "chatterbox.*server.py"

# Force kill if needed
pkill -9 -f "chatterbox.*server.py"

# Free the port
lsof -ti :8004 | xargs kill -9
```

## Performance

Chatterbox-turbo is designed for speed:

- **Latency**: <100ms for short phrases
- **Throughput**: Can handle real-time conversation
- **Memory**: ~200MB RAM usage
- **CPU**: Moderate usage, no GPU required

## Making it Permanent

### macOS (launchd)

Create `~/Library/LaunchAgents/com.chatterbox-turbo.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" 
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.chatterbox-turbo</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/YOUR_USERNAME/chatterbox-turbo/bin/python</string>
        <string>/Users/YOUR_USERNAME/chatterbox-turbo/server/server.py</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/chatterbox-turbo.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/chatterbox-turbo.error.log</string>
</dict>
</plist>
```

Load with:
```bash
launchctl load ~/Library/LaunchAgents/com.chatterbox-turbo.plist
```

### Linux (systemd)

Create `/etc/systemd/system/chatterbox-turbo.service`:

```ini
[Unit]
Description=Chatterbox-turbo TTS Server
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/home/YOUR_USERNAME/chatterbox-turbo/server
ExecStart=/home/YOUR_USERNAME/chatterbox-turbo/bin/python server.py
Restart=always
Environment="PATH=/home/YOUR_USERNAME/chatterbox-turbo/bin:/usr/local/bin:/usr/bin:/bin"

[Install]
WantedBy=multi-user.target
```

Enable with:
```bash
sudo systemctl enable chatterbox-turbo
sudo systemctl start chatterbox-turbo
```

## Comparison with Other TTS Options

### vs. OpenAI TTS
- **Privacy**: Chatterbox-turbo wins (fully local)
- **Quality**: OpenAI TTS is better
- **Cost**: Chatterbox-turbo is free, OpenAI charges per use
- **Setup**: OpenAI is easier (just API key)

### vs. Kokoro TTS
- **Setup**: Chatterbox-turbo is easier (no Docker required by default)
- **Voices**: Kokoro has more voices (20+ vs 1)
- **Quality**: Similar quality
- **Speed**: Similar performance

### vs. NeMo TTS
- **Setup**: Chatterbox-turbo is much easier
- **Quality**: NeMo TTS is better
- **Speed**: NeMo TTS is faster on GPU
- **Hardware**: Chatterbox-turbo works well on CPU

## Integration Examples

### With Whisper STT (Most Popular)

```bash
# Start both services
./scripts/start-whisper.sh
./scripts/start-chatterbox.sh

# Configure
export VOICEMODE_STT_BASE_URL="http://127.0.0.1:2022/v1"
export VOICEMODE_TTS_BASE_URL="http://127.0.0.1:8004/v1"

# Use
claude
```

### With Parakeet STT (Fastest)

```bash
# Start both services
./scripts/start-parakeet.sh
./scripts/start-chatterbox.sh

# Configure
export VOICEMODE_STT_BASE_URL="http://127.0.0.1:2022/v1"
export VOICEMODE_TTS_BASE_URL="http://127.0.0.1:8004/v1"

# Use
claude
```

### With OpenAI STT

```bash
# Start Chatterbox-turbo
./scripts/start-chatterbox.sh

# Configure (only TTS, STT uses OpenAI)
export OPENAI_API_KEY="your-api-key"
export VOICEMODE_TTS_BASE_URL="http://127.0.0.1:8004/v1"

# Use
claude
```

## Uninstall

To completely remove Chatterbox-turbo:

```bash
# Stop the server
./scripts/stop-chatterbox.sh

# Remove the virtual environment
rm -rf ~/chatterbox-turbo

# Remove logs
rm -f /tmp/chatterbox-turbo.log

# Remove from shell profile if you added it permanently
# Edit ~/.bashrc or ~/.zshrc and remove export VOICEMODE_TTS_BASE_URL lines
```

## Resources

- [Chatterbox-tts GitHub](https://github.com/resemble-ai/chatterbox)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Voice Mode MCP Documentation](https://voice-mode.readthedocs.io)
- [Claude Code Documentation](https://docs.anthropic.com/claude-code)

## Support

If you encounter issues:

1. Check the logs: `tail -f /tmp/chatterbox-turbo.log`
2. Verify the server is running: `curl http://127.0.0.1:8004/health`
3. Ensure environment variables are set correctly
4. Try restarting the server: `./scripts/stop-chatterbox.sh && ./scripts/start-chatterbox.sh`

---

**Note**: This setup uses the fixed version of Chatterbox-turbo server that properly implements the OpenAI API specification for JSON request bodies, ensuring compatibility with Voice Mode MCP and other standard TTS clients.
