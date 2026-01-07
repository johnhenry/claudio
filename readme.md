# Claudio - Voice-Controlled Claude Code Setup

This is now a plugin in the [clapplications marketplace](https://github.com/johnhenry/clapplications).

Comprehensive guides and scripts for setting up voice-controlled Claude Code with local speech recognition and text-to-speech.

## Overview

This repository contains setup guides and scripts for configuring voice-controlled Claude Code using:

**Speech-to-Text (STT):**
- **whisper.cpp** - Fast local speech recognition
- **Parakeet ASR** - Alternative speech recognition server

**Text-to-Speech (TTS):**
- **Chatterbox-turbo** - Easy local TTS (Recommended)
- **Kokoro** - Multi-voice local TTS
- **OpenAI TTS** - Cloud-based TTS

**Infrastructure:**
- **Voice Mode MCP** - Natural voice interface for Claude Code
- **Claude Code** - AI coding assistant with voice capabilities

## Quick Start

### Recommended: Full Local Setup (Whisper + Chatterbox-turbo)

The easiest way to get fully local voice mode:

```bash
# 1. Start speech recognition
./scripts/start-whisper.sh

# 2. Start text-to-speech
./scripts/start-chatterbox.sh

# 3. Follow the on-screen instructions to configure and use
```

### Alternative Options

#### Option 1: Whisper.cpp Only (STT)
```bash
./scripts/start-whisper.sh
```

#### Option 2: Parakeet ASR (STT)
```bash
./scripts/start-parakeet.sh
```

## Documentation

### Main Guides
- [Claudio Configuration](docs/claudio.md) - **Start here!** Complete voice mode setup guide
- [Whisper.cpp Setup Guide](docs/whisper-setup.md) - Complete setup for whisper.cpp
- [Parakeet ASR Setup Guide](docs/readme-parakeet-asr.md) - Alternative ASR solution
- [Chatterbox-turbo TTS Guide](docs/readme-chatterbox-turbo.md) - Local TTS setup (Recommended)

### Additional Resources
- [NVIDIA Solutions Guide](docs/readme-nvidia-complete.md) - NVIDIA GPU-accelerated options
- [Whisper Details](docs/readme-whisper.md) - Additional whisper.cpp information

## Scripts

### Speech-to-Text
- `scripts/start-whisper.sh` - Start whisper.cpp server
- `scripts/stop-whisper.sh` - Stop whisper.cpp server
- `scripts/start-parakeet.sh` - Start Parakeet ASR server
- `scripts/stop-parakeet.sh` - Stop Parakeet ASR server

### Text-to-Speech
- `scripts/start-chatterbox.sh` - Start Chatterbox-turbo TTS server (Recommended)
- `scripts/stop-chatterbox.sh` - Stop Chatterbox-turbo TTS server

## Requirements

- macOS or Linux (Windows via WSL2)
- Claude Code installed (`npm install -g @anthropic-ai/claude-code`)
- Git
- C/C++ compiler
- CMake
- ~2GB disk space for models
- Microphone access

## License

MIT