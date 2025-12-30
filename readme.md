# Claudio - Voice-Controlled Claude Code Setup

This is now a plugin in the [clapplications marketplace](https://github.com/johnhenry/clapplications).

Comprehensive guides and scripts for setting up voice-controlled Claude Code with local speech recognition.

## Overview

This repository contains setup guides and scripts for configuring voice-controlled Claude Code using:
- **whisper.cpp** - Fast local speech recognition
- **Parakeet ASR** - Alternative speech recognition server  
- **Voice Mode MCP** - Natural voice interface for Claude Code
- **Claude Code** - AI coding assistant with voice capabilities

## Quick Start

Choose your preferred speech recognition solution:

### Option 1: Whisper.cpp (Recommended)
```bash
./scripts/start-whisper.sh
```

### Option 2: Parakeet ASR
```bash
./scripts/start-parakeet.sh
```

## Documentation

- [Whisper.cpp Setup Guide](docs/whisper-setup.md) - Complete setup for whisper.cpp
- [Parakeet ASR Setup Guide](docs/readme-parakeet-asr.md) - Alternative ASR solution
- [NVIDIA Solutions Guide](docs/readme-nvidia-complete.md) - NVIDIA GPU-accelerated options
- [Whisper Details](docs/readme-whisper.md) - Additional whisper.cpp information
- [Claudio Configuration](docs/claudio.md) - Voice Mode configuration details

## Scripts

- `scripts/start-whisper.sh` - Start whisper.cpp server
- `scripts/stop-whisper.sh` - Stop whisper.cpp server
- `scripts/start-parakeet.sh` - Start Parakeet ASR server
- `scripts/stop-parakeet.sh` - Stop Parakeet ASR server

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