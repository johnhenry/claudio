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
    
    # Wait for server to start
    sleep 2
    
    if pgrep -x "whisper-server" > /dev/null; then
        echo "✓ Whisper server started successfully"
    else
        echo "✗ Failed to start whisper server. Check /tmp/whisper.log for errors"
        exit 1
    fi
fi

# Set environment variable
export VOICEMODE_STT_BASE_URL="http://127.0.0.1:2022/v1"
echo "✓ Environment variable set: VOICEMODE_STT_BASE_URL=$VOICEMODE_STT_BASE_URL"

# Check server is responding
if curl -s http://127.0.0.1:2022 > /dev/null 2>&1; then
    echo "✓ Whisper server is responding on port 2022"
else
    echo "✗ Whisper server is not responding"
    exit 1
fi

# Check Voice Mode MCP is connected (if claude is available)
if command -v claude &> /dev/null; then
    if claude mcp list | grep -q "voice-mode.*✓ Connected"; then
        echo "✓ Voice Mode MCP is connected"
    else
        echo "⚠ Voice Mode MCP might not be connected (check with: claude mcp list)"
    fi
else
    echo "ℹ Claude Code command not found in current PATH"
fi

echo ""
echo "========================================="
echo "Setup complete! Voice Mode is ready."
echo "========================================="
echo ""
echo "To use voice in Claude Code:"
echo "1. Start a new terminal"
echo "2. Run: export VOICEMODE_STT_BASE_URL=\"http://127.0.0.1:2022/v1\""
echo "3. Run: claude"
echo "4. Say 'Let's have a voice conversation' to test"
echo ""
echo "To stop whisper server: pkill -f whisper-server"
echo "To view logs: tail -f /tmp/whisper.log"