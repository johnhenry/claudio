#!/bin/bash
# Parakeet Voice Mode Setup Script

echo "Starting Parakeet Voice Mode Setup..."

# Activate virtual environment and check if server is already running
if pgrep -f "server.py" > /dev/null; then
    echo "✓ Parakeet server is already running"
else
    echo "Starting Parakeet server..."
    cd ~/parakeet-asr
    source bin/activate
    nohup python server.py > /tmp/parakeet.log 2>&1 &
    
    # Wait for server to start
    sleep 5
    
    if pgrep -f "server.py" > /dev/null; then
        echo "✓ Parakeet server started successfully"
    else
        echo "✗ Failed to start Parakeet server. Check /tmp/parakeet.log"
        exit 1
    fi
fi

# Set environment variable
export VOICEMODE_STT_BASE_URL="http://127.0.0.1:2022/v1"
echo "✓ Environment variable set: VOICEMODE_STT_BASE_URL=$VOICEMODE_STT_BASE_URL"

# Check server is responding
if curl -s http://127.0.0.1:2022 > /dev/null 2>&1; then
    echo "✓ Parakeet server is responding on port 2022"
else
    echo "✗ Parakeet server is not responding"
    exit 1
fi

# Check Voice Mode MCP
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
echo "Performance:"
echo "  CPU mode: ~15x real-time transcription on M2 Mac"
echo "  (GPU would be ~3386x real-time if available)"
echo ""
echo "To use voice in Claude Code:"
echo "1. Open a NEW terminal window"
echo "2. Run: export VOICEMODE_STT_BASE_URL=\"http://127.0.0.1:2022/v1\""
echo "3. Run: claude"
echo "4. Say 'Let's have a voice conversation' to test"
echo ""
echo "Model loaded: NVIDIA Parakeet-TDT-0.6B-v2 (2.3GB)"
echo "To stop server: pkill -f server.py"
echo "To view logs: tail -f /tmp/parakeet.log"