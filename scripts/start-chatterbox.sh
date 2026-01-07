#!/bin/bash
# Chatterbox-turbo TTS Setup Script

echo "Starting Chatterbox-turbo TTS Setup..."

# Check if chatterbox-tts server is already running
if pgrep -f "chatterbox.*server.py" > /dev/null; then
    echo "✓ Chatterbox-turbo server is already running"
else
    echo "Starting Chatterbox-turbo server..."
    
    # Check if virtual environment exists
    if [ ! -d ~/chatterbox-turbo ]; then
        echo "Setting up Chatterbox-turbo for the first time..."
        
        # Create virtual environment
        python3 -m venv ~/chatterbox-turbo
        source ~/chatterbox-turbo/bin/activate
        
        # Install dependencies
        echo "Installing dependencies..."
        pip install -q --upgrade pip
        
        # Copy server.py to the virtual environment
        mkdir -p ~/chatterbox-turbo/server
        cp "$(dirname "$0")/../providers/chatterbox-turbo/chatterbox-tts/server.py" ~/chatterbox-turbo/server/
        cp "$(dirname "$0")/../providers/chatterbox-turbo/requirements.txt" ~/chatterbox-turbo/
        
        # Install requirements
        cd ~/chatterbox-turbo
        pip install -q -r requirements.txt
        
        echo "✓ Chatterbox-turbo environment created"
    else
        source ~/chatterbox-turbo/bin/activate
    fi
    
    # Start server
    cd ~/chatterbox-turbo/server
    nohup python server.py > /tmp/chatterbox-turbo.log 2>&1 &
    
    # Wait for server to start
    sleep 3
    
    if pgrep -f "chatterbox.*server.py" > /dev/null; then
        echo "✓ Chatterbox-turbo server started successfully"
    else
        echo "✗ Failed to start Chatterbox-turbo server. Check /tmp/chatterbox-turbo.log"
        exit 1
    fi
fi

# Set environment variable
export VOICEMODE_TTS_BASE_URL="http://127.0.0.1:8004/v1"
echo "✓ Environment variable set: VOICEMODE_TTS_BASE_URL=$VOICEMODE_TTS_BASE_URL"

# Check server is responding
sleep 2
if curl -s http://127.0.0.1:8004/health > /dev/null 2>&1; then
    echo "✓ Chatterbox-turbo server is responding on port 8004"
else
    echo "⚠ Chatterbox-turbo server might not be ready yet. Checking..."
    sleep 3
    if curl -s http://127.0.0.1:8004/health > /dev/null 2>&1; then
        echo "✓ Chatterbox-turbo server is now responding"
    else
        echo "✗ Chatterbox-turbo server is not responding. Check /tmp/chatterbox-turbo.log"
        exit 1
    fi
fi

echo ""
echo "========================================="
echo "Setup complete! Chatterbox-turbo is ready."
echo "========================================="
echo ""
echo "Performance:"
echo "  Fast, lightweight local TTS"
echo "  OpenAI-compatible API on port 8004"
echo ""
echo "To use with Claude Code:"
echo "1. Open a NEW terminal window"
echo "2. Run: export VOICEMODE_TTS_BASE_URL=\"http://127.0.0.1:8004/v1\""
echo "3. Combine with STT (Whisper/Parakeet): export VOICEMODE_STT_BASE_URL=\"...\""
echo "4. Run: claude"
echo "5. Say 'Let's have a voice conversation' to test"
echo ""
echo "To stop server: pkill -f 'chatterbox.*server.py'"
echo "To view logs: tail -f /tmp/chatterbox-turbo.log"
