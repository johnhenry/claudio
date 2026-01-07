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
        
        # Check if python3 is available
        if ! command -v python3 &> /dev/null; then
            echo "✗ python3 not found. Please install Python 3.8 or higher."
            exit 1
        fi
        
        # Create virtual environment
        python3 -m venv ~/chatterbox-turbo
        if [ $? -ne 0 ]; then
            echo "✗ Failed to create virtual environment"
            exit 1
        fi
        
        source ~/chatterbox-turbo/bin/activate
        
        # Install dependencies
        echo "Installing dependencies..."
        pip install -q --upgrade pip
        
        # Copy server.py to the virtual environment
        mkdir -p ~/chatterbox-turbo/server
        
        # Check if source files exist
        SCRIPT_DIR="$(dirname "$0")"
        SERVER_FILE="$SCRIPT_DIR/../providers/chatterbox-turbo/chatterbox-tts/server.py"
        REQ_FILE="$SCRIPT_DIR/../providers/chatterbox-turbo/requirements.txt"
        
        if [ ! -f "$SERVER_FILE" ]; then
            echo "✗ server.py not found at $SERVER_FILE"
            exit 1
        fi
        
        if [ ! -f "$REQ_FILE" ]; then
            echo "✗ requirements.txt not found at $REQ_FILE"
            exit 1
        fi
        
        cp "$SERVER_FILE" ~/chatterbox-turbo/server/
        cp "$REQ_FILE" ~/chatterbox-turbo/
        
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
echo "3. Combine with STT (Whisper/Parakeet): export VOICEMODE_STT_BASE_URL=\"http://127.0.0.1:2022/v1\""
echo "4. Run: claude"
echo "5. Say 'Let's have a voice conversation' to test"
echo ""
echo "To stop server: ./scripts/stop-chatterbox.sh"
echo "To view logs: tail -f /tmp/chatterbox-turbo.log"
