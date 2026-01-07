#!/bin/bash
# Stop Chatterbox-turbo TTS Server Script

echo "Stopping Chatterbox-turbo TTS Server..."

# Find and kill Chatterbox-turbo server process
if pgrep -f "chatterbox.*server.py" > /dev/null; then
    echo "Found Chatterbox-turbo server process, stopping..."
    pkill -f "chatterbox.*server.py"
    sleep 2
    
    # Verify it's stopped
    if pgrep -f "chatterbox.*server.py" > /dev/null; then
        echo "⚠ Server still running, forcing stop..."
        pkill -9 -f "chatterbox.*server.py"
        sleep 1
    fi
    
    if ! pgrep -f "chatterbox.*server.py" > /dev/null; then
        echo "✓ Chatterbox-turbo server stopped successfully"
    else
        echo "✗ Failed to stop Chatterbox-turbo server"
        exit 1
    fi
else
    echo "ℹ Chatterbox-turbo server is not running"
fi

# Check if port 8004 is still in use
if lsof -i :8004 > /dev/null 2>&1; then
    echo "⚠ Port 8004 is still in use by:"
    lsof -i :8004
    echo "Attempting to free port..."
    # Cross-platform compatible: check if there are PIDs before piping to xargs
    PIDS=$(lsof -ti :8004 2>/dev/null)
    if [ -n "$PIDS" ]; then
        echo "$PIDS" | xargs kill -9 2>/dev/null
    fi
    sleep 1
    if ! lsof -i :8004 > /dev/null 2>&1; then
        echo "✓ Port 8004 freed"
    fi
else
    echo "✓ Port 8004 is free"
fi

# Clean up log files (optional)
if [ -f /tmp/chatterbox-turbo.log ]; then
    echo "Cleaning up log file..."
    rm -f /tmp/chatterbox-turbo.log
    echo "✓ Log file removed"
fi

echo ""
echo "========================================="
echo "Chatterbox-turbo TTS Server stopped"
echo "========================================="
echo ""
echo "To restart Chatterbox-turbo:"
echo "  ./start-chatterbox.sh"
echo ""
echo "To use other TTS options:"
echo "  Kokoro: Follow setup in docs/claudio.md"
echo "  OpenAI: Set OPENAI_API_KEY environment variable"
