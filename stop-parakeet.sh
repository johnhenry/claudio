#!/bin/bash
# Stop Parakeet ASR Server Script

echo "Stopping Parakeet ASR Server..."

# Find and kill Parakeet server process
if pgrep -f "server.py" > /dev/null; then
    echo "Found Parakeet server process, stopping..."
    pkill -f "server.py"
    sleep 2
    
    # Verify it's stopped
    if pgrep -f "server.py" > /dev/null; then
        echo "⚠ Server still running, forcing stop..."
        pkill -9 -f "server.py"
        sleep 1
    fi
    
    if ! pgrep -f "server.py" > /dev/null; then
        echo "✓ Parakeet server stopped successfully"
    else
        echo "✗ Failed to stop Parakeet server"
        exit 1
    fi
else
    echo "ℹ Parakeet server is not running"
fi

# Check if port 2022 is still in use
if lsof -i :2022 > /dev/null 2>&1; then
    echo "⚠ Port 2022 is still in use by:"
    lsof -i :2022
    echo "Attempting to free port..."
    lsof -ti :2022 | xargs kill -9 2>/dev/null
    sleep 1
    if ! lsof -i :2022 > /dev/null 2>&1; then
        echo "✓ Port 2022 freed"
    fi
else
    echo "✓ Port 2022 is free"
fi

# Clean up log files (optional)
if [ -f /tmp/parakeet.log ]; then
    echo "Cleaning up log file..."
    rm -f /tmp/parakeet.log
    echo "✓ Log file removed"
fi

# Clean up any Python cache
if [ -d ~/parakeet-asr/__pycache__ ]; then
    rm -rf ~/parakeet-asr/__pycache__
    echo "✓ Python cache cleaned"
fi

echo ""
echo "========================================="
echo "Parakeet ASR Server stopped"
echo "========================================="
echo ""
echo "To restart Parakeet:"
echo "  ./start-parakeet.sh"
echo ""
echo "To use Whisper instead:"
echo "  ./start-whisper.sh"