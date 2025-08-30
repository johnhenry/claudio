#!/bin/bash
# Stop Whisper Server Script

echo "Stopping Whisper Server..."

# Find and kill whisper-server process
if pgrep -f "whisper-server" > /dev/null; then
    echo "Found whisper-server process, stopping..."
    pkill -f "whisper-server"
    sleep 2
    
    # Verify it's stopped
    if pgrep -f "whisper-server" > /dev/null; then
        echo "⚠ Server still running, forcing stop..."
        pkill -9 -f "whisper-server"
        sleep 1
    fi
    
    if ! pgrep -f "whisper-server" > /dev/null; then
        echo "✓ Whisper server stopped successfully"
    else
        echo "✗ Failed to stop whisper server"
        exit 1
    fi
else
    echo "ℹ Whisper server is not running"
fi

# Also check for whisper-cli process
if pgrep -f "whisper-cli" > /dev/null; then
    echo "Found whisper-cli process, stopping..."
    pkill -f "whisper-cli"
    echo "✓ Whisper-cli stopped"
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
if [ -f /tmp/whisper.log ]; then
    echo "Cleaning up log file..."
    rm -f /tmp/whisper.log
    echo "✓ Log file removed"
fi

if [ -f /tmp/whisper-server.log ]; then
    rm -f /tmp/whisper-server.log
    echo "✓ Server log file removed"
fi

if [ -f /tmp/whisper-server.error.log ]; then
    rm -f /tmp/whisper-server.error.log
    echo "✓ Error log file removed"
fi

echo ""
echo "========================================="
echo "Whisper Server stopped"
echo "========================================="
echo ""
echo "Note: whisper.cpp has been uninstalled from this system."
echo "To use Whisper again, you'll need to reinstall it:"
echo "  Follow instructions in readme.md"
echo ""
echo "To use Parakeet instead:"
echo "  ./start-parakeet.sh"