# Providers Directory

This directory contains TTS (Text-to-Speech) provider implementations for the Claudio voice mode plugin.

## Structure

Each provider is organized in its own directory with the following structure:

```
providers/
├── provider-name/
│   ├── requirements.txt       # Python dependencies
│   └── provider-implementation/
│       └── server.py          # FastAPI server implementing OpenAI-compatible TTS API
```

## Available Providers

### Chatterbox-turbo

**Location**: `providers/chatterbox-turbo/`

Fast, lightweight local TTS with OpenAI-compatible API.

**Features**:
- Fixed version with proper JSON body parsing
- Pydantic model validation
- OpenAI API compatibility
- Easy setup via automated scripts

**Setup**: Use `./scripts/start-chatterbox.sh` or see [detailed documentation](../docs/readme-chatterbox-turbo.md)

**Port**: 8004

## Using Providers

### With Automated Scripts

The easiest way to use providers is through the provided scripts:

```bash
# Start Chatterbox-turbo
./scripts/start-chatterbox.sh

# Stop Chatterbox-turbo
./scripts/stop-chatterbox.sh
```

### Manual Installation

To manually install a provider:

```bash
# 1. Create virtual environment
python3 -m venv ~/provider-name
source ~/provider-name/bin/activate

# 2. Install dependencies
cd providers/provider-name
pip install -r requirements.txt

# 3. Copy and run server
mkdir -p ~/provider-name/server
cp provider-implementation/server.py ~/provider-name/server/
cd ~/provider-name/server
python server.py
```

## Adding New Providers

To add a new TTS provider:

1. Create a new directory under `providers/`
2. Add `requirements.txt` with dependencies
3. Create the provider implementation with:
   - FastAPI server
   - OpenAI-compatible `/v1/audio/speech` endpoint
   - Health check endpoint (`/health`)
4. Add start/stop scripts to `scripts/`
5. Add documentation to `docs/`

### Required API Endpoints

All providers must implement:

#### POST /v1/audio/speech

Request body (JSON):
```json
{
  "input": "Text to synthesize",
  "model": "tts-1",
  "voice": "default",
  "response_format": "mp3"
}
```

Response: Audio file (WAV, MP3, or other audio format)

#### GET /health

Response:
```json
{
  "status": "healthy",
  "model": "provider-name"
}
```

## Technical Notes

### The Chatterbox-turbo Fix

The original chatterbox-turbo server had a bug where it expected query parameters instead of JSON body. This caused compatibility issues with Voice Mode MCP.

**Original (buggy) implementation**:
```python
@app.post("/v1/audio/speech")
async def generate_speech(
    input: str,  # Query parameter
    model_name: str = "tts-1",
    ...
):
```

**Fixed implementation**:
```python
class TTSRequest(BaseModel):
    input: str
    model: str = "tts-1"
    ...

@app.post("/v1/audio/speech")
async def generate_speech(request: TTSRequest):  # JSON body
```

This fix ensures proper OpenAI API compatibility and seamless integration with Voice Mode MCP.

## License

MIT - See main repository LICENSE file.
