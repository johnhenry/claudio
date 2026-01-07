#!/usr/bin/env python3
"""
Chatterbox TTS Server - OpenAI-compatible TTS API
Fixed to accept JSON body instead of query parameters
"""
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import uvicorn
from chatterbox.tts_turbo import ChatterboxTurboTTS
import io


class TTSRequest(BaseModel):
    input: str
    model: str = "tts-1"
    voice: str = "default"
    response_format: str = "mp3"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the model
    try:
        model = ChatterboxTurboTTS()
        app.state.model = model
        print("✓ Chatterbox-turbo model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load Chatterbox-turbo model: {e}")
        raise RuntimeError(f"Failed to initialize TTS model: {e}")
    
    yield
    
    # Shutdown: Clean up resources
    app.state.model = None
    print("✓ Chatterbox-turbo model unloaded")


app = FastAPI(title="Chatterbox TTS Server", lifespan=lifespan)


@app.post("/v1/audio/speech")
async def generate_speech(request: TTSRequest):
    """OpenAI-compatible TTS endpoint"""
    model = app.state.model
    
    if not request.input:
        raise HTTPException(status_code=400, detail="No input text provided")
    
    try:
        # Generate audio
        wav = model.generate(request.input)
        
        # Convert to bytes
        # Note: ChatterboxTurboTTS currently only supports WAV output.
        # The response_format parameter is accepted for API compatibility
        # but the actual format is always WAV regardless of the request.
        buffer = io.BytesIO()
        wav.export(buffer, format="wav")
        audio_bytes = buffer.getvalue()
        
        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f'attachment; filename="speech.wav"'
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate speech: {str(e)}"
        )


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "chatterbox-turbo"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8004)
