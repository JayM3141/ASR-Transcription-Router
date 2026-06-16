# CrisperWhisper

## Overview

CrisperWhisper is an advanced speech-to-text transcription system based on OpenAI's Whisper model, optimized for verbatim transcription with precise word-level timestamps. The project provides accurate transcription including filler words ("um", "uh"), reduced hallucinations, and precise timing even around disfluencies and pauses. It achieved 1st place on the OpenASR Leaderboard for verbatim datasets and was accepted at INTERSPEECH 2024.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Core Application Design
- **Python-based ML pipeline**: The application uses PyTorch and Hugging Face Transformers for speech recognition
- **Model source**: Uses a custom fork of Hugging Face Transformers (`nyrahealth/transformers@crisper_whisper`) with modifications for improved timestamp accuracy
- **Pre-trained model**: Loads the `nyrahealth/CrisperWhisper` model from Hugging Face Hub

### Processing Pipeline
- **Audio processing**: Uses `torchaudio` for audio loading and transformation, with `moviepy` for video file support
- **Inference pipeline**: Hugging Face `pipeline` API with chunked processing (30-second chunks, batch size 16)
- **Timestamp refinement**: Custom post-processing in `utils.py` adjusts pause timings between words for more natural segmentation

### User Interfaces
1. **CLI tool** (`transcribe.py`): Simple command-line interface for transcribing audio files
2. **Streamlit web app** (`app.py`): Interactive web interface with microphone recording support via `streamlit_mic_recorder`

### Device Handling
- Automatic GPU detection with CUDA support
- Falls back to CPU with float32 precision when GPU unavailable
- Uses float16 precision on GPU for faster inference

## External Dependencies

### ML/AI Services
- **Hugging Face Hub**: Model hosting and download (`nyrahealth/CrisperWhisper`)
- **Custom Transformers Fork**: `git+https://github.com/nyrahealth/transformers.git@crisper_whisper` - Modified Hugging Face Transformers with CrisperWhisper-specific features

### Key Python Libraries
- **PyTorch + torchaudio**: Core ML framework and audio processing
- **Transformers + Accelerate**: Model loading and inference optimization
- **Streamlit**: Web application framework
- **moviepy**: Video file audio extraction
- **librosa/scipy**: Audio file I/O and processing