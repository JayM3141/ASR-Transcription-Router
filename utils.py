import contextlib
import importlib
import io
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchaudio
import torchaudio.transforms as T


# ---------------------------------------------------------------------------
# Device / dtype helpers  (previously duplicated in app.py and transcribe.py)
# ---------------------------------------------------------------------------
HAS_CUDA: bool = torch.cuda.is_available()


def get_device() -> str:
    """Return the best available torch device string."""
    return "cuda:0" if HAS_CUDA else "cpu"


def get_torch_dtype() -> torch.dtype:
    """Return float16 on CUDA, float32 on CPU."""
    return torch.float16 if HAS_CUDA else torch.float32


def get_compute_dtype(prefer_bfloat16: bool = False) -> torch.dtype:
    """Return a compute dtype suitable for the current hardware.

    Several model loaders (Voxtral, Granite, GLM-ASR) prefer bfloat16 on GPU;
    pass ``prefer_bfloat16=True`` for those.
    """
    if not HAS_CUDA:
        return torch.float32
    return torch.bfloat16 if prefer_bfloat16 else torch.float16


# ---------------------------------------------------------------------------
# Optional-import helper  (replaces repeated try/except blocks in app.py)
# ---------------------------------------------------------------------------
def check_optional_import(module_name: str) -> Tuple[Optional[Any], bool]:
    """Try to import *module_name*; return ``(module, True)`` or ``(None, False)``."""
    try:
        return importlib.import_module(module_name), True
    except Exception:
        return None, False


def check_optional_attr(module_name: str, attr: str) -> Tuple[Optional[Any], bool]:
    """Import *module_name* and look up *attr*; return ``(obj, True)`` or ``(None, False)``."""
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, attr), True
    except Exception:
        return None, False


# ---------------------------------------------------------------------------
# Audio processing helpers
# ---------------------------------------------------------------------------
def audio_bytes_to_waveform(audio_bytes: bytes, target_sr: int = 16000) -> torch.Tensor:
    """Decode *audio_bytes* into a mono float32 waveform at *target_sr* Hz.

    Returns a tensor of shape ``(1, num_samples)``.
    """
    audio_stream = io.BytesIO(audio_bytes)
    waveform, sr = torchaudio.load(audio_stream, backend="ffmpeg")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.to(torch.float32)
    if sr != target_sr:
        waveform = T.Resample(sr, target_sr)(waveform)
    return waveform


def audio_bytes_to_numpy(audio_bytes: bytes, target_sr: int = 16000):
    """Convenience wrapper: decode audio bytes and return a 1-D numpy array."""
    return audio_bytes_to_waveform(audio_bytes, target_sr)[0].numpy()


def save_waveform(waveform: torch.Tensor, path: str, sample_rate: int = 16000) -> None:
    """Persist a waveform tensor to *path* via torchaudio."""
    torchaudio.save(path, waveform, sample_rate=sample_rate, backend="ffmpeg")


# ---------------------------------------------------------------------------
# Temp-file context manager  (replaces repeated write+unlink in API funcs)
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def temp_audio_file(audio_bytes: bytes, suffix: str = ".wav"):
    """Write *audio_bytes* to a temp file, yield the path, then clean up."""
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(fd, audio_bytes)
        os.close(fd)
        yield tmp_path
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Whisper pipeline builder  (previously duplicated in app.py & transcribe.py)
# ---------------------------------------------------------------------------
def build_whisper_pipeline(
    model_id: str,
    *,
    device: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    batch_size: int = 1,
    return_timestamps: Any = True,
):
    """Load a Whisper-family model and return a HuggingFace ASR pipeline.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier (e.g. ``"openai/whisper-large-v3"``).
    device, torch_dtype : optional
        Override device / dtype; defaults come from :func:`get_device` /
        :func:`get_torch_dtype`.
    batch_size : int
        Pipeline batch size (``app.py`` uses 1, ``transcribe.py`` uses 16).
    return_timestamps : bool | str
        Passed through to the pipeline (``True`` or ``"word"``).
    """
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    if device is None:
        device = get_device()
    if torch_dtype is None:
        torch_dtype = get_torch_dtype()

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)
    if hasattr(model, "generation_config"):
        model.generation_config.median_filter_width = 3

    processor = AutoProcessor.from_pretrained(model_id)

    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=batch_size,
        return_timestamps=return_timestamps,
        torch_dtype=torch_dtype,
        device=device,
    )


# ---------------------------------------------------------------------------
# Standardised transcription result
# ---------------------------------------------------------------------------
def make_transcription_result(
    text: str, chunks: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Return a normalised ``{"text": ..., "chunks": ...}`` dict."""
    return {"text": text.strip() if text else "", "chunks": chunks or []}


# ---------------------------------------------------------------------------
# Pause-adjustment (original util kept intact)
# ---------------------------------------------------------------------------
def adjust_pauses_for_hf_pipeline_output(pipeline_output, split_threshold=0.12):
    """Distribute pauses up to *split_threshold* evenly between adjacent words."""
    adjusted_chunks = pipeline_output["chunks"].copy()

    for i in range(len(adjusted_chunks) - 1):
        current_chunk = adjusted_chunks[i]
        next_chunk = adjusted_chunks[i + 1]

        current_start, current_end = current_chunk["timestamp"]
        next_start, next_end = next_chunk["timestamp"]
        pause_duration = next_start - current_end

        if pause_duration > 0:
            if pause_duration > split_threshold:
                distribute = split_threshold / 2
            else:
                distribute = pause_duration / 2

            adjusted_chunks[i]["timestamp"] = (current_start, current_end + distribute)
            adjusted_chunks[i + 1]["timestamp"] = (next_start - distribute, next_end)

    pipeline_output["chunks"] = adjusted_chunks
    return pipeline_output
