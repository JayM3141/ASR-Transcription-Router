import argparse
import io
import os
import tempfile
from typing import Any, Dict, List, Optional, Union

import moviepy as mp
import numpy as np
import streamlit as st
import torch
import torchaudio
import torchaudio.transforms as T
from streamlit_mic_recorder import mic_recorder

# ---------------------------------------------------------------------------
# Optional framework detection (graceful – app works without them)
# ---------------------------------------------------------------------------
try:
    import nemo.collections.asr as nemo_asr
    NEMO_AVAILABLE = True
except Exception:
    NEMO_AVAILABLE = False

try:
    import speechbrain  # noqa: F401
    SPEECHBRAIN_AVAILABLE = True
except Exception:
    SPEECHBRAIN_AVAILABLE = False

HAS_CUDA = torch.cuda.is_available()
device: str = "cuda:0" if HAS_CUDA else "cpu"
torch_dtype: torch.dtype = torch.float16 if HAS_CUDA else torch.float32

# ---------------------------------------------------------------------------
# Model type constants
# ---------------------------------------------------------------------------
T_WHISPER      = "transformers_whisper"  # encoder-decoder Whisper-style
T_GENERIC      = "transformers_generic"  # any other HF pipeline model
T_NEMO         = "nemo"                  # NVIDIA NeMo toolkit
T_SPEECHBRAIN  = "speechbrain"           # SpeechBrain library
T_UNSUPPORTED  = "unsupported"           # needs custom framework not yet wired up
API_ASSEMBLYAI   = "api_assemblyai"
API_ELEVENLABS   = "api_elevenlabs"
API_REVAI        = "api_revai"
API_GOOGLE       = "api_google"
API_AZURE        = "api_azure"
API_COHERE       = "api_cohere"
API_SPEECHMATICS = "api_speechmatics"
API_GENERIC      = "api_generic"      # commercial API, no SDK yet

# ---------------------------------------------------------------------------
# Model registry  –  type + per-model metadata
# ---------------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ── CrisperWhisper ──────────────────────────────────────────────────
    "nyrahealth/CrisperWhisper":               {"type": T_WHISPER},
    # ── OpenAI Whisper ──────────────────────────────────────────────────
    "openai/whisper-large-v3-turbo":           {"type": T_WHISPER},
    "openai/whisper-large-v3":                 {"type": T_WHISPER},
    "openai/whisper-large-v2":                 {"type": T_WHISPER},
    "openai/whisper-medium":                   {"type": T_WHISPER},
    "openai/whisper-small":                    {"type": T_WHISPER},
    "openai/whisper-base":                     {"type": T_WHISPER},
    "openai/whisper-tiny":                     {"type": T_WHISPER},
    # ── Distil-Whisper ──────────────────────────────────────────────────
    "distil-whisper/distil-large-v3.5":        {"type": T_WHISPER},
    "distil-whisper/distil-large-v3":          {"type": T_WHISPER},
    "distil-whisper/distil-medium.en":         {"type": T_WHISPER},
    "distil-whisper/distil-small.en":          {"type": T_WHISPER},
    "efficient-speech/lite-whisper-large-v3-acc": {"type": T_WHISPER},
    # ── NVIDIA NeMo ─────────────────────────────────────────────────────
    "nvidia/canary-qwen-2.5b":                 {"type": T_NEMO, "nemo_class": "EncDecMultiTaskModel", "requires_gpu": False},
    "nvidia/canary-1b-v2":                     {"type": T_NEMO, "nemo_class": "EncDecMultiTaskModel", "requires_gpu": False},
    "nvidia/canary-1b-flash":                  {"type": T_NEMO, "nemo_class": "EncDecMultiTaskModel", "requires_gpu": False},
    "nvidia/canary-1b":                        {"type": T_NEMO, "nemo_class": "EncDecMultiTaskModel", "requires_gpu": False},
    "nvidia/canary-180m-flash":                {"type": T_NEMO, "nemo_class": "EncDecMultiTaskModel", "requires_gpu": False},
    "nvidia/parakeet-tdt-0.6b-v3":             {"type": T_NEMO, "nemo_class": "EncDecRNNTBPEModel",   "requires_gpu": False},
    "nvidia/parakeet-tdt-1.1b":                {"type": T_NEMO, "nemo_class": "EncDecRNNTBPEModel",   "requires_gpu": False},
    "nvidia/parakeet-tdt-0.6b-v2":             {"type": T_NEMO, "nemo_class": "EncDecRNNTBPEModel",   "requires_gpu": False},
    "nvidia/parakeet-rnnt-1.1b":               {"type": T_NEMO, "nemo_class": "EncDecRNNTBPEModel",   "requires_gpu": False},
    "nvidia/parakeet-ctc-1.1b":                {"type": T_NEMO, "nemo_class": "EncDecCTCModelBPE",    "requires_gpu": False},
    "nvidia/stt_en_conformer_transducer_small": {"type": T_NEMO, "nemo_class": "EncDecRNNTBPEModel",  "requires_gpu": False},
    # ── Meta / Facebook ─────────────────────────────────────────────────
    "facebook/wav2vec2-large-960h-lv60-self":  {"type": T_GENERIC},
    "facebook/wav2vec2-base-960h":             {"type": T_GENERIC},
    "facebook/mms-300m":                       {"type": T_GENERIC},
    "facebook/mms-1b-fl102":                   {"type": T_GENERIC},
    "facebook/seamless-m4t-v2-large":          {"type": T_GENERIC},
    "facebook/hubert-large-ls960-ft":          {"type": T_GENERIC},
    "facebook/hubert-xlarge-ls960-ft":         {"type": T_GENERIC},
    "facebook/wav2vec2-base-10k-voxpopuli-ft-en": {"type": T_GENERIC},
    # ── Microsoft ───────────────────────────────────────────────────────
    "microsoft/speecht5_asr":                  {"type": T_GENERIC},
    "microsoft/Phi-4-multimodal-instruct":     {"type": T_UNSUPPORTED,
                                                "reason": "Phi-4 multimodal requires a custom chat-style inference pipeline (not a standard ASR pipeline). Use the Azure Speech API or a Whisper model instead."},
    "microsoft/azure-speech-05-2026":          {"type": API_AZURE,        "label": "Azure Speech"},
    # ── Google ──────────────────────────────────────────────────────────
    "google/chirp_3":                          {"type": API_GOOGLE,       "label": "Google Chirp 3"},
    "google/chirp_2":                          {"type": API_GOOGLE,       "label": "Google Chirp 2"},
    # ── IBM Granite ─────────────────────────────────────────────────────
    "ibm-granite/granite-speech-4.1-2b-nar":  {"type": T_GENERIC},
    "ibm-granite/granite-speech-4.1-2b":      {"type": T_GENERIC},
    "ibm-granite/granite-speech-3.3-8b":      {"type": T_GENERIC},
    "ibm-granite/granite-speech-3.3-2b":      {"type": T_GENERIC},
    "ibm-granite/granite-4.0-1b-speech":      {"type": T_GENERIC},
    # ── Mistral ─────────────────────────────────────────────────────────
    "mistralai/Voxtral-Small-24B-2507":        {"type": T_UNSUPPORTED,
                                                "reason": "Voxtral is a multimodal LLM requiring vLLM or custom chat-style inference — not a standard ASR pipeline. Use a Whisper or Distil-Whisper model instead."},
    "mistralai/Voxtral-Mini-3B-2507":          {"type": T_UNSUPPORTED,
                                                "reason": "Voxtral is a multimodal LLM requiring vLLM or custom chat-style inference — not a standard ASR pipeline. Use a Whisper or Distil-Whisper model instead."},
    # ── Qwen ────────────────────────────────────────────────────────────
    "Qwen/Qwen3-ASR-1.7B":                     {"type": T_GENERIC},
    "Qwen/Qwen3-ASR-0.6B":                     {"type": T_GENERIC},
    # ── Kyutai ──────────────────────────────────────────────────────────
    "kyutai/stt-2.6b-en":                      {"type": T_GENERIC},
    # ── Boson AI ────────────────────────────────────────────────────────
    "bosonai/higgs-audio-v3-8b-stt-v2":        {"type": T_GENERIC},
    # ── ZAI / GLM ───────────────────────────────────────────────────────
    "zai-org/GLM-ASR-Nano-2512":               {"type": T_GENERIC},
    # ── ESPnet / OWSM ───────────────────────────────────────────────────
    "espnet/owsm_ctc_v4_1B":                   {"type": T_UNSUPPORTED,
                                                "reason": "ESPnet OWSM models use the ESPnet2 framework (`pip install espnet`). They are not loadable via the standard HuggingFace Transformers pipeline."},
    "pyf98/owsm_ctc_v3.1_1B":                 {"type": T_UNSUPPORTED,
                                                "reason": "ESPnet OWSM models use the ESPnet2 framework (`pip install espnet`). They are not loadable via the standard HuggingFace Transformers pipeline."},
    # ── SoundsgoodAI ────────────────────────────────────────────────────
    "soundsgoodai/Zipformer-transducer-XL-290M": {"type": T_UNSUPPORTED,
                                                   "reason": "Zipformer-transducer models use the k2/icefall framework and are not loadable via the standard HuggingFace Transformers pipeline."},
    # ── Useful Sensors ──────────────────────────────────────────────────
    "usefulsensors/moonshine-streaming-medium": {"type": T_GENERIC},
    # ── Wav2Vec2 / HuBERT ───────────────────────────────────────────────
    "jonatasgrosman/wav2vec2-large-xlsr-53-english": {"type": T_GENERIC},
    # ── SpeechBrain ─────────────────────────────────────────────────────
    "speechbrain/asr-conformer-transformerlm-librispeech": {"type": T_SPEECHBRAIN, "sb_class": "EncoderDecoderASR"},
    "speechbrain/asr-wav2vec2-librispeech":    {"type": T_SPEECHBRAIN, "sb_class": "EncoderASR"},
    # ── Commercial APIs ─────────────────────────────────────────────────
    "assemblyai/universal-3-pro":              {"type": API_ASSEMBLYAI,   "label": "AssemblyAI Universal-3 Pro"},
    "aquavoice/avalon-v1-en":                  {"type": API_GENERIC,      "label": "Aqua Voice Avalon"},
    "CohereLabs/cohere-transcribe-03-2026":    {"type": API_COHERE,       "label": "Cohere Transcribe"},
    "elevenlabs/scribe_v1":                    {"type": API_ELEVENLABS,   "label": "ElevenLabs Scribe v1"},
    "elevenlabs/scribe_v2":                    {"type": API_ELEVENLABS,   "label": "ElevenLabs Scribe v2"},
    "reson8/resonant-1":                       {"type": API_GENERIC,      "label": "Reson8 Resonant-1"},
    "reson8/resonant-1-flash":                 {"type": API_GENERIC,      "label": "Reson8 Resonant-1 Flash"},
    "revai/fusion":                            {"type": API_REVAI,        "label": "Rev AI Fusion"},
    "smallestai/pulse":                        {"type": API_GENERIC,      "label": "SmallestAI Pulse"},
    "speechmatics/enhanced":                   {"type": API_SPEECHMATICS, "label": "Speechmatics Enhanced"},
    "zoom/scribe_v1":                          {"type": API_GENERIC,      "label": "Zoom Scribe v1"},
}

# API types that need keys (shown in sidebar)
API_KEY_CONFIG = {
    API_ASSEMBLYAI:   {"label": "AssemblyAI",   "key": "assemblyai_key",   "url": "https://www.assemblyai.com/app"},
    API_ELEVENLABS:   {"label": "ElevenLabs",   "key": "elevenlabs_key",   "url": "https://elevenlabs.io/settings/api-keys"},
    API_REVAI:        {"label": "Rev AI",        "key": "revai_key",        "url": "https://www.rev.ai/access_token"},
    API_COHERE:       {"label": "Cohere",        "key": "cohere_key",       "url": "https://dashboard.cohere.com/api-keys"},
    API_GOOGLE:       {"label": "Google Cloud",  "key": "google_key",       "url": "https://console.cloud.google.com/apis/credentials"},
    API_AZURE:        {"label": "Azure Speech",  "key": "azure_key",        "url": "https://portal.azure.com"},
    API_SPEECHMATICS: {"label": "Speechmatics",  "key": "speechmatics_key", "url": "https://portal.speechmatics.com/manage-access/"},
    API_GENERIC:      {"label": "API",           "key": "generic_key",      "url": ""},
}

# ---------------------------------------------------------------------------
# Model groups for the sidebar dropdown
# ---------------------------------------------------------------------------
MODEL_GROUPS: Dict[str, List[str]] = {
    "⭐ CrisperWhisper":           ["nyrahealth/CrisperWhisper"],
    "🤫 OpenAI Whisper":           ["openai/whisper-large-v3-turbo","openai/whisper-large-v3","openai/whisper-large-v2","openai/whisper-medium","openai/whisper-small","openai/whisper-base","openai/whisper-tiny"],
    "⚡ Distil-Whisper":           ["distil-whisper/distil-large-v3.5","distil-whisper/distil-large-v3","distil-whisper/distil-medium.en","distil-whisper/distil-small.en","efficient-speech/lite-whisper-large-v3-acc"],
    "🟢 NVIDIA Parakeet/Canary":   ["nvidia/canary-qwen-2.5b","nvidia/canary-1b-v2","nvidia/canary-1b-flash","nvidia/canary-1b","nvidia/canary-180m-flash","nvidia/parakeet-tdt-0.6b-v3","nvidia/parakeet-tdt-1.1b","nvidia/parakeet-tdt-0.6b-v2","nvidia/parakeet-rnnt-1.1b","nvidia/parakeet-ctc-1.1b","nvidia/stt_en_conformer_transducer_small"],
    "🦙 Meta / Facebook":          ["facebook/wav2vec2-large-960h-lv60-self","facebook/wav2vec2-base-960h","facebook/mms-300m","facebook/mms-1b-fl102","facebook/seamless-m4t-v2-large","facebook/wav2vec2-base-10k-voxpopuli-ft-en"],
    "🪟 Microsoft":                ["microsoft/speecht5_asr","microsoft/Phi-4-multimodal-instruct","microsoft/azure-speech-05-2026"],
    "🔍 Google":                   ["google/chirp_3","google/chirp_2"],
    "💎 IBM Granite Speech":       ["ibm-granite/granite-speech-4.1-2b-nar","ibm-granite/granite-speech-4.1-2b","ibm-granite/granite-speech-3.3-8b","ibm-granite/granite-speech-3.3-2b","ibm-granite/granite-4.0-1b-speech"],
    "🌊 Mistral AI":               ["mistralai/Voxtral-Mini-3B-2507","mistralai/Voxtral-Small-24B-2507"],
    "🐲 Qwen":                     ["Qwen/Qwen3-ASR-1.7B","Qwen/Qwen3-ASR-0.6B"],
    "🎙️ ElevenLabs":              ["elevenlabs/scribe_v2","elevenlabs/scribe_v1"],
    "🎵 Kyutai":                   ["kyutai/stt-2.6b-en"],
    "📡 AssemblyAI":               ["assemblyai/universal-3-pro"],
    "🔊 Aqua Voice":               ["aquavoice/avalon-v1-en"],
    "🤖 Boson AI":                 ["bosonai/higgs-audio-v3-8b-stt-v2"],
    "🧪 Cohere":                   ["CohereLabs/cohere-transcribe-03-2026"],
    "🔉 Reson8":                   ["reson8/resonant-1","reson8/resonant-1-flash"],
    "📝 Rev AI":                   ["revai/fusion"],
    "🎤 Speechmatics":             ["speechmatics/enhanced"],
    "🔬 Smallest AI":              ["smallestai/pulse"],
    "🌙 Useful Sensors":           ["usefulsensors/moonshine-streaming-medium"],
    "📹 Zoom":                     ["zoom/scribe_v1"],
    "🔵 ZAI / GLM":                ["zai-org/GLM-ASR-Nano-2512"],
    "🎓 ESPnet / OWSM":            ["espnet/owsm_ctc_v4_1B","pyf98/owsm_ctc_v3.1_1B"],
    "🔈 SoundsgoodAI":             ["soundsgoodai/Zipformer-transducer-XL-290M"],
    "🧠 Wav2Vec2 / HuBERT":        ["facebook/hubert-large-ls960-ft","facebook/hubert-xlarge-ls960-ft","jonatasgrosman/wav2vec2-large-xlsr-53-english"],
    "🗣️ SpeechBrain":              ["speechbrain/asr-wav2vec2-librispeech","speechbrain/asr-conformer-transformerlm-librispeech"],
}

DEFAULT_MODEL = "nyrahealth/CrisperWhisper"

# All models flat list
ALL_MODELS: List[str] = [m for models in MODEL_GROUPS.values() for m in models]

def get_model_info(model_id: str) -> Dict[str, Any]:
    return MODEL_REGISTRY.get(model_id, {"type": T_GENERIC})

def is_api_type(t: str) -> bool:
    return t.startswith("api_")


# ---------------------------------------------------------------------------
# Pipeline / model loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model… this may take a few minutes on first run.")
def load_transformers_pipeline(model_id: str, arch: str) -> Any:
    from transformers import pipeline as hf_pipeline
    if arch == T_WHISPER:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(device)
        if hasattr(model, "generation_config"):
            model.generation_config.median_filter_width = 3
        processor = AutoProcessor.from_pretrained(model_id)
        return hf_pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=30,
            batch_size=1,
            return_timestamps=True,
            torch_dtype=torch_dtype,
            device=device,
        )
    else:
        return hf_pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=device,
            torch_dtype=torch_dtype,
        )


@st.cache_resource(show_spinner="Loading SpeechBrain model… this may take a few minutes on first run.")
def load_speechbrain_model(model_id: str, sb_class: str) -> Any:
    if not SPEECHBRAIN_AVAILABLE:
        raise RuntimeError(
            "SpeechBrain is not installed. Install it with:\n"
            "`pip install speechbrain`\n\nThen restart the app."
        )
    try:
        from speechbrain.inference.ASR import EncoderASR, EncoderDecoderASR
    except ImportError:
        from speechbrain.pretrained import EncoderASR, EncoderDecoderASR  # type: ignore[no-redef]
    cls_map = {"EncoderASR": EncoderASR, "EncoderDecoderASR": EncoderDecoderASR}
    cls = cls_map.get(sb_class, EncoderASR)
    savedir = f"pretrained_models/{model_id.replace('/', '_')}"
    return cls.from_hparams(source=model_id, savedir=savedir)


@st.cache_resource(show_spinner="Loading NeMo model… this may take a few minutes on first run.")
def load_nemo_model(model_id: str, nemo_class: str) -> Any:
    if not NEMO_AVAILABLE:
        raise RuntimeError(
            "NeMo toolkit is not installed. Install it with:\n"
            "`pip install nemo_toolkit[asr]`\n\n"
            "Then restart the app."
        )
    cls_map = {
        "EncDecCTCModelBPE":    nemo_asr.models.EncDecCTCModelBPE,
        "EncDecRNNTBPEModel":   nemo_asr.models.EncDecRNNTBPEModel,
        "EncDecMultiTaskModel": nemo_asr.models.EncDecMultiTaskModel,
    }
    model_cls = cls_map.get(nemo_class)
    if model_cls is None:
        model_cls = nemo_asr.models.ASRModel
    return model_cls.from_pretrained(model_name=model_id)


# ---------------------------------------------------------------------------
# Audio processing
# ---------------------------------------------------------------------------
def process_audio_bytes(audio_bytes: bytes) -> torch.Tensor:
    audio_stream = io.BytesIO(audio_bytes)
    waveform, sr = torchaudio.load(audio_stream, backend="ffmpeg")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.to(torch.float32)
    if sr != 16000:
        waveform = T.Resample(sr, 16000)(waveform)
    torchaudio.save("sample.wav", waveform, sample_rate=16000, backend="ffmpeg")
    return waveform


def wav_to_black_mp4(wav_path: str, output_path: str, fps: int = 25) -> None:
    waveform, sample_rate = torchaudio.load(wav_path)
    duration: float = waveform.shape[1] / sample_rate
    audio = mp.AudioFileClip(wav_path)
    black_clip = mp.ColorClip((256, 250), color=(0, 0, 0), duration=duration)
    final_clip = black_clip.with_audio(audio)
    final_clip.write_videofile(output_path, fps=fps)


def timestamps_to_vtt(timestamps: List[Dict[str, Any]]) -> str:
    def fmt(t: float) -> str:
        return f"{int(t // 3600)}:{int(t // 60 % 60):02d}:{t % 60:06.3f}"
    vtt = "WEBVTT\n\n"
    for i, word in enumerate(timestamps):
        start, end = word["timestamp"]
        if start is None:
            continue
        if end is None:
            end = (timestamps[i+1]["timestamp"][0] if i+1 < len(timestamps) and timestamps[i+1]["timestamp"][0] else start + 1.0)
        vtt += f"{fmt(start)} --> {fmt(end)}\n{word['text']}\n\n"
    return vtt


# ---------------------------------------------------------------------------
# Transcription dispatch
# ---------------------------------------------------------------------------
def transcribe_transformers(audio_bytes: bytes, pipe: Any) -> Dict[str, Any]:
    waveform = process_audio_bytes(audio_bytes)
    return pipe(waveform[0, :].numpy(), return_timestamps="word")


def transcribe_nemo(audio_bytes: bytes, model: Any) -> Dict[str, Any]:
    waveform = process_audio_bytes(audio_bytes)  # saves sample.wav
    results = model.transcribe(["sample.wav"])
    text = results[0] if isinstance(results[0], str) else results[0].text
    return {"text": text, "chunks": []}


def transcribe_speechbrain(audio_bytes: bytes, model: Any) -> Dict[str, Any]:
    process_audio_bytes(audio_bytes)  # saves sample.wav
    result = model.transcribe_file("sample.wav")
    if isinstance(result, (list, tuple)):
        text = " ".join(str(r) for r in result).strip()
    else:
        text = str(result).strip()
    return {"text": text, "chunks": []}


def transcribe_assemblyai(audio_bytes: bytes, api_key: str) -> Dict[str, Any]:
    import assemblyai as aai
    aai.settings.api_key = api_key
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.best, word_boost=[])
        transcriber = aai.Transcriber(config=config)
        transcript = transcriber.transcribe(tmp_path)
        if transcript.error:
            raise RuntimeError(transcript.error)
        chunks = []
        if transcript.words:
            for w in transcript.words:
                chunks.append({"text": " " + w.text, "timestamp": (w.start / 1000.0, w.end / 1000.0)})
        return {"text": transcript.text or "", "chunks": chunks}
    finally:
        os.unlink(tmp_path)


def transcribe_elevenlabs(audio_bytes: bytes, api_key: str, model_id: str) -> Dict[str, Any]:
    from elevenlabs.client import ElevenLabs
    client = ElevenLabs(api_key=api_key)
    el_model = "scribe_v2" if "v2" in model_id else "scribe_v1"
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        with open(tmp_path, "rb") as audio_file:
            result = client.speech_to_text.convert(audio=audio_file, model_id=el_model)
        chunks = []
        if hasattr(result, "words") and result.words:
            for w in result.words:
                chunks.append({"text": " " + w.text, "timestamp": (w.start, w.end)})
        return {"text": result.text or "", "chunks": chunks}
    finally:
        os.unlink(tmp_path)


def transcribe_revai(audio_bytes: bytes, api_key: str) -> Dict[str, Any]:
    from rev_ai import apiclient
    client = apiclient.RevAiAPIClient(api_key)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        job = client.submit_job_local_file(tmp_path)
        import time
        while True:
            details = client.get_job_details(job.id)
            if details.status.name in ("transcribed", "failed"):
                break
            time.sleep(2)
        if details.status.name == "failed":
            raise RuntimeError(f"Rev AI job failed: {details.failure_detail}")
        transcript_obj = client.get_transcript_object(job.id)
        text_parts, chunks = [], []
        for mono in transcript_obj.monologues:
            for elem in mono.elements:
                if elem.type_ == "text":
                    text_parts.append(elem.value)
                    chunks.append({"text": " " + elem.value, "timestamp": (elem.ts, elem.end_ts)})
        return {"text": " ".join(text_parts), "chunks": chunks}
    finally:
        os.unlink(tmp_path)


def transcribe_audio(
    audio_bytes: bytes,
    model_id: str,
    model_or_pipe: Any,
    api_keys: Dict[str, str],
) -> Dict[str, Any]:
    info = get_model_info(model_id)
    t = info["type"]

    if t in (T_WHISPER, T_GENERIC):
        return transcribe_transformers(audio_bytes, model_or_pipe)
    elif t == T_NEMO:
        return transcribe_nemo(audio_bytes, model_or_pipe)
    elif t == T_SPEECHBRAIN:
        return transcribe_speechbrain(audio_bytes, model_or_pipe)
    elif t == T_UNSUPPORTED:
        raise RuntimeError(active_info.get("reason", "This model requires a custom inference framework not yet integrated."))
    elif t == API_ASSEMBLYAI:
        key = api_keys.get("assemblyai_key", "")
        if not key:
            raise RuntimeError("AssemblyAI API key is required. Enter it in the sidebar.")
        return transcribe_assemblyai(audio_bytes, key)
    elif t == API_ELEVENLABS:
        key = api_keys.get("elevenlabs_key", "")
        if not key:
            raise RuntimeError("ElevenLabs API key is required. Enter it in the sidebar.")
        return transcribe_elevenlabs(audio_bytes, key, model_id)
    elif t == API_REVAI:
        key = api_keys.get("revai_key", "")
        if not key:
            raise RuntimeError("Rev AI API key is required. Enter it in the sidebar.")
        return transcribe_revai(audio_bytes, key)
    else:
        provider_label = info.get("label", model_id)
        raise RuntimeError(
            f"**{provider_label}** is a commercial API model. "
            "Direct API integration for this provider is not yet implemented. "
            "Please use the custom model ID field to enter a HuggingFace-hosted alternative, "
            "or contact the provider for API access."
        )


# ---------------------------------------------------------------------------
# CLI arg (kept for workflow launch compat – used as sidebar default)
# ---------------------------------------------------------------------------
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL)
    return parser.parse_args()


args = parse_arguments()
cli_default = args.model_id if args.model_id in ALL_MODELS else DEFAULT_MODEL

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Model Settings")
st.sidebar.caption(
    "Select any model from the [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard). "
    "Models are downloaded and cached on first use."
)

group_display: List[str] = []
for models in MODEL_GROUPS.values():
    group_display.extend(models)

default_idx = group_display.index(cli_default) if cli_default in group_display else 0

selected_model = st.sidebar.selectbox(
    "Select a model",
    options=group_display,
    index=default_idx,
    format_func=lambda m: f"{m}  [{get_model_info(m)['type'].upper().replace('TRANSFORMERS_','').replace('_',' ')}]",
    help="Switch between ASR models.",
)

st.sidebar.markdown("---")
custom_model_id = st.sidebar.text_input(
    "Or enter any HuggingFace model ID",
    value="",
    placeholder="e.g. openai/whisper-medium",
)

active_model = custom_model_id.strip() if custom_model_id.strip() else selected_model
active_info = get_model_info(active_model)
active_type = active_info["type"]

st.sidebar.markdown(f"**Active:** `{active_model}`")
if not is_api_type(active_type):
    st.sidebar.markdown(f"[🔗 View on HuggingFace](https://huggingface.co/{active_model})")

# ── SpeechBrain info banner ───────────────────────────────────────────
if active_type == T_SPEECHBRAIN:
    if not SPEECHBRAIN_AVAILABLE:
        st.sidebar.warning("⚠️ **SpeechBrain not installed.**\n\nRun: `pip install speechbrain`\n\nThen restart the app.")
    else:
        st.sidebar.info("🗣️ SpeechBrain model — runs on CPU")

# ── Unsupported model banner ──────────────────────────────────────────
if active_type == T_UNSUPPORTED:
    st.sidebar.warning("⚠️ **Needs custom framework** — see main panel for details.")

# ── NeMo info banner ──────────────────────────────────────────────────
if active_type == T_NEMO:
    if not NEMO_AVAILABLE:
        st.sidebar.warning(
            "⚠️ **NeMo not installed.**\n\n"
            "Run: `pip install nemo_toolkit[asr]`\n\n"
            "Then restart the app."
        )
    else:
        gpu_note = "GPU available ✅" if HAS_CUDA else "CPU only — no CUDA needed for CPU path"
        st.sidebar.info(f"NeMo model — {gpu_note}")

# ── API key inputs ─────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("**🔑 API Keys** *(stored in session only)*")

if "api_keys" not in st.session_state:
    st.session_state.api_keys = {}

# Show key input for active API model, plus allow pre-filling others
api_types_present = {get_model_info(m)["type"] for m in ALL_MODELS if is_api_type(get_model_info(m)["type"])}
for api_type in sorted(api_types_present):
    cfg = API_KEY_CONFIG.get(api_type)
    if cfg is None:
        continue
    is_active = active_type == api_type
    existing = st.session_state.api_keys.get(cfg["key"], "")
    expanded = is_active and not existing
    label = f"{'🔴' if is_active and not existing else '🟢' if existing else '⚪'} {cfg['label']} API Key"
    with st.sidebar.expander(label, expanded=expanded):
        val = st.text_input(
            f"{cfg['label']} key",
            value=existing,
            type="password",
            key=f"input_{cfg['key']}",
            label_visibility="collapsed",
        )
        if val:
            st.session_state.api_keys[cfg["key"]] = val
        if cfg.get("url"):
            st.markdown(f"[Get API key ↗]({cfg['url']})")

# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
model_badge = f"`{active_model}`"
if active_type == T_NEMO:
    model_badge += " 🟢 NeMo"
elif active_type == T_SPEECHBRAIN:
    model_badge += " 🗣️ SpeechBrain"
elif active_type == T_UNSUPPORTED:
    model_badge += " ⚠️ Needs custom framework"
elif is_api_type(active_type):
    model_badge += " 🔑 API"

st.title("Open ASR Leaderboard Transcription 🦻")
st.markdown(
    f"Transcribing with {model_badge} &nbsp;|&nbsp; "
    f"[View leaderboard ↗](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)"
)

# Block early if requirements clearly not met
if active_type == T_NEMO and not NEMO_AVAILABLE:
    st.error(
        "**NeMo toolkit is not installed.**\n\n"
        "Install it with: `pip install nemo_toolkit[asr]`\n\n"
        "Once installed, restart the app."
    )
    st.stop()

if active_type == T_SPEECHBRAIN and not SPEECHBRAIN_AVAILABLE:
    st.error(
        "**SpeechBrain is not installed.**\n\n"
        "Install it with: `pip install speechbrain`\n\n"
        "Once installed, restart the app."
    )
    st.stop()

if active_type == T_UNSUPPORTED:
    reason = active_info.get("reason", "This model requires a custom inference framework.")
    st.warning(
        f"**`{active_model}` is not directly loadable via this app.**\n\n{reason}"
    )
    st.stop()

# Load model/pipeline (skipped for pure API models and unsupported models)
model_or_pipe = None
if active_type not in (T_UNSUPPORTED,) and not is_api_type(active_type):
    try:
        if active_type == T_NEMO:
            model_or_pipe = load_nemo_model(active_model, active_info.get("nemo_class", "ASRModel"))
        elif active_type == T_SPEECHBRAIN:
            model_or_pipe = load_speechbrain_model(active_model, active_info.get("sb_class", "EncoderASR"))
        else:
            model_or_pipe = load_transformers_pipeline(active_model, active_type)
    except Exception as load_err:
        st.error(f"**Failed to load `{active_model}`:**\n\n{load_err}")
        st.stop()

st.write("🎙️ Record audio below or 📁 upload an audio file.")

# Audio recorder
audio = mic_recorder(
    start_prompt="Start recording",
    stop_prompt="Stop recording",
    just_once=False,
    use_container_width=False,
    format="wav",
    callback=None,
    args=(),
    kwargs={},
    key=None,
)
audio_bytes: Optional[bytes] = audio["bytes"] if audio else None

# File upload
audio_file = st.file_uploader(
    "Or upload an audio file",
    type=["wav", "mp3", "ogg", "flac", "m4a", "aac", "amr", "opus", "wma", "webm"],
)
if audio_file is not None:
    audio_bytes = audio_file.getvalue()

if audio_bytes:
    with st.spinner("Transcribing…"):
        try:
            transcription = transcribe_audio(
                audio_bytes,
                active_model,
                model_or_pipe,
                st.session_state.api_keys,
            )
            chunks = transcription.get("chunks") or []

            if chunks:
                vtt = timestamps_to_vtt(chunks)
                with open("subtitles.vtt", "w") as f:
                    f.write(vtt)
                wav_to_black_mp4("sample.wav", "video.mp4")
                st.video("video.mp4", subtitles="subtitles.vtt")

            st.subheader("Transcription")
            st.markdown(
                f"""<div style="background:#f0f0f0;padding:12px;border-radius:6px;">
                    <p style="font-size:16px;color:#333;margin:0;">
                        {transcription.get("text","").strip()}
                    </p></div>""",
                unsafe_allow_html=True,
            )

            if chunks:
                with st.expander("Word-level timestamps"):
                    import pandas as pd
                    rows = []
                    for c in chunks:
                        s, e = c["timestamp"]
                        if s is None:
                            continue
                        rows.append({"Word": c["text"].strip(), "Start": f"{s:.2f}s", "End": f"{e:.2f}s" if e else "—"})
                    if rows:
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"**Transcription error:** {e}")

# Footer
st.markdown(
    "<hr><footer><p style='text-align:center;'>© 2024 nyra health GmbH</p></footer>",
    unsafe_allow_html=True,
)
