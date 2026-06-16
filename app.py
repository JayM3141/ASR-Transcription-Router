import argparse
import io
import os
import runpy
import subprocess
import sys
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

try:
    from transformers import VoxtralForConditionalGeneration  # noqa: F401
    VOXTRAL_AVAILABLE = True
except Exception:
    VOXTRAL_AVAILABLE = False

try:
    from transformers import GlmAsrForConditionalGeneration  # noqa: F401
    GLM_ASR_AVAILABLE = True
except Exception:
    GLM_ASR_AVAILABLE = False

try:
    import moshi  # noqa: F401
    MOSHI_AVAILABLE = True
except Exception:
    MOSHI_AVAILABLE = False

try:
    import qwen_asr  # noqa: F401
    QWEN_ASR_AVAILABLE = True
except Exception:
    QWEN_ASR_AVAILABLE = False

try:
    import espnet2  # noqa: F401
    ESPNET_AVAILABLE = True
except Exception:
    ESPNET_AVAILABLE = False

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
T_VOXTRAL      = "voxtral"               # VoxtralForConditionalGeneration (mistral-common)
T_GENERATE     = "generate"              # AutoModelForCausalLM / GlmAsrForConditionalGeneration
T_GRANITE      = "granite"               # AutoModelForSpeechSeq2Seq + chat template
T_TRUST        = "trust_remote"          # AutoModel trust_remote_code=True
T_QWEN_ASR     = "qwen_asr"             # qwen_asr.Qwen3ASRModel
T_KYUTAI       = "kyutai"               # moshi streaming ASR
T_ESPNET       = "espnet"               # espnet2 / OWSM CTC
T_ZIPFORMER    = "zipformer"            # k2/icefall (requires CUDA + k2)
API_ASSEMBLYAI   = "api_assemblyai"
API_ELEVENLABS   = "api_elevenlabs"
API_REVAI        = "api_revai"
API_GOOGLE       = "api_google"
API_AZURE        = "api_azure"
API_COHERE       = "api_cohere"
API_SPEECHMATICS = "api_speechmatics"
API_GENERIC      = "api_generic"

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
    # LiteASR: trust_remote_code + whisper-large-v3-turbo processor
    "efficient-speech/lite-whisper-large-v3-acc": {"type": T_TRUST, "lite_asr": True},
    # ── NVIDIA NeMo ─────────────────────────────────────────────────────
    "nvidia/canary-qwen-2.5b":                 {"type": T_NEMO, "nemo_class": "EncDecMultiTaskModel"},
    "nvidia/canary-1b-v2":                     {"type": T_NEMO, "nemo_class": "EncDecMultiTaskModel"},
    "nvidia/canary-1b-flash":                  {"type": T_NEMO, "nemo_class": "EncDecMultiTaskModel"},
    "nvidia/canary-1b":                        {"type": T_NEMO, "nemo_class": "EncDecMultiTaskModel"},
    "nvidia/canary-180m-flash":                {"type": T_NEMO, "nemo_class": "EncDecMultiTaskModel"},
    "nvidia/parakeet-tdt-0.6b-v3":             {"type": T_NEMO, "nemo_class": "EncDecRNNTBPEModel"},
    "nvidia/parakeet-tdt-1.1b":                {"type": T_NEMO, "nemo_class": "EncDecRNNTBPEModel"},
    "nvidia/parakeet-tdt-0.6b-v2":             {"type": T_NEMO, "nemo_class": "EncDecRNNTBPEModel"},
    "nvidia/parakeet-rnnt-1.1b":               {"type": T_NEMO, "nemo_class": "EncDecRNNTBPEModel"},
    "nvidia/parakeet-ctc-1.1b":                {"type": T_NEMO, "nemo_class": "EncDecCTCModelBPE"},
    "nvidia/stt_en_conformer_transducer_small": {"type": T_NEMO, "nemo_class": "EncDecRNNTBPEModel"},
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
    # Phi-4: AutoModelForCausalLM + processor.apply_transcription_request
    "microsoft/Phi-4-multimodal-instruct":     {"type": T_GENERATE, "gen_class": "AutoModelForCausalLM"},
    "microsoft/azure-speech-05-2026":          {"type": API_AZURE,        "label": "Azure Speech"},
    # ── Google ──────────────────────────────────────────────────────────
    "google/chirp_3":                          {"type": API_GOOGLE,       "label": "Google Chirp 3"},
    "google/chirp_2":                          {"type": API_GOOGLE,       "label": "Google Chirp 2"},
    # ── IBM Granite Speech (chat-template style) ─────────────────────────
    "ibm-granite/granite-speech-4.1-2b-nar":  {"type": T_GRANITE},
    "ibm-granite/granite-speech-4.1-2b":      {"type": T_GRANITE},
    "ibm-granite/granite-speech-3.3-8b":      {"type": T_GRANITE},
    "ibm-granite/granite-speech-3.3-2b":      {"type": T_GRANITE},
    "ibm-granite/granite-4.0-1b-speech":      {"type": T_GRANITE},
    # ── Mistral Voxtral ─────────────────────────────────────────────────
    "mistralai/Voxtral-Small-24B-2507":        {"type": T_VOXTRAL},
    "mistralai/Voxtral-Mini-3B-2507":          {"type": T_VOXTRAL},
    # ── Qwen3-ASR (qwen_asr package) ────────────────────────────────────
    "Qwen/Qwen3-ASR-1.7B":                     {"type": T_QWEN_ASR},
    "Qwen/Qwen3-ASR-0.6B":                     {"type": T_QWEN_ASR},
    # ── Kyutai STT (moshi library) ───────────────────────────────────────
    "kyutai/stt-2.6b-en":                      {"type": T_KYUTAI},
    # ── Boson AI HiggsAudio (trust_remote_code + transcribe.py) ─────────
    "bosonai/higgs-audio-v3-8b-stt-v2":        {"type": T_TRUST, "higgs": True},
    # ── ZAI / GLM ASR (GlmAsrForConditionalGeneration) ──────────────────
    "zai-org/GLM-ASR-Nano-2512":               {"type": T_GENERATE, "gen_class": "GlmAsrForConditionalGeneration"},
    # ── ESPnet / OWSM (espnet2 framework) ───────────────────────────────
    "espnet/owsm_ctc_v4_1B":                   {"type": T_ESPNET},
    "pyf98/owsm_ctc_v3.1_1B":                 {"type": T_ESPNET},
    # ── SoundsgoodAI Zipformer (k2/icefall, CUDA required) ──────────────
    "soundsgoodai/Zipformer-transducer-XL-290M": {"type": T_ZIPFORMER},
    # ── Useful Sensors Moonshine ─────────────────────────────────────────
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
    "🌊 Mistral Voxtral":          ["mistralai/Voxtral-Mini-3B-2507","mistralai/Voxtral-Small-24B-2507"],
    "🐲 Qwen3-ASR":                ["Qwen/Qwen3-ASR-0.6B","Qwen/Qwen3-ASR-1.7B"],
    "🎵 Kyutai":                   ["kyutai/stt-2.6b-en"],
    "🤖 Boson AI HiggsAudio":      ["bosonai/higgs-audio-v3-8b-stt-v2"],
    "🔵 ZAI / GLM ASR":            ["zai-org/GLM-ASR-Nano-2512"],
    "🎓 ESPnet / OWSM":            ["espnet/owsm_ctc_v4_1B","pyf98/owsm_ctc_v3.1_1B"],
    "🔈 SoundsgoodAI Zipformer":   ["soundsgoodai/Zipformer-transducer-XL-290M"],
    "🌙 Useful Sensors":           ["usefulsensors/moonshine-streaming-medium"],
    "🧠 Wav2Vec2 / HuBERT":        ["facebook/hubert-large-ls960-ft","facebook/hubert-xlarge-ls960-ft","jonatasgrosman/wav2vec2-large-xlsr-53-english"],
    "🗣️ SpeechBrain":              ["speechbrain/asr-wav2vec2-librispeech","speechbrain/asr-conformer-transformerlm-librispeech"],
    "🎙️ ElevenLabs":              ["elevenlabs/scribe_v2","elevenlabs/scribe_v1"],
    "📡 AssemblyAI":               ["assemblyai/universal-3-pro"],
    "🔊 Aqua Voice":               ["aquavoice/avalon-v1-en"],
    "🧪 Cohere":                   ["CohereLabs/cohere-transcribe-03-2026"],
    "🔉 Reson8":                   ["reson8/resonant-1","reson8/resonant-1-flash"],
    "📝 Rev AI":                   ["revai/fusion"],
    "🎤 Speechmatics":             ["speechmatics/enhanced"],
    "🔬 Smallest AI":              ["smallestai/pulse"],
    "📹 Zoom":                     ["zoom/scribe_v1"],
}

DEFAULT_MODEL = "nyrahealth/CrisperWhisper"

ALL_MODELS: List[str] = [m for models in MODEL_GROUPS.values() for m in models]

def get_model_info(model_id: str) -> Dict[str, Any]:
    return MODEL_REGISTRY.get(model_id, {"type": T_GENERIC})

def is_api_type(t: str) -> bool:
    return t.startswith("api_")


# ---------------------------------------------------------------------------
# Pipeline / model loading functions
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
        raise RuntimeError("SpeechBrain is not installed. Run: pip install speechbrain")
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
        raise RuntimeError("NeMo toolkit is not installed. Run: pip install nemo_toolkit[asr]")
    cls_map = {
        "EncDecCTCModelBPE":    nemo_asr.models.EncDecCTCModelBPE,
        "EncDecRNNTBPEModel":   nemo_asr.models.EncDecRNNTBPEModel,
        "EncDecMultiTaskModel": nemo_asr.models.EncDecMultiTaskModel,
    }
    model_cls = cls_map.get(nemo_class, nemo_asr.models.ASRModel)
    return model_cls.from_pretrained(model_name=model_id)


@st.cache_resource(show_spinner="Loading Voxtral model… (large model, may take several minutes)")
def load_voxtral_model(model_id: str) -> Any:
    """Load Voxtral using VoxtralForConditionalGeneration per the leaderboard backend."""
    if not VOXTRAL_AVAILABLE:
        raise RuntimeError(
            "VoxtralForConditionalGeneration is not available in your transformers version.\n"
            "Run: pip install transformers --upgrade\n"
            "Also required: pip install 'mistral-common[audio]>=1.9.0'"
        )
    try:
        from transformers import VoxtralForConditionalGeneration, AutoProcessor
    except ImportError as e:
        raise RuntimeError(f"Failed to import Voxtral classes: {e}")
    proc = AutoProcessor.from_pretrained(model_id)
    model = VoxtralForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if HAS_CUDA else torch.float32,
        device_map=device,
    )
    model.eval()
    return (model, proc)


@st.cache_resource(show_spinner="Loading model… this may take a few minutes on first run.")
def load_generate_model(model_id: str, gen_class: str = "AutoModelForCausalLM") -> Any:
    """Load generative models (Phi-4, GLM ASR) using apply_transcription_request pattern."""
    import transformers as _tr
    proc = _tr.AutoProcessor.from_pretrained(model_id)
    ModelCls = getattr(_tr, gen_class, None)
    if ModelCls is None:
        raise RuntimeError(
            f"Model class `{gen_class}` not found in your transformers version ({_tr.__version__}).\n"
            f"Run: pip install transformers --upgrade"
        )
    model = ModelCls.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if HAS_CUDA else torch.float32,
        device_map=device,
    )
    model.eval()
    return (model, proc)


@st.cache_resource(show_spinner="Loading IBM Granite Speech model… this may take a few minutes.")
def load_granite_model(model_id: str) -> Any:
    """Load IBM Granite Speech using chat-template style inference per the leaderboard."""
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
    proc = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if HAS_CUDA else torch.float32,
        device_map=device,
    )
    model.eval()
    return (model, proc)


@st.cache_resource(show_spinner="Loading model with trust_remote_code… this may take a few minutes.")
def load_trust_remote_model(model_id: str, info: Dict[str, Any]) -> Any:
    """Load models that require trust_remote_code=True (HiggsAudio, LiteASR)."""
    from transformers import AutoModel, AutoProcessor
    is_lite = info.get("lite_asr", False)
    if is_lite:
        proc = AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")
    else:
        try:
            proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        except Exception:
            proc = AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if HAS_CUDA else torch.float32,
        trust_remote_code=True,
        device_map=device,
    )
    model.eval()
    return (model, proc)


@st.cache_resource(show_spinner="Loading Qwen3-ASR model… this may take a few minutes.")
def load_qwen_asr_model(model_id: str) -> Any:
    """Load Qwen3-ASR via the qwen_asr package per the leaderboard."""
    if not QWEN_ASR_AVAILABLE:
        # Try to install qwen_asr
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "qwen_asr", "--no-deps", "-q"],
                timeout=120,
            )
        except Exception:
            pass
        try:
            import qwen_asr as _qa  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "qwen_asr package is not installed.\n"
                "Run: pip install qwen_asr\n"
                "Then restart the app."
            )
    from qwen_asr import Qwen3ASRModel
    model = Qwen3ASRModel.from_pretrained(
        model_id,
        dtype=torch.bfloat16 if HAS_CUDA else torch.float32,
        device_map=device,
    )
    return model


@st.cache_resource(show_spinner="Loading Kyutai STT model via moshi… this may take several minutes.")
def load_kyutai_model(model_id: str) -> Any:
    """Load Kyutai STT using the moshi library per the leaderboard."""
    if not MOSHI_AVAILABLE:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "moshi", "julius", "-q"],
                timeout=180,
            )
        except Exception:
            pass
        try:
            import moshi  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "moshi package is not installed.\n"
                "Run: pip install moshi julius\n"
                "Then restart the app."
            )
    from moshi import models as moshi_models
    moshi_device = "cuda" if HAS_CUDA else "cpu"
    moshi_dtype = torch.bfloat16 if HAS_CUDA else torch.float32
    info = moshi_models.loaders.CheckpointInfo.from_hf_repo(model_id)
    mimi = info.get_mimi(device=moshi_device)
    tokenizer = info.get_text_tokenizer()
    lm = info.get_moshi(device=moshi_device, dtype=moshi_dtype)
    lm_gen = moshi_models.LMGen(lm, temp=0, temp_text=0.0)
    padding_token_id = info.raw_config.get("text_padding_token_id", 3)
    silence_prefix_s = info.stt_config.get("audio_silence_prefix_seconds", 1.0)
    delay_s = info.stt_config.get("audio_delay_seconds", 5.0)
    return (mimi, tokenizer, lm, lm_gen, padding_token_id, silence_prefix_s, delay_s)


@st.cache_resource(show_spinner="Loading ESPnet OWSM model… this may take several minutes.")
def load_espnet_model(model_id: str) -> Any:
    """Load ESPnet OWSM CTC model using espnet2 per the leaderboard."""
    if not ESPNET_AVAILABLE:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install",
                 "espnet @ git+https://github.com/espnet/espnet",
                 "espnet_model_zoo @ git+https://github.com/espnet/espnet_model_zoo",
                 "-q"],
                timeout=300,
            )
        except Exception:
            pass
        try:
            import espnet2  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "ESPnet is not installed.\n"
                "Run: pip install 'espnet @ git+https://github.com/espnet/espnet'\n"
                "Then restart the app."
            )
    from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch
    load_args: Dict[str, Any] = {
        "device": "cpu",
        "dtype": "float32",
        "lang_sym": "<eng>",
        "task_sym": "<asr>",
    }
    if os.path.exists(model_id):
        load_args["s2t_model_file"] = model_id
    else:
        load_args["model_tag"] = model_id
    return Speech2TextGreedySearch.from_pretrained(**load_args)


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
# Transcription functions per model type
# ---------------------------------------------------------------------------
def transcribe_transformers(audio_bytes: bytes, pipe: Any) -> Dict[str, Any]:
    waveform = process_audio_bytes(audio_bytes)
    return pipe(waveform[0, :].numpy(), return_timestamps="word")


def transcribe_nemo(audio_bytes: bytes, model: Any) -> Dict[str, Any]:
    process_audio_bytes(audio_bytes)
    results = model.transcribe(["sample.wav"])
    text = results[0] if isinstance(results[0], str) else results[0].text
    return {"text": text, "chunks": []}


def transcribe_speechbrain(audio_bytes: bytes, model: Any) -> Dict[str, Any]:
    process_audio_bytes(audio_bytes)
    result = model.transcribe_file("sample.wav")
    if isinstance(result, (list, tuple)):
        text = " ".join(str(r) for r in result).strip()
    else:
        text = str(result).strip()
    return {"text": text, "chunks": []}


def transcribe_voxtral(audio_bytes: bytes, model_and_proc: Any, model_id: str) -> Dict[str, Any]:
    """Voxtral: processor.apply_transcription_request per leaderboard voxtral/run_eval.py"""
    model, proc = model_and_proc
    waveform = process_audio_bytes(audio_bytes)
    audio_np = waveform[0].numpy()
    inputs = proc.apply_transcription_request(
        language="en",
        audio=audio_np,
        model_id=model_id,
    )
    inputs = inputs.to(model.device, dtype=torch.bfloat16 if HAS_CUDA else torch.float32)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=448)
    decoded = proc.batch_decode(
        outputs[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    return {"text": decoded[0].strip() if decoded else "", "chunks": []}


def transcribe_generate(audio_bytes: bytes, model_and_proc: Any) -> Dict[str, Any]:
    """Phi-4 / GLM ASR: processor.apply_transcription_request(audios) per leaderboard phi/ and glm_asr/."""
    model, proc = model_and_proc
    waveform = process_audio_bytes(audio_bytes)
    audio_np = waveform[0].numpy()
    inputs = proc.apply_transcription_request([audio_np])
    inputs = inputs.to(model.device, dtype=model.dtype)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=448, do_sample=False)
    input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
    decoded = proc.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
    return {"text": decoded[0].strip() if decoded else "", "chunks": []}


def transcribe_granite(audio_bytes: bytes, model_and_proc: Any) -> Dict[str, Any]:
    """IBM Granite Speech: chat-template + AutoModelForSpeechSeq2Seq per leaderboard granite/run_eval.py"""
    model, proc = model_and_proc
    tokenizer = proc.tokenizer
    waveform = process_audio_bytes(audio_bytes)
    audio_np = waveform[0].numpy()

    chat = [
        {
            "role": "system",
            "content": (
                "Knowledge Cutoff Date: April 2024.\n"
                "Today's Date: December 19, 2024.\n"
                "You are Granite, developed by IBM. You are a helpful AI assistant"
            ),
        },
        {
            "role": "user",
            "content": "<|audio|>can you transcribe the speech into a written format?",
        },
    ]
    text_prompt = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )

    model_inputs = proc(
        [text_prompt],
        [audio_np],
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **model_inputs,
            bos_token_id=tokenizer.bos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=448,
        )

    num_input_tokens = model_inputs["input_ids"].shape[-1]
    new_tokens = outputs[:, num_input_tokens:]
    text_out = tokenizer.batch_decode(new_tokens, add_special_tokens=False, skip_special_tokens=True)
    return {"text": text_out[0].strip() if text_out else "", "chunks": []}


def transcribe_trust_remote(audio_bytes: bytes, model_and_proc: Any, info: Dict[str, Any]) -> Dict[str, Any]:
    """HiggsAudio / LiteASR: AutoModel trust_remote_code per leaderboard higgs_audio/ and liteASR/."""
    model, proc = model_and_proc
    waveform = process_audio_bytes(audio_bytes)
    audio_np = waveform[0].numpy()

    is_higgs = info.get("higgs", False)
    is_lite = info.get("lite_asr", False)

    if is_higgs:
        # HiggsAudio: dynamically load transcribe_batch from the model repo
        from transformers.utils import cached_file
        from transformers import AutoTokenizer
        model_id = info.get("_model_id", "bosonai/higgs-audio-v3-8b-stt-v2")
        path = cached_file(model_id, "transcribe.py")
        module_dir = os.path.dirname(path)
        sys.path.insert(0, module_dir)
        try:
            module_globals = runpy.run_path(path)
        finally:
            if module_dir in sys.path:
                sys.path.remove(module_dir)
        transcribe_batch_fn = module_globals["transcribe_batch"]
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        results = transcribe_batch_fn(model, tokenizer, [(audio_np, 16000)])
        text = results[0] if results else ""
        return {"text": text.strip() if text else "", "chunks": []}
    else:
        # LiteASR: standard whisper-processor + model.generate()
        inputs = proc(audio_np, return_tensors="pt", sampling_rate=16000)
        inputs = inputs.to(model.device, dtype=torch.float16 if HAS_CUDA else torch.float32)
        with torch.no_grad():
            if hasattr(model, "can_generate") and model.can_generate():
                outputs = model.generate(**inputs, max_new_tokens=224)
                text = proc.batch_decode(outputs, skip_special_tokens=True)[0]
            else:
                logits = model(**inputs).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                text = proc.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return {"text": text.strip(), "chunks": []}


def transcribe_qwen_asr(audio_bytes: bytes, model: Any) -> Dict[str, Any]:
    """Qwen3-ASR: Qwen3ASRModel.transcribe per leaderboard qwen/run_eval.py"""
    waveform = process_audio_bytes(audio_bytes)
    audio_np = waveform[0].numpy()
    audio_inputs = [(audio_np, 16000)]
    results = model.transcribe(audio=audio_inputs, language="English")
    text = results[0].text if results else ""
    return {"text": text.strip(), "chunks": []}


def transcribe_kyutai(audio_bytes: bytes, model_components: Any) -> Dict[str, Any]:
    """Kyutai STT: moshi streaming inference per leaderboard kyutai/run_eval.py"""
    import julius
    mimi, tokenizer, _lm, lm_gen, padding_token_id, silence_prefix_s, delay_s = model_components
    moshi_device = "cuda" if HAS_CUDA else "cpu"

    waveform = process_audio_bytes(audio_bytes)
    audio_16k = waveform[0]
    # Resample from 16kHz to 24kHz (moshi native rate)
    audio_24k = julius.resample.resample_frac(audio_16k, old_sr=16000, new_sr=24000)
    # Pad with silence prefix and delay
    audio_padded = torch.nn.functional.pad(
        audio_24k,
        (int(silence_prefix_s * 24000), int(delay_s * 24000)),
    ).unsqueeze(0).to(moshi_device)

    mimi_frame_size = mimi.frame_size
    target_len = audio_padded.shape[-1]
    if target_len % mimi_frame_size != 0:
        pad_len = mimi_frame_size - (target_len % mimi_frame_size)
        audio_padded = torch.nn.functional.pad(audio_padded, (0, pad_len))

    text_tokens: List[int] = []
    with torch.inference_mode():
        # Encode audio into mimi codes frame by frame and run language model
        with mimi.streaming(1):
            for offset in range(0, audio_padded.shape[-1], mimi_frame_size):
                chunk = audio_padded[:, :, offset:offset + mimi_frame_size]
                codes = mimi.encode(chunk)  # (1, K, T)
                for t in range(codes.shape[-1]):
                    audio_tokens = codes[:, :, t:t+1]
                    lm_gen_out = lm_gen.step(audio_tokens)
                    if lm_gen_out is not None:
                        text_t = lm_gen_out[0, 0].item()
                        if text_t != padding_token_id:
                            text_tokens.append(text_t)

    text = tokenizer.decode(text_tokens) if text_tokens else ""
    return {"text": text.strip(), "chunks": []}


def transcribe_espnet(audio_bytes: bytes, model: Any) -> Dict[str, Any]:
    """ESPnet OWSM CTC: Speech2TextGreedySearch.batch_decode per leaderboard espnet/run_eval.py"""
    waveform = process_audio_bytes(audio_bytes)
    audio_np = waveform[0].numpy()
    with torch.inference_mode():
        pred_text = model.batch_decode(
            [audio_np],
            batch_size=1,
            context_len_in_secs=4,
        )
    text = pred_text[0] if pred_text else ""
    return {"text": text.strip(), "chunks": []}


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
    import time
    client = apiclient.RevAiAPIClient(api_key)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        job = client.submit_job_local_file(tmp_path)
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
    active_info: Dict[str, Any],
) -> Dict[str, Any]:
    t = active_info["type"]

    if t in (T_WHISPER, T_GENERIC):
        return transcribe_transformers(audio_bytes, model_or_pipe)
    elif t == T_NEMO:
        return transcribe_nemo(audio_bytes, model_or_pipe)
    elif t == T_SPEECHBRAIN:
        return transcribe_speechbrain(audio_bytes, model_or_pipe)
    elif t == T_VOXTRAL:
        return transcribe_voxtral(audio_bytes, model_or_pipe, model_id)
    elif t == T_GENERATE:
        return transcribe_generate(audio_bytes, model_or_pipe)
    elif t == T_GRANITE:
        return transcribe_granite(audio_bytes, model_or_pipe)
    elif t == T_TRUST:
        _info = dict(active_info)
        _info["_model_id"] = model_id
        return transcribe_trust_remote(audio_bytes, model_or_pipe, _info)
    elif t == T_QWEN_ASR:
        return transcribe_qwen_asr(audio_bytes, model_or_pipe)
    elif t == T_KYUTAI:
        return transcribe_kyutai(audio_bytes, model_or_pipe)
    elif t == T_ESPNET:
        return transcribe_espnet(audio_bytes, model_or_pipe)
    elif t == T_ZIPFORMER:
        raise RuntimeError(
            "**Zipformer (k2/icefall)** requires the `k2` library which is only available "
            "for CUDA 12.4 (GPU). This model cannot run on CPU-only systems.\n\n"
            "To use this model, you need a machine with an NVIDIA GPU and CUDA 12.4 installed."
        )
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
        provider_label = active_info.get("label", model_id)
        raise RuntimeError(
            f"**{provider_label}** is a commercial API model. "
            "Please contact the provider for API access."
        )


# ---------------------------------------------------------------------------
# CLI arg (kept for workflow launch compat)
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

TYPE_BADGE = {
    T_WHISPER:     "WHISPER",
    T_GENERIC:     "HF",
    T_NEMO:        "NeMo",
    T_SPEECHBRAIN: "SpeechBrain",
    T_VOXTRAL:     "Voxtral",
    T_GENERATE:    "Generate",
    T_GRANITE:     "Granite",
    T_TRUST:       "TrustRemote",
    T_QWEN_ASR:    "Qwen-ASR",
    T_KYUTAI:      "Kyutai",
    T_ESPNET:      "ESPnet",
    T_ZIPFORMER:   "Zipformer",
}

selected_model = st.sidebar.selectbox(
    "Select a model",
    options=group_display,
    index=default_idx,
    format_func=lambda m: f"{m}  [{TYPE_BADGE.get(get_model_info(m)['type'], get_model_info(m)['type'].upper())}]",
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

# ── Framework info banners ────────────────────────────────────────────
if active_type == T_SPEECHBRAIN:
    if not SPEECHBRAIN_AVAILABLE:
        st.sidebar.warning("⚠️ SpeechBrain not installed. Run: `pip install speechbrain`")
    else:
        st.sidebar.info("🗣️ SpeechBrain model")

if active_type == T_NEMO:
    if not NEMO_AVAILABLE:
        st.sidebar.warning("⚠️ NeMo not installed. Run: `pip install nemo_toolkit[asr]`")
    else:
        st.sidebar.info(f"🟢 NeMo model — {'GPU ✅' if HAS_CUDA else 'CPU'}")

if active_type == T_VOXTRAL:
    if not VOXTRAL_AVAILABLE:
        st.sidebar.warning("⚠️ Upgrade transformers: `pip install transformers --upgrade`\nAlso: `pip install 'mistral-common[audio]>=1.9.0'`")
    else:
        st.sidebar.info("🌊 Voxtral — VoxtralForConditionalGeneration")

if active_type == T_GENERATE:
    gen_cls = active_info.get("gen_class", "AutoModelForCausalLM")
    st.sidebar.info(f"⚡ Loaded via `{gen_cls}` + `apply_transcription_request`")

if active_type == T_GRANITE:
    st.sidebar.info("💎 IBM Granite Speech — chat-template inference")

if active_type == T_TRUST:
    st.sidebar.info("🔒 Loaded with `trust_remote_code=True`")

if active_type == T_QWEN_ASR:
    if not QWEN_ASR_AVAILABLE:
        st.sidebar.warning("⚠️ qwen_asr package not installed. Will attempt install on load.")
    else:
        st.sidebar.info("🐲 Qwen3-ASR via qwen_asr package")

if active_type == T_KYUTAI:
    if not MOSHI_AVAILABLE:
        st.sidebar.warning("⚠️ moshi package not installed. Will attempt install on load.")
    else:
        st.sidebar.info("🎵 Kyutai STT via moshi streaming inference")

if active_type == T_ESPNET:
    if not ESPNET_AVAILABLE:
        st.sidebar.warning("⚠️ ESPnet not installed. Will attempt install on load.")
    else:
        st.sidebar.info("🎓 ESPnet OWSM CTC via espnet2")

if active_type == T_ZIPFORMER:
    st.sidebar.error("🔴 Zipformer requires k2 (CUDA 12.4 + GPU). Cannot run on CPU-only systems.")

# ── API key inputs ─────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("**🔑 API Keys** *(stored in session only)*")

if "api_keys" not in st.session_state:
    st.session_state.api_keys = {}

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
badge_map = {
    T_NEMO:        " 🟢 NeMo",
    T_SPEECHBRAIN: " 🗣️ SpeechBrain",
    T_VOXTRAL:     " 🌊 Voxtral",
    T_GENERATE:    " ⚡ Generate",
    T_GRANITE:     " 💎 Granite",
    T_TRUST:       " 🔒 TrustRemote",
    T_QWEN_ASR:    " 🐲 Qwen-ASR",
    T_KYUTAI:      " 🎵 Kyutai",
    T_ESPNET:      " 🎓 ESPnet",
    T_ZIPFORMER:   " ⚠️ Needs GPU+CUDA",
}
if is_api_type(active_type):
    model_badge += " 🔑 API"
elif active_type in badge_map:
    model_badge += badge_map[active_type]

st.title("Open ASR Leaderboard Transcription 🦻")
st.markdown(
    f"Transcribing with {model_badge} &nbsp;|&nbsp; "
    f"[View leaderboard ↗](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)"
)

# Block early if requirements clearly not met
if active_type == T_NEMO and not NEMO_AVAILABLE:
    st.error("**NeMo toolkit is not installed.**\n\nInstall it with: `pip install nemo_toolkit[asr]`\n\nThen restart the app.")
    st.stop()

if active_type == T_SPEECHBRAIN and not SPEECHBRAIN_AVAILABLE:
    st.error("**SpeechBrain is not installed.**\n\nInstall it with: `pip install speechbrain`\n\nThen restart the app.")
    st.stop()

if active_type == T_ZIPFORMER:
    st.error(
        "**Zipformer (k2/icefall) requires a CUDA-capable GPU.**\n\n"
        "The `k2` library is only distributed for CUDA 12.4 (GPU) and cannot run on CPU-only systems. "
        "To use this model, you need an NVIDIA GPU with CUDA 12.4 installed.\n\n"
        "**Alternative:** Try a Whisper or Distil-Whisper model for CPU-compatible transcription."
    )
    st.stop()

# Load model/pipeline (skipped for pure API models)
model_or_pipe = None
if not is_api_type(active_type):
    try:
        if active_type == T_NEMO:
            model_or_pipe = load_nemo_model(active_model, active_info.get("nemo_class", "ASRModel"))
        elif active_type == T_SPEECHBRAIN:
            model_or_pipe = load_speechbrain_model(active_model, active_info.get("sb_class", "EncoderASR"))
        elif active_type == T_VOXTRAL:
            model_or_pipe = load_voxtral_model(active_model)
        elif active_type == T_GENERATE:
            model_or_pipe = load_generate_model(active_model, active_info.get("gen_class", "AutoModelForCausalLM"))
        elif active_type == T_GRANITE:
            model_or_pipe = load_granite_model(active_model)
        elif active_type == T_TRUST:
            model_or_pipe = load_trust_remote_model(active_model, active_info)
        elif active_type == T_QWEN_ASR:
            model_or_pipe = load_qwen_asr_model(active_model)
        elif active_type == T_KYUTAI:
            model_or_pipe = load_kyutai_model(active_model)
        elif active_type == T_ESPNET:
            model_or_pipe = load_espnet_model(active_model)
        else:
            model_or_pipe = load_transformers_pipeline(active_model, active_type)
    except Exception as load_err:
        st.error(f"**Failed to load `{active_model}`:**\n\n{load_err}")
        st.stop()

st.write("🎙️ Record audio below or 📁 upload an audio file.")

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
                active_info,
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

st.markdown(
    "<hr><footer><p style='text-align:center;'>© 2024 nyra health GmbH</p></footer>",
    unsafe_allow_html=True,
)
