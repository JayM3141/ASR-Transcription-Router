import io
import sys
import types
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Stub heavy third-party imports that app.py uses at import time so we can
# import the module's pure helpers without needing the real libraries.
# ---------------------------------------------------------------------------
_STUBS = {}


def _ensure_stub(name):
    if name not in sys.modules:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        _STUBS[name] = mod
    return sys.modules[name]


# A dict subclass that also supports attribute-style access (like Streamlit's
# real SessionState).
class _AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name)


# streamlit and its sub-modules
_st = _ensure_stub("streamlit")
_st.cache_resource = lambda **kw: (lambda fn: fn)
_st.sidebar = mock.MagicMock()
_st.session_state = _AttrDict()
_st.title = mock.MagicMock()
_st.markdown = mock.MagicMock()
_st.write = mock.MagicMock()
_st.file_uploader = mock.MagicMock(return_value=None)
_st.video = mock.MagicMock()
_st.subheader = mock.MagicMock()
_st.spinner = mock.MagicMock(return_value=mock.MagicMock(__enter__=mock.MagicMock(), __exit__=mock.MagicMock()))
_st.stop = mock.MagicMock()
_st.error = mock.MagicMock()
_st.expander = mock.MagicMock()
_st.text_input = mock.MagicMock(return_value="")
_st.selectbox = mock.MagicMock(return_value="nyrahealth/CrisperWhisper")
_st.dataframe = mock.MagicMock()
_ensure_stub("streamlit_mic_recorder")
sys.modules["streamlit_mic_recorder"].mic_recorder = mock.MagicMock(return_value=None)

# moviepy
_mp = _ensure_stub("moviepy")
_mp.AudioFileClip = mock.MagicMock()
_mp.ColorClip = mock.MagicMock()

# torch / torchaudio (we need real torch for process_audio_bytes tests)
import torch
import torchaudio
import torchaudio.transforms as T

# Patch sys.argv to prevent argparse from choking on pytest args
with mock.patch("sys.argv", ["app.py"]):
    import app


# ---------------------------------------------------------------------------
# Tests for get_model_info
# ---------------------------------------------------------------------------
class TestGetModelInfo:
    def test_known_model(self):
        info = app.get_model_info("openai/whisper-large-v3")
        assert info["type"] == app.T_WHISPER

    def test_unknown_model_defaults_to_generic(self):
        info = app.get_model_info("some/unknown-model")
        assert info["type"] == app.T_GENERIC

    def test_nemo_model(self):
        info = app.get_model_info("nvidia/canary-1b-v2")
        assert info["type"] == app.T_NEMO
        assert info["nemo_class"] == "EncDecMultiTaskModel"

    def test_speechbrain_model(self):
        info = app.get_model_info("speechbrain/asr-wav2vec2-librispeech")
        assert info["type"] == app.T_SPEECHBRAIN
        assert info["sb_class"] == "EncoderASR"

    def test_voxtral_model(self):
        info = app.get_model_info("mistralai/Voxtral-Mini-3B-2507")
        assert info["type"] == app.T_VOXTRAL

    def test_granite_model(self):
        info = app.get_model_info("ibm-granite/granite-speech-4.1-2b")
        assert info["type"] == app.T_GRANITE

    def test_api_model(self):
        info = app.get_model_info("assemblyai/universal-3-pro")
        assert info["type"] == app.API_ASSEMBLYAI

    def test_generate_model(self):
        info = app.get_model_info("microsoft/Phi-4-multimodal-instruct")
        assert info["type"] == app.T_GENERATE
        assert info["gen_class"] == "AutoModelForCausalLM"

    def test_trust_model(self):
        info = app.get_model_info("efficient-speech/lite-whisper-large-v3-acc")
        assert info["type"] == app.T_TRUST
        assert info.get("lite_asr") is True

    def test_qwen_asr_model(self):
        info = app.get_model_info("Qwen/Qwen3-ASR-1.7B")
        assert info["type"] == app.T_QWEN_ASR

    def test_kyutai_model(self):
        info = app.get_model_info("kyutai/stt-2.6b-en")
        assert info["type"] == app.T_KYUTAI

    def test_espnet_model(self):
        info = app.get_model_info("espnet/owsm_ctc_v4_1B")
        assert info["type"] == app.T_ESPNET

    def test_zipformer_model(self):
        info = app.get_model_info("soundsgoodai/Zipformer-transducer-XL-290M")
        assert info["type"] == app.T_ZIPFORMER


# ---------------------------------------------------------------------------
# Tests for is_api_type
# ---------------------------------------------------------------------------
class TestIsApiType:
    def test_api_assemblyai(self):
        assert app.is_api_type(app.API_ASSEMBLYAI) is True

    def test_api_elevenlabs(self):
        assert app.is_api_type(app.API_ELEVENLABS) is True

    def test_api_revai(self):
        assert app.is_api_type(app.API_REVAI) is True

    def test_api_google(self):
        assert app.is_api_type(app.API_GOOGLE) is True

    def test_api_azure(self):
        assert app.is_api_type(app.API_AZURE) is True

    def test_api_cohere(self):
        assert app.is_api_type(app.API_COHERE) is True

    def test_api_speechmatics(self):
        assert app.is_api_type(app.API_SPEECHMATICS) is True

    def test_api_generic(self):
        assert app.is_api_type(app.API_GENERIC) is True

    def test_non_api_whisper(self):
        assert app.is_api_type(app.T_WHISPER) is False

    def test_non_api_nemo(self):
        assert app.is_api_type(app.T_NEMO) is False

    def test_non_api_generic(self):
        assert app.is_api_type(app.T_GENERIC) is False

    def test_non_api_arbitrary(self):
        assert app.is_api_type("some_random_type") is False

    def test_empty_string(self):
        assert app.is_api_type("") is False


# ---------------------------------------------------------------------------
# Tests for timestamps_to_vtt
# ---------------------------------------------------------------------------
class TestTimestampsToVtt:
    def test_empty_list(self):
        result = app.timestamps_to_vtt([])
        assert result == "WEBVTT\n\n"

    def test_single_word(self):
        timestamps = [{"text": "hello", "timestamp": (0.0, 0.5)}]
        result = app.timestamps_to_vtt(timestamps)
        assert "WEBVTT" in result
        assert "0:00:00.000 --> 0:00:00.500" in result
        assert "hello" in result

    def test_multiple_words(self):
        timestamps = [
            {"text": "hello", "timestamp": (0.0, 0.5)},
            {"text": "world", "timestamp": (0.5, 1.0)},
        ]
        result = app.timestamps_to_vtt(timestamps)
        lines = result.strip().split("\n")
        assert lines[0] == "WEBVTT"
        assert "hello" in result
        assert "world" in result

    def test_skips_none_start(self):
        timestamps = [
            {"text": "hello", "timestamp": (None, 0.5)},
            {"text": "world", "timestamp": (0.5, 1.0)},
        ]
        result = app.timestamps_to_vtt(timestamps)
        assert "hello" not in result
        assert "world" in result

    def test_none_end_uses_next_word_start(self):
        timestamps = [
            {"text": "hello", "timestamp": (0.0, None)},
            {"text": "world", "timestamp": (0.5, 1.0)},
        ]
        result = app.timestamps_to_vtt(timestamps)
        assert "0:00:00.000 --> 0:00:00.500" in result
        assert "hello" in result

    def test_none_end_last_word_uses_fallback(self):
        timestamps = [
            {"text": "hello", "timestamp": (0.0, None)},
        ]
        result = app.timestamps_to_vtt(timestamps)
        # fallback: start + 1.0
        assert "0:00:00.000 --> 0:00:01.000" in result

    def test_large_timestamp_formatting(self):
        timestamps = [{"text": "late", "timestamp": (3661.5, 3662.0)}]
        result = app.timestamps_to_vtt(timestamps)
        # 3661.5s = 1h 1m 1.5s
        assert "1:01:01.500" in result

    def test_zero_minute_formatting(self):
        timestamps = [{"text": "word", "timestamp": (5.123, 6.789)}]
        result = app.timestamps_to_vtt(timestamps)
        assert "0:00:05.123" in result
        assert "0:00:06.789" in result


# ---------------------------------------------------------------------------
# Tests for MODEL_REGISTRY / MODEL_GROUPS consistency
# ---------------------------------------------------------------------------
class TestModelRegistryConsistency:
    def test_all_grouped_models_in_registry(self):
        for group_name, models in app.MODEL_GROUPS.items():
            for model_id in models:
                assert model_id in app.MODEL_REGISTRY, (
                    f"Model '{model_id}' in group '{group_name}' not found in MODEL_REGISTRY"
                )

    def test_all_models_list_matches_groups(self):
        expected = [m for models in app.MODEL_GROUPS.values() for m in models]
        assert app.ALL_MODELS == expected

    def test_default_model_in_all_models(self):
        assert app.DEFAULT_MODEL in app.ALL_MODELS

    def test_all_registry_entries_have_type(self):
        for model_id, info in app.MODEL_REGISTRY.items():
            assert "type" in info, f"Model '{model_id}' missing 'type' in registry"

    def test_api_key_config_covers_api_types(self):
        api_types_in_registry = {
            info["type"]
            for info in app.MODEL_REGISTRY.values()
            if app.is_api_type(info["type"])
        }
        for api_type in api_types_in_registry:
            assert api_type in app.API_KEY_CONFIG, (
                f"API type '{api_type}' not in API_KEY_CONFIG"
            )

    def test_api_key_config_entries_have_required_fields(self):
        for api_type, cfg in app.API_KEY_CONFIG.items():
            assert "label" in cfg, f"API config '{api_type}' missing 'label'"
            assert "key" in cfg, f"API config '{api_type}' missing 'key'"
            assert "url" in cfg, f"API config '{api_type}' missing 'url'"


# ---------------------------------------------------------------------------
# Tests for process_audio_bytes
# ---------------------------------------------------------------------------
class TestProcessAudioBytes:
    def _mock_load(self, channels=1, sample_rate=16000, n_samples=8000):
        """Return a mock torchaudio.load that yields a specific waveform."""
        waveform = torch.randn(channels, n_samples, dtype=torch.float32)
        return mock.MagicMock(return_value=(waveform, sample_rate))

    def test_mono_16khz_passthrough(self):
        with mock.patch("app.torchaudio.load", self._mock_load(channels=1, sample_rate=16000)), \
             mock.patch("app.torchaudio.save"):
            result = app.process_audio_bytes(b"fake")
        assert result.shape[0] == 1
        assert result.dtype == torch.float32

    def test_stereo_downmixed_to_mono(self):
        with mock.patch("app.torchaudio.load", self._mock_load(channels=2, sample_rate=16000)), \
             mock.patch("app.torchaudio.save"):
            result = app.process_audio_bytes(b"fake")
        assert result.shape[0] == 1

    def test_resamples_to_16khz(self):
        with mock.patch("app.torchaudio.load", self._mock_load(channels=1, sample_rate=44100, n_samples=22050)), \
             mock.patch("app.torchaudio.save"):
            result = app.process_audio_bytes(b"fake")
        # After resampling from 44100 -> 16000 (0.5s of audio)
        expected_samples = int(0.5 * 16000)
        assert abs(result.shape[1] - expected_samples) < 10

    def test_saves_to_sample_wav(self):
        with mock.patch("app.torchaudio.load", self._mock_load()), \
             mock.patch("app.torchaudio.save") as mock_save:
            app.process_audio_bytes(b"fake")
            mock_save.assert_called_once()
            args, kwargs = mock_save.call_args
            assert args[0] == "sample.wav"
            assert kwargs.get("sample_rate") == 16000
            assert kwargs.get("backend") == "ffmpeg"


# ---------------------------------------------------------------------------
# Tests for transcribe_audio (router)
# ---------------------------------------------------------------------------
class TestTranscribeAudioRouter:
    def _dummy_result(self):
        return {"text": "test transcription", "chunks": []}

    def test_routes_whisper(self):
        with mock.patch.object(app, "transcribe_transformers", return_value=self._dummy_result()) as m:
            result = app.transcribe_audio(
                b"audio", "openai/whisper-large-v3", mock.MagicMock(),
                {}, {"type": app.T_WHISPER},
            )
            m.assert_called_once()
            assert result["text"] == "test transcription"

    def test_routes_generic(self):
        with mock.patch.object(app, "transcribe_transformers", return_value=self._dummy_result()) as m:
            app.transcribe_audio(
                b"audio", "facebook/wav2vec2-base-960h", mock.MagicMock(),
                {}, {"type": app.T_GENERIC},
            )
            m.assert_called_once()

    def test_routes_nemo(self):
        with mock.patch.object(app, "transcribe_nemo", return_value=self._dummy_result()) as m:
            app.transcribe_audio(
                b"audio", "nvidia/canary-1b-v2", mock.MagicMock(),
                {}, {"type": app.T_NEMO},
            )
            m.assert_called_once()

    def test_routes_speechbrain(self):
        with mock.patch.object(app, "transcribe_speechbrain", return_value=self._dummy_result()) as m:
            app.transcribe_audio(
                b"audio", "speechbrain/asr-wav2vec2-librispeech", mock.MagicMock(),
                {}, {"type": app.T_SPEECHBRAIN},
            )
            m.assert_called_once()

    def test_routes_voxtral(self):
        with mock.patch.object(app, "transcribe_voxtral", return_value=self._dummy_result()) as m:
            app.transcribe_audio(
                b"audio", "mistralai/Voxtral-Mini-3B-2507", mock.MagicMock(),
                {}, {"type": app.T_VOXTRAL},
            )
            m.assert_called_once()

    def test_routes_generate(self):
        with mock.patch.object(app, "transcribe_generate", return_value=self._dummy_result()) as m:
            app.transcribe_audio(
                b"audio", "microsoft/Phi-4-multimodal-instruct", mock.MagicMock(),
                {}, {"type": app.T_GENERATE},
            )
            m.assert_called_once()

    def test_routes_granite(self):
        with mock.patch.object(app, "transcribe_granite", return_value=self._dummy_result()) as m:
            app.transcribe_audio(
                b"audio", "ibm-granite/granite-speech-4.1-2b", mock.MagicMock(),
                {}, {"type": app.T_GRANITE},
            )
            m.assert_called_once()

    def test_routes_trust_remote(self):
        with mock.patch.object(app, "transcribe_trust_remote", return_value=self._dummy_result()) as m:
            app.transcribe_audio(
                b"audio", "bosonai/higgs-audio-v3-8b-stt-v2", mock.MagicMock(),
                {}, {"type": app.T_TRUST, "higgs": True},
            )
            m.assert_called_once()

    def test_routes_qwen_asr(self):
        with mock.patch.object(app, "transcribe_qwen_asr", return_value=self._dummy_result()) as m:
            app.transcribe_audio(
                b"audio", "Qwen/Qwen3-ASR-1.7B", mock.MagicMock(),
                {}, {"type": app.T_QWEN_ASR},
            )
            m.assert_called_once()

    def test_routes_kyutai(self):
        with mock.patch.object(app, "transcribe_kyutai", return_value=self._dummy_result()) as m:
            app.transcribe_audio(
                b"audio", "kyutai/stt-2.6b-en", mock.MagicMock(),
                {}, {"type": app.T_KYUTAI},
            )
            m.assert_called_once()

    def test_routes_espnet(self):
        with mock.patch.object(app, "transcribe_espnet", return_value=self._dummy_result()) as m:
            app.transcribe_audio(
                b"audio", "espnet/owsm_ctc_v4_1B", mock.MagicMock(),
                {}, {"type": app.T_ESPNET},
            )
            m.assert_called_once()

    def test_routes_assemblyai_with_key(self):
        with mock.patch.object(app, "transcribe_assemblyai", return_value=self._dummy_result()) as m:
            app.transcribe_audio(
                b"audio", "assemblyai/universal-3-pro", None,
                {"assemblyai_key": "test-key"}, {"type": app.API_ASSEMBLYAI},
            )
            m.assert_called_once_with(b"audio", "test-key")

    def test_assemblyai_missing_key_raises(self):
        with pytest.raises(RuntimeError, match="AssemblyAI API key"):
            app.transcribe_audio(
                b"audio", "assemblyai/universal-3-pro", None,
                {}, {"type": app.API_ASSEMBLYAI},
            )

    def test_elevenlabs_missing_key_raises(self):
        with pytest.raises(RuntimeError, match="ElevenLabs API key"):
            app.transcribe_audio(
                b"audio", "elevenlabs/scribe_v2", None,
                {}, {"type": app.API_ELEVENLABS},
            )

    def test_revai_missing_key_raises(self):
        with pytest.raises(RuntimeError, match="Rev AI API key"):
            app.transcribe_audio(
                b"audio", "revai/fusion", None,
                {}, {"type": app.API_REVAI},
            )

    def test_zipformer_raises(self):
        with pytest.raises(RuntimeError, match="Zipformer"):
            app.transcribe_audio(
                b"audio", "soundsgoodai/Zipformer-transducer-XL-290M", None,
                {}, {"type": app.T_ZIPFORMER},
            )

    def test_unknown_api_type_raises(self):
        with pytest.raises(RuntimeError, match="commercial API"):
            app.transcribe_audio(
                b"audio", "some/model", None,
                {}, {"type": "api_unknown", "label": "UnknownAPI"},
            )


# ---------------------------------------------------------------------------
# Tests for parse_arguments
# ---------------------------------------------------------------------------
class TestParseArguments:
    def test_default_model(self):
        with mock.patch("sys.argv", ["app.py"]):
            args = app.parse_arguments()
            assert args.model_id == app.DEFAULT_MODEL

    def test_custom_model(self):
        with mock.patch("sys.argv", ["app.py", "--model_id", "openai/whisper-tiny"]):
            args = app.parse_arguments()
            assert args.model_id == "openai/whisper-tiny"


# ---------------------------------------------------------------------------
# Tests for wav_to_black_mp4
# ---------------------------------------------------------------------------
class TestWavToBlackMp4:
    def test_creates_video(self):
        wav_path = "/fake/test.wav"
        output_path = "/fake/test.mp4"

        fake_waveform = torch.randn(1, 16000)  # 1 second at 16kHz
        mock_audio = mock.MagicMock()
        mock_black = mock.MagicMock()
        mock_final = mock.MagicMock()
        mock_black.with_audio.return_value = mock_final

        with mock.patch("app.torchaudio.load", return_value=(fake_waveform, 16000)), \
             mock.patch("app.mp.AudioFileClip", return_value=mock_audio) as mock_afc, \
             mock.patch("app.mp.ColorClip", return_value=mock_black) as mock_cc:
            app.wav_to_black_mp4(wav_path, output_path, fps=25)
            mock_afc.assert_called_once_with(wav_path)
            mock_cc.assert_called_once()
            mock_black.with_audio.assert_called_once_with(mock_audio)
            mock_final.write_videofile.assert_called_once_with(output_path, fps=25)


# ---------------------------------------------------------------------------
# Tests for type constants
# ---------------------------------------------------------------------------
class TestTypeConstants:
    def test_type_constants_are_unique(self):
        types = [
            app.T_WHISPER, app.T_GENERIC, app.T_NEMO, app.T_SPEECHBRAIN,
            app.T_VOXTRAL, app.T_GENERATE, app.T_GRANITE, app.T_TRUST,
            app.T_QWEN_ASR, app.T_KYUTAI, app.T_ESPNET, app.T_ZIPFORMER,
            app.API_ASSEMBLYAI, app.API_ELEVENLABS, app.API_REVAI,
            app.API_GOOGLE, app.API_AZURE, app.API_COHERE,
            app.API_SPEECHMATICS, app.API_GENERIC,
        ]
        assert len(types) == len(set(types)), "Duplicate type constants found"

    def test_api_types_start_with_api_prefix(self):
        for t in [app.API_ASSEMBLYAI, app.API_ELEVENLABS, app.API_REVAI,
                   app.API_GOOGLE, app.API_AZURE, app.API_COHERE,
                   app.API_SPEECHMATICS, app.API_GENERIC]:
            assert t.startswith("api_")

    def test_model_types_do_not_start_with_api(self):
        for t in [app.T_WHISPER, app.T_GENERIC, app.T_NEMO, app.T_SPEECHBRAIN,
                   app.T_VOXTRAL, app.T_GENERATE, app.T_GRANITE, app.T_TRUST,
                   app.T_QWEN_ASR, app.T_KYUTAI, app.T_ESPNET, app.T_ZIPFORMER]:
            assert not t.startswith("api_")
