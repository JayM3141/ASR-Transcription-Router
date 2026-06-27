import os
import sys
from unittest import mock

import pytest

import transcribe


class TestTranscribeAudioCLI:
    def test_transcribe_audio_calls_pipeline(self):
        mock_pipe = mock.MagicMock(return_value={"text": "hello world", "chunks": []})
        with mock.patch("transcribe.AutoModelForSpeechSeq2Seq") as mock_model_cls, \
             mock.patch("transcribe.AutoProcessor") as mock_proc_cls, \
             mock.patch("transcribe.pipeline", return_value=mock_pipe):
            mock_model = mock.MagicMock()
            mock_model_cls.from_pretrained.return_value = mock_model
            mock_proc = mock.MagicMock()
            mock_proc_cls.from_pretrained.return_value = mock_proc

            result = transcribe.transcribe_audio("test.wav")

            mock_model_cls.from_pretrained.assert_called_once()
            mock_proc_cls.from_pretrained.assert_called_once()
            mock_pipe.assert_called_once_with("test.wav")
            assert result == {"text": "hello world", "chunks": []}

    def test_transcribe_audio_uses_correct_model_id(self):
        mock_pipe = mock.MagicMock(return_value={"text": ""})
        with mock.patch("transcribe.AutoModelForSpeechSeq2Seq") as mock_model_cls, \
             mock.patch("transcribe.AutoProcessor") as mock_proc_cls, \
             mock.patch("transcribe.pipeline", return_value=mock_pipe):
            mock_model_cls.from_pretrained.return_value = mock.MagicMock()
            mock_proc_cls.from_pretrained.return_value = mock.MagicMock()

            transcribe.transcribe_audio("test.wav")

            model_id = mock_model_cls.from_pretrained.call_args[0][0]
            assert model_id == "nyrahealth/CrisperWhisper"


class TestMain:
    def test_main_missing_file(self, tmp_path):
        nonexistent = str(tmp_path / "no_such_file.wav")
        with mock.patch("sys.argv", ["transcribe.py", "--f", nonexistent]):
            with pytest.raises(SystemExit) as exc_info:
                transcribe.main()
            assert exc_info.value.code == 1

    def test_main_successful_transcription(self, tmp_path):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake wav data")

        with mock.patch("sys.argv", ["transcribe.py", "--f", str(audio_file)]), \
             mock.patch.object(
                 transcribe, "transcribe_audio",
                 return_value={"text": "hello world"},
             ), \
             mock.patch("builtins.print") as mock_print:
            transcribe.main()
            mock_print.assert_any_call("Transcription:")
            mock_print.assert_any_call("hello world")

    def test_main_transcription_error(self, tmp_path):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake wav data")

        with mock.patch("sys.argv", ["transcribe.py", "--f", str(audio_file)]), \
             mock.patch.object(
                 transcribe, "transcribe_audio",
                 side_effect=RuntimeError("model load failed"),
             ):
            with pytest.raises(SystemExit) as exc_info:
                transcribe.main()
            assert exc_info.value.code == 1

    def test_main_requires_f_argument(self):
        with mock.patch("sys.argv", ["transcribe.py"]):
            with pytest.raises(SystemExit) as exc_info:
                transcribe.main()
            assert exc_info.value.code == 2  # argparse exits with 2 for missing required args
