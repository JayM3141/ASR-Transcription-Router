import copy

from utils import adjust_pauses_for_hf_pipeline_output


class TestAdjustPausesForHfPipelineOutput:
    """Tests for adjust_pauses_for_hf_pipeline_output."""

    def test_no_chunks_returns_empty(self):
        pipeline_output = {"text": "", "chunks": []}
        result = adjust_pauses_for_hf_pipeline_output(pipeline_output)
        assert result["chunks"] == []

    def test_single_chunk_unchanged(self):
        pipeline_output = {
            "text": "hello",
            "chunks": [{"text": "hello", "timestamp": (0.0, 0.5)}],
        }
        result = adjust_pauses_for_hf_pipeline_output(pipeline_output)
        assert result["chunks"][0]["timestamp"] == (0.0, 0.5)

    def test_adjacent_words_no_pause(self):
        pipeline_output = {
            "text": "hello world",
            "chunks": [
                {"text": "hello", "timestamp": (0.0, 0.5)},
                {"text": "world", "timestamp": (0.5, 1.0)},
            ],
        }
        result = adjust_pauses_for_hf_pipeline_output(pipeline_output)
        assert result["chunks"][0]["timestamp"] == (0.0, 0.5)
        assert result["chunks"][1]["timestamp"] == (0.5, 1.0)

    def test_small_pause_split_evenly(self):
        pipeline_output = {
            "text": "hello world",
            "chunks": [
                {"text": "hello", "timestamp": (0.0, 0.5)},
                {"text": "world", "timestamp": (0.6, 1.0)},
            ],
        }
        result = adjust_pauses_for_hf_pipeline_output(pipeline_output)
        # pause = 0.1 < split_threshold(0.12), so distribute = 0.1/2 = 0.05
        assert result["chunks"][0]["timestamp"] == (0.0, 0.55)
        assert result["chunks"][1]["timestamp"] == (0.55, 1.0)

    def test_large_pause_capped_at_threshold(self):
        pipeline_output = {
            "text": "hello world",
            "chunks": [
                {"text": "hello", "timestamp": (0.0, 0.5)},
                {"text": "world", "timestamp": (1.0, 1.5)},
            ],
        }
        result = adjust_pauses_for_hf_pipeline_output(pipeline_output)
        # pause = 0.5 > split_threshold(0.12), so distribute = 0.12/2 = 0.06
        assert result["chunks"][0]["timestamp"] == (0.0, 0.56)
        assert result["chunks"][1]["timestamp"] == (0.94, 1.5)

    def test_custom_split_threshold(self):
        pipeline_output = {
            "text": "hello world",
            "chunks": [
                {"text": "hello", "timestamp": (0.0, 0.5)},
                {"text": "world", "timestamp": (1.0, 1.5)},
            ],
        }
        result = adjust_pauses_for_hf_pipeline_output(pipeline_output, split_threshold=0.20)
        # pause = 0.5 > 0.20, distribute = 0.20/2 = 0.10
        assert result["chunks"][0]["timestamp"] == (0.0, 0.6)
        assert result["chunks"][1]["timestamp"] == (0.9, 1.5)

    def test_multiple_chunks_with_pauses(self):
        pipeline_output = {
            "text": "a b c",
            "chunks": [
                {"text": "a", "timestamp": (0.0, 0.2)},
                {"text": "b", "timestamp": (0.4, 0.6)},
                {"text": "c", "timestamp": (0.8, 1.0)},
            ],
        }
        result = adjust_pauses_for_hf_pipeline_output(pipeline_output)
        # pause between a-b: 0.2 > 0.12 -> distribute = 0.06
        # a end: 0.2 + 0.06 = 0.26, b start: 0.4 - 0.06 = 0.34
        assert abs(result["chunks"][0]["timestamp"][0] - 0.0) < 1e-9
        assert abs(result["chunks"][0]["timestamp"][1] - 0.26) < 1e-9
        assert abs(result["chunks"][1]["timestamp"][0] - 0.34) < 1e-9
        # pause between b-c: b_end=0.6, c_start=0.8 -> pause=0.2 > 0.12 -> distribute=0.06
        # b end: 0.6 + 0.06 = 0.66, c start: 0.8 - 0.06 = 0.74
        assert abs(result["chunks"][1]["timestamp"][1] - 0.66) < 1e-9
        assert abs(result["chunks"][2]["timestamp"][0] - 0.74) < 1e-9
        assert abs(result["chunks"][2]["timestamp"][1] - 1.0) < 1e-9

    def test_negative_pause_not_adjusted(self):
        """Overlapping timestamps (negative pause) should not be adjusted."""
        pipeline_output = {
            "text": "hello world",
            "chunks": [
                {"text": "hello", "timestamp": (0.0, 0.6)},
                {"text": "world", "timestamp": (0.5, 1.0)},
            ],
        }
        result = adjust_pauses_for_hf_pipeline_output(pipeline_output)
        assert result["chunks"][0]["timestamp"] == (0.0, 0.6)
        assert result["chunks"][1]["timestamp"] == (0.5, 1.0)

    def test_zero_pause_not_adjusted(self):
        """Exactly zero pause should not be adjusted."""
        pipeline_output = {
            "text": "hello world",
            "chunks": [
                {"text": "hello", "timestamp": (0.0, 0.5)},
                {"text": "world", "timestamp": (0.5, 1.0)},
            ],
        }
        result = adjust_pauses_for_hf_pipeline_output(pipeline_output)
        assert result["chunks"][0]["timestamp"] == (0.0, 0.5)
        assert result["chunks"][1]["timestamp"] == (0.5, 1.0)

    def test_returns_modified_input(self):
        """The function modifies and returns the same dict."""
        pipeline_output = {
            "text": "hello",
            "chunks": [{"text": "hello", "timestamp": (0.0, 0.5)}],
        }
        result = adjust_pauses_for_hf_pipeline_output(pipeline_output)
        assert result is pipeline_output

    def test_pause_exactly_at_threshold(self):
        """Pause exactly equal to split_threshold is not > threshold, so distribute = pause/2."""
        pipeline_output = {
            "text": "hello world",
            "chunks": [
                {"text": "hello", "timestamp": (0.0, 0.5)},
                {"text": "world", "timestamp": (0.62, 1.0)},
            ],
        }
        result = adjust_pauses_for_hf_pipeline_output(pipeline_output)
        # pause = 0.12 == split_threshold(0.12), not > threshold
        # distribute = 0.12 / 2 = 0.06
        assert abs(result["chunks"][0]["timestamp"][1] - 0.56) < 1e-9
        assert abs(result["chunks"][1]["timestamp"][0] - 0.56) < 1e-9

    def test_text_field_preserved(self):
        pipeline_output = {
            "text": "hello world",
            "chunks": [
                {"text": "hello", "timestamp": (0.0, 0.5)},
                {"text": "world", "timestamp": (0.7, 1.0)},
            ],
        }
        result = adjust_pauses_for_hf_pipeline_output(pipeline_output)
        assert result["text"] == "hello world"
        assert result["chunks"][0]["text"] == "hello"
        assert result["chunks"][1]["text"] == "world"
