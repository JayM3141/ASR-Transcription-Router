import argparse
import logging
import os
import sys
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

logger = logging.getLogger(__name__)


def transcribe_audio(file_path):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "nyrahealth/CrisperWhisper"  # You can change this to a different model if needed

    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{model_id}': {e}") from e
    model.to(device)

    try:
        processor = AutoProcessor.from_pretrained(model_id)
    except Exception as e:
        raise RuntimeError(f"Failed to load processor for '{model_id}': {e}") from e

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps="word",
        torch_dtype=torch_dtype,
        device=device,
    )

    result = pipe(file_path)
    if result is None:
        raise RuntimeError("Transcription pipeline returned no result.")
    return result


def main():
    parser = argparse.ArgumentParser(description="Transcribe an audio file.")
    parser.add_argument("--f", type=str, required=True, help="Path to the audio file")
    args = parser.parse_args()

    if not os.path.exists(args.f):
        print(f"Error: The file '{args.f}' does not exist.")
        sys.exit(1)

    try:
        transcription = transcribe_audio(args.f)
        print("Transcription:")
        print(transcription["text"])
    except Exception as e:
        logger.error("Transcription failed", exc_info=True)
        print(f"An error occurred while transcribing the audio: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
