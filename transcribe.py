import argparse
import os
import sys

from utils import build_whisper_pipeline


def transcribe_audio(file_path):
    pipe = build_whisper_pipeline(
        "nyrahealth/CrisperWhisper",
        batch_size=16,
        return_timestamps="word",
    )
    return pipe(file_path)


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
        print(f"An error occurred while transcribing the audio: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
