import argparse
import os
from pathlib import Path
from omnizart.cli.cli import download_checkpoints
from omnizart.cli.music import transcribe as music_transcribe
from omnizart.cli.drum import transcribe as drum_transcribe
from omnizart.cli.chord import transcribe as chord_transcribe
from omnizart import MODULE_PATH

MODEL_MAP = {
    "music": music_transcribe,
    "drum": drum_transcribe,
    "chord": chord_transcribe,
}

# Known checkpoints to check for (any one is enough to skip downloading)
EXPECTED_CHECKPOINTS = [
    "checkpoints/music/music_piano/variables/variables.data-00000-of-00001",
    "checkpoints/chord/chord_v1/variables/variables.data-00000-of-00001",
    "checkpoints/drum/drum_keras/variables/variables.data-00000-of-00001"
]


def checkpoints_exist(base_path):
    music_piano_ckpt = base_path / "omnizart/checkpoints/music/music_piano/variables/variables.data-00000-of-00001"
    print(f"Checking for music piano checkpoint at: {music_piano_ckpt}")
    return music_piano_ckpt.exists()


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio using Omnizart.")
    parser.add_argument("-i", "--input", required=True, help="Path to input WAV file")
    parser.add_argument("-o", "--output", default="./", help="Output file or directory for results")
    parser.add_argument("-m", "--model", default="music", choices=MODEL_MAP.keys(), help="Model type to use")
    parser.add_argument("--checkpoints-path", default=str(Path(MODULE_PATH).parent), help="Path to omnizart module directory for checkpoints")

    args = parser.parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    checkpoints_path = Path(args.checkpoints_path).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    if input_path.suffix.lower() != ".wav":
        raise ValueError(f"Input must be a .wav file, got: {input_path.suffix}")

    # 1. Check and download checkpoints if necessary
    if not checkpoints_exist(checkpoints_path):
        print("Downloading model checkpoints...")
        download_checkpoints.callback(output_path=str(checkpoints_path))
    else:
        print("Checkpoints already exist. Skipping download.")

    # 2. Transcribe using selected model
    print(f"Transcribing using model: {args.model}")
    MODEL_MAP[args.model].callback(
        input_audio=str(input_path),
        model_path=None,
        output=str(output_path)
    )

    print(f"Done. Output should be at: {output_path}")


if __name__ == "__main__":
    main()
