"""Generate WAV audio files for any benchmark's turns using OpenAI TTS.

Usage:
    uv run python scripts/generate_audio.py restaurant_bench
    uv run python scripts/generate_audio.py travel_bench --voice nova
    uv run python scripts/generate_audio.py product_bench --model tts-1
"""

import argparse
import importlib
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def load_benchmark_turns(benchmark_name: str) -> list[dict]:
    """Load turns from a benchmark by name."""
    try:
        module = importlib.import_module(f"benchmarks.{benchmark_name}.turns")
        return module.turns
    except ModuleNotFoundError:
        print(f"Error: Benchmark '{benchmark_name}' not found.")
        print(f"Available benchmarks:")
        benchmarks_dir = Path(__file__).resolve().parent.parent / "benchmarks"
        for d in sorted(benchmarks_dir.iterdir()):
            if d.is_dir() and (d / "turns.py").exists() and not d.name.startswith("_"):
                print(f"  {d.name}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Generate TTS audio for a benchmark")
    parser.add_argument("benchmark", help="Benchmark name (e.g., restaurant_bench)")
    parser.add_argument("--voice", default="echo", help="OpenAI TTS voice (default: echo)")
    parser.add_argument("--model", default="tts-1-hd", help="OpenAI TTS model (default: tts-1-hd)")
    parser.add_argument("--force", action="store_true", help="Regenerate existing files")
    args = parser.parse_args()

    turns = load_benchmark_turns(args.benchmark)

    audio_dir = Path(__file__).resolve().parent.parent / "benchmarks" / args.benchmark / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI()

    generated = 0
    skipped = 0

    for i, turn in enumerate(turns):
        out_path = audio_dir / f"turn_{i:03d}.wav"

        if out_path.exists() and not args.force:
            print(f"  [skip] {out_path.name} already exists")
            skipped += 1
            continue

        text = turn["input"]
        print(f"  [{i:02d}/{len(turns)-1}] Generating {out_path.name}: {text[:60]}...")

        response = client.audio.speech.create(
            model=args.model,
            voice=args.voice,
            input=text,
            response_format="wav",
        )
        response.stream_to_file(str(out_path))
        generated += 1

    print(f"\nDone — {generated} generated, {skipped} skipped, {len(turns)} total turns in {audio_dir}")


if __name__ == "__main__":
    main()
