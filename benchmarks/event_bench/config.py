"""Configuration for the event planning benchmark."""
from pathlib import Path

from .turns import turns
from .tools import ToolsSchemaForTest
from .system import system_instruction


class BenchmarkConfig:
    """Configuration for the event planning benchmark."""

    name = "event_bench"
    description = "29-turn event planning benchmark: venue+catering+guest count changes, mid-sentence self-corrections, vague pronoun resolution, wrong-math correction, multi-request reversals, ambiguous add-on disambiguation, hypothetical reasoning, phone swap, retroactive date change, 3 false memory traps, cross-entity state tracking"
    hf_repo = "arcada-labs/event-bench"

    turns = turns
    tools_schema = ToolsSchemaForTest
    system_instruction = system_instruction

    _benchmark_dir = Path(__file__).parent
    audio_dir = _benchmark_dir / "audio"

    @classmethod
    def get_audio_path(cls, turn_index: int) -> Path:
        """Get the audio file path for a specific turn, downloading from HF if needed."""
        if not cls.audio_dir.exists() or not any(cls.audio_dir.glob("*.wav")):
            from audio_arena.data import ensure_audio
            cls.audio_dir = ensure_audio(cls._benchmark_dir, cls.hf_repo)
        return cls.audio_dir / f"turn_{turn_index:03d}.wav"
