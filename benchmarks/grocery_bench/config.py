"""Configuration for the grocery benchmark."""
from pathlib import Path

from .turns import turns
from .tools import ToolsSchemaForTest
from .system import system_instruction


class BenchmarkConfig:
    """Configuration for the grocery benchmark."""

    name = "grocery_bench"
    description = ("30-turn grocery ordering benchmark with 15 difficulty enhancements: "
                   "3-item turn, relative-math quantity, conditional addition, "
                   "chained corrections, ambiguous 'both', revert removal, "
                   "second subtotal after mods, 'same as first' recall, "
                   "partial name reference, phone number correction, "
                   "audio false start, homophone collision (flower/flour), "
                   "fifteen/fifty audio confusion, conditional removal by "
                   "price threshold, plus vague pronoun, mid-sentence "
                   "self-correction, false memory trap, item removal, "
                   "swap operation, retroactive qty change, "
                   "full order reconciliation")
    hf_repo = "arcada-labs/grocery-bench"

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
