"""Configuration for the appointment scheduling benchmark."""
from pathlib import Path

from .turns import turns
from .tools import ToolsSchemaForTest
from .system import system_instruction


class BenchmarkConfig:
    """Configuration for the appointment scheduling benchmark."""

    name = "appointment_bench"
    description = "25-turn dual-appointment benchmark: two patients (Daniel/Danielle Nolan), two doctors (Perry/Barry), phone number swap+revert, 3 false memory traps, slot-taken error recovery, cross-entity state tracking"
    hf_repo = "arcada-labs/appointment-bench"

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
