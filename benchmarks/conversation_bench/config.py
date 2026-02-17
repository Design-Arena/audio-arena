"""Configuration for the conversation bench benchmark."""
from pathlib import Path

from .turns import turns
from .tools import ToolsSchemaForTest
from .system import system_instruction


class BenchmarkConfig:
    """Configuration for the conversation bench benchmark."""

    # Benchmark metadata
    name = "conversation_bench"
    description = "75-turn hard benchmark with ~12K token knowledge base and 9 tools"

    # Benchmark data
    turns = turns
    tools_schema = ToolsSchemaForTest

    # Audio directory path
    audio_dir = Path(__file__).parent / "audio"

    # System prompt
    system_instruction = system_instruction

    @classmethod
    def get_audio_path(cls, turn_index: int) -> Path:
        """Get the audio file path for a specific turn."""
        return cls.audio_dir / f"turn_{turn_index:03d}.wav"
