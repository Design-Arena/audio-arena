"""Configuration for the appointment scheduling benchmark."""
from pathlib import Path

from .turns import turns
from .tools import ToolsSchemaForTest
from .system import system_instruction


class BenchmarkConfig:
    """Configuration for the appointment scheduling benchmark."""

    name = "appointment_bench"
    description = "25-turn dual-appointment benchmark: two patients (Daniel/Danielle Nolan), two doctors (Perry/Barry), phone number swap+revert, 4 false memory traps, slot-taken error recovery, cross-entity state tracking"
    hf_repo = "arcada-labs/appointment-bench"

    turns = turns
    tools_schema = ToolsSchemaForTest
    system_instruction = system_instruction

    _benchmark_dir = Path(__file__).parent
    audio_dir = _benchmark_dir / "audio"
    real_audio_dir = _benchmark_dir / "real_audio"
    use_real_audio = False
    real_audio_speaker = None

    def get_audio_path(self, turn_index: int) -> Path:
        """Get the audio file path for a specific turn."""
        if self.use_real_audio:
            audio_dir = self.real_audio_dir / self.real_audio_speaker
        else:
            audio_dir = self.audio_dir
        if not audio_dir.exists() or not any(audio_dir.glob("*.wav")):
            source = f"real audio ({self.real_audio_speaker})" if self.use_real_audio else "generated audio"
            raise FileNotFoundError(
                f"No {source} files found in {audio_dir}."
            )
        return audio_dir / f"turn_{turn_index:03d}.wav"

    def list_speakers(self) -> list[str]:
        """List available real audio speakers."""
        if not self.real_audio_dir.exists():
            return []
        return sorted(
            d.name for d in self.real_audio_dir.iterdir()
            if d.is_dir() and any(d.glob("*.wav"))
        )
