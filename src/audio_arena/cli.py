"""Audio Arena: Multi-turn voice AI evaluation framework CLI.

Usage:
    uv run audio-arena run grocery_bench --model gpt-realtime
    uv run audio-arena run grocery_bench --model nova-sonic --rehydrate
    uv run audio-arena judge runs/grocery_bench/20251213T123456_gpt-realtime
    uv run audio-arena list-benchmarks
"""

import asyncio
import importlib
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import click
from dotenv import load_dotenv
from loguru import logger

from audio_arena.pipelines.base import BasePipeline

# Load environment variables from .env file
load_dotenv()


# ============================================================================
# Service Aliases
# ============================================================================

SERVICE_ALIASES = {
    "openai": "pipecat.services.openai.llm.OpenAILLMService",
    "openai-realtime": "audio_arena.pipelines.openai_realtime.OpenAIRealtimeLLMServiceExplicitToolResult",
    "openrouter": "pipecat.services.openai.llm.OpenAILLMService",  # OpenRouter uses OpenAI-compatible API
    "anthropic": "pipecat.services.anthropic.llm.AnthropicLLMService",
    "google": "pipecat.services.google.llm.GoogleLLMService",
    "gemini-live": "audio_arena.pipelines.realtime.GeminiLiveLLMServiceWithReconnection",
    "bedrock": "pipecat.services.aws.llm.AWSBedrockLLMService",
    "groq": "pipecat.services.groq.llm.GroqLLMService",
    "cerebras": "pipecat.services.cerebras.llm.CerebrasLLMService",
    "ultravox-realtime": "pipecat.services.ultravox.llm.UltravoxRealtimeLLMService",
}


# ============================================================================
# Model Aliases — friendly names → (actual_api_model, default_service, default_pipeline)
# ============================================================================

MODEL_ALIASES: dict[str, tuple[str, Optional[str], Optional[str]]] = {
    "gpt-realtime": ("gpt-realtime", "openai-realtime", None),
    "gemini-native-audio": (
        "gemini-2.5-flash-native-audio-preview-12-2025",
        "gemini-live",
        None,
    ),
    "ultravox": ("ultravox-v0.7", "ultravox-realtime", None),
    "grok-realtime": ("grok-realtime", None, None),
    "nova-sonic": ("amazon.nova-2-sonic-v1:0", None, "nova-sonic"),
}


# ============================================================================
# Pipeline Registry
# ============================================================================

PIPELINE_CLASSES = {
    "text": "audio_arena.pipelines.text.TextPipeline",
    "realtime": "audio_arena.pipelines.realtime.RealtimePipeline",
    "grok-realtime": "audio_arena.pipelines.grok_realtime.GrokRealtimePipeline",
    "nova-sonic": "audio_arena.pipelines.nova_sonic.NovaSonicPipeline",
}

REHYDRATED_TURN_RUNS_DIRNAME = "turn_runs"


# ============================================================================
# Utility Functions
# ============================================================================


def resolve_model_alias(
    model: str,
    service: Optional[str] = None,
    pipeline: Optional[str] = None,
) -> tuple[str, Optional[str], Optional[str]]:
    """Resolve a model alias to (actual_model, service, pipeline).

    If the model is a known alias, fills in service/pipeline defaults
    (only when not explicitly provided by the caller).
    """
    if model in MODEL_ALIASES:
        actual_model, default_service, default_pipeline = MODEL_ALIASES[model]
        return (
            actual_model,
            service or default_service,
            pipeline or default_pipeline,
        )
    return model, service, pipeline


def load_service_class(service: str) -> type:
    """Load service class from fully qualified name or alias."""
    class_name = SERVICE_ALIASES.get(service, service)
    module_name, cls_name = class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)


def load_benchmark(name: str):
    """Load benchmark by name from benchmarks/ directory."""
    try:
        module = importlib.import_module(f"benchmarks.{name}.config")
        return module.BenchmarkConfig
    except ModuleNotFoundError as e:
        raise click.UsageError(f"Benchmark '{name}' not found: {e}")


def list_available_benchmarks() -> list[str]:
    """Discover available benchmarks by scanning benchmarks/ directory."""
    # Find the benchmarks directory relative to the package or current working directory
    cwd_benchmarks = Path.cwd() / "benchmarks"

    benchmarks = []
    if cwd_benchmarks.exists():
        for d in cwd_benchmarks.iterdir():
            if d.is_dir() and not d.name.startswith("_") and (d / "config.py").exists():
                benchmarks.append(d.name)

    return sorted(benchmarks)


def load_benchmark_kb_text(name: str) -> Optional[str]:
    """Load benchmark knowledge base text when the benchmark provides one."""
    try:
        module = importlib.import_module(f"benchmarks.{name}.config")
    except ModuleNotFoundError:
        return None

    module_path = getattr(module, "__file__", None)
    if not module_path:
        return None

    kb_path = Path(module_path).resolve().parent / "data" / "knowledge_base.txt"
    if not kb_path.exists():
        return None

    return kb_path.read_text(encoding="utf-8")


def load_prompt_visible_kb_text(name: str) -> Optional[str]:
    """Load the prompt-visible KB text exposed by a benchmark system module."""
    try:
        module = importlib.import_module(f"benchmarks.{name}.system")
    except ModuleNotFoundError:
        return None

    return getattr(module, "prompt_visible_knowledge_base", None)


def get_pipeline_class(pipeline_type: str) -> type:
    """Load pipeline class by type name."""
    class_name = PIPELINE_CLASSES.get(pipeline_type)
    if not class_name:
        raise click.UsageError(
            f"Unknown pipeline: {pipeline_type}. Available: {list(PIPELINE_CLASSES.keys())}"
        )
    module_name, cls_name = class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)


def infer_pipeline(model: str) -> str:
    """Infer default pipeline from model name pattern."""
    m = model.lower()
    # Grok realtime uses dedicated pipeline for xAI-specific protocol handling
    if m.startswith("grok") and "realtime" in m:
        return "grok-realtime"
    if "realtime" in m:
        return "realtime"
    if "native-audio" in m or "live" in m:
        return "realtime"
    if "ultravox" in m:
        return "realtime"
    if "nova-sonic" in m or "nova_sonic" in m:
        return "nova-sonic"
    return "text"


def turn_uses_oracle_post_tool_continuation(
    *,
    pipeline_type: str,
    turn: dict[str, Any],
) -> bool:
    """Return True for rehydrated realtime turns with any scripted tool call."""
    required_call = turn.get("required_function_call")
    return pipeline_type in {"realtime", "grok-realtime"} and required_call is not None


def build_oracle_continuation_history(
    golden_history: Optional[list[dict[str, Any]]],
    turn: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return seeded history for the fresh oracle continuation session."""
    history = list(golden_history or [])
    history.append(
        {
            "input": turn.get("input", ""),
            "required_function_call": turn.get("required_function_call"),
            "function_call_response": turn.get("function_call_response"),
            # Omit assistant text so the continuation session generates it fresh.
            "golden_text": "",
        }
    )
    return history


def merge_oracle_continuation_artifacts(
    *,
    transcript_path: Path,
    live_tool_capture: dict[str, Any],
) -> None:
    """Rewrite the single-turn transcript row with live + oracle metadata."""
    if not transcript_path.exists():
        raise RuntimeError(f"Missing continuation transcript: {transcript_path}")

    lines = [
        json.loads(line)
        for line in transcript_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) != 1:
        raise RuntimeError(
            f"Expected exactly 1 transcript row in {transcript_path}, got {len(lines)}"
        )

    row = lines[0]
    actual_tool_calls = live_tool_capture.get("actual_tool_calls")
    actual_tool_results = live_tool_capture.get("actual_tool_results")
    if actual_tool_calls is None:
        actual_tool_call = live_tool_capture.get("actual_tool_call")
        actual_tool_calls = [actual_tool_call] if actual_tool_call else []
    if actual_tool_results is None:
        actual_tool_result = live_tool_capture.get("actual_tool_result")
        if actual_tool_result is not None:
            actual_tool_results = [
                {
                    "name": (
                        actual_tool_calls[0].get("name") if actual_tool_calls else None
                    ),
                    "response": actual_tool_result,
                }
            ]
        else:
            actual_tool_results = []
    row["tool_calls"] = actual_tool_calls
    row["tool_results"] = actual_tool_results

    row["oracle_continuation"] = {
        "used": True,
        "tool_name_correct": live_tool_capture.get("tool_name_correct"),
        "tool_args_correct": live_tool_capture.get("tool_args_correct"),
        "tool_use_pass": live_tool_capture.get("tool_use_pass"),
        "oracle_tool_calls": live_tool_capture.get("oracle_tool_calls", []),
        "oracle_tool_results": live_tool_capture.get("oracle_tool_results", []),
    }

    transcript_path.write_text(
        json.dumps(row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_missing_tool_capture(turn: dict[str, Any]) -> dict[str, Any]:
    """Return a failed live tool-use verdict when no tool call was captured."""
    required_call = turn.get("required_function_call")
    oracle_calls = (
        required_call
        if isinstance(required_call, list)
        else [required_call] if required_call else []
    )
    response = turn.get("function_call_response")
    oracle_results = (
        response
        if isinstance(response, list)
        else [response] if response is not None else []
    )
    return {
        "actual_tool_calls": [],
        "actual_tool_results": [],
        "actual_tool_call": None,
        "actual_tool_result": None,
        "tool_name_correct": False,
        "tool_args_correct": False if required_call else None,
        "tool_use_pass": False,
        "oracle_tool_calls": oracle_calls,
        "oracle_tool_results": oracle_results,
    }


def _load_single_transcript_row(transcript_path: Path) -> dict[str, Any]:
    """Load exactly one transcript row from a per-turn transcript artifact."""
    lines = [
        json.loads(line)
        for line in transcript_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) != 1:
        raise RuntimeError(
            f"Expected exactly 1 transcript row in {transcript_path}, got {len(lines)}"
        )
    return lines[0]


def build_live_tool_capture_from_transcript(
    *,
    turn: dict[str, Any],
    transcript_path: Path,
) -> dict[str, Any]:
    """Build tool-capture metadata from a completed live capture transcript row."""
    row = _load_single_transcript_row(transcript_path)
    actual_tool_calls = list(row.get("tool_calls", []) or [])
    actual_tool_results = list(row.get("tool_results", []) or [])
    required_call = turn.get("required_function_call")
    oracle_calls = (
        required_call
        if isinstance(required_call, list)
        else [required_call] if required_call else []
    )
    response = turn.get("function_call_response")
    oracle_results = (
        response
        if isinstance(response, list)
        else [response] if response is not None else []
    )

    if not actual_tool_calls:
        return build_missing_tool_capture(turn)

    remaining_actual_indices = set(range(len(actual_tool_calls)))
    matched_expected = 0
    for expected_call in oracle_calls:
        for actual_index, actual_call in enumerate(actual_tool_calls):
            if actual_index not in remaining_actual_indices:
                continue
            if actual_call.get("name") != expected_call.get("name"):
                continue
            if not BasePipeline._tool_args_match(
                expected_call.get("args", {}),
                actual_call.get("args", {}),
            ):
                continue
            remaining_actual_indices.remove(actual_index)
            matched_expected += 1
            break

    matching_name_calls = [
        actual_call
        for actual_call in actual_tool_calls
        if any(
            actual_call.get("name") == oracle_call.get("name")
            for oracle_call in oracle_calls
        )
    ]

    tool_name_correct = len(matching_name_calls) == len(actual_tool_calls)
    tool_args_correct = matched_expected == len(oracle_calls)
    return {
        "actual_tool_calls": actual_tool_calls,
        "actual_tool_results": actual_tool_results,
        "actual_tool_call": actual_tool_calls[0] if actual_tool_calls else None,
        "actual_tool_result": (
            actual_tool_results[0].get("response") if actual_tool_results else None
        ),
        "tool_name_correct": tool_name_correct,
        "tool_args_correct": tool_args_correct,
        "tool_use_pass": bool(tool_name_correct and tool_args_correct),
        "oracle_tool_calls": oracle_calls,
        "oracle_tool_results": oracle_results,
    }


def get_disable_vad_status_messages(
    *,
    disable_vad: bool,
    rehydrate: bool,
    pipeline_type: str,
    service: Optional[str],
    model: str,
) -> list[str]:
    """Return user-visible status/warning messages for --disable-vad behavior."""
    if not disable_vad:
        return []

    if pipeline_type != "realtime":
        return [
            "[disable-vad] Ignored: --disable-vad only applies to the realtime pipeline.",
        ]

    service_name = (service or "").lower()
    if service_name != "openai-realtime":
        return [
            f"[disable-vad] Ignored: supported only for --service openai-realtime (got: {service or 'none'}).",
        ]

    model_name = model.lower()
    is_openai_realtime_model = model_name.startswith("gpt") and "realtime" in model_name
    if not is_openai_realtime_model:
        return [
            f"[disable-vad] Ignored: model '{model}' is not an OpenAI realtime model.",
        ]

    if rehydrate:
        return [
            "[disable-vad] Active: server-side VAD disabled for OpenAI Realtime.",
            "[disable-vad] Active: using manual input_audio_buffer.commit/response.create turn handling.",
            "[disable-vad] Rehydration still seeds prior turns with conversation.item.create.",
        ]

    return [
        "[disable-vad] Active: server-side VAD disabled for OpenAI Realtime.",
        "[disable-vad] Note: prior-turn rehydration is separate and does not use response.create input.",
    ]


def create_run_directory(benchmark_name: str, model: str) -> Path:
    """Create timestamped run directory."""
    import uuid

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    # Add a short unique suffix to prevent collisions in parallel runs
    unique_suffix = str(uuid.uuid4())[:8]
    # Sanitize model name for filesystem (replace / and :)
    safe_model = model.replace("/", "_").replace(":", "_")
    run_dir = (
        Path("runs") / benchmark_name / f"{timestamp}_{safe_model}_{unique_suffix}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


_logging_initialized = False


def setup_logging(run_dir: Path, verbose: bool = False):
    """Configure logging to both console and run directory.

    The global logger.remove() + console handler setup only runs once so that
    parallel callers (e.g. run_all_benchmarks.py) don't destroy each other's
    per-turn sinks.  Each call still adds its own per-run file sink.
    """
    global _logging_initialized
    if not _logging_initialized:
        logger.remove()
        logger.add(
            sys.stderr,
            level="INFO" if not verbose else "DEBUG",
            format="<level>{message}</level>",
        )
        _logging_initialized = True

    logger.add(
        run_dir / "run.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} {level} {name}: {message}",
    )


def add_turn_logging_sink(turn_run_dir: Path) -> int:
    """Add a per-turn file sink scoped to a rehydrated worker task."""
    turn_dir_str = str(turn_run_dir)
    return logger.add(
        turn_run_dir / "run.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} {level} {name}: {message}",
        filter=lambda record: record["extra"].get("rehydration_turn_dir")
        == turn_dir_str,
    )


def build_rehydrated_turn_run_dir(run_dir: Path, turn_index: int, width: int) -> Path:
    """Return the isolated artifact directory for a rehydrated target turn."""
    return run_dir / REHYDRATED_TURN_RUNS_DIRNAME / f"turn_{turn_index:0{width}d}"


def read_jsonl_records(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts."""
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def finalize_rehydrated_run_artifacts(
    *,
    run_dir: Path,
    model: str,
    target_indices: list[int],
    turn_results: dict[int, dict],
    parallel: int,
    disable_vad: bool,
    real_audio_speaker: Optional[str],
) -> dict:
    """Merge canonical parent artifacts from isolated per-turn rehydrated outputs."""
    merged_records: list[dict] = []
    seen_turns: dict[int, Path] = {}
    failed_turns: list[int] = []
    manifest_entries: list[dict] = []

    for turn_index in sorted(target_indices):
        result = turn_results.get(turn_index)
        if result is None:
            raise RuntimeError(
                f"Missing execution result metadata for rehydrated turn {turn_index}."
            )

        turn_run_dir = Path(result["turn_run_dir"])
        transcript_path = turn_run_dir / "transcript.jsonl"
        runtime_path = turn_run_dir / "runtime.json"
        conversation_wav_path = turn_run_dir / "conversation.wav"
        success = bool(result.get("success"))

        manifest_entry = {
            "turn": turn_index,
            "turn_run_dir": str(turn_run_dir),
            "success": success,
            "error": result.get("error"),
            "transcript_path": str(transcript_path),
            "transcript_exists": transcript_path.exists(),
            "runtime_path": str(runtime_path),
            "runtime_exists": runtime_path.exists(),
            "conversation_wav_path": str(conversation_wav_path),
            "conversation_wav_exists": conversation_wav_path.exists(),
        }

        if success:
            if not transcript_path.exists():
                raise RuntimeError(
                    f"Successful rehydrated turn {turn_index} is missing transcript.jsonl in {turn_run_dir}."
                )
            records = read_jsonl_records(transcript_path)
            if len(records) != 1:
                raise RuntimeError(
                    f"Successful rehydrated turn {turn_index} must have exactly one transcript row; "
                    f"found {len(records)} in {transcript_path}."
                )
            record = records[0]
            actual_turn = record.get("turn")
            if actual_turn in seen_turns:
                raise RuntimeError(
                    f"Duplicate transcript rows found for rehydrated turn {actual_turn}: "
                    f"{seen_turns[actual_turn]} and {transcript_path}."
                )
            if actual_turn != turn_index:
                raise RuntimeError(
                    f"Rehydrated turn {turn_index} wrote transcript row for turn {actual_turn} "
                    f"in {transcript_path}."
                )
            seen_turns[actual_turn] = transcript_path
            merged_records.append(record)
        else:
            failed_turns.append(turn_index)

        manifest_entries.append(manifest_entry)

    merged_records.sort(key=lambda record: record["turn"])
    transcript_path = run_dir / "transcript.jsonl"
    transcript_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n" for record in merged_records
        ),
        encoding="utf-8",
    )

    turn_taking_skip_reason = "Per-turn rehydrated runs keep audio isolated in turn_runs/; parent conversation.wav is intentionally omitted."
    runtime = {
        "model_name": model,
        "turns": len(merged_records),
        "total_attempted": len(target_indices),
        "failed_turns": failed_turns,
        "mode": "rehydrated",
        "parallel": parallel,
        "disable_vad": disable_vad,
        "audio_source": (
            f"real_audio/{real_audio_speaker}" if real_audio_speaker else "tts"
        ),
        "turn_artifact_layout": "per_turn_subdirs",
        "turn_taking_supported": False,
        "turn_taking_skip_reason": turn_taking_skip_reason,
        "note": "Single-step rehydration: each turn evaluated independently with golden prior context",
    }
    (run_dir / "runtime.json").write_text(
        json.dumps(runtime, indent=2), encoding="utf-8"
    )

    manifest = {
        "model_name": model,
        "mode": "rehydrated",
        "parallel": parallel,
        "target_turns": sorted(target_indices),
        "turn_artifact_layout": "per_turn_subdirs",
        "turn_taking_supported": False,
        "turn_taking_skip_reason": turn_taking_skip_reason,
        "turns": manifest_entries,
    }
    (run_dir / "rehydrated_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return runtime


# ============================================================================
# CLI Commands
# ============================================================================


@click.group()
def cli():
    """Audio Arena: Multi-turn voice AI evaluation framework."""
    pass


@cli.command()
@click.argument("benchmark_name")
@click.option(
    "--model", required=True, help="Model name (e.g., gpt-4o, claude-sonnet-4-5)"
)
@click.option("--service", help="Service class name or alias (e.g., openai, anthropic)")
@click.option(
    "--pipeline",
    help="Pipeline type (text, realtime, nova-sonic). Auto-detected if not specified.",
)
@click.option("--only-turns", help="Comma-separated turn indices to run (e.g., 0,1,2)")
@click.option(
    "--rehydrate",
    is_flag=True,
    help="Single-step rehydration mode: evaluate each turn independently with golden prior context.",
)
@click.option(
    "--parallel",
    type=int,
    default=1,
    help="Max concurrent turns in rehydrated mode (default: 1 = sequential). Ignored for normal runs.",
)
@click.option(
    "--disable-vad",
    is_flag=True,
    help="Disable server-side VAD for compatible realtime models (manual input_audio_buffer.commit/response.create flow).",
)
@click.option(
    "--juice",
    type=int,
    default=None,
    help="OpenAI Realtime backdoor reasoning-effort override.",
)
@click.option(
    "--verbosity",
    type=int,
    default=None,
    help="OpenAI Realtime backdoor verbosity override.",
)
@click.option(
    "--real-audio",
    "real_audio_speaker",
    default=None,
    help='Use real (human-recorded) audio. Pass a speaker name (e.g., "person1") or "all" to run every speaker.',
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option(
    "--skip-audio",
    is_flag=True,
    help="Skip saving conversation.wav audio recording (saves disk space).",
)
def run(
    benchmark_name: str,
    model: str,
    service: Optional[str],
    pipeline: Optional[str],
    only_turns: Optional[str],
    rehydrate: bool,
    parallel: int,
    disable_vad: bool,
    juice: Optional[int],
    verbosity: Optional[int],
    real_audio_speaker: Optional[str],
    verbose: bool,
    skip_audio: bool,
):
    """Run a benchmark against an LLM.

    Model aliases (auto-fill service & pipeline):
        gpt-realtime, gemini-native-audio, ultravox, grok-realtime, nova-sonic

    Examples:
        uv run audio-arena run conversation_bench --model gpt-realtime
        uv run audio-arena run conversation_bench --model gpt-realtime --real-audio person1
        uv run audio-arena run conversation_bench --model gpt-realtime --real-audio all
        uv run audio-arena run conversation_bench --model nova-sonic
        uv run audio-arena run conversation_bench --model gemini-native-audio --rehydrate
        uv run audio-arena run conversation_bench --model claude-sonnet-4-5 --service anthropic
    """
    model, service, pipeline = resolve_model_alias(model, service, pipeline)

    if skip_audio:
        os.environ["SKIP_AUDIO_RECORDING"] = "1"

    if real_audio_speaker and real_audio_speaker.lower() == "all":
        _run_all_speakers(
            benchmark_name,
            model,
            service,
            pipeline,
            only_turns,
            rehydrate,
            parallel,
            disable_vad,
            juice,
            verbosity,
            verbose,
        )
        return

    if rehydrate:
        asyncio.run(
            _run_rehydrated(
                benchmark_name,
                model,
                service,
                pipeline,
                only_turns,
                verbose,
                parallel,
                disable_vad=disable_vad,
                juice=juice,
                verbosity=verbosity,
                real_audio_speaker=real_audio_speaker,
            )
        )
    else:
        asyncio.run(
            _run(
                benchmark_name,
                model,
                service,
                pipeline,
                only_turns,
                verbose,
                disable_vad=disable_vad,
                juice=juice,
                verbosity=verbosity,
                real_audio_speaker=real_audio_speaker,
            )
        )


def _run_all_speakers(
    benchmark_name: str,
    model: str,
    service: Optional[str],
    pipeline: Optional[str],
    only_turns: Optional[str],
    rehydrate: bool,
    parallel: int,
    disable_vad: bool,
    juice: Optional[int],
    verbosity: Optional[int],
    verbose: bool,
):
    """Run the benchmark once per available real-audio speaker."""
    BenchmarkConfig = load_benchmark(benchmark_name)
    benchmark = BenchmarkConfig()
    speakers = benchmark.list_speakers()
    if not speakers:
        # Try downloading the full real_audio/ tree from HF
        _download_real_audio(benchmark, speaker=None)
        speakers = benchmark.list_speakers()
    if not speakers:
        raise click.UsageError(
            f"No real audio speakers found for {benchmark_name}. "
            f"Add recordings to benchmarks/{benchmark_name}/real_audio/<speaker>/ "
            f"or upload them to HF first."
        )
    click.echo(f"Running all speakers: {speakers}")
    for speaker in speakers:
        click.echo(f"\n{'='*60}")
        click.echo(f"Speaker: {speaker}")
        click.echo(f"{'='*60}")
        if rehydrate:
            asyncio.run(
                _run_rehydrated(
                    benchmark_name,
                    model,
                    service,
                    pipeline,
                    only_turns,
                    verbose,
                    parallel,
                    disable_vad=disable_vad,
                    juice=juice,
                    verbosity=verbosity,
                    real_audio_speaker=speaker,
                )
            )
        else:
            asyncio.run(
                _run(
                    benchmark_name,
                    model,
                    service,
                    pipeline,
                    only_turns,
                    verbose,
                    disable_vad=disable_vad,
                    juice=juice,
                    verbosity=verbosity,
                    real_audio_speaker=speaker,
                )
            )


def _download_real_audio(benchmark, speaker: Optional[str] = None):
    """Download real audio from HF if not present locally."""
    hf_repo = getattr(benchmark, "hf_repo", None)
    if not hf_repo:
        return

    try:
        from huggingface_hub import hf_hub_download, list_repo_tree
    except ImportError:
        click.echo("huggingface_hub not installed — cannot auto-download real audio.")
        return

    if speaker:
        include = f"real_audio/{speaker}/*.wav"
    else:
        include = "real_audio/**/*.wav"

    target_dir = benchmark._benchmark_dir
    click.echo(f"Downloading real audio from HF ({hf_repo}) ...")

    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=hf_repo,
            repo_type="dataset",
            allow_patterns=[include],
            local_dir=str(target_dir),
        )
        click.echo("Download complete.")
    except Exception as e:
        click.echo(f"Could not download real audio from HF: {e}")


def _setup_real_audio(benchmark, speaker: str):
    """Configure benchmark for real audio: download if needed, validate coverage."""
    speaker_dir = benchmark.real_audio_dir / speaker
    if not speaker_dir.exists() or not any(speaker_dir.glob("*.wav")):
        _download_real_audio(benchmark, speaker=speaker)

    if not speaker_dir.exists() or not any(speaker_dir.glob("*.wav")):
        raise click.UsageError(
            f"No real audio files found for speaker '{speaker}' at {speaker_dir}. "
            f"Record audio files as {speaker_dir}/turn_000.wav, turn_001.wav, ... "
            f"or upload to HF and re-run."
        )

    total_turns = len(benchmark.turns)
    missing = []
    for i in range(total_turns):
        if not (speaker_dir / f"turn_{i:03d}.wav").exists():
            missing.append(f"turn_{i:03d}.wav")
    if missing:
        raise click.UsageError(
            f"Speaker '{speaker}' is missing {len(missing)}/{total_turns} turn files "
            f"in {speaker_dir}: {', '.join(missing[:5])}"
            + (f" ... and {len(missing)-5} more" if len(missing) > 5 else "")
        )

    benchmark.use_real_audio = True
    benchmark.real_audio_speaker = speaker
    click.echo(f"Using real audio: speaker={speaker} ({total_turns} turns)")


async def _run(
    benchmark_name: str,
    model: str,
    service: Optional[str],
    pipeline_type: Optional[str],
    only_turns: Optional[str],
    verbose: bool,
    disable_vad: bool = False,
    juice: Optional[int] = None,
    verbosity: Optional[int] = None,
    real_audio_speaker: Optional[str] = None,
):
    """Async implementation of the run command."""
    # Load benchmark
    BenchmarkConfig = load_benchmark(benchmark_name)
    benchmark = BenchmarkConfig()

    if real_audio_speaker:
        _setup_real_audio(benchmark, real_audio_speaker)

    # Infer pipeline if not specified
    if not pipeline_type:
        pipeline_type = infer_pipeline(model)
        click.echo(f"Auto-detected pipeline: {pipeline_type}")

    pipeline_cls = get_pipeline_class(pipeline_type)

    # Check if pipeline requires a service
    requires_service = getattr(pipeline_cls, "requires_service", True)
    if requires_service and not service:
        raise click.UsageError(f"--service is required for {pipeline_type} pipeline")

    for msg in get_disable_vad_status_messages(
        disable_vad=disable_vad,
        rehydrate=False,
        pipeline_type=pipeline_type,
        service=service,
        model=model,
    ):
        click.echo(msg)

    # Load service class if provided
    service_class = load_service_class(service) if service else None

    # Create output directory
    run_dir = create_run_directory(benchmark_name, model)
    click.echo(f"Output directory: {run_dir}")

    # Setup logging
    setup_logging(run_dir, verbose)

    # Create recorder
    from audio_arena.recording.transcript_recorder import TranscriptRecorder

    recorder = TranscriptRecorder(run_dir, model)

    # Parse turn indices if provided
    turn_indices = None
    if only_turns:
        turn_indices = [int(i.strip()) for i in only_turns.split(",")]
        click.echo(f"Running only turns: {turn_indices}")

    # Run the pipeline
    try:
        pipeline_instance = pipeline_cls(benchmark)
        await pipeline_instance.run(
            recorder=recorder,
            model=model,
            service_class=service_class,
            service_name=service,
            turn_indices=turn_indices,
            disable_vad=disable_vad,
            juice=juice,
            verbosity=verbosity,
        )
        # Save audio source metadata
        runtime_path = run_dir / "runtime.json"
        if runtime_path.exists():
            runtime = json.loads(runtime_path.read_text())
        else:
            runtime = {}
        runtime["audio_source"] = (
            f"real_audio/{real_audio_speaker}" if real_audio_speaker else "tts"
        )
        runtime_path.write_text(json.dumps(runtime, indent=2), encoding="utf-8")

        click.echo(f"Completed benchmark run")
        click.echo(f"  Transcript: {run_dir / 'transcript.jsonl'}")
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        raise click.ClickException(str(e))
    finally:
        recorder.close()


async def _run_rehydrated(
    benchmark_name: str,
    model: str,
    service: Optional[str],
    pipeline_type: Optional[str],
    only_turns: Optional[str],
    verbose: bool,
    max_parallel: int = 1,
    disable_vad: bool = False,
    juice: Optional[int] = None,
    verbosity: Optional[int] = None,
    real_audio_speaker: Optional[str] = None,
):
    """Run benchmark in single-step rehydration mode.

    Each turn is evaluated independently: the model receives golden conversation
    history for all prior turns, then live audio/text for the target turn only.
    A fresh API session is created per turn to ensure complete isolation.

    When max_parallel > 1, turns run concurrently up to the given limit.
    """
    from audio_arena.recording.transcript_recorder import TranscriptRecorder

    BenchmarkConfig = load_benchmark(benchmark_name)
    benchmark = BenchmarkConfig()

    if real_audio_speaker:
        _setup_real_audio(benchmark, real_audio_speaker)

    all_turns = benchmark.turns

    if not pipeline_type:
        pipeline_type = infer_pipeline(model)
        click.echo(f"Auto-detected pipeline: {pipeline_type}")

    pipeline_cls = get_pipeline_class(pipeline_type)

    requires_service = getattr(pipeline_cls, "requires_service", True)
    if requires_service and not service:
        raise click.UsageError(f"--service is required for {pipeline_type} pipeline")

    for msg in get_disable_vad_status_messages(
        disable_vad=disable_vad,
        rehydrate=True,
        pipeline_type=pipeline_type,
        service=service,
        model=model,
    ):
        click.echo(msg)

    service_class = load_service_class(service) if service else None

    run_dir = create_run_directory(benchmark_name, model)
    click.echo(f"Output directory: {run_dir}")
    mode_desc = f"single-step rehydration ({len(all_turns)} total turns"
    if max_parallel > 1:
        mode_desc += f", {max_parallel} concurrent"
    mode_desc += ")"
    click.echo(f"Mode: {mode_desc}")

    setup_logging(run_dir, verbose)

    target_indices = list(range(len(all_turns)))
    if only_turns:
        target_indices = [int(i.strip()) for i in only_turns.split(",")]
        click.echo(f"Evaluating turns: {target_indices}")

    results: dict[int, dict] = {}
    turn_dir_width = max(3, len(str(len(all_turns) - 1)))

    async def _run_single_turn(
        semaphore: asyncio.Semaphore,
        target_idx: int,
    ):
        async with semaphore:
            golden_history = all_turns[:target_idx] if target_idx > 0 else None
            target_turn = all_turns[target_idx]
            turn_run_dir = build_rehydrated_turn_run_dir(
                run_dir, target_idx, turn_dir_width
            )
            turn_run_dir.mkdir(parents=True, exist_ok=True)
            click.echo(
                f"[Rehydration] Turn {target_idx}/{len(all_turns) - 1}"
                + (
                    f" (rehydrating {len(golden_history)} golden turns)"
                    if golden_history
                    else ""
                )
                + f" -> {turn_run_dir.relative_to(run_dir)}"
            )

            turn_sink_id = add_turn_logging_sink(turn_run_dir)
            use_oracle_split = turn_uses_oracle_post_tool_continuation(
                pipeline_type=pipeline_type,
                turn=target_turn,
            )

            try:
                with logger.contextualize(
                    rehydration_turn_dir=str(turn_run_dir),
                    rehydration_target_turn=target_idx,
                ):
                    if use_oracle_split:
                        capture_run_dir = turn_run_dir / "tool_capture"
                        capture_benchmark = BenchmarkConfig()
                        if real_audio_speaker:
                            _setup_real_audio(capture_benchmark, real_audio_speaker)
                        capture_recorder = TranscriptRecorder(capture_run_dir, model)
                        capture_pipeline = pipeline_cls(capture_benchmark)

                        try:
                            await capture_pipeline.run(
                                recorder=capture_recorder,
                                model=model,
                                service_class=service_class,
                                service_name=service,
                                turn_indices=[target_idx],
                                rehydration_turns=golden_history,
                                disable_vad=disable_vad,
                                juice=juice,
                                verbosity=verbosity,
                                stop_after_first_tool_call=isinstance(
                                    target_turn.get("required_function_call"),
                                    dict,
                                ),
                            )
                        finally:
                            capture_recorder.close()

                        if isinstance(target_turn.get("required_function_call"), dict):
                            live_tool_capture = (
                                capture_pipeline.get_captured_tool_phase()
                                or build_missing_tool_capture(target_turn)
                            )
                        else:
                            live_tool_capture = build_live_tool_capture_from_transcript(
                                turn=target_turn,
                                transcript_path=capture_run_dir / "transcript.jsonl",
                            )

                        continuation_benchmark = BenchmarkConfig()
                        if real_audio_speaker:
                            _setup_real_audio(
                                continuation_benchmark, real_audio_speaker
                            )
                        continuation_recorder = TranscriptRecorder(turn_run_dir, model)
                        continuation_pipeline = pipeline_cls(continuation_benchmark)
                        try:
                            await continuation_pipeline.run(
                                recorder=continuation_recorder,
                                model=model,
                                service_class=service_class,
                                service_name=service,
                                turn_indices=[target_idx],
                                rehydration_turns=build_oracle_continuation_history(
                                    golden_history,
                                    target_turn,
                                ),
                                disable_vad=disable_vad,
                                juice=juice,
                                verbosity=verbosity,
                                oracle_continuation_only=True,
                            )
                        finally:
                            continuation_recorder.close()

                        merge_oracle_continuation_artifacts(
                            transcript_path=turn_run_dir / "transcript.jsonl",
                            live_tool_capture=live_tool_capture,
                        )
                    else:
                        turn_benchmark = BenchmarkConfig()
                        if real_audio_speaker:
                            _setup_real_audio(turn_benchmark, real_audio_speaker)
                        recorder = TranscriptRecorder(turn_run_dir, model)
                        pipeline_instance = pipeline_cls(turn_benchmark)
                        try:
                            await pipeline_instance.run(
                                recorder=recorder,
                                model=model,
                                service_class=service_class,
                                service_name=service,
                                turn_indices=[target_idx],
                                rehydration_turns=golden_history,
                                disable_vad=disable_vad,
                                juice=juice,
                                verbosity=verbosity,
                            )
                        finally:
                            recorder.close()
                results[target_idx] = {
                    "success": True,
                    "turn_run_dir": str(turn_run_dir),
                    "error": None,
                }
                click.echo(f"[Rehydration] Turn {target_idx} OK")
            except Exception as e:
                with logger.contextualize(
                    rehydration_turn_dir=str(turn_run_dir),
                    rehydration_target_turn=target_idx,
                ):
                    logger.exception(f"Turn {target_idx} failed: {e}")
                click.echo(f"  Turn {target_idx} FAILED: {e}")
                results[target_idx] = {
                    "success": False,
                    "turn_run_dir": str(turn_run_dir),
                    "error": str(e),
                }
            finally:
                recorder.close()
                try:
                    logger.remove(turn_sink_id)
                except ValueError:
                    pass

    semaphore = asyncio.Semaphore(max_parallel)
    tasks = [_run_single_turn(semaphore, idx) for idx in target_indices]
    await asyncio.gather(*tasks)

    runtime = finalize_rehydrated_run_artifacts(
        run_dir=run_dir,
        model=model,
        target_indices=target_indices,
        turn_results=results,
        parallel=max_parallel,
        disable_vad=disable_vad,
        real_audio_speaker=real_audio_speaker,
    )
    succeeded = runtime["turns"]
    failed_turns = runtime["failed_turns"]

    click.echo(
        f"\nCompleted rehydrated run: {succeeded}/{len(target_indices)} turns succeeded"
    )
    if failed_turns:
        click.echo(f"  Failed turns: {failed_turns}")
    click.echo(f"  Transcript: {run_dir / 'transcript.jsonl'}")


@cli.command()
@click.argument("run_dir", type=click.Path(exists=True))
@click.option(
    "--only-turns", help="Comma-separated turn indices to judge (e.g., 0,1,2)"
)
@click.option(
    "--judge",
    "judge_backend",
    type=click.Choice(["claude", "openai"], case_sensitive=False),
    default=None,
    help="Judge backend to use. Defaults to 'claude' for all benchmarks.",
)
@click.option(
    "--judge-model",
    default=None,
    help="Model for judging (default: claude-opus-4-5 for Claude, gpt-5.2 for OpenAI)",
)
@click.option(
    "--skip-turn-taking",
    is_flag=True,
    help="Skip audio turn-taking analysis (faster; all turns count as turn_taking=True)",
)
@click.option("--debug", is_flag=True, help="Enable debug logging")
def judge(
    run_dir: str,
    only_turns: Optional[str],
    judge_backend: Optional[str],
    judge_model: Optional[str],
    skip_turn_taking: bool,
    debug: bool,
):
    """Judge a completed benchmark run.

    Examples:
        uv run audio-arena judge runs/grocery_bench/20251213T123456_gpt-4o
        uv run audio-arena judge runs/conversation_bench/... --judge claude
        uv run audio-arena judge runs/appointment_bench/... --judge openai --judge-model gpt-4.1
    """
    run_path = Path(run_dir)

    # Infer benchmark from path: runs/{benchmark}/{timestamp}_{model}/
    benchmark_name = run_path.parent.name

    if judge_backend is None:
        judge_backend = "claude"
    click.echo(f"Using {judge_backend} judge for {benchmark_name}")

    # Load transcript
    transcript_path = run_path / "transcript.jsonl"
    if not transcript_path.exists():
        raise click.UsageError(f"No transcript found at {transcript_path}")

    # Parse turn indices
    turn_indices_set: Optional[set[int]] = None
    if only_turns:
        turn_indices_set = {int(i.strip()) for i in only_turns.split(",")}

    # Load benchmark for expected turns and get_relevant_dimensions
    get_relevant_dimensions_fn = None
    kb_text = None
    prompt_visible_kb_text = None
    try:
        BenchmarkConfig = load_benchmark(benchmark_name)
        benchmark = BenchmarkConfig()
        expected_turns = benchmark.turns
        benchmark_turns_module = importlib.import_module(
            f"benchmarks.{benchmark_name}.turns"
        )
        get_relevant_dimensions_fn = getattr(
            benchmark_turns_module, "get_relevant_dimensions", None
        )
        kb_text = load_benchmark_kb_text(benchmark_name)
        prompt_visible_kb_text = load_prompt_visible_kb_text(benchmark_name)
    except Exception:
        click.echo(
            f"Could not load benchmark '{benchmark_name}', using shared turns module"
        )
        from benchmarks.conversation_bench.turns import turns as expected_turns

    # Load shared utilities
    from audio_arena.judging.llm_judge import load_transcript, write_outputs

    records = load_transcript(run_path)
    if turn_indices_set is not None:
        records = [r for r in records if r["turn"] in turn_indices_set]

    if judge_backend == "openai":
        from audio_arena.judging.openai_judge import (
            judge_with_openai,
            OPENAI_JUDGE_MODEL,
        )

        effective_model = judge_model or OPENAI_JUDGE_MODEL
        try:
            result = asyncio.run(
                judge_with_openai(
                    run_path,
                    only_turns=turn_indices_set,
                    debug=debug,
                    expected_turns=expected_turns,
                    skip_turn_taking=skip_turn_taking,
                    get_relevant_dimensions_fn=get_relevant_dimensions_fn,
                    model=judge_model,
                    kb_text=kb_text,
                    prompt_visible_kb_text=prompt_visible_kb_text,
                )
            )
        except Exception as e:
            raise click.ClickException(f"Judgment failed: {e}")

        write_outputs(
            run_path,
            records,
            result["judgments"],
            result["summary"],
            result["model_name"],
            result.get("realignment_notes", ""),
            result.get("function_tracking", {}),
            result.get("turn_taking_analysis"),
            expected_turns=expected_turns,
            judge_name="openai",
            judge_version=result.get("judge_version"),
            judge_model=result.get("judge_model", effective_model),
            realignment_applied=result.get("cross_turn_realignment_applied"),
            turn_taking_supported=result.get("turn_taking_supported"),
            turn_taking_skip_reason=result.get("turn_taking_skip_reason"),
        )
        summary_file = "openai_summary.json"

    else:
        from audio_arena.judging.llm_judge import judge_with_claude

        try:
            result = asyncio.run(
                judge_with_claude(
                    run_path,
                    only_turns=turn_indices_set,
                    debug=debug,
                    expected_turns=expected_turns,
                    skip_turn_taking=skip_turn_taking,
                    get_relevant_dimensions_fn=get_relevant_dimensions_fn,
                    kb_text=kb_text,
                    prompt_visible_kb_text=prompt_visible_kb_text,
                )
            )
        except Exception as e:
            raise click.ClickException(f"Judgment failed: {e}")

        write_outputs(
            run_path,
            records,
            result["judgments"],
            result["summary"],
            result["model_name"],
            result.get("realignment_notes", ""),
            result.get("function_tracking", {}),
            result.get("turn_taking_analysis"),
            expected_turns=expected_turns,
            judge_name="claude",
            judge_version=result.get("judge_version"),
            realignment_applied=result.get("cross_turn_realignment_applied"),
            turn_taking_supported=result.get("turn_taking_supported"),
            turn_taking_skip_reason=result.get("turn_taking_skip_reason"),
        )
        summary_file = "claude_summary.json"

    # Print summary
    summary_path = run_path / summary_file
    summary = json.loads(summary_path.read_text())
    passes = summary.get("passes", summary.get("claude_passes", {}))
    total = summary.get("turns_scored", 0)

    turn_taking_supported = summary.get("turn_taking_supported", True)
    turn_taking_skip_reason = summary.get("turn_taking_skip_reason")
    if turn_taking_supported:
        click.echo(f"\nJudged {total} turns (with turn-taking analysis)")
        click.echo(f"  Turn-taking: {passes.get('turn_taking', total)}/{total}")
    else:
        suffix = f": {turn_taking_skip_reason}" if turn_taking_skip_reason else ""
        click.echo(f"\nJudged {total} turns (without turn-taking analysis{suffix})")
    click.echo(f"  Tool use: {passes.get('tool_use_correct', 0)}/{total}")
    click.echo(
        f"  Instruction following: {passes.get('instruction_following', 0)}/{total}"
    )
    click.echo(f"  KB grounding: {passes.get('kb_grounding', 0)}/{total}")

    category_totals = summary.get("category_totals", {})
    amb_total = category_totals.get("ambiguity_handling", 0)
    state_total = category_totals.get("state_tracking", 0)
    if amb_total:
        click.echo(
            f"  Ambiguity handling: {passes.get('ambiguity_handling', 0)}/{amb_total}"
        )
    if state_total:
        click.echo(f"  State tracking: {passes.get('state_tracking', 0)}/{state_total}")

    turn_taking_failures = summary.get("turn_taking_failures", [])
    if turn_taking_failures:
        click.echo(f"\nTurn-taking failures: {turn_taking_failures}")


@cli.command("list-benchmarks")
def list_benchmarks():
    """List available benchmarks."""
    benchmarks = list_available_benchmarks()
    if not benchmarks:
        click.echo("No benchmarks found in benchmarks/ directory")
        return

    click.echo("Available benchmarks:")
    for name in benchmarks:
        try:
            BenchmarkConfig = load_benchmark(name)
            description = getattr(BenchmarkConfig, "description", "")
            click.echo(f"  {name}: {description}")
        except Exception:
            click.echo(f"  {name}")


@cli.command("list-speakers")
@click.argument("benchmark_name")
def list_speakers_cmd(benchmark_name: str):
    """List available real audio speakers for a benchmark."""
    BenchmarkConfig = load_benchmark(benchmark_name)
    benchmark = BenchmarkConfig()
    speakers = benchmark.list_speakers()
    if not speakers:
        click.echo(f"No real audio speakers found for {benchmark_name}.")
        click.echo(
            f"  Expected location: benchmarks/{benchmark_name}/real_audio/<speaker>/"
        )
        return
    click.echo(f"Available speakers for {benchmark_name}:")
    total_turns = len(benchmark.turns)
    for name in speakers:
        speaker_dir = benchmark.real_audio_dir / name
        wav_count = len(list(speaker_dir.glob("*.wav")))
        status = (
            "complete"
            if wav_count >= total_turns
            else f"{wav_count}/{total_turns} turns"
        )
        click.echo(f"  {name}: {status}")


@cli.command("list-pipelines")
def list_pipelines():
    """List available pipelines."""
    click.echo("Available pipelines:")
    for name, cls_path in PIPELINE_CLASSES.items():
        click.echo(f"  {name}: {cls_path}")


@cli.command("list-aliases")
def list_aliases():
    """List service aliases."""
    click.echo("Service aliases:")
    for alias, cls_path in SERVICE_ALIASES.items():
        click.echo(f"  {alias}: {cls_path}")


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
