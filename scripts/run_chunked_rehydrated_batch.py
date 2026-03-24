import argparse
import json
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from audio_arena.cli import finalize_rehydrated_run_artifacts, load_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a rehydrated benchmark in smaller only-turns chunks, then merge the "
            "per-turn artifacts into one synthetic parent run and optionally judge it."
        )
    )
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--service", required=True)
    parser.add_argument("--chunk-size", type=int, default=12)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--run-label", default="chunked")
    parser.add_argument("--judge", default="openai")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--disable-vad", action="store_true", default=False)
    parser.add_argument("--skip-turn-taking", action="store_true", default=False)
    parser.add_argument("--no-judge", action="store_true", default=False)
    return parser.parse_args()


def extract_run_dir(stdout_text: str, stderr_text: str) -> Path | None:
    for line in (stdout_text + "\n" + stderr_text).splitlines():
        if line.startswith("Output directory: "):
            return (Path.cwd() / line.split(": ", 1)[1].strip()).resolve()
    return None


def chunk_turns(turns: list[int], chunk_size: int) -> list[list[int]]:
    return [turns[index : index + chunk_size] for index in range(0, len(turns), chunk_size)]


def transcript_completed(turn_run_dir: Path) -> bool:
    transcript_path = turn_run_dir / "transcript.jsonl"
    return transcript_path.exists() and transcript_path.stat().st_size > 0


def run_chunk(
    *,
    benchmark: str,
    model: str,
    service: str,
    turn_indices: list[int],
    parallel: int,
    disable_vad: bool,
    log_path: Path,
) -> tuple[subprocess.CompletedProcess[str], Path | None]:
    command = [
        "uv",
        "run",
        "audio-arena",
        "run",
        benchmark,
        "--model",
        model,
        "--service",
        service,
        "--rehydrate",
        "--parallel",
        str(parallel),
        "--only-turns",
        ",".join(str(turn_index) for turn_index in turn_indices),
    ]
    if disable_vad:
        command.append("--disable-vad")

    result = subprocess.run(
        command,
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
    )
    log_path.write_text(
        "COMMAND:\n"
        + " ".join(command)
        + "\n\nSTDOUT:\n"
        + result.stdout
        + "\n\nSTDERR:\n"
        + result.stderr,
        encoding="utf-8",
    )
    return result, extract_run_dir(result.stdout, result.stderr)


def main() -> None:
    args = parse_args()

    benchmark_config = load_benchmark(args.benchmark)
    total_turns = len(benchmark_config.turns)
    pending_turns = list(range(total_turns))

    started_at = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    synthetic_run_dir = (
        Path.cwd()
        / "runs"
        / args.benchmark
        / f"{started_at}_{args.model}_{args.run_label}_{uuid.uuid4().hex[:8]}"
    ).resolve()
    synthetic_run_dir.mkdir(parents=True, exist_ok=True)

    completed_turn_dirs: dict[int, Path] = {}
    chunk_history: list[dict] = []
    chunk_index = 0

    while pending_turns:
        current_chunk = pending_turns[: args.chunk_size]
        chunk_log_path = synthetic_run_dir / f"chunk_{chunk_index:03d}.log"
        result, chunk_run_dir = run_chunk(
            benchmark=args.benchmark,
            model=args.model,
            service=args.service,
            turn_indices=current_chunk,
            parallel=args.parallel,
            disable_vad=args.disable_vad,
            log_path=chunk_log_path,
        )

        chunk_record = {
            "chunk_index": chunk_index,
            "turns_requested": current_chunk,
            "returncode": result.returncode,
            "chunk_run_dir": str(chunk_run_dir) if chunk_run_dir else None,
            "log_path": str(chunk_log_path),
        }

        if chunk_run_dir is None:
            chunk_history.append(chunk_record)
            raise RuntimeError(
                f"Failed to parse chunk run directory for chunk {chunk_index}. "
                f"See {chunk_log_path}"
            )

        completed_in_chunk: list[int] = []
        missing_in_chunk: list[int] = []
        for turn_index in current_chunk:
            turn_run_dir = chunk_run_dir / "turn_runs" / f"turn_{turn_index:03d}"
            if transcript_completed(turn_run_dir):
                completed_turn_dirs[turn_index] = turn_run_dir
                completed_in_chunk.append(turn_index)
            else:
                missing_in_chunk.append(turn_index)

        chunk_record["completed_turns"] = completed_in_chunk
        chunk_record["missing_turns"] = missing_in_chunk
        chunk_history.append(chunk_record)

        if not completed_in_chunk:
            (synthetic_run_dir / "chunk_history.json").write_text(
                json.dumps(chunk_history, indent=2),
                encoding="utf-8",
            )
            raise RuntimeError(
                f"Chunk {chunk_index} made no progress for turns {current_chunk}. "
                f"See {chunk_log_path}"
            )

        pending_turns = [turn for turn in pending_turns if turn not in completed_turn_dirs]
        (synthetic_run_dir / "chunk_history.json").write_text(
            json.dumps(chunk_history, indent=2),
            encoding="utf-8",
        )

        print(
            f"[chunked] model={args.model} chunk={chunk_index} "
            f"completed={len(completed_in_chunk)}/{len(current_chunk)} "
            f"total_completed={len(completed_turn_dirs)}/{total_turns}",
            flush=True,
        )
        chunk_index += 1

    turn_results = {
        turn_index: {
            "success": True,
            "turn_run_dir": str(turn_run_dir),
            "error": None,
        }
        for turn_index, turn_run_dir in completed_turn_dirs.items()
    }
    runtime = finalize_rehydrated_run_artifacts(
        run_dir=synthetic_run_dir,
        model=args.model,
        target_indices=list(range(total_turns)),
        turn_results=turn_results,
        parallel=args.parallel,
        disable_vad=args.disable_vad,
        real_audio_speaker=None,
    )
    (synthetic_run_dir / "chunk_metadata.json").write_text(
        json.dumps(
            {
                "benchmark": args.benchmark,
                "model": args.model,
                "service": args.service,
                "chunk_size": args.chunk_size,
                "parallel": args.parallel,
                "disable_vad": args.disable_vad,
                "run_label": args.run_label,
                "started_at": started_at,
                "runtime": runtime,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if not args.no_judge:
        judge_command = [
            "uv",
            "run",
            "audio-arena",
            "judge",
            str(synthetic_run_dir),
            "--judge",
            args.judge,
        ]
        if args.judge_model:
            judge_command.extend(["--judge-model", args.judge_model])
        if args.skip_turn_taking:
            judge_command.append("--skip-turn-taking")

        judge_result = subprocess.run(
            judge_command,
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
        )
        (synthetic_run_dir / "judge.log").write_text(
            "COMMAND:\n"
            + " ".join(judge_command)
            + "\n\nSTDOUT:\n"
            + judge_result.stdout
            + "\n\nSTDERR:\n"
            + judge_result.stderr,
            encoding="utf-8",
        )
        if judge_result.returncode != 0:
            raise RuntimeError(
                f"Judge failed for {synthetic_run_dir}. See {synthetic_run_dir / 'judge.log'}"
            )

    print(
        json.dumps(
            {
                "synthetic_run_dir": str(synthetic_run_dir),
                "turns_completed": len(completed_turn_dirs),
                "judge_ran": not args.no_judge,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
