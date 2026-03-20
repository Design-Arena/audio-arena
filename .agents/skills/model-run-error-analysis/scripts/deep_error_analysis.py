import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ERROR_MODE_DESCRIPTIONS = {
    "premature_end_session": (
        "The model terminates the session instead of answering or completing the workflow. "
        "This usually manifests as instruction-following failures and can also truncate multi-step tool tasks."
    ),
    "empty_response": (
        "The model returns no usable response after retries. This is an execution failure rather than a content error."
    ),
    "partial_multi_tool_execution": (
        "The model starts a batch, fallback, or multi-step tool workflow but stops before all required actions are complete."
    ),
    "state_memory_failure": (
        "The model loses track of previously established conversation state, such as the known user name, prior actions, or previously given facts."
    ),
    "ambiguity_handling_failure": (
        "The model mishandles ambiguity by over-clarifying when the intent is clear or by guessing when it should first disambiguate."
    ),
    "knowledge_grounding_error": (
        "The model gives schedule, logistics, pricing, or speaker facts that do not match the benchmark source of truth."
    ),
    "tool_or_action_selection_error": (
        "The model chooses the wrong tool behavior, misses a required tool call, or performs an extra action not requested by the user."
    ),
    "other": (
        "The failure does not fit the main candidate buckets cleanly and needs direct inspection of the datapoints to determine the true cause."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a deep error analysis markdown report from judged runs."
    )
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=None,
        help="Comparison output directory containing run_results.csv.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=[],
        help="Judged run directory containing openai_judged.jsonl. Repeat for multiple runs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit output markdown path.",
    )
    return parser.parse_args()


def load_run_dirs(args: argparse.Namespace) -> list[Path]:
    if args.run_dir:
        return [path.expanduser().resolve() for path in args.run_dir]

    if args.comparison_dir is None:
        raise ValueError("Provide either --comparison-dir or at least one --run-dir.")

    comparison_dir = args.comparison_dir.expanduser().resolve()
    run_results_path = comparison_dir / "run_results.csv"
    if not run_results_path.exists():
        raise FileNotFoundError(f"Missing {run_results_path}")
    run_results_df = pd.read_csv(run_results_path)
    return [Path(path).expanduser().resolve() for path in run_results_df["run_dir"].tolist()]


def load_rows(run_dir: Path) -> list[dict]:
    judged_path = run_dir / "openai_judged.jsonl"
    if not judged_path.exists():
        raise FileNotFoundError(f"Missing {judged_path}")
    return [
        json.loads(line)
        for line in judged_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def serialize_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def truncate_text(value: object, limit: int = 320) -> str:
    text = str(value).replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def classify_error_mode(row: dict) -> str:
    assistant_text = str(row.get("assistant_text", ""))
    judge_reasoning = str(row.get("judge_reasoning", "")).lower()
    user_text = str(row.get("user_text", "")).lower()

    if assistant_text.startswith("[MODEL_ENDED_SESSION]") or "ended the session" in judge_reasoning:
        return "premature_end_session"
    if assistant_text.startswith("[EMPTY_RESPONSE"):
        return "empty_response"
    if (
        "multi-step tool sequence was expected" in judge_reasoning
        or "multi-step tool use was expected" in judge_reasoning
        or "only registered one" in judge_reasoning
        or "only added" in judge_reasoning
        or "only canceled" in judge_reasoning
        or "fallback" in judge_reasoning
        or "batch request" in judge_reasoning
    ):
        return "partial_multi_tool_execution"
    if (
        "asked for the name again" in judge_reasoning
        or "already provided earlier" in judge_reasoning
        or "should remember" in judge_reasoning
        or "state-caused missed tool call" in judge_reasoning
        or "should use conversation state" in judge_reasoning
        or "what dietary preference i originally registered" in user_text
    ):
        return "state_memory_failure"
    if "ambiguous" in judge_reasoning or "not genuinely ambiguous" in judge_reasoning:
        return "ambiguity_handling_failure"
    if (
        "factual error" in judge_reasoning
        or "hallucinated" in judge_reasoning
        or "unsupported" in judge_reasoning
        or "wrong number" in judge_reasoning
        or "incorrect lineup" in judge_reasoning
        or "schedule factual error" in judge_reasoning
        or "capability error" in judge_reasoning
        or "does not match the provided program" in judge_reasoning
        or "mismatch" in judge_reasoning
    ):
        return "knowledge_grounding_error"
    if "function call was expected" in judge_reasoning or "did not call the tool" in judge_reasoning:
        return "tool_or_action_selection_error"
    return "other"


def failed_dimensions(row: dict) -> list[str]:
    return [dimension for dimension, value in row.get("scores", {}).items() if value is False]


def exact_top_lines(run_rows: list[dict]) -> dict:
    failed_rows = 0
    ended_session = 0
    empty_response = 0
    dimension_counter = Counter()

    for row in run_rows:
        failed = failed_dimensions(row)
        if failed:
            failed_rows += 1
            for dimension in failed:
                dimension_counter[dimension] += 1
        assistant_text = str(row.get("assistant_text", ""))
        if assistant_text.startswith("[MODEL_ENDED_SESSION]"):
            ended_session += 1
        if assistant_text.startswith("[EMPTY_RESPONSE"):
            empty_response += 1

    return {
        "failed_rows": failed_rows,
        "ended_session": ended_session,
        "empty_response": empty_response,
        "dimension_counter": dimension_counter,
    }


def representative_examples(run_rows: list[dict], limit_per_mode: int = 5) -> dict[str, list[dict]]:
    examples: dict[str, list[dict]] = defaultdict(list)
    for row in run_rows:
        failed = failed_dimensions(row)
        if not failed:
            continue
        error_mode = classify_error_mode(row)
        if len(examples[error_mode]) >= limit_per_mode:
            continue
        examples[error_mode].append(
            {
                "turn": row["turn"],
                "user_text": row["user_text"],
                "assistant_text": row["assistant_text"],
                "judge_reasoning": row.get("judge_reasoning", ""),
                "tool_calls": row.get("tool_calls", []),
                "tool_results": row.get("tool_results", []),
                "failed_dimensions": failed,
            }
        )
    return dict(examples)


def build_failure_review_rows(run_payloads: dict[str, list[dict]]) -> pd.DataFrame:
    review_rows: list[dict] = []
    for model_name, run_rows in run_payloads.items():
        for row in run_rows:
            failed = failed_dimensions(row)
            if not failed:
                continue
            review_rows.append(
                {
                    "model_name": model_name,
                    "turn": row["turn"],
                    "failed_dimensions": ",".join(failed),
                    "error_mode_candidate": classify_error_mode(row),
                    "user_text": row.get("user_text", ""),
                    "assistant_text": row.get("assistant_text", ""),
                    "judge_reasoning": row.get("judge_reasoning", ""),
                    "scores_json": serialize_json(row.get("scores", {})),
                    "tool_calls_json": serialize_json(row.get("tool_calls", [])),
                    "tool_results_json": serialize_json(row.get("tool_results", [])),
                }
            )
    review_df = pd.DataFrame(review_rows)
    if review_df.empty:
        return review_df
    return review_df.sort_values(
        ["model_name", "turn"],
        ascending=[True, True],
    ).reset_index(drop=True)


def model_name_from_rows(run_rows: list[dict], run_dir: Path) -> str:
    if run_rows:
        return str(run_rows[0].get("model_name") or run_dir.name)
    return run_dir.name


def build_run_payloads(run_dirs: list[Path]) -> dict[str, list[dict]]:
    base_names: list[str] = []
    loaded_rows: list[list[dict]] = []

    for run_dir in run_dirs:
        run_rows = load_rows(run_dir)
        loaded_rows.append(run_rows)
        base_names.append(model_name_from_rows(run_rows, run_dir))

    name_counts = Counter(base_names)
    seen_counts: Counter[str] = Counter()
    run_payloads: dict[str, list[dict]] = {}

    for run_dir, run_rows, base_name in zip(run_dirs, loaded_rows, base_names, strict=True):
        if name_counts[base_name] == 1:
            run_label = base_name
        else:
            seen_counts[base_name] += 1
            run_label = f"{base_name} run {seen_counts[base_name]} ({run_dir.name})"
        run_payloads[run_label] = run_rows

    return run_payloads


def shared_turn_analysis(run_payloads: dict[str, list[dict]]) -> tuple[list[int], dict[str, dict[str, list[int]]]]:
    failed_turns_by_model: dict[str, set[int]] = {}
    for model_name, run_rows in run_payloads.items():
        failed_turns_by_model[model_name] = {
            row["turn"] for row in run_rows if failed_dimensions(row)
        }

    all_turns = set()
    for turns in failed_turns_by_model.values():
        all_turns |= turns

    failed_by_all = sorted(
        turn
        for turn in all_turns
        if all(turn in failed_turns_by_model[model_name] for model_name in failed_turns_by_model)
    )

    unique_deltas: dict[str, dict[str, list[int]]] = {}
    for model_name, failed_turns in failed_turns_by_model.items():
        other_failed_turns = set().union(
            *(turns for other_model, turns in failed_turns_by_model.items() if other_model != model_name)
        )
        unique_fail = sorted(failed_turns - other_failed_turns)
        unique_pass = sorted((set(range(75)) - failed_turns) & other_failed_turns)
        unique_deltas[model_name] = {
            "unique_fail": unique_fail,
            "unique_pass": unique_pass,
        }
    return failed_by_all, unique_deltas


def write_report(
    *,
    output_path: Path,
    run_dirs: list[Path],
    run_payloads: dict[str, list[dict]],
) -> str:
    model_order = list(run_payloads.keys())
    top_lines = {
        model_name: exact_top_lines(run_payloads[model_name]) for model_name in model_order
    }
    examples = {
        model_name: representative_examples(run_payloads[model_name]) for model_name in model_order
    }
    failed_by_all, unique_deltas = shared_turn_analysis(run_payloads)

    lines: list[str] = []
    lines.append("# Deep Error Analysis")
    lines.append("")
    lines.append("## Run Set")
    for model_name, run_dir in zip(model_order, run_dirs):
        lines.append(f"- `{model_name}`: `{run_dir}`")
    lines.append("")
    lines.append("## Top-Line Comparison")
    lines.append("")
    lines.append("| Model | Failed rows | `MODEL_ENDED_SESSION` | `EMPTY_RESPONSE` | Top failed dimensions |")
    lines.append("| --- | ---: | ---: | ---: | --- |")
    for model_name in model_order:
        stats = top_lines[model_name]
        top_dims = ", ".join(
            f"`{dimension}` {count}"
            for dimension, count in stats["dimension_counter"].most_common(3)
        )
        lines.append(
            f"| `{model_name}` | {stats['failed_rows']} | {stats['ended_session']} | "
            f"{stats['empty_response']} | {top_dims} |"
        )
    lines.append("")
    lines.append("## Shared Failures")
    lines.append("")
    lines.append(f"- Failed by all models: `{', '.join(str(turn) for turn in failed_by_all)}`")
    lines.append("")

    for model_name in model_order:
        stats = top_lines[model_name]
        model_examples = examples[model_name]
        lines.append(f"## `{model_name}`")
        lines.append("")
        lines.append(
            f"- Failed rows: `{stats['failed_rows']}`"
        )
        lines.append(
            f"- `MODEL_ENDED_SESSION`: `{stats['ended_session']}`"
        )
        lines.append(
            f"- `EMPTY_RESPONSE`: `{stats['empty_response']}`"
        )
        lines.append("")
        lines.append("Top failed dimensions:")
        for dimension, count in stats["dimension_counter"].most_common():
            percentage = (count / stats["failed_rows"] * 100) if stats["failed_rows"] else 0.0
            lines.append(f"- `{dimension}`: `{count}` ({percentage:.1f}% of failed rows)")
        lines.append("")

        mode_counts = Counter()
        for row in run_payloads[model_name]:
            if failed_dimensions(row):
                mode_counts[classify_error_mode(row)] += 1
        lines.append("Main error modes:")
        for error_mode, count in mode_counts.most_common(5):
            lines.append(f"- `{error_mode}`: `{count}`")
        lines.append("")

        for error_mode, count in mode_counts.most_common(4):
            lines.append(f"### `{error_mode}`")
            lines.append("")
            lines.append(ERROR_MODE_DESCRIPTIONS.get(error_mode, ERROR_MODE_DESCRIPTIONS["other"]))
            lines.append("")
            for example in model_examples.get(error_mode, []):
                lines.append(
                    f"- Turn `{example['turn']}` "
                    f"(`{', '.join(example['failed_dimensions'])}`)"
                )
                lines.append(f"  - User: {truncate_text(example['user_text'])}")
                lines.append(f"  - Assistant: {truncate_text(example['assistant_text'])}")
                lines.append(
                    f"  - Judge: {truncate_text(example['judge_reasoning'])}"
                )
                if example["tool_calls"]:
                    lines.append(
                        f"  - Tool calls: {truncate_text(serialize_json(example['tool_calls']), 260)}"
                    )
                if example["tool_results"]:
                    lines.append(
                        f"  - Tool results: {truncate_text(serialize_json(example['tool_results']), 260)}"
                    )
            lines.append("")

        lines.append("Unique deltas:")
        lines.append(
            f"- Unique fail turns: `{', '.join(str(turn) for turn in unique_deltas[model_name]['unique_fail']) or 'none'}`"
        )
        lines.append(
            f"- Unique pass turns: `{', '.join(str(turn) for turn in unique_deltas[model_name]['unique_pass']) or 'none'}`"
        )
        lines.append("")

    report_text = "\n".join(lines).rstrip() + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")
    failure_review_df = build_failure_review_rows(run_payloads)
    if not failure_review_df.empty:
        failure_review_df.to_csv(
            output_path.parent / "failure_review_rows.csv",
            index=False,
        )
        failure_review_jsonl_path = output_path.parent / "failure_review_rows.jsonl"
        with failure_review_jsonl_path.open("w", encoding="utf-8") as handle:
            for record in failure_review_df.to_dict(orient="records"):
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    return report_text


def main() -> None:
    args = parse_args()
    run_dirs = load_run_dirs(args)
    run_payloads = build_run_payloads(run_dirs)

    if args.output is not None:
        output_path = args.output.expanduser().resolve()
    elif args.comparison_dir is not None:
        output_path = args.comparison_dir.expanduser().resolve() / "deep_error_analysis.md"
    else:
        benchmark_name = run_dirs[0].parent.name if run_dirs else "benchmark"
        output_path = (
            Path.cwd()
            / "results"
            / benchmark_name
            / "deep_error_analysis.md"
        )

    report_text = write_report(
        output_path=output_path,
        run_dirs=run_dirs,
        run_payloads=run_payloads,
    )
    summary_lines: list[str] = []
    for model_name, run_rows in run_payloads.items():
        stats = exact_top_lines(run_rows)
        top_dimensions = ", ".join(
            f"{dimension}={count}"
            for dimension, count in stats["dimension_counter"].most_common(3)
        )
        summary_lines.append(
            f"{model_name}: failed_rows={stats['failed_rows']}, "
            f"ended_session={stats['ended_session']}, "
            f"empty_response={stats['empty_response']}, "
            f"top_failed_dimensions={top_dimensions}"
        )

    print(f"Saved markdown report: {output_path}")
    failure_csv_path = output_path.parent / "failure_review_rows.csv"
    if failure_csv_path.exists():
        print(f"Saved review packet CSV: {failure_csv_path}")
    failure_jsonl_path = output_path.parent / "failure_review_rows.jsonl"
    if failure_jsonl_path.exists():
        print(f"Saved review packet JSONL: {failure_jsonl_path}")
    print("Summary:")
    for line in summary_lines:
        print(f"- {line}")


if __name__ == "__main__":
    main()
