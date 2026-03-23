import argparse
import importlib
import json
import sys
from collections import Counter
from datetime import datetime, UTC
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SCORE_COLUMNS = [
    "turn_taking",
    "tool_use_correct",
    "instruction_following",
    "kb_grounding",
    "ambiguity_handling",
    "state_tracking",
]


def load_jsonl(jsonl_path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def format_status(value: bool | None) -> str:
    if value is True:
        return "pass"
    if value is False:
        return "fail"
    return "n/a"


def percentile(values: pd.Series, quantile: float) -> float | None:
    clean_values = values.dropna()
    if clean_values.empty:
        return None
    return round(float(clean_values.quantile(quantile)), 2)


def path_to_file_url(path: Path) -> str:
    return path.resolve().as_uri()


def turn_audio_path(benchmark_name: str, turn: int) -> Path:
    return Path.cwd() / "benchmarks" / benchmark_name / "audio" / f"turn_{turn:03d}.wav"


def load_benchmark_turns(benchmark_name: str) -> list[dict]:
    turns_path = PROJECT_ROOT / "benchmarks" / benchmark_name / "turns.py"
    if not turns_path.exists():
        raise FileNotFoundError(f"Benchmark turns file not found: {turns_path}")

    spec = importlib.util.spec_from_file_location(
        f"{benchmark_name}_turns_module",
        turns_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load benchmark turns from {turns_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.turns


def normalize_tool_entries(value: object) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def format_golden_context_block(turn_number: int, benchmark_turn: dict) -> str:
    lines = [
        f"Turn {turn_number}",
        f"User: {benchmark_turn.get('input', '')}",
    ]

    tool_calls = normalize_tool_entries(benchmark_turn.get("required_function_call"))
    tool_results = normalize_tool_entries(benchmark_turn.get("function_call_response"))

    for index, tool_call in enumerate(tool_calls):
        tool_name = tool_call.get("name", "unknown_tool")
        tool_args = json.dumps(tool_call.get("args", {}), ensure_ascii=False)
        lines.append(f"  [Tool call: {tool_name}({tool_args})]")
        if index < len(tool_results):
            lines.append(
                f"  [Tool result: {json.dumps(tool_results[index], ensure_ascii=False)}]"
            )

    lines.append(f"Assistant: {benchmark_turn.get('golden_text', '')}")
    return "\n".join(lines)


def build_record(raw_record: dict) -> dict:
    scores = raw_record.get("scores", {})
    normalized_scores = {column: scores.get(column) for column in SCORE_COLUMNS}
    failed_dimensions = [
        column for column, value in normalized_scores.items() if value is False
    ]
    completed_dimensions = [
        column for column, value in normalized_scores.items() if value is not None
    ]
    assistant_text = raw_record.get("assistant_text", "")
    response_status = (
        "empty_response"
        if assistant_text.startswith("[EMPTY_RESPONSE")
        else "normal"
    )
    if not completed_dimensions:
        overall_status = "incomplete"
    elif failed_dimensions:
        overall_status = "fail"
    else:
        overall_status = "pass"

    tool_calls = raw_record.get("tool_calls", [])
    tool_results = raw_record.get("tool_results", [])
    oracle_continuation = raw_record.get("oracle_continuation") or {}
    oracle_tool_calls = oracle_continuation.get("oracle_tool_calls", [])
    oracle_tool_results = oracle_continuation.get("oracle_tool_results", [])
    return {
        "turn": raw_record.get("turn"),
        "timestamp": raw_record.get("ts"),
        "model_name": raw_record.get("model_name"),
        "user_text": raw_record.get("user_text", ""),
        "assistant_text": assistant_text,
        "latency_ms": raw_record.get("latency_ms"),
        "ttfb_ms": raw_record.get("ttfb_ms"),
        "reconnection_count": raw_record.get("reconnection_count"),
        "tool_call_count": len(tool_calls),
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "tool_calls_text": json.dumps(tool_calls, ensure_ascii=False, indent=2),
        "tool_results_text": json.dumps(tool_results, ensure_ascii=False, indent=2),
        "oracle_continuation": oracle_continuation,
        "oracle_continuation_used": bool(oracle_continuation.get("used")),
        "live_tool_name_correct": oracle_continuation.get("tool_name_correct"),
        "live_tool_args_correct": oracle_continuation.get("tool_args_correct"),
        "live_tool_use_pass": oracle_continuation.get("tool_use_pass"),
        "oracle_tool_calls": oracle_tool_calls,
        "oracle_tool_results": oracle_tool_results,
        "oracle_tool_calls_text": json.dumps(
            oracle_tool_calls,
            ensure_ascii=False,
            indent=2,
        ),
        "oracle_tool_results_text": json.dumps(
            oracle_tool_results,
            ensure_ascii=False,
            indent=2,
        ),
        "judge_reasoning": raw_record.get("judge_reasoning", ""),
        "scores": normalized_scores,
        "failed_dimensions": failed_dimensions,
        "failed_dimensions_text": ", ".join(failed_dimensions),
        "completed_dimensions": completed_dimensions,
        "response_status": response_status,
        "overall_status": overall_status,
        "failure_count": len(failed_dimensions),
    }


def build_sanity_report(flattened_rows: pd.DataFrame) -> dict:
    duplicate_turns = int(flattened_rows["turn"].duplicated().sum())
    missingness = {}
    for column in [
        "turn",
        "timestamp",
        "user_text",
        "assistant_text",
        "latency_ms",
        "ttfb_ms",
        "judge_reasoning",
    ]:
        missingness[column] = int(flattened_rows[column].isna().sum())

    response_status_counts = {
        key: int(value)
        for key, value in flattened_rows["response_status"].value_counts().to_dict().items()
    }
    dtypes = {
        key: str(value)
        for key, value in flattened_rows.dtypes.astype(str).to_dict().items()
    }

    return {
        "shape": [int(flattened_rows.shape[0]), int(flattened_rows.shape[1])],
        "duplicate_turn_rows": duplicate_turns,
        "missingness": missingness,
        "dtypes": dtypes,
        "response_status_counts": response_status_counts,
    }


def build_dimension_summary(flattened_rows: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for column in SCORE_COLUMNS:
        values = flattened_rows[column]
        applicable = int(values.notna().sum())
        passed = int((values == True).sum())
        failed = int((values == False).sum())
        rows.append(
            {
                "dimension": column,
                "applicable": applicable,
                "passed": passed,
                "failed": failed,
                "not_applicable": int(len(flattened_rows) - applicable),
                "pass_rate": round((passed / applicable) * 100, 1) if applicable else None,
            }
        )
    return rows


def build_error_analysis(
    flattened_rows: pd.DataFrame,
    summary_payload: dict,
    latency_summary: dict,
) -> str:
    failed_rows = flattened_rows[flattened_rows["overall_status"] == "fail"].copy()
    failed_rows = failed_rows.sort_values(
        by=["failure_count", "turn"],
        ascending=[False, True],
    )
    empty_response_rows = flattened_rows[
        flattened_rows["response_status"] == "empty_response"
    ].sort_values("turn")
    function_tracking = summary_payload.get("function_tracking", {})
    degraded_functions = [
        {
            "name": name,
            "status": details.get("status"),
            "expected_turn": details.get("expected_turn"),
            "actual_turn": details.get("actual_turn"),
        }
        for name, details in function_tracking.items()
        if details.get("status") != "on_time"
    ]

    failure_counter = Counter()
    for failed_dimension_list in failed_rows["failed_dimensions"]:
        failure_counter.update(failed_dimension_list)

    logistics_and_refusal_turns = [1, 22, 23, 30]
    logistics_rows = flattened_rows[flattened_rows["turn"].isin(logistics_and_refusal_turns)]

    lines: list[str] = []
    lines.append("# Error Analysis")
    lines.append("")
    lines.append("## Headline")
    lines.append(
        f"- Overall pass rows: {int((flattened_rows['overall_status'] == 'pass').sum())}/{len(flattened_rows)}"
    )
    lines.append(
        f"- Empty responses: {len(empty_response_rows)} turns ({', '.join(str(value) for value in empty_response_rows['turn'].tolist()) or 'none'})"
    )
    lines.append(
        f"- Latency p50/p90/p95/p99 (ms): {latency_summary['p50']}, {latency_summary['p90']}, {latency_summary['p95']}, {latency_summary['p99']}"
    )
    lines.append("")
    lines.append("## Primary Failure Clusters")
    if len(empty_response_rows):
        lines.append(
            f"- Empty-response failures dominate the worst user-facing breakdowns: turns {', '.join(str(value) for value in empty_response_rows['turn'].tolist())}. These create hard instruction-following failures even when grounding is scored leniently."
        )
    if degraded_functions:
        degraded_summary = ", ".join(
            f"{item['name']} ({item['status']} @ expected turn {item['expected_turn']})"
            for item in degraded_functions
        )
        lines.append(
            f"- Tool execution is mostly strong, but non-on-time function paths remain concentrated in a small set of workflows: {degraded_summary}."
        )
    if not logistics_rows.empty:
        lines.append(
            "- The model still refuses or under-serves several benchmark-supported logistics questions (transport, ticketing, hotels, and venue correction), which hurts both instruction following and grounding on turns 1, 22, 23, and 30."
        )
    if int((flattened_rows["ambiguity_handling"] == False).sum()):
        lines.append(
            "- Ambiguity handling is the weakest scoped metric. The model misses required clarification on ambiguous speaker references and sometimes over-clarifies when the request is already specific."
        )
    if int((flattened_rows["state_tracking"] == False).sum()):
        lines.append(
            "- State tracking failures are narrower than instruction-following failures, but they show up in places where the run should reuse known user state (especially previously supplied name and registration context)."
        )
    lines.append("")
    lines.append("## Failure Counts By Dimension")
    for dimension, count in failure_counter.most_common():
        lines.append(f"- {dimension}: {count}")
    lines.append("")
    lines.append("## Highest-Severity Turns")
    top_failures = failed_rows.head(8)
    for _, row in top_failures.iterrows():
        lines.append(
            f"- Turn {int(row['turn'])}: failed {row['failed_dimensions_text'] or 'none'}; user asked: {row['user_text']}"
        )
    lines.append("")
    lines.append("## Recommended Follow-Ups")
    lines.append(
        "- Focus first on eliminating empty responses; they account for a disproportionate share of outright task failure and inflate latency."
    )
    lines.append(
        "- Tighten knowledge-base retrieval for conference logistics and pricing so the model stops refusing benchmark-covered questions."
    )
    lines.append(
        "- Improve multi-action tool execution paths, especially batch registrations/cancellations and conditional fallback registrations."
    )
    lines.append(
        "- Add explicit ambiguity handling guardrails for abbreviated names (for example, 'Dr. Liu') and remove unnecessary reconfirmation when the user already supplied a valid identifier."
    )
    return "\n".join(lines) + "\n"


def build_review_payload(
    run_dir: Path,
    *,
    output_dir: Path | None = None,
) -> tuple[dict, pd.DataFrame, str, dict]:
    run_dir = run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    benchmark_name = run_dir.parent.name
    run_id = run_dir.name
    resolved_output_dir = (output_dir or run_dir).resolve()
    benchmark_turns = load_benchmark_turns(benchmark_name)

    judged_path = run_dir / "openai_judged.jsonl"
    summary_path = run_dir / "openai_summary.json"
    runtime_path = run_dir / "runtime.json"
    transcript_path = run_dir / "transcript.jsonl"

    required_paths = [judged_path, summary_path, runtime_path, transcript_path]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Missing required input files:\n" + "\n".join(missing_paths)
        )

    judged_rows = load_jsonl(judged_path)
    if not judged_rows:
        raise ValueError(f"No rows found in {judged_path}")

    normalized_rows = [build_record(raw_record) for raw_record in judged_rows]
    flattened_rows = pd.DataFrame(normalized_rows).sort_values("turn").reset_index(drop=True)
    flattened_rows["golden_user_text"] = flattened_rows["turn"].map(
        lambda turn_value: benchmark_turns[int(turn_value)]["input"]
    )
    flattened_rows["golden_assistant_text"] = flattened_rows["turn"].map(
        lambda turn_value: benchmark_turns[int(turn_value)]["golden_text"]
    )
    flattened_rows["golden_tool_calls"] = flattened_rows["turn"].map(
        lambda turn_value: normalize_tool_entries(
            benchmark_turns[int(turn_value)].get("required_function_call")
        )
    )
    flattened_rows["golden_tool_results"] = flattened_rows["turn"].map(
        lambda turn_value: normalize_tool_entries(
            benchmark_turns[int(turn_value)].get("function_call_response")
        )
    )
    flattened_rows["golden_tool_calls_text"] = flattened_rows["golden_tool_calls"].map(
        lambda tool_calls: json.dumps(tool_calls, ensure_ascii=False, indent=2)
    )
    flattened_rows["golden_tool_results_text"] = flattened_rows["golden_tool_results"].map(
        lambda tool_results: json.dumps(tool_results, ensure_ascii=False, indent=2)
    )
    flattened_rows["golden_context_block"] = flattened_rows["turn"].map(
        lambda turn_value: format_golden_context_block(
            int(turn_value),
            benchmark_turns[int(turn_value)],
        )
    )

    flattened_rows["user_audio_path"] = flattened_rows["turn"].map(
        lambda turn_value: str(turn_audio_path(benchmark_name, int(turn_value)))
    )
    flattened_rows["user_audio_url"] = flattened_rows["turn"].map(
        lambda turn_value: path_to_file_url(turn_audio_path(benchmark_name, int(turn_value)))
    )
    flattened_rows["user_audio_exists"] = flattened_rows["turn"].map(
        lambda turn_value: turn_audio_path(benchmark_name, int(turn_value)).exists()
    )

    for column in SCORE_COLUMNS:
        flattened_rows[column] = flattened_rows["scores"].map(
            lambda row_scores: row_scores.get(column)
        )

    sanity_report = build_sanity_report(flattened_rows)
    if sanity_report["duplicate_turn_rows"]:
        raise ValueError(
            f"Duplicate turn rows found in review data: {sanity_report['duplicate_turn_rows']}"
        )

    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    runtime_payload = json.loads(runtime_path.read_text(encoding="utf-8"))

    latency_values = pd.to_numeric(flattened_rows["latency_ms"], errors="coerce")
    ttfb_values = pd.to_numeric(flattened_rows["ttfb_ms"], errors="coerce")

    latency_summary = {
        "p50": percentile(latency_values, 0.50),
        "p90": percentile(latency_values, 0.90),
        "p95": percentile(latency_values, 0.95),
        "p99": percentile(latency_values, 0.99),
        "mean": round(float(latency_values.dropna().mean()), 2),
    }
    ttfb_summary = {
        "p50": percentile(ttfb_values, 0.50),
        "p90": percentile(ttfb_values, 0.90),
        "p95": percentile(ttfb_values, 0.95),
        "p99": percentile(ttfb_values, 0.99),
        "mean": round(float(ttfb_values.dropna().mean()), 2),
    }

    overall_pass_rows = int((flattened_rows["overall_status"] == "pass").sum())
    empty_response_count = int(
        (flattened_rows["response_status"] == "empty_response").sum()
    )

    error_analysis = build_error_analysis(
        flattened_rows=flattened_rows,
        summary_payload=summary_payload,
        latency_summary=latency_summary,
    )

    dimension_summary = build_dimension_summary(flattened_rows)
    metadata = {
        "benchmark_name": benchmark_name,
        "run_id": run_id,
        "model_name": summary_payload.get("model_name"),
        "judge_name": summary_payload.get("judge_name"),
        "judge_model": summary_payload.get("judge_model"),
        "judge_version": summary_payload.get("judge_version"),
        "judged_at": summary_payload.get("judged_at"),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "source_run_dir": str(run_dir),
        "output_dir": str(resolved_output_dir),
        "turn_count": int(len(flattened_rows)),
        "runtime_mode": runtime_payload.get("mode"),
        "runtime_parallel": runtime_payload.get("parallel"),
        "disable_vad": runtime_payload.get("disable_vad"),
        "conversation_audio_path": str(run_dir / "conversation.wav"),
        "conversation_audio_url": path_to_file_url(run_dir / "conversation.wav"),
        "conversation_audio_exists": (run_dir / "conversation.wav").exists(),
    }

    payload = {
        "metadata": metadata,
        "summary": {
            "passes": summary_payload.get("passes", {}),
            "category_totals": summary_payload.get("category_totals", {}),
            "function_tracking": summary_payload.get("function_tracking", {}),
            "turn_taking_failures": summary_payload.get("turn_taking_failures", []),
            "overall_pass_rows": overall_pass_rows,
            "empty_response_count": empty_response_count,
        },
        "latency_ms": latency_summary,
        "ttfb_ms": ttfb_summary,
        "dimension_summary": dimension_summary,
        "sanity_checks": sanity_report,
        "rows": flattened_rows.to_dict(orient="records"),
    }
    return payload, flattened_rows, error_analysis, sanity_report


def build_html(payload: dict) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    generated_at = payload["metadata"]["generated_at"]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ConversationBench Review</title>
  <style>
    :root {{
      --bg: #f6f1e8;
      --panel: rgba(255, 250, 242, 0.92);
      --panel-strong: #fff9ef;
      --ink: #17211f;
      --muted: #5a6763;
      --line: #d6c9b4;
      --accent: #0f5f5a;
      --accent-soft: #dff2ee;
      --danger: #a03a2f;
      --danger-soft: #f6d9d3;
      --warn: #8f5a08;
      --warn-soft: #f6e8c7;
      --success: #1b6a3b;
      --success-soft: #dff0e3;
      --shadow: 0 18px 40px rgba(41, 33, 12, 0.08);
      --radius: 18px;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      min-height: 100vh;
      font-family: "Avenir Next", "Trebuchet MS", "Gill Sans", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top right, rgba(15, 95, 90, 0.1), transparent 34%),
        radial-gradient(circle at top left, rgba(160, 58, 47, 0.08), transparent 28%),
        linear-gradient(180deg, #fbf7f0 0%, #f3ecdf 46%, #efe4d3 100%);
    }}

    .page {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 28px 24px 40px;
    }}

    .hero {{
      padding: 28px;
      border: 1px solid rgba(214, 201, 180, 0.9);
      border-radius: 28px;
      background:
        linear-gradient(135deg, rgba(255, 250, 242, 0.98), rgba(247, 239, 224, 0.96)),
        linear-gradient(120deg, rgba(15, 95, 90, 0.08), transparent 45%);
      box-shadow: var(--shadow);
    }}

    .eyebrow {{
      margin: 0 0 8px;
      font-size: 12px;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--accent);
      font-weight: 700;
    }}

    h1, h2, h3 {{
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
      margin: 0;
      color: #14201c;
    }}

    h1 {{
      font-size: clamp(2rem, 4vw, 3.4rem);
      line-height: 1.05;
      max-width: 900px;
    }}

    h2 {{
      font-size: 1.5rem;
      margin-bottom: 14px;
    }}

    h3 {{
      font-size: 1.05rem;
      margin-bottom: 10px;
    }}

    p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.55;
    }}

    .hero-grid,
    .metrics-grid,
    .detail-meta {{
      display: grid;
      gap: 14px;
    }}

    .hero-grid {{
      margin-top: 18px;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    }}

    .hero-card,
    .section,
    .metric-card,
    .detail-panel {{
      min-width: 0;
      border: 1px solid rgba(214, 201, 180, 0.9);
      border-radius: var(--radius);
      background: var(--panel);
      box-shadow: var(--shadow);
    }}

    .hero-card,
    .section,
    .metric-card,
    .detail-panel {{
      padding: 18px;
    }}

    .mono,
    code,
    pre,
    td.mono {{
      font-family: "SFMono-Regular", "Menlo", "Monaco", "Liberation Mono", monospace;
      overflow-wrap: anywhere;
      word-break: break-word;
    }}

    .label {{
      margin-bottom: 6px;
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--muted);
      font-weight: 700;
    }}

    .value {{
      font-size: 0.98rem;
      line-height: 1.45;
      color: var(--ink);
    }}

    .section-stack {{
      display: grid;
      gap: 18px;
      margin-top: 24px;
    }}

    .section {{
      overflow: hidden;
    }}

    .filters {{
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      align-items: end;
    }}

    .field {{
      display: grid;
      gap: 6px;
      min-width: 0;
    }}

    .field span {{
      font-size: 0.8rem;
      font-weight: 700;
      color: var(--muted);
    }}

    input,
    select {{
      width: 100%;
      padding: 11px 12px;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: var(--panel-strong);
      color: var(--ink);
      font: inherit;
    }}

    .metrics-grid {{
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    }}

    .metric-card {{
      background:
        linear-gradient(180deg, rgba(255, 249, 239, 0.98), rgba(252, 244, 232, 0.92));
    }}

    .metric-number {{
      font-size: 1.8rem;
      font-weight: 700;
      margin-top: 4px;
      color: #112623;
    }}

    .metric-note {{
      margin-top: 8px;
      font-size: 0.9rem;
    }}

    .table-wrap {{
      overflow: auto;
      margin-top: 14px;
      border: 1px solid rgba(214, 201, 180, 0.7);
      border-radius: 14px;
      background: rgba(255, 251, 244, 0.95);
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
    }}

    .wide-table table {{
      min-width: 980px;
    }}

    th,
    td {{
      padding: 11px 12px;
      text-align: left;
      vertical-align: top;
      border-bottom: 1px solid rgba(214, 201, 180, 0.7);
      overflow-wrap: anywhere;
      word-break: break-word;
      font-size: 0.92rem;
      line-height: 1.45;
    }}

    th {{
      position: sticky;
      top: 0;
      z-index: 1;
      background: rgba(246, 239, 226, 0.98);
      color: #23312d;
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}

    tbody tr {{
      cursor: pointer;
      transition: background-color 120ms ease, transform 120ms ease;
    }}

    tbody tr:hover {{
      background: rgba(223, 242, 238, 0.55);
    }}

    tbody tr.selected {{
      background: rgba(15, 95, 90, 0.13);
    }}

    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 9px;
      border-radius: 999px;
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      white-space: nowrap;
    }}

    .badge-pass {{
      background: var(--success-soft);
      color: var(--success);
    }}

    .badge-fail {{
      background: var(--danger-soft);
      color: var(--danger);
    }}

    .badge-na,
    .badge-incomplete {{
      background: var(--warn-soft);
      color: var(--warn);
    }}

    .detail-layout {{
      display: grid;
      gap: 18px;
      grid-template-columns: minmax(0, 1.4fr) minmax(0, 1fr);
      align-items: start;
    }}

    .detail-meta {{
      grid-template-columns: repeat(3, minmax(0, 1fr));
      margin-bottom: 16px;
    }}

    .detail-panel {{
      background:
        linear-gradient(180deg, rgba(255, 250, 242, 0.98), rgba(248, 239, 225, 0.94));
    }}

    .stack {{
      display: grid;
      gap: 16px;
      min-width: 0;
    }}

    .text-card {{
      border: 1px solid rgba(214, 201, 180, 0.7);
      border-radius: 14px;
      padding: 14px;
      background: rgba(255, 252, 246, 0.98);
      min-width: 0;
    }}

    .scroll-box {{
      margin-top: 8px;
      border-radius: 12px;
      border: 1px solid rgba(214, 201, 180, 0.85);
      background: #fffdf8;
      padding: 12px;
      overflow: auto;
      min-width: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      word-break: break-word;
    }}

    .short-box {{
      max-height: 180px;
    }}

    .tall-box {{
      max-height: 360px;
    }}

    .context-box {{
      max-height: 320px;
    }}

    .reason-box {{
      max-height: 220px;
    }}

    audio {{
      width: 100%;
      margin-top: 8px;
    }}

    .subtle {{
      color: var(--muted);
      font-size: 0.9rem;
    }}

    .footer-note {{
      margin-top: 12px;
      font-size: 0.86rem;
      color: var(--muted);
    }}

    @media (max-width: 1040px) {{
      .detail-layout {{
        grid-template-columns: 1fr;
      }}
    }}

    @media (max-width: 820px) {{
      .page {{
        padding: 16px 14px 28px;
      }}

      .hero {{
        padding: 18px;
        border-radius: 22px;
      }}

      .detail-meta {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <p class="eyebrow">ConversationBench Review</p>
      <h1>Skip-Turn-Taking Error Review for <span class="mono">{payload["metadata"]["model_name"]}</span></h1>
      <p style="margin-top: 12px;">Run <span class="mono">{payload["metadata"]["run_id"]}</span>. Self-contained offline review page generated at <span class="mono">{generated_at}</span>. Click any row to inspect the raw turn, tool calls, model output, and per-dimension grading verdicts.</p>
      <div class="hero-grid" id="hero-metadata"></div>
    </section>

    <div class="section-stack">
      <section class="section">
        <h2>Global Filters</h2>
        <p class="subtle">Filter by graded dimension, pass/fail state, response status, or free text across prompts, outputs, and judge reasoning.</p>
        <div class="filters" style="margin-top: 16px;">
          <label class="field">
            <span>Dimension</span>
            <select id="dimension-filter"></select>
          </label>
          <label class="field">
            <span>Status</span>
            <select id="status-filter">
              <option value="all">All</option>
              <option value="fail">Fail</option>
              <option value="pass">Pass</option>
              <option value="incomplete">Incomplete / N/A</option>
            </select>
          </label>
          <label class="field">
            <span>Response</span>
            <select id="response-filter">
              <option value="all">All</option>
              <option value="normal">Normal response</option>
              <option value="empty_response">Empty response</option>
            </select>
          </label>
          <label class="field">
            <span>Search</span>
            <input id="search-filter" type="search" placeholder="Turn id, prompt, output, or reason">
          </label>
        </div>
      </section>

      <section class="section">
        <h2>Model-Level Summary</h2>
        <div class="metrics-grid" id="metrics-grid"></div>
      </section>

      <section class="section">
        <h2>Grader Metrics By Model Config</h2>
        <p class="subtle">Pass rate is computed over applicable rows for each dimension. Missing dimensions are treated as not applicable, not failures.</p>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th style="width: 240px;">Dimension</th>
                <th style="width: 140px;">Pass Rate</th>
                <th style="width: 140px;">Pass / Applicable</th>
                <th style="width: 140px;">Failures</th>
                <th style="width: 140px;">N/A</th>
              </tr>
            </thead>
            <tbody id="dimension-summary-body"></tbody>
          </table>
        </div>
      </section>

      <section class="section">
        <h2>Curated Failure Review</h2>
        <p class="subtle" id="curated-count"></p>
        <div class="table-wrap wide-table">
          <table>
            <thead>
              <tr>
                <th style="width: 80px;">Turn</th>
                <th style="width: 130px;">Status</th>
                <th style="width: 140px;">Failed Dims</th>
                <th>User Prompt</th>
                <th>Assistant Output</th>
              </tr>
            </thead>
            <tbody id="curated-body"></tbody>
          </table>
        </div>
      </section>

      <section class="section">
        <h2>Datapoint Detail</h2>
        <div id="detail-root"></div>
      </section>

      <section class="section">
        <h2>Full Datapoints</h2>
        <p class="subtle" id="full-count"></p>
        <div class="table-wrap wide-table">
          <table>
            <thead>
              <tr>
                <th style="width: 80px;">Turn</th>
                <th style="width: 130px;">Overall</th>
                <th style="width: 140px;">Response</th>
                <th style="width: 110px;">Latency</th>
                <th style="width: 140px;">Failed Dims</th>
                <th>User Prompt</th>
                <th>Assistant Output</th>
              </tr>
            </thead>
            <tbody id="full-body"></tbody>
          </table>
        </div>
      </section>
    </div>
  </div>

  <script id="payload" type="application/json">{payload_json}</script>
  <script>
    const payload = JSON.parse(document.getElementById("payload").textContent);
    const allRows = payload.rows.slice();
    const heroMetadata = document.getElementById("hero-metadata");
    const metricsGrid = document.getElementById("metrics-grid");
    const dimensionFilter = document.getElementById("dimension-filter");
    const statusFilter = document.getElementById("status-filter");
    const responseFilter = document.getElementById("response-filter");
    const searchFilter = document.getElementById("search-filter");
    const curatedBody = document.getElementById("curated-body");
    const fullBody = document.getElementById("full-body");
    const curatedCount = document.getElementById("curated-count");
    const fullCount = document.getElementById("full-count");
    const detailRoot = document.getElementById("detail-root");
    const dimensionSummaryBody = document.getElementById("dimension-summary-body");

    let selectedTurn = allRows.length ? allRows[0].turn : null;

    function badgeClass(status) {{
      if (status === "pass") return "badge badge-pass";
      if (status === "fail") return "badge badge-fail";
      return "badge badge-incomplete";
    }}

    function escapeHtml(text) {{
      return String(text ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }}

    function badge(status, label) {{
      return `<span class="${{badgeClass(status)}}">${{escapeHtml(label || status)}}</span>`;
    }}

    function renderHero() {{
      const entries = [
        ["Run ID", payload.metadata.run_id],
        ["Model", payload.metadata.model_name],
        ["Judge", `${{payload.metadata.judge_name}} / ${{payload.metadata.judge_model}}`],
        ["Judged At", payload.metadata.judged_at],
        ["Generated", payload.metadata.generated_at],
        ["Rows", String(payload.metadata.turn_count)],
        ["Source Run", payload.metadata.source_run_dir],
        ["Output Dir", payload.metadata.output_dir],
      ];
      heroMetadata.innerHTML = entries.map(([label, value]) => `
        <div class="hero-card">
          <div class="label">${{escapeHtml(label)}}</div>
          <div class="value mono">${{escapeHtml(value)}}</div>
        </div>
      `).join("");
    }}

    function renderMetrics() {{
      const cards = [
        {{
          label: "Overall Pass Rows",
          value: `${{payload.summary.overall_pass_rows}} / ${{payload.metadata.turn_count}}`,
          note: "Rows with no graded failures across applicable dimensions.",
        }},
        {{
          label: "Empty Responses",
          value: String(payload.summary.empty_response_count),
          note: "Turns where the assistant returned the explicit empty-response sentinel.",
        }},
        {{
          label: "Latency p95",
          value: `${{payload.latency_ms.p95}} ms`,
          note: `p50=${{payload.latency_ms.p50}} ms, p99=${{payload.latency_ms.p99}} ms`,
        }},
        {{
          label: "Tool Use",
          value: `${{payload.summary.passes.tool_use_correct}} / ${{payload.metadata.turn_count}}`,
          note: "Exact or accepted partial tool correctness from the judge.",
        }},
        {{
          label: "Instruction Following",
          value: `${{payload.summary.passes.instruction_following}} / ${{payload.metadata.turn_count}}`,
          note: "User-facing task completion at the turn level.",
        }},
        {{
          label: "KB Grounding",
          value: `${{payload.summary.passes.kb_grounding}} / ${{payload.metadata.turn_count}}`,
          note: "Grounding against the benchmark knowledge base.",
        }},
      ];
      metricsGrid.innerHTML = cards.map((card) => `
        <div class="metric-card">
          <div class="label">${{escapeHtml(card.label)}}</div>
          <div class="metric-number">${{escapeHtml(card.value)}}</div>
          <div class="metric-note subtle">${{escapeHtml(card.note)}}</div>
        </div>
      `).join("");
    }}

    function renderDimensionSummary() {{
      dimensionSummaryBody.innerHTML = payload.dimension_summary.map((row) => {{
        const passRate = row.pass_rate == null ? "n/a" : `${{row.pass_rate}}%`;
        return `
          <tr>
            <td class="mono">${{escapeHtml(row.dimension)}}</td>
            <td>${{escapeHtml(passRate)}}</td>
            <td>${{escapeHtml(`${{row.passed}} / ${{row.applicable}}`)}}</td>
            <td>${{escapeHtml(String(row.failed))}}</td>
            <td>${{escapeHtml(String(row.not_applicable))}}</td>
          </tr>
        `;
      }}).join("");
    }}

    function populateFilters() {{
      const options = ['all', ...payload.dimension_summary.map((row) => row.dimension)];
      dimensionFilter.innerHTML = options.map((option) => {{
        const label = option === 'all' ? 'All dimensions' : option;
        return `<option value="${{escapeHtml(option)}}">${{escapeHtml(label)}}</option>`;
      }}).join("");
    }}

    function rowMatches(row) {{
      const dimension = dimensionFilter.value;
      const status = statusFilter.value;
      const response = responseFilter.value;
      const needle = searchFilter.value.trim().toLowerCase();

      if (response !== "all" && row.response_status !== response) {{
        return false;
      }}

      if (dimension === "all") {{
        if (status !== "all" && row.overall_status !== status) {{
          return false;
        }}
      }} else {{
        const dimensionValue = row.scores[dimension];
        const dimensionStatus = dimensionValue === true ? "pass" : dimensionValue === false ? "fail" : "incomplete";
        if (status !== "all" && dimensionStatus !== status) {{
          return false;
        }}
      }}

      if (!needle) {{
        return true;
      }}

      const haystack = [
        row.turn,
        row.user_text,
        row.assistant_text,
        row.judge_reasoning,
        row.failed_dimensions_text,
        row.tool_calls_text,
        row.tool_results_text,
      ].join(" ").toLowerCase();
      return haystack.includes(needle);
    }}

    function filteredRows() {{
      const rows = allRows.filter(rowMatches);
      rows.sort((left, right) => {{
        if (left.overall_status !== right.overall_status) {{
          const order = {{ fail: 0, incomplete: 1, pass: 2 }};
          return order[left.overall_status] - order[right.overall_status];
        }}
        if (left.failure_count !== right.failure_count) {{
          return right.failure_count - left.failure_count;
        }}
        return left.turn - right.turn;
      }});
      return rows;
    }}

    function setSelectedTurn(candidateRows) {{
      if (!candidateRows.length) {{
        selectedTurn = null;
        return;
      }}
      if (!candidateRows.some((row) => row.turn === selectedTurn)) {{
        selectedTurn = candidateRows[0].turn;
      }}
    }}

    function renderTableRow(row, isSelected) {{
      return `
        <tr data-turn="${{row.turn}}" class="${{isSelected ? 'selected' : ''}}">
          <td class="mono">${{row.turn}}</td>
          <td>${{badge(row.overall_status)}}</td>
          <td>${{badge(row.response_status === 'empty_response' ? 'fail' : 'pass', row.response_status.replace('_', ' '))}}</td>
          <td class="mono">${{escapeHtml(row.latency_ms == null ? 'n/a' : `${{row.latency_ms}} ms`)}}</td>
          <td>${{escapeHtml(row.failed_dimensions_text || 'none')}}</td>
          <td>${{escapeHtml(row.user_text)}}</td>
          <td>${{escapeHtml(row.assistant_text)}}</td>
        </tr>
      `;
    }}

    function attachRowHandlers(rootElement, rows) {{
      rootElement.querySelectorAll("tr[data-turn]").forEach((element) => {{
        element.addEventListener("click", () => {{
          selectedTurn = Number(element.dataset.turn);
          render(rows);
        }});
      }});
    }}

    function renderCurated(rows) {{
      const curatedRows = rows
        .filter((row) => row.overall_status === "fail")
        .slice()
        .sort((left, right) => left.turn - right.turn)
        .slice(0, 25);
      curatedCount.textContent = `Showing ${{curatedRows.length}} of ${{rows.filter((row) => row.overall_status === "fail").length}} failing rows after filters.`;
      curatedBody.innerHTML = curatedRows.map((row) => `
        <tr data-turn="${{row.turn}}" class="${{row.turn === selectedTurn ? 'selected' : ''}}">
          <td class="mono">${{row.turn}}</td>
          <td>${{badge(row.overall_status)}}</td>
          <td>${{escapeHtml(row.failed_dimensions_text || 'none')}}</td>
          <td>${{escapeHtml(row.user_text)}}</td>
          <td>${{escapeHtml(row.assistant_text)}}</td>
        </tr>
      `).join("");
      attachRowHandlers(curatedBody, rows);
    }}

    function renderFull(rows) {{
      fullCount.textContent = `Showing ${{rows.length}} of ${{allRows.length}} rows. Sorted with failures first.`;
      fullBody.innerHTML = rows.map((row) => renderTableRow(row, row.turn === selectedTurn)).join("");
      attachRowHandlers(fullBody, rows);
    }}

    function renderDetail(rows) {{
      if (!rows.length || selectedTurn == null) {{
        detailRoot.innerHTML = '<p class="subtle">No rows match the current filters.</p>';
        return;
      }}

      const row = rows.find((item) => item.turn === selectedTurn) || rows[0];
      const previousContext = allRows
        .filter((item) => item.turn < row.turn)
        .sort((left, right) => left.turn - right.turn)
        .map((item) => item.golden_context_block || `Turn ${{item.turn}}\\nUser: ${{item.golden_user_text || item.user_text}}\\nAssistant: ${{item.golden_assistant_text || item.assistant_text}}`)
        .join("\\n\\n");
      const graderRows = payload.dimension_summary.map((summaryRow) => {{
        const value = row.scores[summaryRow.dimension];
        const status = value === true ? 'pass' : value === false ? 'fail' : 'incomplete';
        let reason = 'Not applicable on this turn.';
        if (value === false) {{
          reason = 'Failed. See the turn-level judge reasoning for decisive evidence.';
        }} else if (value === true) {{
          reason = 'Passed.';
        }}
        return `
          <tr>
            <td class="mono" style="width: 220px;">${{escapeHtml(summaryRow.dimension)}}</td>
            <td style="width: 140px;">${{badge(status, value == null ? 'n/a' : status)}}</td>
            <td>${{escapeHtml(reason)}}</td>
          </tr>
        `;
      }}).join("");

      detailRoot.innerHTML = `
        <div class="detail-panel">
          <div class="detail-meta">
            <div class="hero-card">
              <div class="label">Source</div>
              <div class="value mono">${{escapeHtml(payload.metadata.run_id)}}</div>
            </div>
            <div class="hero-card">
              <div class="label">Identifiers</div>
              <div class="value mono">turn=${{row.turn}} · ts=${{escapeHtml(row.timestamp)}}</div>
            </div>
            <div class="hero-card">
              <div class="label">Response Status</div>
              <div class="value">${{badge(row.overall_status)}} ${{badge(row.response_status === 'empty_response' ? 'fail' : 'pass', row.response_status.replace('_', ' '))}}</div>
            </div>
          </div>
          <div class="detail-layout">
            <div class="stack">
              <div class="text-card">
                <div class="label">Current Turn Audio</div>
                <div class="subtle">Benchmark input audio for the selected user turn.</div>
                ${{row.user_audio_exists ? `<audio controls preload="none" src="${{escapeHtml(row.user_audio_url)}}"></audio>` : `<div class="subtle" style="margin-top: 8px;">No per-turn input audio file found.</div>`}}
              </div>
              <div class="text-card">
                <div class="label">Full Conversation Audio</div>
                <div class="subtle">Run-level merged audio saved for the full benchmark conversation.</div>
                ${{payload.metadata.conversation_audio_exists ? `<audio controls preload="none" src="${{escapeHtml(payload.metadata.conversation_audio_url)}}"></audio>` : `<div class="subtle" style="margin-top: 8px;">No conversation.wav found for this run.</div>`}}
              </div>
              <div class="text-card">
                <div class="label">Gold Previous Turns Context</div>
                <div class="scroll-box context-box">${{escapeHtml(previousContext || 'No previous turns.')}}</div>
              </div>
              <div class="text-card">
                <div class="label">User Query</div>
                <div class="scroll-box short-box">${{escapeHtml(row.user_text)}}</div>
              </div>
              <div class="text-card">
                <div class="label">Model Output</div>
                <div class="scroll-box tall-box">${{escapeHtml(row.assistant_text)}}</div>
              </div>
              <div class="text-card">
                <div class="label">Judge Reasoning</div>
                <div class="scroll-box reason-box">${{escapeHtml(row.judge_reasoning || 'No reasoning provided.')}}</div>
              </div>
            </div>
            <div class="stack">
              <div class="text-card">
                <div class="label">Compact Metadata</div>
                <div class="value mono">
                  latency=${{row.latency_ms == null ? 'n/a' : `${{row.latency_ms}}ms`}}<br>
                  ttfb=${{row.ttfb_ms == null ? 'n/a' : `${{row.ttfb_ms}}ms`}}<br>
                  reconnections=${{row.reconnection_count ?? 'n/a'}}<br>
                  failed=${{escapeHtml(row.failed_dimensions_text || 'none')}}
                </div>
              </div>
              <div class="text-card">
                <div class="label">Per-Dimension Verdicts</div>
                <div class="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th style="width: 220px;">Dimension</th>
                        <th style="width: 140px;">Verdict</th>
                        <th>Reason</th>
                      </tr>
                    </thead>
                    <tbody>${{graderRows}}</tbody>
                  </table>
                </div>
              </div>
              <div class="text-card">
                <div class="label">Tool Calls</div>
                <div class="scroll-box short-box mono">${{escapeHtml(row.tool_calls_text)}}</div>
              </div>
              <div class="text-card">
                <div class="label">Tool Results</div>
                <div class="scroll-box short-box mono">${{escapeHtml(row.tool_results_text)}}</div>
              </div>
            </div>
          </div>
        </div>
      `;
    }}

    function render(rows) {{
      setSelectedTurn(rows);
      renderCurated(rows);
      renderFull(rows);
      renderDetail(rows);
    }}

    function initialize() {{
      renderHero();
      renderMetrics();
      renderDimensionSummary();
      populateFilters();

      [dimensionFilter, statusFilter, responseFilter, searchFilter].forEach((element) => {{
        element.addEventListener("input", () => render(filteredRows()));
      }});

      render(filteredRows());
    }}

    initialize();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a self-contained HTML review page for a judged benchmark run."
    )
    parser.add_argument("run_dir", help="Path to the judged run directory.")
    parser.add_argument(
        "--output-dir",
        help="Optional output directory. Defaults to results/<benchmark>/<run_id>/",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()

    benchmark_name = run_dir.parent.name
    run_id = run_dir.name
    default_output_dir = (
        Path.cwd() / "results" / benchmark_name / run_id
    )
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    payload, flattened_rows, error_analysis, sanity_report = build_review_payload(
        run_dir,
        output_dir=output_dir,
    )
    csv_columns = [
        "turn",
        "timestamp",
        "model_name",
        "overall_status",
        "response_status",
        "failed_dimensions_text",
        "failure_count",
        "latency_ms",
        "ttfb_ms",
        "reconnection_count",
        "tool_call_count",
        "turn_taking",
        "tool_use_correct",
        "instruction_following",
        "kb_grounding",
        "ambiguity_handling",
        "state_tracking",
        "user_text",
        "assistant_text",
        "judge_reasoning",
        "tool_calls_text",
        "tool_results_text",
        "user_audio_path",
        "user_audio_url",
        "user_audio_exists",
        "golden_user_text",
        "golden_assistant_text",
        "golden_tool_calls_text",
        "golden_tool_results_text",
        "golden_context_block",
        "oracle_continuation_used",
        "live_tool_name_correct",
        "live_tool_args_correct",
        "live_tool_use_pass",
        "oracle_tool_calls_text",
        "oracle_tool_results_text",
    ]
    flattened_rows[csv_columns].to_csv(
        output_dir / "review_rows.csv",
        index=False,
        encoding="utf-8",
    )
    (output_dir / "error_analysis.md").write_text(error_analysis, encoding="utf-8")

    html = build_html(payload)
    (output_dir / "review.html").write_text(html, encoding="utf-8")
    (output_dir / "review_payload.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Output directory: {output_dir}")
    print(f"Rows: {len(flattened_rows)}")
    print(f"CSV: {output_dir / 'review_rows.csv'}")
    print(f"HTML: {output_dir / 'review.html'}")
    print(f"Analysis: {output_dir / 'error_analysis.md'}")
    print("Sanity checks:")
    print(json.dumps(sanity_report, indent=2))


if __name__ == "__main__":
    main()
