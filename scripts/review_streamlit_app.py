import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_experiment_review import SCORE_COLUMNS, build_review_payload, turn_audio_path


TIMESTAMP_FORMAT = "%Y%m%dT%H%M%S"
STATUS_ORDER = ["fail", "incomplete", "pass"]
STATUS_PALETTE = {
    "fail": "#c43c39",
    "incomplete": "#d08700",
    "pass": "#2b8a3e",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "input_path",
        nargs="?",
        help="Path to a review directory, review_payload.json, or judged run directory.",
    )
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


def find_default_input_path() -> str:
    run_sources = discover_review_sources()
    if run_sources:
        return run_sources[0]["input_path"]
    return ""


def parse_run_timestamp(path: Path) -> datetime:
    run_id = path.name
    match = re.search(r"\d{8}T\d{6}", run_id)
    if not match:
        raise ValueError(
            f"Could not find a timestamp like YYYYMMDDTHHMMSS in '{run_id}'"
        )
    timestamp = match.group(0)
    return datetime.strptime(timestamp, TIMESTAMP_FORMAT)


def load_review_metadata(payload_path: Path) -> dict:
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    run_id = metadata.get("run_id") or payload_path.parent.name
    benchmark_name = metadata.get("benchmark_name") or payload_path.parent.parent.name
    model_name = metadata.get("model_name") or "unknown-model"
    timestamp = parse_run_timestamp(Path(run_id))
    return {
        "run_id": run_id,
        "benchmark_name": benchmark_name,
        "model_name": model_name,
        "timestamp": timestamp,
    }


def load_comparison_metadata(comparison_dir: Path) -> dict:
    metadata_path = comparison_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    generated_at = metadata.get("generated_at")
    timestamp = (
        datetime.fromisoformat(generated_at.replace("Z", "+00:00")).replace(tzinfo=None)
        if generated_at
        else parse_run_timestamp(comparison_dir)
    )
    benchmark_name = metadata.get("benchmark") or comparison_dir.parent.name
    models = metadata.get("models", [])
    model_name = ", ".join(models[:3]) if models else "comparison"
    if len(models) > 3:
        model_name = f"{model_name}, +{len(models) - 3}"
    return {
        "run_id": comparison_dir.name,
        "benchmark_name": benchmark_name,
        "model_name": f"comparison | {model_name}",
        "timestamp": timestamp,
    }


def build_run_label(run_source: dict) -> str:
    timestamp = run_source["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"{timestamp} | {run_source['benchmark_name']} | {run_source['model_name']} "
        f"| {run_source['run_id']}"
    )


def discover_review_sources() -> list[dict]:
    discovered_sources: dict[str, dict] = {}

    results_root = Path.cwd() / "results"
    for metadata_path in results_root.glob("**/metadata.json"):
        comparison_dir = metadata_path.parent
        if not (comparison_dir / "run_results.csv").exists():
            continue
        metadata = load_comparison_metadata(comparison_dir)
        discovered_sources[metadata["run_id"]] = {
            **metadata,
            "input_path": str(comparison_dir),
            "source_type": "comparison_dir",
        }

    for payload_path in results_root.glob("**/review_payload.json"):
        metadata = load_review_metadata(payload_path)
        discovered_sources[metadata["run_id"]] = {
            **metadata,
            "input_path": str(payload_path.parent),
            "source_type": "review_dir",
        }

    runs_root = Path.cwd() / "runs"
    for judged_path in runs_root.glob("**/openai_judged.jsonl"):
        run_dir = judged_path.parent
        run_id = run_dir.name
        if run_id in discovered_sources:
            continue
        discovered_sources[run_id] = {
            "run_id": run_id,
            "benchmark_name": run_dir.parent.name,
            "model_name": run_id.split("_")[1] if "_" in run_id else "unknown-model",
            "timestamp": parse_run_timestamp(run_dir),
            "input_path": str(run_dir),
            "source_type": "judged_run_dir",
        }

    sorted_sources = sorted(
        discovered_sources.values(),
        key=lambda run_source: (run_source["timestamp"], run_source["run_id"]),
        reverse=True,
    )
    for run_source in sorted_sources:
        run_source["label"] = build_run_label(run_source)
    return sorted_sources


def load_review_source(input_path: str) -> tuple[dict, str, str]:
    candidate_path = Path(input_path).expanduser().resolve()
    if candidate_path.is_file():
        if candidate_path.name != "review_payload.json":
            raise FileNotFoundError(
                "Expected a review_payload.json file when using a file path."
            )
        payload = json.loads(candidate_path.read_text(encoding="utf-8"))
        review_dir = candidate_path.parent
        analysis_path = review_dir / "error_analysis.md"
        error_analysis = (
            analysis_path.read_text(encoding="utf-8")
            if analysis_path.exists()
            else "No saved error analysis found next to review_payload.json."
        )
        return payload, error_analysis, str(candidate_path)

    if candidate_path.is_dir():
        payload_path = candidate_path / "review_payload.json"
        if payload_path.exists():
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            analysis_path = candidate_path / "error_analysis.md"
            error_analysis = (
                analysis_path.read_text(encoding="utf-8")
                if analysis_path.exists()
                else "No saved error analysis found next to review_payload.json."
            )
            return payload, error_analysis, str(payload_path)

        metadata_path = candidate_path / "metadata.json"
        run_results_path = candidate_path / "run_results.csv"
        if metadata_path.exists() and run_results_path.exists():
            model_summary_path = candidate_path / "model_summary.csv"
            metric_summary_path = candidate_path / "metric_summary.csv"
            commands_path = candidate_path / "commands.csv"
            summary_path = candidate_path / "summary.md"
            comparison_payload = {
                "kind": "comparison",
                "metadata": json.loads(metadata_path.read_text(encoding="utf-8")),
                "run_results": pd.read_csv(run_results_path),
                "model_summary": (
                    pd.read_csv(model_summary_path)
                    if model_summary_path.exists()
                    else pd.DataFrame()
                ),
                "metric_summary": (
                    pd.read_csv(metric_summary_path)
                    if metric_summary_path.exists()
                    else pd.DataFrame()
                ),
                "commands": (
                    pd.read_csv(commands_path) if commands_path.exists() else pd.DataFrame()
                ),
                "summary_markdown": (
                    summary_path.read_text(encoding="utf-8")
                    if summary_path.exists()
                    else "No saved markdown summary found."
                ),
                "plot_paths": {
                    plot_name: candidate_path / plot_name
                    for plot_name in [
                        "pass_rates_by_metric.png",
                        "failed_turns.png",
                        "latency_p95.png",
                        "model_ended_session.png",
                    ]
                    if (candidate_path / plot_name).exists()
                },
            }
            return comparison_payload, comparison_payload["summary_markdown"], str(candidate_path)

        judged_path = candidate_path / "openai_judged.jsonl"
        if judged_path.exists():
            payload, _, error_analysis, _ = build_review_payload(
                candidate_path,
                output_dir=candidate_path,
            )
            return payload, error_analysis, str(candidate_path)

    raise FileNotFoundError(
        "Input must be a review directory, a review_payload.json file, or a judged run directory."
    )


def apply_filters(
    rows_df: pd.DataFrame,
    *,
    dimension: str,
    status: str,
    response_status: str,
    search_text: str,
) -> pd.DataFrame:
    filtered_df = rows_df.copy()

    if response_status != "all":
        filtered_df = filtered_df[filtered_df["response_status"] == response_status]

    if dimension == "all":
        if status != "all":
            filtered_df = filtered_df[filtered_df["overall_status"] == status]
    else:
        dimension_values = filtered_df[dimension]
        dimension_status = pd.Series("incomplete", index=filtered_df.index)
        dimension_status = dimension_status.mask(dimension_values == True, "pass")
        dimension_status = dimension_status.mask(dimension_values == False, "fail")
        if status != "all":
            filtered_df = filtered_df[dimension_status == status]

    needle = search_text.strip().lower()
    if needle:
        search_columns = [
            "user_text",
            "assistant_text",
            "judge_reasoning",
            "failed_dimensions_text",
            "tool_calls_text",
            "tool_results_text",
        ]
        for optional_column in [
            "golden_tool_calls_text",
            "golden_tool_results_text",
            "golden_context_block",
        ]:
            if optional_column in filtered_df.columns:
                search_columns.append(optional_column)
        haystack = filtered_df[search_columns].fillna("").astype(str).agg(
            " ".join,
            axis=1,
        )
        haystack = haystack + " " + filtered_df["turn"].astype(str)
        filtered_df = filtered_df[haystack.str.lower().str.contains(needle, regex=False)]

    status_order = {"fail": 0, "incomplete": 1, "pass": 2}
    filtered_df = filtered_df.assign(
        _status_order=filtered_df["overall_status"].map(status_order).fillna(3)
    )
    filtered_df = filtered_df.sort_values(
        by=["_status_order", "failure_count", "turn"],
        ascending=[True, False, True],
    ).drop(columns="_status_order")
    return filtered_df.reset_index(drop=True)


def format_previous_turns(previous_rows: pd.DataFrame) -> str:
    if previous_rows.empty:
        return "No previous turns."

    parts: list[str] = []
    for _, row in previous_rows.iterrows():
        context_block = row.get("golden_context_block")
        if context_block:
            parts.append(str(context_block))
            continue
        parts.append(
            "\n".join(
                [
                    f"Turn {int(row['turn'])}",
                    f"User: {row.get('golden_user_text', row['user_text'])}",
                    f"Assistant: {row.get('golden_assistant_text', row['assistant_text'])}",
                ]
            )
        )
    return "\n\n".join(parts)


def load_audio_bytes(audio_path: Path) -> bytes | None:
    if not audio_path.exists():
        return None
    return audio_path.read_bytes()


def load_knowledge_base_text(benchmark_name: str) -> tuple[str | None, Path]:
    knowledge_base_path = (
        Path.cwd() / "benchmarks" / benchmark_name / "data" / "knowledge_base.txt"
    )
    if not knowledge_base_path.exists():
        return None, knowledge_base_path
    return knowledge_base_path.read_text(encoding="utf-8"), knowledge_base_path


def step_selected_turn(turn_options: list[int], step: int) -> None:
    if "selected_turn" not in st.session_state or not turn_options:
        return
    current_turn = int(st.session_state["selected_turn"])
    if current_turn not in turn_options:
        st.session_state["selected_turn"] = turn_options[0]
        return
    current_index = turn_options.index(current_turn)
    next_index = max(0, min(len(turn_options) - 1, current_index + step))
    st.session_state["selected_turn"] = turn_options[next_index]


def build_dimension_pass_rate_df(payload: dict) -> pd.DataFrame:
    dimension_df = pd.DataFrame(payload["dimension_summary"]).copy()
    if dimension_df.empty:
        return dimension_df
    dimension_df["pass_rate"] = (
        pd.to_numeric(dimension_df["pass_rate"], errors="coerce").fillna(0.0) / 100.0
    )
    return dimension_df.sort_values("pass_rate", ascending=True).reset_index(drop=True)


def render_visualizations(rows_df: pd.DataFrame, filtered_df: pd.DataFrame, payload: dict) -> None:
    st.caption(
        f"Charts reflect the current filtered set when applicable. Filtered rows: {len(filtered_df)} of {len(rows_df)}."
    )

    status_counts = (
        filtered_df["overall_status"]
        .value_counts()
        .reindex(STATUS_ORDER, fill_value=0)
        .rename_axis("overall_status")
        .reset_index(name="count")
    )

    latency_df = filtered_df.copy()
    latency_df["latency_ms"] = pd.to_numeric(latency_df["latency_ms"], errors="coerce")
    latency_df["ttfb_ms"] = pd.to_numeric(latency_df["ttfb_ms"], errors="coerce")
    latency_df = latency_df.dropna(subset=["latency_ms", "ttfb_ms"])

    col1, col2 = st.columns(2)

    with col1:
        fig, axis = plt.subplots(figsize=(7, 4))
        axis.bar(
            status_counts["overall_status"],
            status_counts["count"],
            color=[STATUS_PALETTE[status] for status in status_counts["overall_status"]],
        )
        axis.set_title("Filtered Row Status Counts")
        axis.set_xlabel("")
        axis.set_ylabel("Rows")
        axis.grid(axis="y", alpha=0.25)
        st.pyplot(fig, clear_figure=True)

    with col2:
        if latency_df.empty:
            st.info("No numeric latency and TTFB values are available for the current filters.")
        else:
            fig, axis = plt.subplots(figsize=(7, 4))
            for status in STATUS_ORDER:
                status_df = latency_df[latency_df["overall_status"] == status]
                if status_df.empty:
                    continue
                axis.scatter(
                    status_df["ttfb_ms"],
                    status_df["latency_ms"],
                    label=status,
                    color=STATUS_PALETTE[status],
                    s=55,
                    alpha=0.8,
                )
            axis.set_title("TTFB vs Latency")
            axis.set_xlabel("TTFB (ms)")
            axis.set_ylabel("Latency (ms)")
            axis.grid(alpha=0.25)
            axis.legend(title="Status")
            st.pyplot(fig, clear_figure=True)

    lower_col, upper_col = st.columns(2)

    with lower_col:
        if latency_df.empty:
            st.info("No latency series is available for the current filters.")
        else:
            fig, axis = plt.subplots(figsize=(7, 4))
            sorted_latency_df = latency_df.sort_values("turn")
            for status in STATUS_ORDER:
                status_df = sorted_latency_df[sorted_latency_df["overall_status"] == status]
                if status_df.empty:
                    continue
                axis.plot(
                    status_df["turn"],
                    status_df["latency_ms"],
                    marker="o",
                    linewidth=1.8,
                    label=status,
                    color=STATUS_PALETTE[status],
                )
            axis.set_title("Latency by Turn")
            axis.set_xlabel("Turn")
            axis.set_ylabel("Latency (ms)")
            axis.grid(alpha=0.25)
            axis.legend(title="Status")
            st.pyplot(fig, clear_figure=True)

    with upper_col:
        dimension_df = build_dimension_pass_rate_df(payload)
        if dimension_df.empty:
            st.info("No dimension summary is available.")
        else:
            fig, axis = plt.subplots(figsize=(7, 4))
            axis.barh(
                dimension_df["dimension"],
                dimension_df["pass_rate"],
                color="#3b6ea8",
            )
            axis.set_title("Dimension Pass Rate")
            axis.set_xlabel("Pass rate")
            axis.set_ylabel("")
            axis.set_xlim(0, 1)
            axis.grid(axis="x", alpha=0.25)
            st.pyplot(fig, clear_figure=True)


def render_comparison_visuals(plot_paths: dict[str, Path]) -> None:
    if not plot_paths:
        st.info("No saved comparison plots were found.")
        return

    ordered_names = [
        "pass_rates_by_metric.png",
        "failed_turns.png",
        "latency_p95.png",
        "model_ended_session.png",
    ]
    available_plots = [plot_paths[name] for name in ordered_names if name in plot_paths]
    for index in range(0, len(available_plots), 2):
        columns = st.columns(2)
        for column, plot_path in zip(columns, available_plots[index : index + 2]):
            with column:
                st.markdown(f"**{plot_path.stem.replace('_', ' ').title()}**")
                st.image(str(plot_path), use_container_width=True)


def render_single_run_view(payload: dict, error_analysis: str, source_path: str) -> None:
    st.sidebar.caption(f"Loaded from: {source_path}")

    rows_df = pd.DataFrame(payload["rows"]).sort_values("turn").reset_index(drop=True)
    if rows_df.empty:
        st.warning("No rows found in the loaded payload.")
        return

    dimension = st.sidebar.selectbox(
        "Dimension",
        options=["all", *SCORE_COLUMNS],
        index=0,
    )
    status = st.sidebar.selectbox(
        "Status",
        options=["all", "fail", "pass", "incomplete"],
        index=0,
    )
    response_status = st.sidebar.selectbox(
        "Response status",
        options=["all", "normal", "empty_response"],
        index=0,
    )
    search_text = st.sidebar.text_input("Search")

    filtered_df = apply_filters(
        rows_df,
        dimension=dimension,
        status=status,
        response_status=response_status,
        search_text=search_text,
    )

    if filtered_df.empty:
        st.warning("No rows match the current filters.")
        return

    turn_options = filtered_df["turn"].astype(int).tolist()
    if (
        "selected_turn" not in st.session_state
        or st.session_state["selected_turn"] not in turn_options
    ):
        st.session_state["selected_turn"] = turn_options[0]

    st.sidebar.selectbox(
        "Selected turn",
        options=turn_options,
        key="selected_turn",
    )
    selected_turn = int(st.session_state["selected_turn"])

    summary_columns = st.columns(6)
    summary_columns[0].metric(
        "Rows",
        str(payload["metadata"]["turn_count"]),
        f"Filtered {len(filtered_df)}",
    )
    summary_columns[1].metric(
        "Overall Pass Rows",
        str(payload["summary"]["overall_pass_rows"]),
    )
    summary_columns[2].metric(
        "Empty Responses",
        str(payload["summary"]["empty_response_count"]),
    )
    summary_columns[3].metric(
        "Instruction",
        str(payload["summary"]["passes"]["instruction_following"]),
    )
    summary_columns[4].metric(
        "Grounding",
        str(payload["summary"]["passes"]["kb_grounding"]),
    )
    summary_columns[5].metric(
        "Tool Use",
        str(payload["summary"]["passes"]["tool_use_correct"]),
    )

    st.subheader("Run Metadata")
    metadata_df = pd.DataFrame(
        {
            "field": list(payload["metadata"].keys()),
            "value": [str(value) for value in payload["metadata"].values()],
        }
    )
    st.dataframe(
        metadata_df,
        use_container_width=True,
        hide_index=True,
    )

    selected_row = rows_df[rows_df["turn"] == selected_turn].iloc[0]
    previous_rows = rows_df[rows_df["turn"] < selected_turn].sort_values("turn")
    previous_turns_text = format_previous_turns(previous_rows)
    benchmark_name = payload["metadata"]["benchmark_name"]
    selected_turn_audio_path = turn_audio_path(benchmark_name, selected_turn)
    selected_turn_audio_bytes = load_audio_bytes(selected_turn_audio_path)
    knowledge_base_text, knowledge_base_path = load_knowledge_base_text(benchmark_name)
    conversation_audio_path = Path(
        payload["metadata"].get(
            "conversation_audio_path",
            str(Path(payload["metadata"]["source_run_dir"]) / "conversation.wav"),
        )
    )
    conversation_audio_bytes = load_audio_bytes(conversation_audio_path)

    selected_turn_index = turn_options.index(selected_turn)
    is_first_filtered_turn = selected_turn_index == 0
    is_last_filtered_turn = selected_turn_index == len(turn_options) - 1

    detail_tab, viz_tab, curated_tab, full_tab, summary_tab, analysis_tab = st.tabs(
        [
            "Selected Turn",
            "Visualizations",
            "Curated Failures",
            "All Datapoints",
            "Dimension Summary",
            "Error Analysis",
        ]
    )

    with detail_tab:
        st.subheader(f"Turn {selected_turn}")
        with st.expander(
            f"Gold previous turns context ({len(previous_rows)} prior turns)",
            expanded=True,
        ):
            st.text_area(
                "Conversation history",
                value=previous_turns_text,
                height=420,
                disabled=True,
                label_visibility="collapsed",
            )

        with st.expander("Knowledge base", expanded=False):
            st.caption(str(knowledge_base_path))
            if knowledge_base_text is None:
                st.info("No knowledge base file was found for this benchmark.")
            else:
                st.text_area(
                    "Knowledge base content",
                    value=knowledge_base_text,
                    height=420,
                    disabled=True,
                    label_visibility="collapsed",
                )

        st.markdown("**Current turn audio**")
        if selected_turn_audio_bytes is None:
            st.info("No per-turn input WAV found for this turn.")
        else:
            st.audio(selected_turn_audio_bytes, format="audio/wav")

        st.markdown("**Full conversation audio**")
        if conversation_audio_bytes is None:
            st.info("No run-level conversation.wav found.")
        else:
            st.audio(conversation_audio_bytes, format="audio/wav")

        content_col, detail_col = st.columns([3, 2])

        with content_col:
            st.markdown("**User query**")
            st.write(selected_row["user_text"])
            st.markdown("**Model output**")
            st.write(selected_row["assistant_text"])
            st.markdown("**Judge reasoning**")
            st.write(selected_row["judge_reasoning"])

        with detail_col:
            compact_metadata_df = pd.DataFrame(
                [
                    ["timestamp", str(selected_row["timestamp"])],
                    ["overall_status", str(selected_row["overall_status"])],
                    ["response_status", str(selected_row["response_status"])],
                    ["latency_ms", str(selected_row["latency_ms"])],
                    ["ttfb_ms", str(selected_row["ttfb_ms"])],
                    ["reconnection_count", str(selected_row["reconnection_count"])],
                    ["failed_dimensions", str(selected_row["failed_dimensions_text"] or "none")],
                ],
                columns=["field", "value"],
            )
            st.markdown("**Compact metadata**")
            st.dataframe(
                compact_metadata_df,
                use_container_width=True,
                hide_index=True,
            )

            score_rows: list[dict] = []
            for column in SCORE_COLUMNS:
                value = selected_row[column]
                if pd.isna(value):
                    verdict = "n/a"
                elif bool(value):
                    verdict = "pass"
                else:
                    verdict = "fail"
                score_rows.append({"dimension": column, "verdict": verdict})
            st.markdown("**Per-dimension verdicts**")
            st.dataframe(
                pd.DataFrame(score_rows),
                use_container_width=True,
                hide_index=True,
            )

            with st.expander("Tool calls", expanded=False):
                st.json(json.loads(selected_row["tool_calls_text"]))
            with st.expander("Tool results", expanded=False):
                st.json(json.loads(selected_row["tool_results_text"]))

        nav_prev_col, nav_status_col, nav_next_col = st.columns([1, 2, 1])
        with nav_prev_col:
            st.button(
                "Previous filtered turn",
                on_click=step_selected_turn,
                args=(turn_options, -1),
                disabled=is_first_filtered_turn,
                use_container_width=True,
            )
        with nav_status_col:
            st.caption(
                f"Filtered turn {selected_turn_index + 1} of {len(turn_options)}"
            )
        with nav_next_col:
            st.button(
                "Next filtered turn",
                on_click=step_selected_turn,
                args=(turn_options, 1),
                disabled=is_last_filtered_turn,
                use_container_width=True,
            )

    with viz_tab:
        render_visualizations(rows_df, filtered_df, payload)

    with curated_tab:
        curated_df = filtered_df[filtered_df["overall_status"] == "fail"].copy()
        curated_df = curated_df.sort_values("turn").head(25)
        st.caption(
            f"Showing {len(curated_df)} of {int((filtered_df['overall_status'] == 'fail').sum())} failing rows after filters. Sorted by turn."
        )
        if curated_df.empty:
            st.info("No failing rows match the current filters.")
        else:
            st.dataframe(
                curated_df[
                    [
                        "turn",
                        "overall_status",
                        "failed_dimensions_text",
                        "user_text",
                        "assistant_text",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
                height=520,
            )

    with full_tab:
        st.caption(
            f"Showing {len(filtered_df)} of {len(rows_df)} rows. Full table keeps failures first, then sorts by severity and turn."
        )
        st.dataframe(
            filtered_df[
                [
                    "turn",
                    "overall_status",
                    "response_status",
                    "latency_ms",
                    "failed_dimensions_text",
                    "user_text",
                    "assistant_text",
                ]
            ],
            use_container_width=True,
            hide_index=True,
            height=620,
        )

    with summary_tab:
        st.caption("Pass rate is computed over applicable rows for each dimension.")
        st.dataframe(
            pd.DataFrame(payload["dimension_summary"]),
            use_container_width=True,
            hide_index=True,
        )

    with analysis_tab:
        st.markdown(error_analysis)


def render_comparison_view(comparison_payload: dict, source_path: str) -> None:
    st.sidebar.caption(f"Loaded comparison from: {source_path}")
    metadata = comparison_payload["metadata"]
    run_results_df = comparison_payload["run_results"].copy()
    model_summary_df = comparison_payload["model_summary"].copy()
    metric_summary_df = comparison_payload["metric_summary"].copy()
    commands_df = comparison_payload["commands"].copy()

    model_options = ["all", *run_results_df["model_name"].dropna().unique().tolist()]
    selected_model = st.sidebar.selectbox("Model filter", options=model_options, index=0)
    selected_phase = "all"
    if not commands_df.empty:
        phase_options = ["all", *commands_df["phase"].dropna().unique().tolist()]
        selected_phase = st.sidebar.selectbox("Command phase", options=phase_options, index=0)

    if selected_model != "all":
        run_results_df = run_results_df[run_results_df["model_name"] == selected_model]
        if not model_summary_df.empty:
            model_summary_df = model_summary_df[model_summary_df["model_name"] == selected_model]
        if not metric_summary_df.empty:
            metric_summary_df = metric_summary_df[metric_summary_df["model_name"] == selected_model]
        if not commands_df.empty:
            commands_df = commands_df[commands_df["model_name"] == selected_model]

    if selected_phase != "all" and not commands_df.empty:
        commands_df = commands_df[commands_df["phase"] == selected_phase]

    st.subheader("Comparison Metadata")
    metadata_df = pd.DataFrame(
        {"field": list(metadata.keys()), "value": [str(value) for value in metadata.values()]}
    )
    st.dataframe(metadata_df, use_container_width=True, hide_index=True)

    top_columns = st.columns(5)
    top_columns[0].metric("Models", str(len(comparison_payload["metadata"].get("models", []))))
    top_columns[1].metric("Runs / model", str(comparison_payload["metadata"].get("runs_per_model", "?")))
    top_columns[2].metric("Total runs", str(len(comparison_payload["run_results"])))
    top_columns[3].metric(
        "Best pass rows",
        str(int(comparison_payload["run_results"]["pass_rows"].max())),
    )
    top_columns[4].metric(
        "Lowest failed rows",
        str(int(comparison_payload["run_results"]["fail_rows"].min())),
    )

    overview_tab, metrics_tab, plots_tab, commands_tab, inspect_tab, summary_tab = st.tabs(
        [
            "Run Results",
            "Metric Summary",
            "Plots",
            "Commands",
            "Inspect Run",
            "Summary",
        ]
    )

    with overview_tab:
        display_columns = [
            "model_name",
            "run_index",
            "run_id",
            "pass_rows",
            "fail_rows",
            "empty_response_count",
            "model_ended_session_count",
            "latency_ms_p95",
            "run_dir",
        ]
        available_columns = [column for column in display_columns if column in run_results_df.columns]
        st.dataframe(
            run_results_df[available_columns].sort_values(["model_name", "run_index"]),
            use_container_width=True,
            hide_index=True,
            height=420,
        )

        if not model_summary_df.empty:
            st.markdown("**Model summary**")
            summary_columns = [
                "model_name",
                "pass_rows_mean",
                "fail_rows_mean",
                "fail_rows_ci_lower_mean",
                "fail_rows_ci_upper_mean",
                "latency_ms_p95_mean",
                "latency_ms_p95_ci_lower_mean",
                "latency_ms_p95_ci_upper_mean",
            ]
            available_summary_columns = [
                column for column in summary_columns if column in model_summary_df.columns
            ]
            st.dataframe(
                model_summary_df[available_summary_columns].sort_values("model_name"),
                use_container_width=True,
                hide_index=True,
            )

    with metrics_tab:
        if metric_summary_df.empty:
            st.info("No metric summary CSV was found.")
        else:
            st.dataframe(
                metric_summary_df.sort_values(["metric", "model_name"]),
                use_container_width=True,
                hide_index=True,
                height=520,
            )

    with plots_tab:
        render_comparison_visuals(comparison_payload["plot_paths"])

    with commands_tab:
        if commands_df.empty:
            st.info("No commands.csv file was found.")
        else:
            st.dataframe(
                commands_df,
                use_container_width=True,
                hide_index=True,
                height=420,
            )

    with inspect_tab:
        if comparison_payload["run_results"].empty:
            st.info("No run results are available to inspect.")
        else:
            run_options = comparison_payload["run_results"]["run_dir"].dropna().tolist()
            if selected_model != "all":
                run_options = run_results_df["run_dir"].dropna().tolist()
            selected_run_dir = st.selectbox(
                "Underlying run directory",
                options=run_options,
                format_func=lambda value: Path(value).name,
            )
            st.caption(selected_run_dir)
            inspect_button = st.button("Load selected run", use_container_width=False)
            if inspect_button:
                st.session_state["comparison_selected_run_dir"] = selected_run_dir

            selected_run_dir = st.session_state.get(
                "comparison_selected_run_dir",
                selected_run_dir,
            )
            try:
                selected_run_payload, selected_error_analysis, selected_source_path = load_review_source(
                    selected_run_dir
                )
            except Exception as exc:
                st.error(f"Failed to load underlying run: {exc}")
            else:
                if selected_run_payload.get("kind") == "comparison":
                    st.error("Selected run unexpectedly resolved to comparison mode.")
                else:
                    render_single_run_view(
                        selected_run_payload,
                        selected_error_analysis,
                        selected_source_path,
                    )

    with summary_tab:
        st.markdown(comparison_payload["summary_markdown"])


def main() -> None:
    cli_args = parse_args()
    run_sources = discover_review_sources()
    default_input_path = cli_args.input_path or find_default_input_path()

    st.set_page_config(
        page_title="ConversationBench Review",
        page_icon="🎧",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("ConversationBench Review App")
    st.caption(
        "Load a generated review directory or a judged run directory. The app keeps the selected turn in session state and shows all prior turns for context."
    )

    input_options = [run_source["input_path"] for run_source in run_sources]
    label_by_path = {
        run_source["input_path"]: run_source["label"] for run_source in run_sources
    }

    if default_input_path and default_input_path not in label_by_path:
        input_options = [default_input_path, *input_options]
        label_by_path[default_input_path] = f"Custom | {default_input_path}"

    selected_index = 0
    if default_input_path in input_options:
        selected_index = input_options.index(default_input_path)

    if run_sources:
        input_path = st.sidebar.selectbox(
            "Review directory, review payload, or judged run directory",
            options=input_options,
            index=selected_index,
            format_func=lambda path: label_by_path[path],
        )
    else:
        input_path = default_input_path

    if not input_path.strip():
        st.info(
            "No review sources were found under results/ or runs/."
        )
        return

    try:
        payload, error_analysis, source_path = load_review_source(input_path)
    except Exception as exc:
        st.error(str(exc))
        return

    if payload.get("kind") == "comparison":
        render_comparison_view(payload, source_path)
    else:
        render_single_run_view(payload, error_analysis, source_path)


if __name__ == "__main__":
    main()
