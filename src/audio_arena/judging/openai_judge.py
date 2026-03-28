"""
OpenAI-based transcript judge (mode-aware realignment + over-clarification handling).

Mirrors the Claude judge but uses OpenAI's Responses API.

Usage via CLI:
    uv run audio-arena judge runs/grocery_bench/20251215T202910_gpt-4o-... --judge openai
    uv run audio-arena judge runs/... --judge openai --judge-model o3
"""

import json
import hashlib
import os
import sys
import time
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional

from .llm_judge import (
    apply_precomputed_oracle_tool_use,
    build_rehydrated_turn_prompt_bundles,
    build_judge_summary,
    build_judge_system_prompt,
    build_judge_user_prompt,
    filter_runtime_failure_records,
    format_turns_for_judge,
    format_rehydrated_turns_for_judge,
    get_turn_taking_support,
    load_transcript,
    uses_cross_turn_realignment,
)


OPENAI_JUDGE_VERSION = "openai-v10-kb-visible-vs-tool-only"
OPENAI_REHYDRATED_JUDGE_VERSION = "openai-v11-rehydrated-oracle-continuation"
OPENAI_JUDGE_MODEL = "gpt-5.2"
OPENAI_JUDGE_TIMEOUT_ENV_VAR = "AUDIO_ARENA_OPENAI_JUDGE_TIMEOUT_SECONDS"
OPENAI_JUDGE_TIMEOUT_SECONDS = 120.0
OPENAI_JUDGE_CONCURRENCY_ENV_VAR = "AUDIO_ARENA_OPENAI_JUDGE_CONCURRENCY"
OPENAI_JUDGE_CONCURRENCY = 8
OPENAI_JUDGE_SERVICE_TIER = "priority"
OPENAI_JUDGE_PROMPT_CACHE_RETENTION = "24h"

OPENAI_JUDGE_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "name": "judge_output",
    "strict": False,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "final_judgments",
            "realignment_notes",
            "function_call_tracking",
        ],
        "properties": {
            "final_judgments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "turn",
                        "reasoning",
                        "turn_taking",
                        "tool_use_correct",
                        "instruction_following",
                        "kb_grounding",
                        "ambiguity_handling",
                        "state_tracking",
                    ],
                    "properties": {
                        "turn": {"type": "integer"},
                        "reasoning": {"type": "string"},
                        "turn_taking": {"type": "boolean"},
                        "tool_use_correct": {"type": ["boolean", "null"]},
                        "instruction_following": {"type": "boolean"},
                        "kb_grounding": {"type": "boolean"},
                        "ambiguity_handling": {"type": ["boolean", "null"]},
                        "state_tracking": {"type": ["boolean", "null"]},
                    },
                },
            },
            "realignment_notes": {"type": "string"},
            "function_call_tracking": {
                "type": "object",
                "additionalProperties": True,
            },
        },
    },
}


def get_openai_judge_timeout_seconds() -> float:
    raw_timeout = os.getenv(OPENAI_JUDGE_TIMEOUT_ENV_VAR)
    if raw_timeout is None:
        return OPENAI_JUDGE_TIMEOUT_SECONDS

    try:
        timeout_seconds = float(raw_timeout)
    except ValueError:
        print(
            f"Ignoring invalid {OPENAI_JUDGE_TIMEOUT_ENV_VAR}={raw_timeout!r}; "
            f"using default {OPENAI_JUDGE_TIMEOUT_SECONDS:.1f}s.",
            file=sys.stderr,
        )
        return OPENAI_JUDGE_TIMEOUT_SECONDS

    if timeout_seconds <= 0:
        print(
            f"Ignoring non-positive {OPENAI_JUDGE_TIMEOUT_ENV_VAR}={raw_timeout!r}; "
            f"using default {OPENAI_JUDGE_TIMEOUT_SECONDS:.1f}s.",
            file=sys.stderr,
        )
        return OPENAI_JUDGE_TIMEOUT_SECONDS

    return timeout_seconds


def get_openai_judge_concurrency() -> int:
    raw_concurrency = os.getenv(OPENAI_JUDGE_CONCURRENCY_ENV_VAR)
    if raw_concurrency is None:
        return OPENAI_JUDGE_CONCURRENCY

    try:
        concurrency = int(raw_concurrency)
    except ValueError:
        print(
            f"Ignoring invalid {OPENAI_JUDGE_CONCURRENCY_ENV_VAR}={raw_concurrency!r}; "
            f"using default {OPENAI_JUDGE_CONCURRENCY}.",
            file=sys.stderr,
        )
        return OPENAI_JUDGE_CONCURRENCY

    if concurrency <= 0:
        print(
            f"Ignoring non-positive {OPENAI_JUDGE_CONCURRENCY_ENV_VAR}={raw_concurrency!r}; "
            f"using default {OPENAI_JUDGE_CONCURRENCY}.",
            file=sys.stderr,
        )
        return OPENAI_JUDGE_CONCURRENCY

    return concurrency


def build_openai_judge_prompt_cache_key(
    run_dir: Path,
    judge_version: str,
    judge_model: str,
    cross_turn_realignment: bool,
) -> str:
    benchmark_name = run_dir.parent.name
    mode = "cross_turn" if cross_turn_realignment else "rehydrated"
    raw_key = f"{judge_version}|{judge_model}|{benchmark_name}|{mode}"
    key_hash = hashlib.sha1(raw_key.encode("utf-8")).hexdigest()[:16]
    return f"aa-judge:{benchmark_name}:{mode}:{key_hash}"


def build_openai_responses_request_kwargs(
    *,
    judge_model: str,
    system_prompt: str,
    prompt: str,
    prompt_cache_key: str,
) -> Dict[str, Any]:
    return {
        "model": judge_model,
        "instructions": system_prompt,
        "input": prompt,
        "temperature": 0,
        "service_tier": OPENAI_JUDGE_SERVICE_TIER,
        "prompt_cache_key": prompt_cache_key,
        "prompt_cache_retention": OPENAI_JUDGE_PROMPT_CACHE_RETENTION,
        "text": {"format": OPENAI_JUDGE_RESPONSE_SCHEMA},
    }


def _parse_openai_judge_response(response_text: str) -> Dict[str, Any]:
    """Extract the judge JSON object from a Responses API result."""
    json_start = response_text.find('{')
    json_end = response_text.rfind('}') + 1

    if json_start == -1 or json_end == 0:
        raise ValueError(f"No JSON found in response: {response_text[:500]}")

    json_str = response_text[json_start:json_end]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response: {e}") from e


async def judge_with_openai(
    run_dir: Path,
    only_turns: Optional[set[int]] = None,
    debug: bool = False,
    expected_turns: Optional[List[Dict[str, Any]]] = None,
    skip_turn_taking: bool = False,
    get_relevant_dimensions_fn=None,
    model: Optional[str] = None,
    kb_text: Optional[str] = None,
    prompt_visible_kb_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Main judging function using OpenAI with mode-aware scoring.

    Args:
        run_dir: Path to the run directory containing transcript.jsonl
        only_turns: Optional set of turn indices to judge
        debug: Enable debug logging
        expected_turns: Optional list of expected turns. If not provided, imports from turns module.
        skip_turn_taking: If True, skip turn-taking analysis
        get_relevant_dimensions_fn: Function to get relevant scoring dimensions for a turn.
        model: OpenAI model to use. Defaults to OPENAI_JUDGE_MODEL.
        kb_text: Optional full oracle knowledge base text for kb_grounding verification.
        prompt_visible_kb_text: Optional prompt-visible KB text that the assistant saw.

    Returns:
        Dict with judgments, realignment_notes, function_tracking, turn_taking_analysis, summary, and model_name.
    """
    try:
        from openai import AsyncOpenAI
    except ImportError:
        print("ERROR: openai package not installed.", file=sys.stderr)
        print("Install with: uv add openai", file=sys.stderr)
        sys.exit(1)

    judge_model = model or OPENAI_JUDGE_MODEL

    records = load_transcript(run_dir)

    if expected_turns is None:
        from benchmarks.conversation_bench.turns import turns as expected_turns

    if only_turns is not None:
        records = [r for r in records if r["turn"] in only_turns]

    records, runtime_excluded_turns = filter_runtime_failure_records(records, run_dir)

    if not records:
        raise ValueError("No turns to judge after excluding runtime-failure turns.")

    model_name = records[0].get("model_name", "unknown")

    cross_turn_realignment = uses_cross_turn_realignment(run_dir)
    judge_version = (
        OPENAI_JUDGE_VERSION if cross_turn_realignment else OPENAI_REHYDRATED_JUDGE_VERSION
    )

    if debug:
        mode_label = "with cross-turn realignment" if cross_turn_realignment else "without cross-turn realignment"
        print(f"Judging {len(records)} turns {mode_label} with OpenAI ({judge_model})...", file=sys.stderr)

    # Run turn-taking analysis when parent-level audio evidence is available.
    turn_taking_data: Optional[Dict[int, Dict[str, Any]]] = None
    turn_taking_analysis = None
    turn_taking_supported, turn_taking_skip_reason = get_turn_taking_support(
        run_dir, skip_turn_taking
    )
    if turn_taking_supported:
        if debug:
            print("Running turn-taking analysis...", file=sys.stderr)
        try:
            from .turn_taking import analyze_turn_taking
            turn_taking_analysis = analyze_turn_taking(run_dir)
            if turn_taking_analysis.error:
                if debug:
                    print(f"Turn-taking analysis error: {turn_taking_analysis.error}", file=sys.stderr)
            else:
                turn_taking_data = {
                    idx: result.to_dict()
                    for idx, result in turn_taking_analysis.per_turn.items()
                }
                if debug and turn_taking_analysis.failed_turns:
                    print(f"Turn-taking failures: {turn_taking_analysis.failed_turns}", file=sys.stderr)
        except Exception as e:
            if debug:
                print(f"Turn-taking analysis failed: {e}", file=sys.stderr)
    elif debug and turn_taking_skip_reason:
        print(f"Turn-taking analysis skipped: {turn_taking_skip_reason}", file=sys.stderr)

    system_prompt = build_judge_system_prompt(cross_turn_realignment)

    client = AsyncOpenAI(timeout=get_openai_judge_timeout_seconds())

    async def _request_judgment(prompt: str) -> Dict[str, Any]:
        prompt_cache_key = build_openai_judge_prompt_cache_key(
            run_dir=run_dir,
            judge_version=judge_version,
            judge_model=judge_model,
            cross_turn_realignment=cross_turn_realignment,
        )
        kwargs = build_openai_responses_request_kwargs(
            judge_model=judge_model,
            system_prompt=system_prompt,
            prompt=prompt,
            prompt_cache_key=prompt_cache_key,
        )

        if debug:
            print(
                f"Sending Responses API request to OpenAI ({judge_model})...",
                file=sys.stderr,
            )

        response = await client.responses.create(**kwargs)
        response_text = response.output_text or ""

        if debug:
            print(f"OpenAI response length: {len(response_text)} chars", file=sys.stderr)
            if response.usage:
                print(
                    f"Tokens: {response.usage.input_tokens} input, {response.usage.output_tokens} output",
                    file=sys.stderr,
                )

        try:
            return _parse_openai_judge_response(response_text)
        except ValueError as e:
            if debug:
                print(f"JSON parse error: {e}", file=sys.stderr)
                print(f"Attempted to parse: {response_text[:500]}...", file=sys.stderr)
            raise

    if cross_turn_realignment:
        formatter = format_turns_for_judge
        formatted_turns = formatter(
            records, expected_turns, only_turns, turn_taking_data,
            get_relevant_dimensions_fn, kb_text=kb_text,
            prompt_visible_kb_text=prompt_visible_kb_text,
        )
        prompt = build_judge_user_prompt(
            formatted_turns,
            [record["turn"] for record in records],
            cross_turn_realignment,
        )
        request_started = time.perf_counter()
        print(
            f"[openai-judge] Requesting combined judgment for {len(records)} turns...",
            file=sys.stderr,
            flush=True,
        )
        result = await _request_judgment(prompt)
        print(
            f"[openai-judge] Combined judgment finished in {time.perf_counter() - request_started:.1f}s.",
            file=sys.stderr,
            flush=True,
        )
        final_judgments = result.get('final_judgments', [])
        realignment_notes = result.get('realignment_notes', '')
        function_tracking = result.get('function_call_tracking', {})
    else:
        final_judgments = []
        realignment_notes = "Cross-turn realignment disabled for rehydrated run."
        function_tracking = {}
        prompt_bundles = build_rehydrated_turn_prompt_bundles(
            records,
            expected_turns,
            turn_taking_data=turn_taking_data,
            get_relevant_dimensions_fn=get_relevant_dimensions_fn,
            kb_text=kb_text,
            prompt_visible_kb_text=prompt_visible_kb_text,
        )
        total_bundles = len(prompt_bundles)
        concurrency = min(get_openai_judge_concurrency(), total_bundles)
        semaphore = asyncio.Semaphore(concurrency)
        print(
            f"[openai-judge] Requesting {total_bundles} rehydrated turn judgments with concurrency={concurrency}...",
            file=sys.stderr,
            flush=True,
        )

        async def _judge_rehydrated_bundle(
            bundle_index: int,
            bundle: Dict[str, Any],
        ) -> Dict[str, Any]:
            async with semaphore:
                if debug:
                    print(
                        f"Judging rehydrated turn {bundle['turn']} in isolation...",
                        file=sys.stderr,
                    )
                print(
                    f"[openai-judge] {bundle_index}/{total_bundles} judging rehydrated turn {bundle['turn']}...",
                    file=sys.stderr,
                    flush=True,
                )
                request_started = time.perf_counter()
                result = await _request_judgment(bundle["prompt"])
                print(
                    f"[openai-judge] {bundle_index}/{total_bundles} finished turn {bundle['turn']} in {time.perf_counter() - request_started:.1f}s.",
                    file=sys.stderr,
                    flush=True,
                )
                judgments_for_turn = result.get("final_judgments", [])
                if len(judgments_for_turn) != 1:
                    raise ValueError(
                        f"Expected exactly 1 judgment for rehydrated turn {bundle['turn']}, got {len(judgments_for_turn)}"
                    )
                return {
                    "bundle_index": bundle_index,
                    "turn": bundle["turn"],
                    "judgment": judgments_for_turn[0],
                }

        bundle_tasks = [
            asyncio.create_task(_judge_rehydrated_bundle(bundle_index, bundle))
            for bundle_index, bundle in enumerate(prompt_bundles, start=1)
        ]
        bundle_results = await asyncio.gather(*bundle_tasks)
        bundle_results.sort(key=lambda result: result["bundle_index"])

        for bundle_result in bundle_results:
            final_judgments.append(bundle_result["judgment"])

    apply_precomputed_oracle_tool_use(final_judgments, records)

    if debug:
        print(f"\nRealignment notes: {realignment_notes}", file=sys.stderr)
        print(f"Function tracking: {json.dumps(function_tracking, indent=2)}", file=sys.stderr)

    judgments = {}
    for j in final_judgments:
        turn_num = j.get('turn')
        if turn_num is not None:
            turn_taking = j.get('turn_taking', True)

            if turn_taking_data and turn_num in turn_taking_data:
                turn_taking = turn_taking_data[turn_num].get('turn_taking', True)

            ambiguity = j.get('ambiguity_handling')
            state = j.get('state_tracking')

            judgments[turn_num] = {
                "scores": {
                    "turn_taking": turn_taking,
                    "tool_use_correct": j.get('tool_use_correct'),
                    "instruction_following": j.get('instruction_following', False),
                    "kb_grounding": j.get('kb_grounding', False),
                    "ambiguity_handling": ambiguity,
                    "state_tracking": state,
                },
                "reasoning": j.get('reasoning', ''),
            }

            if turn_taking_data and turn_num in turn_taking_data:
                issues = turn_taking_data[turn_num].get('issues', [])
                if issues:
                    judgments[turn_num]["turn_taking_issues"] = issues

    expected_turn_numbers = {r["turn"] for r in records}
    judged_turn_numbers = set(judgments.keys())
    missing = expected_turn_numbers - judged_turn_numbers

    if missing:
        raise ValueError(
            f"Failed to get judgments for turns: {sorted(missing)}. "
            f"Expected {len(expected_turn_numbers)} judgments, got {len(judgments)}."
        )

    return {
        "judgments": judgments,
        "realignment_notes": realignment_notes,
        "function_tracking": function_tracking,
        "cross_turn_realignment_applied": cross_turn_realignment,
        "turn_taking_analysis": turn_taking_analysis.to_dict() if turn_taking_analysis else None,
        "summary": build_judge_summary(len(judgments), cross_turn_realignment),
        "model_name": model_name,
        "judge_model": judge_model,
        "judge_version": judge_version,
        "turn_taking_supported": turn_taking_supported,
        "turn_taking_skip_reason": turn_taking_skip_reason,
        "runtime_excluded_turns": runtime_excluded_turns,
    }
