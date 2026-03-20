"""
OpenAI-based transcript judge (mode-aware realignment + over-clarification handling).

Mirrors the Claude judge but uses OpenAI's chat completions API.

Usage via CLI:
    uv run audio-arena judge runs/grocery_bench/20251215T202910_gpt-4o-... --judge openai
    uv run audio-arena judge runs/... --judge openai --judge-model o3
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from .llm_judge import (
    build_rehydrated_turn_prompt_bundles,
    build_judge_summary,
    build_judge_system_prompt,
    build_judge_user_prompt,
    format_turns_for_judge,
    format_rehydrated_turns_for_judge,
    get_turn_taking_support,
    load_transcript,
    uses_cross_turn_realignment,
)


OPENAI_JUDGE_VERSION = "openai-v10-kb-visible-vs-tool-only"
OPENAI_REHYDRATED_JUDGE_VERSION = "openai-v11-rehydrated-oracle-continuation"
OPENAI_JUDGE_MODEL = "gpt-5.2"


def _parse_openai_judge_response(response_text: str) -> Dict[str, Any]:
    """Extract the judge JSON object from a chat completion response."""
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

    if not records:
        raise ValueError("No turns to judge")

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

    client = AsyncOpenAI()

    # o3 / o-series models use developer messages instead of system messages
    is_o_series = judge_model.startswith("o")

    def _build_messages(prompt: str) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if is_o_series:
            messages.append({"role": "developer", "content": system_prompt})
        else:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    async def _request_judgment(prompt: str) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": judge_model,
            "messages": _build_messages(prompt),
        }
        if is_o_series:
            kwargs["reasoning_effort"] = "high"
            kwargs["response_format"] = {"type": "json_object"}
        else:
            kwargs["response_format"] = {"type": "json_object"}
            kwargs["temperature"] = 0

        if debug:
            print(f"Sending request to OpenAI ({judge_model})...", file=sys.stderr)

        response = await client.chat.completions.create(**kwargs)
        response_text = response.choices[0].message.content or ""

        if debug:
            print(f"OpenAI response length: {len(response_text)} chars", file=sys.stderr)
            if response.usage:
                print(
                    f"Tokens: {response.usage.prompt_tokens} prompt, {response.usage.completion_tokens} completion",
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
        result = await _request_judgment(prompt)
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
        for bundle in prompt_bundles:
            if debug:
                print(f"Judging rehydrated turn {bundle['turn']} in isolation...", file=sys.stderr)
            result = await _request_judgment(bundle["prompt"])
            judgments_for_turn = result.get("final_judgments", [])
            if len(judgments_for_turn) != 1:
                raise ValueError(
                    f"Expected exactly 1 judgment for rehydrated turn {bundle['turn']}, got {len(judgments_for_turn)}"
                )
            final_judgments.extend(judgments_for_turn)

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
    }
