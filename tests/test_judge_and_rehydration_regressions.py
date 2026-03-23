import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.openai.realtime import events as rt_events

from audio_arena.judging.llm_judge import (
    build_rehydrated_turn_prompt_bundles,
    build_judge_system_prompt,
    build_judge_user_prompt,
    format_rehydrated_turns_for_judge,
    format_turns_for_judge,
    get_turn_taking_support,
    write_outputs,
)
from audio_arena.cli import (
    build_live_tool_capture_from_transcript,
    build_missing_tool_capture,
    build_oracle_continuation_history,
    merge_oracle_continuation_artifacts,
    turn_uses_oracle_post_tool_continuation,
)
from benchmarks.appointment_bench.turns import turns as appointment_turns
from benchmarks.assistant_bench.turns import turns as assistant_turns
from benchmarks.conversation_bench.turns import turns as conversation_turns
from audio_arena.pipelines.openai_realtime import OpenAIRealtimeLLMServiceExplicitToolResult
from audio_arena.pipelines.realtime import RealtimePipeline
from audio_arena.pipelines.text import TextPipeline
from benchmarks.product_bench.turns import turns as product_turns


class DummyBenchmark:
    def __init__(self):
        self.system_instruction = "Base system prompt."
        self.tools_schema = ToolsSchema(standard_tools=[])
        self.turns = [{"input": "target turn"}]


class JudgeAndRehydrationRegressionTests(unittest.TestCase):
    def test_product_bench_checkout_flow_only_requests_phone_before_add_to_cart(self):
        turn_20 = product_turns[20]
        turn_21 = product_turns[21]

        self.assertIn("phone number", turn_20["golden_text"].lower())
        self.assertNotIn("campus address", turn_20["golden_text"].lower())
        self.assertEqual(
            turn_21["required_function_call"],
            {
                "name": "add_to_cart",
                "args": {
                    "product_id": "X1940",
                    "customer_name": "Ryan Chen",
                    "phone": "201-473-1560",
                    "condition": "open_box",
                    "warranty": "accidental_damage",
                    "student_discount": False,
                },
            },
        )

    def test_freeform_gt_args_are_trimmed_for_subset_matching(self):
        self.assertEqual(
            assistant_turns[6]["required_function_call"]["args"],
            {"to": "alex.reed@meridian.com"},
        )
        self.assertEqual(
            assistant_turns[13]["required_function_call"]["args"],
            {"to": "maria.torres@meridian.com"},
        )
        self.assertEqual(
            conversation_turns[10]["required_function_call"]["args"],
            {"name": "Jennifer Smith"},
        )
        self.assertEqual(
            conversation_turns[15]["required_function_call"]["args"],
            {"name": "Jennifer Smith"},
        )

    def test_multi_call_tool_responses_match_by_args_not_call_order(self):
        benchmark = DummyBenchmark()
        benchmark.turns = [
            {
                "input": "register me for all the Voice sessions",
                "required_function_call": [
                    {"name": "register_for_session", "args": {"name": "Jennifer Smith", "session_id": "920101"}},
                    {"name": "register_for_session", "args": {"name": "Jennifer Smith", "session_id": "920102"}},
                    {"name": "register_for_session", "args": {"name": "Jennifer Smith", "session_id": "920103"}},
                ],
                "function_call_response": [
                    {"status": "success"},
                    {"status": "error", "error_code": "SESSION_FULL"},
                    {"status": "error", "error_code": "SCHEDULE_CONFLICT"},
                ],
            }
        ]
        pipeline = TextPipeline(benchmark)

        first = pipeline._get_turn_tool_response(
            "register_for_session",
            {"name": "Jennifer Smith", "session_id": "920103"},
        )
        second = pipeline._get_turn_tool_response(
            "register_for_session",
            {"name": "Jennifer Smith", "session_id": "920101"},
        )
        third = pipeline._get_turn_tool_response(
            "register_for_session",
            {"name": "Jennifer Smith", "session_id": "920102"},
        )

        self.assertEqual(first, {"status": "error", "error_code": "SCHEDULE_CONFLICT"})
        self.assertEqual(second, {"status": "success"})
        self.assertEqual(third, {"status": "error", "error_code": "SESSION_FULL"})

    def test_wrong_tool_on_single_call_turn_returns_unexpected_tool_error(self):
        benchmark = DummyBenchmark()
        benchmark.turns = [
            {
                "input": "check availability",
                "required_function_call": {
                    "name": "search_venues",
                    "args": {"date": "2025-03-15", "guest_count": 90},
                },
                "function_call_response": {
                    "status": "success",
                    "venues": [{"venue_id": "garden_pavilion"}],
                },
            }
        ]
        pipeline = TextPipeline(benchmark)

        result = pipeline._get_turn_tool_response(
            "cancel_booking",
            {"booking_id": "BK-1"},
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_code"], "UNEXPECTED_TOOL_CALL")
        self.assertEqual(result["called_function"], "cancel_booking")
        self.assertEqual(result["expected_function"], "search_venues")

    def test_unexpected_tool_on_no_tool_turn_returns_unexpected_tool_error(self):
        benchmark = DummyBenchmark()
        benchmark.turns = [
            {
                "input": "what time does it start?",
                "required_function_call": None,
            }
        ]
        pipeline = TextPipeline(benchmark)

        result = pipeline._get_turn_tool_response(
            "lookup_item",
            {"query": "flowers"},
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_code"], "UNEXPECTED_TOOL_CALL")
        self.assertEqual(result["called_function"], "lookup_item")
        self.assertIsNone(result["expected_function"])

    def test_wrong_args_on_single_call_turn_returns_arg_mismatch(self):
        benchmark = DummyBenchmark()
        benchmark.turns = [
            {
                "input": "check availability",
                "required_function_call": {
                    "name": "search_venues",
                    "args": {"date": "2025-03-15", "guest_count": 90},
                },
                "function_call_response": {
                    "status": "success",
                    "venues": [{"venue_id": "garden_pavilion"}],
                },
            }
        ]
        pipeline = TextPipeline(benchmark)

        result = pipeline._get_turn_tool_response(
            "search_venues",
            {"date": "2025-03-16", "guest_count": 90},
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_code"], "ARG_MISMATCH")
        self.assertEqual(result["called_function"], "search_venues")
        self.assertEqual(result["expected_function"], "search_venues")

    def test_subset_matching_allows_extra_freeform_tool_args(self):
        benchmark = DummyBenchmark()
        benchmark.turns = [
            {
                "input": "email maria",
                "required_function_call": {
                    "name": "send_email",
                    "args": {"to": "maria.torres@meridian.com"},
                },
                "function_call_response": {"status": "success", "message_id": "MSG-1"},
            }
        ]
        pipeline = TextPipeline(benchmark)

        result = pipeline._get_turn_tool_response(
            "send_email",
            {
                "to": "maria.torres@meridian.com",
                "subject": "Horizon Project Deadline Update — Correction",
                "body": "Hi Maria, correction to my last email. The deadline is February 15th.",
            },
        )

        self.assertEqual(result, {"status": "success", "message_id": "MSG-1"})

    def test_subset_matching_still_requires_expected_fields(self):
        benchmark = DummyBenchmark()
        benchmark.turns = [
            {
                "input": "email maria",
                "required_function_call": {
                    "name": "send_email",
                    "args": {"to": "maria.torres@meridian.com", "subject": "Required subject"},
                },
                "function_call_response": {"status": "success", "message_id": "MSG-1"},
            }
        ]
        pipeline = TextPipeline(benchmark)

        result = pipeline._get_turn_tool_response(
            "send_email",
            {"to": "maria.torres@meridian.com"},
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_code"], "ARG_MISMATCH")

    def test_end_session_without_scripted_response_uses_implicit_success(self):
        benchmark = DummyBenchmark()
        benchmark.turns = [
            {
                "input": "thanks bye",
                "required_function_call": {
                    "name": "end_session",
                    "args": {},
                },
                "function_call_response": None,
            }
        ]
        pipeline = TextPipeline(benchmark)

        result = pipeline._get_turn_tool_response("end_session", {})

        self.assertEqual(result, {"status": "success"})

    def test_oracle_post_tool_continuation_applies_to_multi_call_realtime_turns(self):
        turn = {
            "input": "do both actions",
            "required_function_call": [
                {"name": "tool_a", "args": {"value": "1"}},
                {"name": "tool_b", "args": {"value": "2"}},
            ],
            "function_call_response": [
                {"status": "success"},
                {"status": "success"},
            ],
        }

        self.assertTrue(
            turn_uses_oracle_post_tool_continuation(
                pipeline_type="realtime",
                turn=turn,
            )
        )

    def test_build_live_tool_capture_from_transcript_for_multi_call_turn(self):
        turn = {
            "input": "do both actions",
            "required_function_call": [
                {"name": "tool_a", "args": {"value": "1"}},
                {"name": "tool_b", "args": {"value": "2"}},
            ],
            "function_call_response": [
                {"status": "success"},
                {"status": "error", "error_code": "SESSION_FULL"},
            ],
        }

        with TemporaryDirectory() as tmpdir:
            transcript_path = Path(tmpdir) / "transcript.jsonl"
            transcript_path.write_text(
                json.dumps(
                    {
                        "turn": 0,
                        "user_text": turn["input"],
                        "assistant_text": "handled",
                        "tool_calls": [
                            {"name": "tool_a", "args": {"value": "1"}},
                            {"name": "tool_b", "args": {"value": "2"}},
                        ],
                        "tool_results": [
                            {"name": "tool_a", "response": {"status": "success"}},
                            {
                                "name": "tool_b",
                                "response": {
                                    "status": "error",
                                    "error_code": "SESSION_FULL",
                                },
                            },
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            capture = build_live_tool_capture_from_transcript(
                turn=turn,
                transcript_path=transcript_path,
            )

        self.assertTrue(capture["tool_name_correct"])
        self.assertTrue(capture["tool_args_correct"])
        self.assertTrue(capture["tool_use_pass"])
        self.assertEqual(len(capture["actual_tool_calls"]), 2)
        self.assertEqual(
            capture["oracle_tool_results"][1]["error_code"],
            "SESSION_FULL",
        )

    def test_merge_oracle_continuation_artifacts_preserves_multi_call_live_capture(self):
        with TemporaryDirectory() as tmpdir:
            transcript_path = Path(tmpdir) / "transcript.jsonl"
            transcript_path.write_text(
                json.dumps(
                    {
                        "turn": 38,
                        "user_text": "register me with fallback",
                        "assistant_text": "oracle continuation answer",
                        "tool_calls": [],
                        "tool_results": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            merge_oracle_continuation_artifacts(
                transcript_path=transcript_path,
                live_tool_capture={
                    "actual_tool_calls": [
                        {
                            "name": "register_for_session",
                            "args": {"name": "Jennifer Smith", "session_id": "923101"},
                        },
                        {
                            "name": "register_for_session",
                            "args": {"name": "Jennifer Smith", "session_id": "923103"},
                        },
                    ],
                    "actual_tool_results": [
                        {
                            "name": "register_for_session",
                            "response": {
                                "status": "error",
                                "error_code": "SESSION_FULL",
                            },
                        },
                        {
                            "name": "register_for_session",
                            "response": {
                                "status": "error",
                                "error_code": "SESSION_FULL",
                            },
                        },
                    ],
                    "tool_name_correct": True,
                    "tool_args_correct": True,
                    "tool_use_pass": True,
                    "oracle_tool_calls": [
                        {
                            "name": "register_for_session",
                            "args": {"name": "Jennifer Smith", "session_id": "923101"},
                        },
                        {
                            "name": "register_for_session",
                            "args": {"name": "Jennifer Smith", "session_id": "923103"},
                        },
                    ],
                    "oracle_tool_results": [
                        {"status": "error", "error_code": "SESSION_FULL"},
                        {"status": "error", "error_code": "SESSION_FULL"},
                    ],
                },
            )

            merged = json.loads(transcript_path.read_text(encoding="utf-8").strip())

        self.assertEqual(len(merged["tool_calls"]), 2)
        self.assertEqual(merged["tool_calls"][1]["args"]["session_id"], "923103")
        self.assertEqual(len(merged["tool_results"]), 2)
        self.assertTrue(merged["oracle_continuation"]["used"])
        self.assertTrue(merged["oracle_continuation"]["tool_use_pass"])

    def test_time_args_match_when_hour_zero_padding_differs(self):
        benchmark = DummyBenchmark()
        benchmark.turns = [
            {
                "input": "book nine-fifteen",
                "required_function_call": {
                    "name": "book_appointment",
                    "args": {"appointment_id": "APT-001", "time": "09:15"},
                },
                "function_call_response": {"status": "success"},
            }
        ]
        pipeline = TextPipeline(benchmark)

        result = pipeline._get_turn_tool_response(
            "book_appointment",
            {"appointment_id": "APT-001", "time": "9:15"},
        )

        self.assertEqual(result, {"status": "success"})

    def test_product_ids_match_regardless_of_order(self):
        benchmark = DummyBenchmark()
        benchmark.turns = [
            {
                "input": "compare these laptops",
                "required_function_call": {
                    "name": "compare_specs",
                    "args": {"product_ids": ["X1490", "S15Pro"]},
                },
                "function_call_response": {"status": "success"},
            }
        ]
        pipeline = TextPipeline(benchmark)

        result = pipeline._get_turn_tool_response(
            "compare_specs",
            {"product_ids": ["S15Pro", "X1490"]},
        )

        self.assertEqual(result, {"status": "success"})

    def test_normalized_tool_args_treat_reordered_dicts_as_identical(self):
        args_a = {"session_id": "920101", "name": "Jennifer Smith"}
        args_b = {"name": "Jennifer Smith", "session_id": "920101"}

        self.assertEqual(
            TextPipeline._normalize_tool_args(args_a),
            TextPipeline._normalize_tool_args(args_b),
        )

    def test_write_outputs_counts_unexpected_tool_call_as_tool_failure(self):
        records = [
            {
                "turn": 0,
                "model_name": "test-model",
                "user_text": "What time does it start?",
                "assistant_text": "Let me check that for you.",
                "tool_calls": [{"name": "lookup_item", "args": {"query": "flowers"}}],
                "tool_results": [
                    {
                        "name": "lookup_item",
                        "response": {
                            "result": {
                                "status": "error",
                                "error_code": "UNEXPECTED_TOOL_CALL",
                            }
                        },
                    }
                ],
            }
        ]
        judgments = {
            0: {
                "scores": {
                    "turn_taking": True,
                    "tool_use_correct": False,
                    "instruction_following": True,
                    "kb_grounding": True,
                    "ambiguity_handling": None,
                    "state_tracking": None,
                },
                "reasoning": "Unexpected tool call on a no-tool turn should fail tool use.",
            }
        }
        expected_turns = [
            {
                "input": "What time does it start?",
                "golden_text": "It starts at 9 AM.",
                "required_function_call": None,
            }
        ]

        with TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            write_outputs(
                run_dir=run_dir,
                records=records,
                judgments=judgments,
                summary="",
                model_name="test-model",
                expected_turns=expected_turns,
                judge_name="openai",
                judge_version="test-version",
                judge_model="test-judge",
            )
            summary = json.loads((run_dir / "openai_summary.json").read_text())

        self.assertEqual(summary["passes"]["tool_use_correct"], 0)

    def test_judge_prompt_allows_benign_benchmark_tool_mismatches(self):
        prompt = build_judge_system_prompt(cross_turn_realignment=False)

        self.assertIn("FALSE if no function call was expected but the assistant still called a tool", prompt)
        self.assertIn("benchmark-generated tool error", prompt)
        self.assertIn("9:15", prompt)
        self.assertIn("end_session({})", prompt)
        self.assertIn("titles, notes, issue descriptions, suggestion text", prompt)
        self.assertIn("judge the rest of the turn against the intended successful action/result", prompt)

    def test_judge_prompt_treats_free_text_args_semantically(self):
        prompt = build_judge_system_prompt(cross_turn_realignment=False)

        self.assertIn("Free-text field normalization", prompt)
        self.assertIn("title case vs sentence case", prompt)
        self.assertIn('In the board room" vs "Board room', prompt)

    def test_judge_prompt_allows_confirm_then_next_step_after_success(self):
        prompt = build_judge_system_prompt(cross_turn_realignment=False)

        self.assertIn("Do NOT fail", prompt)
        self.assertIn("clearly confirms that the completed action is done", prompt)
        self.assertIn("logically next action or follow-up question", prompt)
        self.assertIn("Would you like me to notify the other attendee too?", prompt)

    def test_rehydrated_run_without_parent_wav_skips_turn_taking_with_reason(self):
        with TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "runtime.json").write_text(
                json.dumps(
                    {
                        "mode": "rehydrated",
                        "turn_artifact_layout": "per_turn_subdirs",
                        "turn_taking_skip_reason": "parent conversation.wav intentionally omitted",
                    }
                ),
                encoding="utf-8",
            )

            supported, reason = get_turn_taking_support(run_dir, skip_turn_taking=False)

        self.assertFalse(supported)
        self.assertEqual(reason, "parent conversation.wav intentionally omitted")

    def test_format_turns_for_judge_includes_tool_results(self):
        records = [
            {
                "turn": 0,
                "user_text": "check availability",
                "assistant_text": "The venue is available.",
                "tool_calls": [
                    {
                        "name": "search_venues",
                        "args": {"date": "2025-03-15", "guest_count": 90},
                    }
                ],
                "tool_results": [
                    {
                        "name": "search_venues",
                        "response": {
                            "tool_call_id": "call_123",
                            "result": {"status": "success", "venues": [{"venue_id": "garden_pavilion"}]},
                            "properties": None,
                            "is_duplicate": False,
                        },
                    }
                ],
            }
        ]
        expected_turns = [
            {
                "golden_text": "Let me check availability.",
                "required_function_call": {
                    "name": "search_venues",
                    "args": {"date": "2025-03-15", "guest_count": 90},
                },
                "categories": ["tool_use"],
            }
        ]

        formatted = format_turns_for_judge(
            records,
            expected_turns,
            get_relevant_dimensions_fn=lambda _: ["instruction_following", "kb_grounding", "tool_use_correct"],
        )

        self.assertIn("**Actual Functions**:", formatted)
        self.assertIn("**Actual Function Results**:", formatted)
        self.assertIn("call_123", formatted)
        self.assertIn("garden_pavilion", formatted)

    def test_format_turns_for_judge_includes_tool_use_guidance(self):
        records = [
            {
                "turn": 0,
                "user_text": "change the syrup back to one bottle",
                "assistant_text": "Updated to one bottle.",
                "tool_calls": [],
                "tool_results": [],
            }
        ]
        expected_turns = [
            {
                "golden_text": "Updated to one bottle.",
                "required_function_call": None,
                "categories": ["long_range_memory"],
                "tool_use_guidance": (
                    "Do not require a redundant lookup_item call here; reuse the already-established "
                    "item facts from session state."
                ),
            }
        ]

        formatted = format_turns_for_judge(
            records,
            expected_turns,
            get_relevant_dimensions_fn=lambda _: ["instruction_following", "kb_grounding"],
        )

        self.assertIn("**Tool Use Guidance**:", formatted)
        self.assertIn("reuse the already-established item facts", formatted)

    def test_format_turns_for_judge_distinguishes_prompt_visible_and_full_kb(self):
        records = [
            {
                "turn": 0,
                "user_text": "Do you have a bouquet?",
                "assistant_text": "I know we carry floral items, but let me look up the bouquet details.",
                "tool_calls": [],
                "tool_results": [],
            }
        ]
        expected_turns = [
            {
                "golden_text": "Fresh Flower Bouquet, mixed seasonal for thirty-four ninety-nine.",
                "required_function_call": {"name": "lookup_item", "args": {"query": "flower bouquet"}},
                "categories": ["tool_use"],
            }
        ]

        formatted = format_turns_for_judge(
            records,
            expected_turns,
            get_relevant_dimensions_fn=lambda _: ["instruction_following", "kb_grounding", "tool_use_correct"],
            kb_text="## Full KB\nFresh Flower Bouquet | mixed seasonal | $34.99",
            prompt_visible_kb_text="## Prompt KB\n### Floral\n- Fresh Flower Bouquet",
        )

        self.assertIn("# Prompt-Visible Knowledge Base", formatted)
        self.assertIn("# Full Benchmark Knowledge Base", formatted)
        self.assertIn("Fresh Flower Bouquet | mixed seasonal | $34.99", formatted)
        self.assertIn("### Floral", formatted)

    def test_format_rehydrated_turns_for_judge_uses_golden_history_not_prior_actual_turns(self):
        records = [
            {
                "turn": 0,
                "user_text": "register me for Kevin Zhang",
                "assistant_text": "What name should I use for that registration?",
                "tool_calls": [],
                "tool_results": [],
            },
            {
                "turn": 1,
                "user_text": "What am I registered for?",
                "assistant_text": "You are registered for Kevin Zhang.",
                "tool_calls": [],
                "tool_results": [],
            },
        ]
        expected_turns = [
            {
                "input": "register me for Kevin Zhang",
                "golden_text": "I've registered you for Kevin Zhang.",
                "required_function_call": {
                    "name": "register_for_session",
                    "args": {"name": "Jennifer Smith", "session_id": "916303"},
                },
                "function_call_response": {"status": "success"},
                "categories": ["tool_use", "long_range_memory"],
            },
            {
                "input": "What am I registered for?",
                "golden_text": "You're registered for Kevin Zhang.",
                "required_function_call": None,
                "categories": ["long_range_memory"],
            },
        ]

        formatted = format_rehydrated_turns_for_judge(
            records,
            expected_turns,
            only_turns={1},
            get_relevant_dimensions_fn=lambda turn: ["instruction_following", "kb_grounding", "state_tracking"]
            if "long_range_memory" in turn.get("categories", [])
            else ["instruction_following", "kb_grounding", "tool_use_correct"],
        )

        self.assertIn("# Hydrated Golden Conversation History", formatted)
        self.assertIn("### Golden Turn 0", formatted)
        self.assertIn("**Assistant (Golden)**: I've registered you for Kevin Zhang.", formatted)
        self.assertNotIn("What name should I use for that registration?", formatted)
        self.assertIn("## Turn 1", formatted)
        self.assertIn("**Assistant**: You are registered for Kevin Zhang.", formatted)

    def test_build_rehydrated_turn_prompt_bundles_isolates_actual_target_turns(self):
        records = [
            {
                "turn": 0,
                "user_text": "register me for Kevin Zhang",
                "assistant_text": "What name should I use for that registration?",
                "tool_calls": [],
                "tool_results": [],
            },
            {
                "turn": 1,
                "user_text": "What am I registered for?",
                "assistant_text": "You are registered for Kevin Zhang.",
                "tool_calls": [],
                "tool_results": [],
            },
        ]
        expected_turns = [
            {
                "input": "register me for Kevin Zhang",
                "golden_text": "I've registered you for Kevin Zhang.",
                "required_function_call": {
                    "name": "register_for_session",
                    "args": {"name": "Jennifer Smith", "session_id": "916303"},
                },
                "function_call_response": {"status": "success"},
                "categories": ["tool_use", "long_range_memory"],
            },
            {
                "input": "What am I registered for?",
                "golden_text": "You're registered for Kevin Zhang.",
                "required_function_call": None,
                "categories": ["long_range_memory"],
            },
        ]

        bundles = build_rehydrated_turn_prompt_bundles(
            records,
            expected_turns,
            get_relevant_dimensions_fn=lambda turn: ["instruction_following", "kb_grounding", "state_tracking"]
            if "long_range_memory" in turn.get("categories", [])
            else ["instruction_following", "kb_grounding", "tool_use_correct"],
        )

        self.assertEqual([bundle["turn"] for bundle in bundles], [0, 1])
        self.assertIn("What name should I use for that registration?", bundles[0]["prompt"])
        self.assertNotIn("You are registered for Kevin Zhang.", bundles[0]["prompt"])
        self.assertIn("You are registered for Kevin Zhang.", bundles[1]["prompt"])
        self.assertNotIn("What name should I use for that registration?", bundles[1]["prompt"])

    def test_rehydrated_judge_user_prompt_explicitly_uses_hydrated_history(self):
        prompt = build_judge_user_prompt(
            "## Turn 52\n**Hydrated Prior Context**: Golden Turns 0-51",
            [52],
            cross_turn_realignment=False,
        )

        self.assertIn('Use the "Hydrated Golden Conversation History" section as the ONLY source of prior-turn state', prompt)
        self.assertIn("only Golden Turns with index < N count as prior state", prompt)
        self.assertIn("Do NOT use the actual transcript content from one target turn block as prior state", prompt)

    def test_judge_prompt_allows_non_material_extra_grounded_commentary(self):
        prompt = build_judge_system_prompt(cross_turn_realignment=False)

        self.assertIn(
            "reasonable conversational detail or present-tense commentary",
            prompt,
        )
        self.assertIn(
            "Do NOT fail kb_grounding just because an extra phrase is not literally stated",
            prompt,
        )
        self.assertIn(
            "same-day delivery cutoff from the KB",
            prompt,
        )
        self.assertIn("we should be good for same-day delivery", prompt)
        self.assertIn(
            "arriving 15 minutes early for first-time-patient paperwork",
            prompt,
        )
        self.assertIn(
            "Do NOT fail kb_grounding just because the assistant lacks a hidden catalog detail",
            prompt,
        )
        self.assertIn(
            "If the prompt-visible KB says the store carries floral items",
            prompt,
        )
        self.assertIn(
            '"I don\'t have the bouquet details on hand" before a lookup is not, by itself, a grounding failure',
            prompt,
        )

    def test_text_pipeline_rehydration_keeps_tool_history_in_system_prompt(self):
        pipeline = TextPipeline(DummyBenchmark())
        pipeline._rehydration_turns = [
            {
                "input": "book the event",
                "golden_text": "Your event is booked.",
                "required_function_call": {
                    "name": "book_event",
                    "args": {"name": "Priya Mehta"},
                },
                "function_call_response": {
                    "status": "success",
                    "event_id": "EVT-3001",
                },
            }
        ]

        pipeline._setup_context()
        messages = pipeline.context.get_messages()

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "target turn")
        self.assertIn("[Tool call: book_event", messages[0]["content"])
        self.assertIn("[Tool result:", messages[0]["content"])
        self.assertIn("\"event_id\": \"EVT-3001\"", messages[0]["content"])
        self.assertIn("Assistant: Your event is booked.", messages[0]["content"])

    def test_openai_rehydration_history_uses_conversation_items_with_assistant_output_text(self):
        items = RealtimePipeline._build_openai_rehydration_history_items(
            [
                {
                    "input": "book the event",
                    "golden_text": "Your event is booked.",
                    "required_function_call": {
                        "name": "book_event",
                        "args": {"name": "Priya Mehta"},
                    },
                    "function_call_response": {
                        "status": "success",
                        "event_id": "EVT-3001",
                    },
                }
            ]
        )

        self.assertEqual(len(items), 4)
        self.assertEqual(items[0].type, "message")
        self.assertEqual(items[0].role, "user")
        self.assertEqual(items[0].content[0].type, "input_text")
        self.assertEqual(items[1].type, "function_call")
        self.assertEqual(items[2].type, "function_call_output")
        self.assertEqual(items[3].type, "message")
        self.assertEqual(items[3].role, "assistant")
        self.assertEqual(items[3].content[0].type, "output_text")
        self.assertEqual(items[3].content[0].text, "Your event is booked.")

    def test_openai_rehydration_history_omits_blank_assistant_for_oracle_continuation(self):
        items = RealtimePipeline._build_openai_rehydration_history_items(
            [
                {
                    "input": "book the event",
                    "golden_text": "",
                    "required_function_call": {
                        "name": "book_event",
                        "args": {"name": "Priya Mehta"},
                    },
                    "function_call_response": {
                        "status": "success",
                        "event_id": "EVT-3001",
                    },
                }
            ]
        )

        self.assertEqual([item.type for item in items], ["message", "function_call", "function_call_output"])

    def test_rehydration_reconnect_history_preserves_tool_results_in_prompt(self):
        pipeline = RealtimePipeline(DummyBenchmark())
        pipeline._rehydration_turns = [
            {
                "input": "book the event",
                "golden_text": "Your event is booked.",
                "required_function_call": {
                    "name": "book_event",
                    "args": {"name": "Priya Mehta"},
                },
                "function_call_response": {
                    "status": "success",
                    "event_id": "EVT-3001",
                },
            }
        ]
        pipeline.llm = SimpleNamespace(
            _session_properties=SimpleNamespace(instructions="")
        )

        pipeline._seed_rehydration_history_for_reconnects()

        self.assertEqual(
            pipeline._conversation_history[0]["tool_results"],
            [
                {
                    "name": "book_event",
                    "response": {"status": "success", "event_id": "EVT-3001"},
                }
            ],
        )

        pipeline._update_instructions_with_history()

        self.assertIn("[Tool result:", pipeline.llm._session_properties.instructions)
        self.assertIn("\"event_id\": \"EVT-3001\"", pipeline.llm._session_properties.instructions)

    def test_appointment_multi_call_turns_have_response_lists(self):
        for turn_index in (16, 21):
            turn = appointment_turns[turn_index]
            self.assertIsInstance(turn["required_function_call"], list)
            self.assertIsInstance(turn["function_call_response"], list)
            self.assertEqual(
                len(turn["required_function_call"]),
                len(turn["function_call_response"]),
            )

    def test_turn_uses_oracle_post_tool_continuation_for_any_realtime_tool_turn(self):
        self.assertTrue(
            turn_uses_oracle_post_tool_continuation(
                pipeline_type="realtime",
                turn={
                    "required_function_call": {"name": "book_event", "args": {"name": "Priya"}},
                },
            )
        )
        self.assertTrue(
            turn_uses_oracle_post_tool_continuation(
                pipeline_type="realtime",
                turn={
                    "required_function_call": [
                        {"name": "book_event", "args": {"name": "Priya"}},
                    ],
                },
            )
        )
        self.assertFalse(
            turn_uses_oracle_post_tool_continuation(
                pipeline_type="text",
                turn={
                    "required_function_call": {"name": "book_event", "args": {"name": "Priya"}},
                },
            )
        )

    def test_realtime_aborts_after_unexpected_tool_call_on_no_tool_turn(self):
        pipeline = RealtimePipeline(DummyBenchmark())
        pipeline.turns = [{"input": "what time does it start?", "required_function_call": None}]
        pipeline._rehydration_turns = [{"input": "earlier turn", "golden_text": "earlier answer"}]
        pipeline._captured_tool_phase = {
            "actual_tool_calls": [{"name": "lookup_item", "args": {"query": "flowers"}}],
            "actual_tool_result": {
                "status": "error",
                "error_code": "UNEXPECTED_TOOL_CALL",
            },
        }

        self.assertTrue(pipeline.should_abort_after_tool_capture())

    def test_realtime_does_not_abort_after_unexpected_tool_call_on_non_rehydrated_turn(self):
        pipeline = RealtimePipeline(DummyBenchmark())
        pipeline.turns = [{"input": "what time does it start?", "required_function_call": None}]
        pipeline._captured_tool_phase = {
            "actual_tool_calls": [{"name": "lookup_item", "args": {"query": "flowers"}}],
            "actual_tool_result": {
                "status": "error",
                "error_code": "UNEXPECTED_TOOL_CALL",
            },
        }

        self.assertFalse(pipeline.should_abort_after_tool_capture())

    def test_realtime_does_not_abort_after_expected_tool_call_during_normal_turn(self):
        pipeline = RealtimePipeline(DummyBenchmark())
        pipeline.turns = [
            {
                "input": "book the event",
                "required_function_call": {
                    "name": "book_event",
                    "args": {"name": "Priya"},
                },
            }
        ]
        pipeline._captured_tool_phase = {
            "actual_tool_calls": [{"name": "book_event", "args": {"name": "Priya"}}],
            "actual_tool_result": {"status": "success"},
        }

        self.assertFalse(pipeline.should_abort_after_tool_capture())

    def test_no_tool_rehydrated_tool_result_suppresses_followup_llm(self):
        pipeline = RealtimePipeline(DummyBenchmark())
        pipeline.turns = [{"input": "what time does it start?", "required_function_call": None}]
        pipeline._rehydration_turns = [{"input": "earlier turn", "golden_text": "earlier answer"}]
        params = SimpleNamespace(
            function_name="lookup_item",
            tool_call_id="call_1",
            arguments={"query": "flowers"},
            result_callback=AsyncMock(),
        )

        asyncio.run(pipeline._function_catchall(params))

        _, kwargs = params.result_callback.await_args
        self.assertEqual(kwargs["properties"].run_llm, False)

    def test_build_oracle_continuation_history_appends_current_turn_without_assistant(self):
        history = build_oracle_continuation_history(
            [{"input": "hello", "golden_text": "hi"}],
            {
                "input": "book the event",
                "golden_text": "should not be used",
                "required_function_call": {"name": "book_event", "args": {"name": "Priya"}},
                "function_call_response": {"status": "success", "event_id": "EVT-3001"},
            },
        )

        self.assertEqual(len(history), 2)
        self.assertEqual(history[-1]["input"], "book the event")
        self.assertEqual(history[-1]["golden_text"], "")

    def test_build_missing_tool_capture_marks_live_tool_use_as_failed(self):
        capture = build_missing_tool_capture(
            {
                "required_function_call": {"name": "book_event", "args": {"name": "Priya"}},
                "function_call_response": {"status": "success", "event_id": "EVT-3001"},
            }
        )

        self.assertIsNone(capture["actual_tool_call"])
        self.assertFalse(capture["tool_name_correct"])
        self.assertFalse(capture["tool_args_correct"])
        self.assertEqual(capture["oracle_tool_results"], [{"status": "success", "event_id": "EVT-3001"}])

    def test_merge_oracle_continuation_artifacts_rewrites_single_turn_transcript(self):
        with TemporaryDirectory() as tmpdir:
            transcript_path = Path(tmpdir) / "transcript.jsonl"
            transcript_path.write_text(
                json.dumps(
                    {
                        "turn": 7,
                        "user_text": "book the event",
                        "assistant_text": "Your event is booked.",
                        "tool_calls": [],
                        "tool_results": [],
                    }
                ) + "\n",
                encoding="utf-8",
            )

            merge_oracle_continuation_artifacts(
                transcript_path=transcript_path,
                live_tool_capture={
                    "actual_tool_call": {"name": "cancel_action", "args": {"name": "Priya"}},
                    "actual_tool_result": {"status": "error", "error_code": "UNEXPECTED_TOOL_CALL"},
                    "tool_name_correct": False,
                    "tool_args_correct": False,
                    "tool_use_pass": False,
                    "oracle_tool_calls": [{"name": "book_event", "args": {"name": "Priya"}}],
                    "oracle_tool_results": [{"status": "success", "event_id": "EVT-3001"}],
                },
            )

            row = json.loads(transcript_path.read_text(encoding="utf-8").strip())
            self.assertEqual(row["tool_calls"], [{"name": "cancel_action", "args": {"name": "Priya"}}])
            self.assertEqual(
                row["oracle_continuation"]["oracle_tool_results"],
                [{"status": "success", "event_id": "EVT-3001"}],
            )

    def test_format_rehydrated_turns_for_judge_surfaces_oracle_continuation_metadata(self):
        formatted = format_rehydrated_turns_for_judge(
            [
                {
                    "turn": 2,
                    "user_text": "book the event",
                    "assistant_text": "Your event is booked.",
                    "tool_calls": [{"name": "cancel_action", "args": {"name": "Priya"}}],
                    "tool_results": [{"name": "cancel_action", "response": {"status": "error"}}],
                    "oracle_continuation": {
                        "used": True,
                        "tool_name_correct": False,
                        "tool_args_correct": False,
                        "tool_use_pass": False,
                        "oracle_tool_calls": [{"name": "book_event", "args": {"name": "Priya"}}],
                        "oracle_tool_results": [{"status": "success", "event_id": "EVT-3001"}],
                    },
                }
            ],
            [
                {"input": "hi", "golden_text": "hello"},
                {"input": "warm up", "golden_text": "sure"},
                {
                    "input": "book the event",
                    "golden_text": "Your event is booked.",
                    "required_function_call": {"name": "book_event", "args": {"name": "Priya"}},
                },
            ],
        )

        self.assertIn("Oracle Continuation", formatted)
        self.assertIn("Oracle Seeded Tool Results", formatted)
        self.assertIn("Actual Functions", formatted)
        self.assertIn("Live Tool Capture Results (tool_use_correct only)", formatted)
        self.assertIn(
            "source of truth for the assistant response",
            formatted,
        )

    def test_rehydrated_judge_prompt_tells_oracle_turns_to_use_seeded_tool_results(self):
        prompt = build_judge_user_prompt(
            formatted_turns="**Oracle Continuation**: yes",
            turn_numbers=[6],
            cross_turn_realignment=False,
        )

        self.assertIn("Oracle Continuation", prompt)
        self.assertIn("Oracle Seeded Tool Calls", prompt)
        self.assertIn("source of truth for the assistant response", prompt)

    def test_manual_turn_handling_ignores_duplicate_stop_after_first_commit(self):
        service = OpenAIRealtimeLLMServiceExplicitToolResult.__new__(
            OpenAIRealtimeLLMServiceExplicitToolResult
        )
        service._session_properties = SimpleNamespace(
            audio=SimpleNamespace(
                input=SimpleNamespace(turn_detection=False)
            )
        )
        service._awaiting_manual_audio_commit = False
        service._manual_turn_input_committed = True
        service._manual_response_in_flight = False
        service._last_manual_commit_monotonic = 0.0
        service._pending_manual_tool_results = []
        service._awaiting_manual_tool_continuation_start = False

        called = {"send_client_event": 0}

        async def fake_send_client_event(_event):
            called["send_client_event"] += 1

        service.send_client_event = fake_send_client_event

        import asyncio

        asyncio.run(service._handle_user_stopped_speaking(None))

        self.assertEqual(called["send_client_event"], 0)
        self.assertFalse(service._awaiting_manual_audio_commit)

    def test_retry_current_turn_resets_manual_no_vad_state_before_requeue(self):
        pipeline = RealtimePipeline(DummyBenchmark())
        pipeline.needs_turn_retry = True
        pipeline.turn_idx = 0
        pipeline.current_turn_audio_path = "/tmp/turn.wav"
        pipeline.paced_input = Mock()
        pipeline._reset_openai_manual_turn_state = Mock()
        pipeline._start_turn_watchdog = Mock()
        pipeline._get_audio_duration = Mock(return_value=1.25)

        with patch("audio_arena.pipelines.realtime.asyncio.sleep", new=AsyncMock()):
            import asyncio

            asyncio.run(pipeline._retry_current_turn())

        pipeline._reset_openai_manual_turn_state.assert_called_once_with()
        pipeline.paced_input.enqueue_wav_file.assert_called_once_with("/tmp/turn.wav")
        pipeline._start_turn_watchdog.assert_called_once_with(1.25)
        self.assertFalse(pipeline.needs_turn_retry)

    def test_tool_output_ack_waits_for_response_done_before_continuation(self):
        service = OpenAIRealtimeLLMServiceExplicitToolResult.__new__(
            OpenAIRealtimeLLMServiceExplicitToolResult
        )
        service._pending_response_create = True
        service._waiting_for_response_done_before_response_create = True
        service._pending_tool_output_item_ids = {"tool-item-1"}
        service.send_client_event = AsyncMock()

        evt = SimpleNamespace(
            item=SimpleNamespace(
                id="tool-item-1",
                type="function_call_output",
                call_id="call_1",
            )
        )

        import asyncio

        asyncio.run(service._handle_tool_output_item_event(evt, phase="added"))

        service.send_client_event.assert_not_awaited()
        self.assertTrue(service._pending_response_create)
        self.assertFalse(service._pending_tool_output_item_ids)

        service._waiting_for_response_done_before_response_create = False
        asyncio.run(service._maybe_send_deferred_response_create(trigger="response.done"))

        service.send_client_event.assert_awaited_once()
        sent_event = service.send_client_event.await_args.args[0]
        self.assertIsInstance(sent_event, rt_events.ResponseCreateEvent)
        self.assertFalse(service._pending_response_create)

    def test_function_call_output_marks_deferred_state_before_send(self):
        service = OpenAIRealtimeLLMServiceExplicitToolResult.__new__(
            OpenAIRealtimeLLMServiceExplicitToolResult
        )
        service._context = object()
        service._pending_function_calls = {"call_1": SimpleNamespace(name="get_quote")}
        service._completed_tool_calls = set()
        service._get_last_tool_result = lambda: {"status": "success", "quote": 12750}
        service._pending_response_create = False
        service._waiting_for_response_done_before_response_create = False
        service._pending_tool_output_item_ids = set()
        service.run_function_calls = AsyncMock()

        async def fake_send_client_event(event):
            if isinstance(event, rt_events.ConversationItemCreateEvent):
                self.assertTrue(service._pending_response_create)
                self.assertTrue(service._waiting_for_response_done_before_response_create)
                self.assertIn(event.item.id, service._pending_tool_output_item_ids)

        service.send_client_event = fake_send_client_event

        evt = SimpleNamespace(
            call_id="call_1",
            arguments=json.dumps({"event_type": "wedding"}),
        )

        import asyncio

        asyncio.run(service._handle_evt_function_call_arguments_done(evt))

        self.assertIn("call_1", service._completed_tool_calls)
        service.run_function_calls.assert_awaited_once()
        self.assertTrue(service._pending_response_create)
        self.assertTrue(service._waiting_for_response_done_before_response_create)
        self.assertEqual(len(service._pending_tool_output_item_ids), 1)

    def test_openai_realtime_session_update_payload_preserves_null_turn_detection(self):
        service = OpenAIRealtimeLLMServiceExplicitToolResult.__new__(
            OpenAIRealtimeLLMServiceExplicitToolResult
        )
        service._session_backdoor_options = {"juice": 64, "verbosity": 2}

        payload = service._build_session_update_payload(
            rt_events.SessionProperties(
                instructions="Use the benchmark policy.",
                audio=rt_events.AudioConfiguration(
                    input=rt_events.AudioInput(turn_detection=None)
                ),
            )
        )

        self.assertEqual(payload["type"], "session.update")
        self.assertEqual(payload["session"]["instructions"], "Use the benchmark policy.")
        self.assertIsNone(payload["session"]["audio"]["input"]["turn_detection"])
        self.assertEqual(
            payload["session"]["backdoor_options"],
            {"juice": 64, "verbosity": 2},
        )

    def test_openai_realtime_manual_turn_handling_treats_null_turn_detection_as_disabled(self):
        service = OpenAIRealtimeLLMServiceExplicitToolResult.__new__(
            OpenAIRealtimeLLMServiceExplicitToolResult
        )
        service._session_properties = SimpleNamespace(
            audio=rt_events.AudioConfiguration(
                input=rt_events.AudioInput(turn_detection=None)
            )
        )

        self.assertTrue(service._manual_turn_handling_active())

    def test_ws_event_sanitizer_truncates_large_audio_payloads(self):
        payload = {
            "type": "input_audio_buffer.append",
            "audio": "A" * 500,
            "nested": {"delta": "B" * 400},
            "small_delta": "ok",
        }

        sanitized = OpenAIRealtimeLLMServiceExplicitToolResult._sanitize_ws_event_payload(
            payload
        )

        self.assertEqual(sanitized["type"], "input_audio_buffer.append")
        self.assertEqual(sanitized["small_delta"], "ok")
        self.assertEqual(sanitized["audio"]["length"], 500)
        self.assertTrue(sanitized["audio"]["__truncated__"])
        self.assertEqual(sanitized["nested"]["delta"]["length"], 400)


if __name__ == "__main__":
    unittest.main()
