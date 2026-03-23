import unittest
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for path in [PROJECT_ROOT, SCRIPTS_DIR]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from build_experiment_review import build_record
from review_streamlit_app import build_phase_review_rows


class ReviewStreamlitPhaseRowsTests(unittest.TestCase):
    def test_build_record_preserves_oracle_continuation_fields(self):
        record = build_record(
            {
                "turn": 7,
                "user_text": "Book it",
                "assistant_text": "Booked.",
                "tool_calls": [{"name": "wrong_tool", "args": {"name": "Priya"}}],
                "tool_results": [{"name": "wrong_tool", "response": {"status": "error"}}],
                "scores": {
                    "tool_use_correct": False,
                    "instruction_following": True,
                    "kb_grounding": True,
                    "turn_taking": True,
                    "ambiguity_handling": None,
                    "state_tracking": True,
                },
                "oracle_continuation": {
                    "used": True,
                    "tool_name_correct": False,
                    "tool_args_correct": False,
                    "tool_use_pass": False,
                    "oracle_tool_calls": [{"name": "book_event", "args": {"name": "Priya"}}],
                    "oracle_tool_results": [{"status": "success", "event_id": "EVT-3001"}],
                },
            }
        )

        self.assertTrue(record["oracle_continuation_used"])
        self.assertFalse(record["live_tool_use_pass"])
        self.assertEqual(record["oracle_tool_calls"][0]["name"], "book_event")
        self.assertIn("EVT-3001", record["oracle_tool_results_text"])

    def test_build_phase_review_rows_splits_oracle_continuation_turn(self):
        rows_df = pd.DataFrame(
            [
                {
                    "turn": 7,
                    "user_text": "Book it",
                    "assistant_text": "Booked.",
                    "response_status": "normal",
                    "latency_ms": 1200,
                    "overall_status": "fail",
                    "failed_dimensions_text": "tool_use_correct",
                    "tool_calls_text": '[{"name":"wrong_tool"}]',
                    "oracle_continuation_used": True,
                    "live_tool_use_pass": False,
                    "turn_taking": True,
                    "instruction_following": True,
                    "kb_grounding": True,
                    "ambiguity_handling": None,
                    "state_tracking": True,
                },
                {
                    "turn": 8,
                    "user_text": "Thanks",
                    "assistant_text": "You are welcome.",
                    "response_status": "normal",
                    "latency_ms": 400,
                    "overall_status": "pass",
                    "failed_dimensions_text": "",
                    "tool_calls_text": "[]",
                    "oracle_continuation_used": False,
                    "live_tool_use_pass": None,
                    "turn_taking": True,
                    "instruction_following": True,
                    "kb_grounding": True,
                    "ambiguity_handling": None,
                    "state_tracking": None,
                },
            ]
        )

        phase_rows = build_phase_review_rows(rows_df)

        self.assertEqual(len(phase_rows), 3)
        self.assertEqual(
            phase_rows["phase_label"].tolist(),
            ["Tool Phase", "Post-Tool Continuation", "Turn"],
        )
        tool_phase = phase_rows.iloc[0]
        continuation_phase = phase_rows.iloc[1]
        self.assertEqual(tool_phase["phase_status"], "fail")
        self.assertEqual(tool_phase["failed_dimensions_text"], "tool_use_correct")
        self.assertEqual(continuation_phase["phase_status"], "pass")
        self.assertEqual(continuation_phase["failed_dimensions_text"], "")


if __name__ == "__main__":
    unittest.main()
