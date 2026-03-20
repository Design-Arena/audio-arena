"""Base pipeline class for multi-turn evaluation.

The pipeline owns EVERYTHING - the CLI/runner just calls pipeline.run().
Each pipeline type (text, realtime, nova-sonic) handles its own specifics.
"""

import asyncio
import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from pipecat.frames.frames import MetricsFrame
from pipecat.metrics.metrics import LLMUsageMetricsData, TTFBMetricsData
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.llm_service import FunctionCallParams

from audio_arena.recording.transcript_recorder import TranscriptRecorder


class BasePipeline(ABC):
    """Base class for all pipelines. Owns all execution state and logic.

    The pipeline is responsible for:
    1. Creating and configuring the LLM service
    2. Setting up context with system prompt and tools
    3. Building the pipeline with all processors
    4. Managing turn flow (queuing turns, detecting end-of-turn)
    5. Recording transcripts and metrics

    Subclasses implement the abstract methods to customize behavior.
    """

    # Set to False for pipelines that create their own LLM (e.g., Nova Sonic)
    requires_service = True

    def __init__(self, benchmark):
        """Initialize the pipeline.

        Args:
            benchmark: A BenchmarkConfig instance with turns, tools, and system instruction.
        """
        self.benchmark = benchmark
        self.turns = benchmark.turns
        self.turn_idx = 0
        self.done = False
        self.recorder: Optional[TranscriptRecorder] = None
        self.task: Optional[PipelineTask] = None
        self.context: Optional[LLMContext] = None
        self.llm: Optional[FrameProcessor] = None
        self.model_name: Optional[str] = None
        self.service_name: Optional[str] = None
        self._disable_vad: bool = False
        self._turn_indices: Optional[List[int]] = None
        # Golden turns to inject as context before the target turn (single-step rehydration)
        self._rehydration_turns: Optional[List[Dict[str, Any]]] = None
        # Track tool calls to detect duplicates within a turn
        self._seen_tool_calls: set = set()
        # Track tool_call_ids that are duplicates (for filtering in ToolCallRecorder)
        self._duplicate_tool_call_ids: set = set()
        # Track which response index we're on for multi-step tool chains
        self._tool_response_idx: int = 0
        # Track which scripted multi-call responses have already been used this turn
        self._consumed_tool_response_indices: set[int] = set()
        # Last tool result (for explicit delivery to GPT/Grok Realtime APIs; see docstring below)
        self._last_tool_result: Optional[Dict[str, Any]] = None
        self._tool_capture_only: bool = False
        self._oracle_continuation_only: bool = False
        self._captured_tool_phase: Optional[Dict[str, Any]] = None
        self._juice: Optional[int] = None
        self._verbosity: Optional[int] = None

    @staticmethod
    def build_rehydration_history(
        golden_turns: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Build rehydration context from golden turns for single-step evaluation.

        Converts golden turn definitions into two formats:
        - A messages list (user/assistant pairs) for text pipeline LLMContext injection.
        - A formatted instruction string for realtime pipeline system-instruction enrichment.

        Args:
            golden_turns: The benchmark turns list sliced to ``turns[0:target_turn_idx]``.

        Returns:
            ``(messages, instruction_string)`` tuple.
        """
        messages: List[Dict[str, Any]] = []
        lines = [
            "\n\n--- CONVERSATION HISTORY (GOLDEN) ---",
            "The following is the conversation so far. The user's name, "
            "preferences, and any actions you have taken (tool calls, "
            "registrations, schedule changes, etc.) are still in effect. "
            "Continue naturally.",
            "",
        ]

        for i, turn in enumerate(golden_turns):
            user_input = turn.get("input", "")
            golden_text = turn.get("golden_text", "")
            fc = turn.get("required_function_call")
            fc_response = turn.get("function_call_response")
            has_golden_assistant = bool(turn.get("golden_text"))

            messages.append({"role": "user", "content": user_input})
            lines.append(f"User: {user_input}")

            if fc is not None:
                calls = fc if isinstance(fc, list) else [fc]
                responses = (
                    fc_response
                    if isinstance(fc_response, list)
                    else [fc_response] if fc_response is not None else []
                )
                for j, call in enumerate(calls):
                    lines.append(
                        f"  [Tool call: {call['name']}({json.dumps(call.get('args', {}))})]"
                    )
                    if j < len(responses):
                        lines.append(f"  [Tool result: {json.dumps(responses[j])}]")

            if has_golden_assistant:
                messages.append({"role": "assistant", "content": golden_text})
                lines.append(f"Assistant: {golden_text}")
                lines.append("")
            elif fc is not None:
                lines.append("")

        instruction_string = "\n".join(lines)
        return messages, instruction_string

    @property
    def effective_turns(self) -> List[dict]:
        """Get the turns to run (filtered by turn_indices if set)."""
        if self._turn_indices is not None:
            return [self.turns[i] for i in self._turn_indices if i < len(self.turns)]
        return self.turns

    async def run(
        self,
        recorder: TranscriptRecorder,
        model: str,
        service_class: Optional[type] = None,
        service_name: Optional[str] = None,
        turn_indices: Optional[List[int]] = None,
        rehydration_turns: Optional[List[Dict[str, Any]]] = None,
        disable_vad: bool = False,
        juice: Optional[int] = None,
        verbosity: Optional[int] = None,
        stop_after_first_tool_call: bool = False,
        oracle_continuation_only: bool = False,
    ) -> None:
        """Run the complete benchmark. Pipeline handles everything internally.

        Args:
            recorder: TranscriptRecorder for saving results.
            model: Model name/identifier.
            service_class: LLM service class (required unless pipeline sets requires_service=False).
            service_name: Service name/alias (e.g., "openai", "openrouter").
            turn_indices: Optional list of turn indices to run (for debugging).
            rehydration_turns: Optional golden turns to inject as prior context.
                When set, the pipeline runs in single-step rehydration mode: the golden
                history is injected into the model context, and only the target turn(s)
                specified by ``turn_indices`` are executed live.
            disable_vad: Disable server-side VAD for compatible realtime pipelines.
            juice: Optional OpenAI Realtime backdoor reasoning-effort override.
            verbosity: Optional OpenAI Realtime backdoor verbosity override.
            stop_after_first_tool_call: Stop the run immediately after capturing the
                first live tool call/result for the target turn.
            oracle_continuation_only: Run a continuation-only turn where the current
                user/tool/tool-result state is already seeded and the model should
                produce only the post-tool assistant response.
        """
        self.recorder = recorder
        self.model_name = model
        self.service_name = service_name  # Store for use in _create_llm overrides
        self._turn_indices = turn_indices
        self._rehydration_turns = rehydration_turns
        self._disable_vad = disable_vad
        self._tool_capture_only = stop_after_first_tool_call
        self._oracle_continuation_only = oracle_continuation_only
        self._captured_tool_phase = None
        self._juice = juice
        self._verbosity = verbosity

        # Create LLM service
        self.llm = self._create_llm(service_class, model)

        # Setup (pipeline-specific)
        self._setup_context()
        self._setup_llm()
        self._build_task()

        # Initialize first turn BEFORE queueing
        self.recorder.start_turn(self._get_actual_turn_index(0))

        # Queue first turn and run
        await self._queue_first_turn()
        runner = PipelineRunner(handle_sigint=True)
        try:
            await runner.run(self.task)
        finally:
            await self._cleanup_after_run()

    async def _cleanup_after_run(self) -> None:
        """Hook for pipeline-specific shutdown cleanup after runner exit."""
        pass

    def _get_actual_turn_index(self, effective_index: int) -> int:
        """Convert effective turn index to actual turn index."""
        if self._turn_indices is not None:
            return self._turn_indices[effective_index]
        return effective_index

    def _get_current_turn(self) -> dict:
        """Get the current turn data."""
        return self.effective_turns[self.turn_idx]

    def get_captured_tool_phase(self) -> Optional[Dict[str, Any]]:
        """Return metadata captured during a tool-capture-only run."""
        return self._captured_tool_phase

    def capture_tool_phase(
        self,
        function_name: str,
        arguments: Dict[str, Any],
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Capture live tool-use metadata for the current turn."""
        self._captured_tool_phase = self._build_tool_capture_metadata(
            function_name,
            arguments,
            result,
        )
        return self._captured_tool_phase

    def should_abort_after_tool_capture(self) -> bool:
        """Return True when realtime services should stop after the first tool call."""
        return self._tool_capture_only and self._captured_tool_phase is not None

    async def abort_after_tool_capture(self) -> None:
        """Terminate a capture-only run once the live tool call is recorded."""
        if self.done:
            return
        logger.info("[ToolCapture] Aborting run after first live tool call capture")
        self.done = True
        if self.task is not None:
            await self.task.cancel()

    def _build_tool_capture_metadata(
        self,
        function_name: str,
        arguments: Dict[str, Any],
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Summarize the first live tool call for artifact merging."""
        current_turn = self._get_current_turn()
        required_calls = current_turn.get("required_function_call")
        custom_response = current_turn.get("function_call_response")

        oracle_calls = required_calls if isinstance(required_calls, list) else [required_calls] if required_calls else []
        oracle_results = (
            custom_response
            if isinstance(custom_response, list)
            else [custom_response] if custom_response is not None else []
        )

        tool_name_correct: Optional[bool] = None
        tool_args_correct: Optional[bool] = None

        if required_calls is None:
            tool_name_correct = False
            tool_args_correct = None
        elif isinstance(required_calls, dict):
            expected_name = required_calls.get("name")
            expected_args = required_calls.get("args", {})
            tool_name_correct = function_name == expected_name
            tool_args_correct = (
                self._tool_args_match(expected_args, arguments)
                if tool_name_correct
                else False
            )
        else:
            matching_names = [
                call for call in required_calls if call.get("name") == function_name
            ]
            tool_name_correct = bool(matching_names)
            tool_args_correct = any(
                self._tool_args_match(call.get("args", {}), arguments)
                for call in matching_names
            ) if matching_names else False

        return {
            "actual_tool_call": {
                "name": function_name,
                "args": arguments or {},
            },
            "actual_tool_result": result,
            "tool_name_correct": tool_name_correct,
            "tool_args_correct": tool_args_correct,
            "tool_use_pass": bool(tool_name_correct and tool_args_correct),
            "oracle_tool_calls": oracle_calls,
            "oracle_tool_results": oracle_results,
        }

    def _create_llm(
        self, service_class: Optional[type], model: str
    ) -> FrameProcessor:
        """Create LLM service. Override for pipelines that create their own.

        Args:
            service_class: LLM service class to instantiate.
            model: Model name/identifier.

        Returns:
            Configured LLM service instance.

        Note:
            Subclasses can access self.service_name if needed for service-specific config.
        """
        if service_class is None:
            raise ValueError("--service is required for this pipeline")

        # Build kwargs with API keys based on service class name
        kwargs: Dict[str, Any] = {"model": model}
        class_name = service_class.__name__
        model_lower = model.lower()
        service_name_lower = (self.service_name or "").lower()

        # Handle OpenRouter (uses OpenAI-compatible API with different base URL and API key)
        if service_name_lower == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise EnvironmentError("OPENROUTER_API_KEY environment variable is required")
            kwargs["api_key"] = api_key
            kwargs["base_url"] = "https://openrouter.ai/api/v1"
            logger.info(f"Using OpenRouter with base_url={kwargs['base_url']}")
            return service_class(**kwargs)

        if "Anthropic" in class_name:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise EnvironmentError("ANTHROPIC_API_KEY environment variable is required")
            kwargs["api_key"] = api_key
        elif "Groq" in class_name:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise EnvironmentError("GROQ_API_KEY environment variable is required")
            kwargs["api_key"] = api_key
        elif "Cerebras" in class_name:
            api_key = os.getenv("CEREBRAS_API_KEY")
            if not api_key:
                raise EnvironmentError("CEREBRAS_API_KEY environment variable is required")
            kwargs["api_key"] = api_key
        elif "OpenAI" in class_name:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("OPENAI_API_KEY environment variable is required")
            kwargs["api_key"] = api_key

            # Configure gpt-5 series models: set reasoning effort and priority tier
            if model_lower.startswith("gpt-5"):
                from pipecat.services.openai.llm import OpenAILLMService
                # gpt-5.1 and gpt-5.2 use "none", other gpt-5 models use "minimal"
                if model_lower.startswith("gpt-5.1") or model_lower.startswith("gpt-5.2"):
                    reasoning_effort = "none"
                else:
                    reasoning_effort = "minimal"
                kwargs["params"] = OpenAILLMService.InputParams(
                    service_tier="priority",
                    extra={"reasoning_effort": reasoning_effort},
                )
                logger.info(f"Configured {model} with reasoning_effort={reasoning_effort}, service_tier=priority")

        elif "Google" in class_name or "Gemini" in class_name:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise EnvironmentError("GOOGLE_API_KEY environment variable is required")
            kwargs["api_key"] = api_key

            # Configure gemini-3 series models: use minimal thinking
            if "gemini-3" in model_lower:
                from google.genai import types
                from pipecat.services.google.llm import GoogleLLMService
                kwargs["params"] = GoogleLLMService.InputParams(
                    extra={
                        "thinking_config": types.ThinkingConfig(
                            thinking_level="MINIMAL",
                            include_thoughts=True,
                        )
                    }
                )
                logger.info(f"Configured {model} with thinking_level=MINIMAL")

        elif "Bedrock" in class_name:
            # AWS Bedrock uses AWS credentials from environment
            access_key = os.getenv("AWS_ACCESS_KEY_ID")
            secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            if not (access_key and secret_key):
                raise EnvironmentError(
                    "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables are required"
                )
            kwargs["aws_access_key_id"] = access_key
            kwargs["aws_secret_access_key"] = secret_key
            session_token = os.getenv("AWS_SESSION_TOKEN")
            if session_token:
                kwargs["aws_session_token"] = session_token
            region = os.getenv("AWS_REGION", "us-east-1")
            kwargs["region"] = region

        return service_class(**kwargs)

    async def _on_turn_end(self, assistant_text: str) -> None:
        """Called when assistant finishes. Base handles common logic.

        Args:
            assistant_text: The assistant's response text.
        """
        if self.done:
            return

        # Get the actual turn index for recording
        actual_turn_idx = self._get_actual_turn_index(self.turn_idx)

        # Record turn (common)
        self.recorder.write_turn(
            user_text=self.effective_turns[self.turn_idx].get("input", ""),
            assistant_text=assistant_text,
        )

        # Advance (common)
        self.turn_idx += 1

        # Reset tool call tracking for the new turn
        self._seen_tool_calls.clear()
        self._duplicate_tool_call_ids.clear()
        self._tool_response_idx = 0
        self._consumed_tool_response_indices.clear()

        if self.turn_idx < len(self.effective_turns):
            # Start next turn
            actual_next_idx = self._get_actual_turn_index(self.turn_idx)
            self.recorder.start_turn(actual_next_idx)
            await self._queue_next_turn()
        else:
            # All turns complete
            logger.info("Conversation complete")
            self.done = True
            self.recorder.write_summary()
            await self.task.cancel()

    def _handle_metrics(self, frame: MetricsFrame) -> None:
        """Common metrics handling."""
        for md in frame.data:
            if isinstance(md, LLMUsageMetricsData):
                self.recorder.record_usage_metrics(md.value, getattr(md, "model", None))
            elif isinstance(md, TTFBMetricsData):
                self.recorder.record_ttfb(md.value)

    async def _function_catchall(self, params: FunctionCallParams) -> None:
        """Common function handler: returns result (from turn data or default), handles end_session.

        Tool call recording is handled by ToolCallRecorder in the pipeline. This handler
        returns the result and handles the special end_session tool.

        Duplicate tool calls (same function + args) are detected and skipped to prevent
        context pollution.

        Explicit results for APIs (only GPT / Grok Realtime):
        Only the OpenAI Realtime protocol (used by OpenAI and xAI for realtime/voice)
        requires the client to push the tool result over the WebSocket (conversation.item.create
        with function_call_output, then response.create). Other providers (text/chat,
        Gemini Live, Ultravox, Nova Sonic) use request/response or other flows where
        Pipecat delivers the handler result automatically; no extra send is needed.
        For GPT and Grok we set self._last_tool_result and pass a getter into their
        services so they send the actual payload (e.g. from the turn's function_call_response)
        instead of a hardcoded {"status": "success"}.

        Ordering guarantee (tool result before model speaks):
        Pipecat awaits this handler before injecting the result and letting the model
        continue. The model only gets to generate speech/text AFTER result_callback() is
        called. If you add real async work (e.g. API calls) to compute the tool response,
        complete that work first, then call result_callback(result). Do not call
        result_callback from a background task or before the result is ready.
        """
        # Create a stable key for duplicate detection (function_name + args)
        call_key = (
            params.function_name,
            self._normalize_tool_args(params.arguments or {}),
        )

        # Check for duplicate tool call
        if call_key in self._seen_tool_calls:
            tool_call_id = getattr(params, 'tool_call_id', None)
            logger.warning(
                f"Skipping duplicate tool call: {params.function_name} "
                f"(tool_call_id={tool_call_id})"
            )
            # Track this tool_call_id as a duplicate so ToolCallRecorder can filter it
            if tool_call_id:
                self._duplicate_tool_call_ids.add(tool_call_id)
            # Return a result to satisfy the API, but mark it as skipped
            skip_result = {"status": "duplicate_skipped"}
            self._last_tool_result = skip_result
            await params.result_callback(skip_result)
            return

        # Track this call
        self._seen_tool_calls.add(call_key)

        result = self._get_turn_tool_response(
            params.function_name,
            params.arguments or {},
        )
        self._last_tool_result = result
        if self._tool_capture_only and self._captured_tool_phase is None:
            self._captured_tool_phase = self._build_tool_capture_metadata(
                params.function_name,
                params.arguments or {},
                result,
            )
        await params.result_callback(result)

        # end_session tool: gracefully terminate the run
        if params.function_name == "end_session":
            logger.info("end_session tool called - gracefully ending run")
            self.done = True
            # Small delay to let tool call frames propagate through ToolCallRecorder
            await asyncio.sleep(0.05)
            # Write the current turn and all remaining turns with empty responses
            # so that the evaluation still sees every turn (model ended early).
            for idx in range(self.turn_idx, len(self.effective_turns)):
                actual_idx = self._get_actual_turn_index(idx)
                if idx != self.turn_idx:
                    self.recorder.start_turn(actual_idx)
                self.recorder.write_turn(
                    user_text=self.effective_turns[idx].get("input", ""),
                    assistant_text="[MODEL_ENDED_SESSION]" if idx == self.turn_idx else "[MODEL_ENDED_SESSION_EARLY]",
                )
                logger.info(
                    f"end_session: wrote {'current' if idx == self.turn_idx else 'remaining'} "
                    f"turn {actual_idx} with empty response"
                )
            self.recorder.write_summary()
            # Cancel the pipeline task to exit cleanly
            await self.task.cancel()

    @staticmethod
    def _normalize_tool_args(args: Any) -> str:
        """Return a stable string key for tool arguments."""
        try:
            return json.dumps(args or {}, sort_keys=True, separators=(",", ":"))
        except TypeError:
            return str(args or {})

    @staticmethod
    def _normalize_time_string(value: str) -> str:
        """Normalize simple HH:MM strings so 9:15 and 09:15 compare equal."""
        parts = value.split(":")
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            return value
        hour, minute = parts
        if len(minute) != 2:
            return value
        return f"{int(hour):02d}:{minute}"

    @classmethod
    def _canonicalize_tool_arg_value(
        cls,
        value: Any,
        *,
        key: Optional[str] = None,
    ) -> Any:
        """Canonicalize a tool argument value for conservative semantic matching."""
        if isinstance(value, dict):
            return {
                sub_key: cls._canonicalize_tool_arg_value(sub_value, key=sub_key)
                for sub_key, sub_value in sorted(value.items())
            }

        if isinstance(value, list):
            normalized_items = [
                cls._canonicalize_tool_arg_value(item, key=key)
                for item in value
            ]
            if key == "product_ids" and all(
                isinstance(item, (str, int, float, bool)) for item in normalized_items
            ):
                return sorted(normalized_items)
            return normalized_items

        if isinstance(value, str):
            stripped = value.strip()
            return cls._normalize_time_string(stripped)

        return value

    @classmethod
    def _tool_args_match(
        cls,
        expected_args: Dict[str, Any],
        actual_args: Dict[str, Any],
    ) -> bool:
        """Return True for conservative semantic matches the runtime should accept."""
        expected = cls._canonicalize_tool_arg_value(expected_args or {})
        actual = cls._canonicalize_tool_arg_value(actual_args or {})
        return expected == actual

    def _get_turn_tool_response(
        self,
        function_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Resolve the scripted tool response for the current turn.

        For multi-call turns, prefer matching by `(tool_name, args)` so the
        benchmark remains correct even if the model calls required tools in a
        different order than the turn spec lists them.
        """
        result: Dict[str, Any] = {"status": "success"}
        if self.turn_idx >= len(self.effective_turns):
            return result

        current_turn = self.effective_turns[self.turn_idx]
        custom_response = current_turn.get("function_call_response")
        required_calls = current_turn.get("required_function_call")
        if required_calls is None:
            return self._build_tool_error(
                function_name=function_name,
                arguments=arguments,
                error_code="UNEXPECTED_TOOL_CALL",
                message="No tool call is expected on this turn.",
                expected_function=None,
                expected_args=None,
            )

        if isinstance(required_calls, dict):
            expected_name = required_calls.get("name")
            if function_name != expected_name:
                return self._build_tool_error(
                    function_name=function_name,
                    arguments=arguments,
                    error_code="UNEXPECTED_TOOL_CALL",
                    message="The called tool does not match the scripted tool for this turn.",
                    expected_function=expected_name,
                    expected_args=required_calls.get("args"),
                )
            expected_args = required_calls.get("args", {})
            if not self._tool_args_match(expected_args, arguments):
                return self._build_tool_error(
                    function_name=function_name,
                    arguments=arguments,
                    error_code="ARG_MISMATCH",
                    message="The tool name matches, but the arguments do not match the scripted tool call for this turn.",
                    expected_function=expected_name,
                    expected_args=expected_args,
                )
            if custom_response is None and function_name == "end_session":
                return result
            if custom_response is None:
                return self._build_tool_error(
                    function_name=function_name,
                    arguments=arguments,
                    error_code="UNEXPECTED_TOOL_CALL",
                    message="This tool call does not match the scripted tool behavior for the turn.",
                    expected_function=expected_name,
                    expected_args=expected_args,
                )
            return custom_response if not isinstance(custom_response, list) else custom_response[0]

        normalized_args = self._normalize_tool_args(arguments)

        if not isinstance(custom_response, list):
            return self._build_tool_error(
                function_name=function_name,
                arguments=arguments,
                error_code="UNEXPECTED_TOOL_CALL",
                message="The scripted tool responses for this turn are inconsistent.",
                expected_function=None,
                expected_args=None,
            )

        if isinstance(required_calls, list) and len(required_calls) == len(custom_response):
            for idx, required_call in enumerate(required_calls):
                if idx in self._consumed_tool_response_indices:
                    continue
                if required_call.get("name") != function_name:
                    continue
                if self._tool_args_match(required_call.get("args", {}), arguments):
                    self._consumed_tool_response_indices.add(idx)
                    self._tool_response_idx = max(self._tool_response_idx, idx + 1)
                    return custom_response[idx]

        matching_calls = [
            required_call
            for required_call in required_calls
            if required_call.get("name") == function_name
        ]
        if matching_calls:
            return self._build_tool_error(
                function_name=function_name,
                arguments=arguments,
                error_code="ARG_MISMATCH",
                message="The tool name matches, but the arguments do not match any scripted tool call for this turn.",
                expected_function=function_name,
                expected_args=[call.get("args", {}) for call in matching_calls],
            )

        expected_names = [required_call.get("name") for required_call in required_calls]
        return self._build_tool_error(
            function_name=function_name,
            arguments=arguments,
            error_code="UNEXPECTED_TOOL_CALL",
            message="The called tool does not match any scripted tool for this turn.",
            expected_function=expected_names,
            expected_args=[call.get("args", {}) for call in required_calls],
        )

    @staticmethod
    def _build_tool_error(
        *,
        function_name: str,
        arguments: Dict[str, Any],
        error_code: str,
        message: str,
        expected_function: Any,
        expected_args: Any,
    ) -> Dict[str, Any]:
        """Build a benchmark-side tool error that the grader can score explicitly."""
        return {
            "status": "error",
            "error_code": error_code,
            "message": message,
            "called_function": function_name,
            "called_args": arguments or {},
            "expected_function": expected_function,
            "expected_args": expected_args,
        }

    # ---- Abstract methods (pipeline-specific) ----

    @abstractmethod
    def _setup_context(self) -> None:
        """Create LLMContext with system prompt and tools."""
        pass

    @abstractmethod
    def _setup_llm(self) -> None:
        """Configure LLM (register functions, set callbacks)."""
        pass

    @abstractmethod
    def _build_task(self) -> None:
        """Build Pipeline and PipelineTask with all processors."""
        pass

    @abstractmethod
    async def _queue_first_turn(self) -> None:
        """Queue the first turn to start the conversation."""
        pass

    @abstractmethod
    async def _queue_next_turn(self) -> None:
        """Queue the next turn (called from _on_turn_end)."""
        pass
