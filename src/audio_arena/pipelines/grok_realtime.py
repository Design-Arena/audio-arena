"""Grok Realtime pipeline for xAI Grok Voice Agent API.

This pipeline extends the realtime pipeline to support xAI's Grok speech-to-speech API,
which is compatible with OpenAI's Realtime API but has several protocol differences:

- Sends "ping" events that must be ignored
- Sends "conversation.created" before "session.created"
- Tool format uses nested "function" key (Chat Completions style)
- Function calls are returned in response.done output array
- Some validation fields have non-standard values

Usage:
    uv run audio-arena run conversation_bench --model grok-realtime --service openai-realtime
"""

import json
import os
from typing import Callable, Optional

from loguru import logger

from pipecat.frames.frames import LLMFullResponseEndFrame
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.llm_service import FunctionCallFromLLM
from pipecat.services.openai.realtime import events as rt_events
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService

from audio_arena.pipelines.realtime import RealtimePipeline


class XAIRealtimeLLMService(OpenAIRealtimeLLMService):
    """xAI Grok Voice Agent API service.

    Extends OpenAI Realtime service to handle xAI-specific protocol differences:
    - "ping" events that xAI sends but OpenAI doesn't
    - "conversation.created" event before session.created
    - Tool format conversion (flat to nested "function" key)
    - Function calls in response.done output array
    - Validation workarounds for non-standard field values
    """

    def __init__(self, get_last_tool_result: Optional[Callable[[], dict]] = None, **kwargs):
        super().__init__(**kwargs)
        self._get_last_tool_result = get_last_tool_result
        # Track if we're using manual turn handling (VAD disabled)
        session_props = kwargs.get("session_properties")
        self._manual_turn_handling = (
            session_props and
            session_props.audio and
            session_props.audio.input and
            session_props.audio.input.turn_detection is False
        )

    async def _handle_user_stopped_speaking(self, frame):
        """Override to manually commit and create response when VAD is disabled."""
        if self._manual_turn_handling:
            logger.info("[xAI] User stopped speaking, committing audio and creating response")
            await self.send_client_event(rt_events.InputAudioBufferCommitEvent())
            await self.send_client_event(rt_events.ResponseCreateEvent())
        else:
            await super()._handle_user_stopped_speaking(frame)

    async def _handle_xai_response_done(self, raw_event):
        """Handle xAI's response.done format which includes function calls in the output array.

        We run each tool via run_function_calls() (so pipeline/transcript see it), then
        explicitly send the tool result back to xAI with conversation.item.create
        (function_call_output). Otherwise the audio model never receives the result and
        cannot continue the conversation with the tool output in context. We then send
        response.create so the model only speaks after getting the result(s).
        """
        response = raw_event.get("response", {})
        output_items = response.get("output", [])
        function_call_ids_handled = []

        for item in output_items:
            item_type = item.get("type")
            if item_type == "function_call":
                call_id = item.get("call_id")
                func_name = item.get("name")
                arguments_str = item.get("arguments", "{}")

                logger.info(f"[xAI] Function call detected in response.done: {func_name}")
                logger.debug(f"[xAI]   call_id: {call_id}")
                logger.debug(f"[xAI]   arguments: {arguments_str}")

                try:
                    args = json.loads(arguments_str)
                    function_calls = [
                        FunctionCallFromLLM(
                            context=self._context,
                            tool_call_id=call_id,
                            function_name=func_name,
                            arguments=args,
                        )
                    ]
                    await self.run_function_calls(function_calls)
                    function_call_ids_handled.append(call_id)
                    logger.info(f"[xAI] Executed function call: {func_name}")

                    # Send this tool's result to xAI so the model gets it before continuing.
                    tool_result = (
                        self._get_last_tool_result()
                        if self._get_last_tool_result
                        else {"status": "success"}
                    )
                    output_json = json.dumps(tool_result)
                    create_ev = rt_events.ConversationItemCreateEvent(
                        item=rt_events.ConversationItem(
                            type="function_call_output",
                            call_id=call_id,
                            output=output_json,
                        )
                    )
                    await self.send_client_event(create_ev)
                    logger.info(f"[xAI] Sent function_call_output for call_id={call_id}")
                except Exception as e:
                    logger.error(f"[xAI] Failed to execute function call {func_name}: {e}")

        # Trigger model to continue with the tool result(s) in context.
        if function_call_ids_handled:
            await self.send_client_event(rt_events.ResponseCreateEvent())
            logger.info("[xAI] Sent response.create to continue after tool result(s)")

    async def send_client_event(self, event):
        """Override to convert tools to xAI format (Chat Completions style with nested 'function' key)."""
        if hasattr(event, 'type') and event.type == "session.update":
            session = getattr(event, 'session', None)
            if session and hasattr(session, 'tools') and session.tools:
                converted_tools = []
                for tool in session.tools:
                    # xAI expects: {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
                    # OpenAI Realtime sends: {"type": "function", "name": ..., "description": ..., "parameters": ...}
                    if isinstance(tool, dict) and tool.get("type") == "function" and "function" not in tool:
                        converted_tool = {
                            "type": "function",
                            "function": {
                                "name": tool.get("name"),
                                "description": tool.get("description"),
                                "parameters": tool.get("parameters", {}),
                            }
                        }
                        converted_tools.append(converted_tool)
                    else:
                        converted_tools.append(tool)
                session.tools = converted_tools
                logger.info(f"[xAI] Converted {len(converted_tools)} tools to xAI format")
        await super().send_client_event(event)

    async def _update_settings(self):
        """Override to log the tools being sent to xAI."""
        if self._session_properties.tools:
            logger.debug(f"[xAI] Tools in session_properties: {type(self._session_properties.tools)}")
            if hasattr(self._session_properties.tools, 'standard_tools'):
                for t in self._session_properties.tools.standard_tools:
                    logger.debug(f"[xAI]   - Tool: {t.name}")
        else:
            logger.warning("[xAI] No tools in session_properties!")

        await super()._update_settings()
        logger.debug("[xAI] Session update sent to xAI")

    async def _receive_task_handler(self):
        """Override to handle xAI-specific events like ping."""
        async for message in self._websocket:
            try:
                raw_event = json.loads(message)
                event_type = raw_event.get("type")

                logger.info(f"[xAI] Received event: {event_type}")

                # Handle xAI-specific events
                if event_type == "ping":
                    logger.debug("Received xAI ping event, ignoring")
                    continue

                if event_type == "conversation.created":
                    logger.info("[xAI] Conversation created, treating as session ready")
                    await self._update_settings()
                    continue

                if event_type == "session.updated":
                    # xAI returns non-standard values like tool_choice="not implemented"
                    session_data = raw_event.get("session", {})
                    tools_in_response = session_data.get("tools", [])
                    logger.debug(f"[xAI] session.updated - tools count: {len(tools_in_response)}")

                    # Handle without pydantic validation
                    logger.info("[xAI] Session updated")
                    self._api_session_ready = True
                    if self._run_llm_when_api_session_ready:
                        self._run_llm_when_api_session_ready = False
                        await self._create_response()
                    continue

                if event_type == "response.created":
                    # xAI sends empty usage object {} which fails pydantic validation
                    logger.info("[xAI] Response created")
                    continue

                if event_type == "input_audio_buffer.committed":
                    logger.debug("[xAI] Audio buffer committed")
                    continue

                # xAI sends these events with different formats that fail pydantic validation
                # Handle them minimally to avoid error spam in logs
                if event_type == "response.content_part.added":
                    logger.debug("[xAI] Content part added")
                    continue

                if event_type == "response.content_part.done":
                    logger.debug("[xAI] Content part done")
                    continue

                if event_type == "response.output_item.added":
                    logger.debug("[xAI] Output item added")
                    continue

                if event_type == "response.output_item.done":
                    logger.debug("[xAI] Output item done")
                    continue

                if event_type == "response.function_call_arguments.delta":
                    # xAI sends function call args progressively but format differs
                    logger.debug("[xAI] Function call arguments delta")
                    continue

                if event_type == "response.function_call_arguments.done":
                    # xAI format differs from OpenAI, handle in response.done instead
                    logger.debug("[xAI] Function call arguments done")
                    continue

                # Handle conversation.item.added for tool-related items that fail pydantic
                if event_type == "conversation.item.added":
                    item = raw_event.get("item", {})
                    item_role = item.get("role")
                    item_type = item.get("type")
                    # Skip validation for tool-related items (function_call, function_call_output)
                    if item_role == "tool" or item_type in ("function_call", "function_call_output"):
                        logger.debug(f"[xAI] Conversation item added (type={item_type}, role={item_role})")
                        continue
                    # For user/assistant items, fall through to standard parser

                if event_type == "response.done":
                    # xAI includes function calls in response.done output
                    await self._handle_xai_response_done(raw_event)
                    await self.push_frame(LLMFullResponseEndFrame())
                    self._current_assistant_response = None
                    continue

                # Use standard parser for other events
                evt = rt_events.parse_server_event(message)
                if evt.type == "session.created":
                    await self._handle_evt_session_created(evt)
                elif evt.type == "session.updated":
                    await self._handle_evt_session_updated(evt)
                elif evt.type == "response.output_audio.delta":
                    await self._handle_evt_audio_delta(evt)
                elif evt.type == "response.output_audio.done":
                    await self._handle_evt_audio_done(evt)
                elif evt.type == "conversation.item.added":
                    await self._handle_evt_conversation_item_added(evt)
                elif evt.type == "conversation.item.done":
                    await self._handle_evt_conversation_item_done(evt)
                elif evt.type == "conversation.item.input_audio_transcription.delta":
                    await self._handle_evt_input_audio_transcription_delta(evt)
                elif evt.type == "conversation.item.input_audio_transcription.completed":
                    await self.handle_evt_input_audio_transcription_completed(evt)
                elif evt.type == "conversation.item.retrieved":
                    await self._handle_conversation_item_retrieved(evt)
                elif evt.type == "response.done":
                    await self._handle_evt_response_done(evt)
                elif evt.type == "input_audio_buffer.speech_started":
                    await self._handle_evt_speech_started(evt)
                elif evt.type == "input_audio_buffer.speech_stopped":
                    await self._handle_evt_speech_stopped(evt)
                elif evt.type == "response.output_text.delta":
                    await self._handle_evt_text_delta(evt)
                elif evt.type == "response.output_audio_transcript.delta":
                    await self._handle_evt_audio_transcript_delta(evt)
                elif evt.type == "response.function_call_arguments.done":
                    await self._handle_evt_function_call_arguments_done(evt)
                elif evt.type == "error":
                    if not await self._maybe_handle_evt_retrieve_conversation_item_error(evt):
                        await self._handle_evt_error(evt)
                        return
                else:
                    logger.debug(f"Ignoring unhandled event type: {evt.type}")
            except Exception as e:
                logger.warning(f"Error processing xAI event: {e}")


class GrokRealtimePipeline(RealtimePipeline):
    """Pipeline for xAI Grok Voice Agent API.

    Extends RealtimePipeline to use XAI-specific configuration:
    - XAI_API_KEY environment variable
    - wss://api.x.ai/v1/realtime endpoint
    - XAIRealtimeLLMService for protocol handling
    - xAI-specific VAD settings
    """

    requires_service = False  # This pipeline creates its own XAIRealtimeLLMService

    def _create_llm(
        self, service_class: Optional[type], model: str
    ) -> FrameProcessor:
        """Create xAI Grok Voice Agent LLM service.

        Args:
            service_class: Ignored - always uses XAIRealtimeLLMService.
            model: Model name (e.g., "grok-realtime").

        Returns:
            Configured XAIRealtimeLLMService instance.
        """
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise EnvironmentError("XAI_API_KEY environment variable is required for Grok models")

        base_url = "wss://api.x.ai/v1/realtime"
        logger.info(f"Using xAI Grok Voice Agent API at {base_url}")

        system_instruction = getattr(self.benchmark, "system_instruction", "")
        tools = getattr(self.benchmark, "tools_schema", None)

        # xAI server VAD with longer silence duration to reduce false interruptions
        # Note: xAI doesn't support disabling VAD (turn_detection=False causes error)
        audio_config = rt_events.AudioConfiguration(
            input=rt_events.AudioInput(
                turn_detection=rt_events.TurnDetection(
                    type="server_vad",
                    threshold=0.7,  # Higher threshold = less sensitive to speech detection
                    prefix_padding_ms=500,  # More padding before speech
                    silence_duration_ms=800,  # Longer silence before turn ends
                )
            )
        )

        session_props = rt_events.SessionProperties(
            instructions=system_instruction,
            tools=tools,
            audio=audio_config,
        )

        return XAIRealtimeLLMService(
            api_key=api_key,
            model=model,
            base_url=base_url,
            system_instruction=system_instruction,
            session_properties=session_props,
            get_last_tool_result=lambda: getattr(
                self, "_last_tool_result", {"status": "success"}
            ),
        )
