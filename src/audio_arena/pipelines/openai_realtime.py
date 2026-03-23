"""OpenAI Realtime helpers for explicit tool-result delivery and history seeding.

This file exists because OpenAI Realtime needs client-side event choreography that
the generic pipeline path does not handle:
- after a tool call, send ``function_call_output`` as a conversation item
- only send ``response.create`` after the active response has finished
- when server VAD is disabled, manually commit input audio and trigger a response
- seed prior turns into the live session for rehydrated runs
"""

import asyncio
import json
import time
import uuid
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Optional

from loguru import logger

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallFromLLM
from pipecat.services.openai.realtime import events as rt_events
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService


class ReconnectOnDisconnectMixin:
    """Mixin for OpenAI-protocol LLM services that auto-reconnect on unexpected WS close.

    Provides:
    - ``_init_reconnection_callbacks(on_reconnecting, on_reconnected)`` — call from ``__init__``
    - ``_reconnect_on_disconnect()`` — reopen the WebSocket and fire callbacks
    - ``_handle_ws_close()`` — post-loop check; call at the end of ``_receive_task_handler``

    Used by both ``OpenAIRealtimeLLMServiceExplicitToolResult`` and ``XAIRealtimeLLMService``.
    """

    _on_reconnecting: Optional[Callable[[], None]]
    _on_reconnected: Optional[Callable[[], None]]

    def _init_reconnection_callbacks(
        self,
        on_reconnecting: Optional[Callable[[], None]] = None,
        on_reconnected: Optional[Callable[[], None]] = None,
    ):
        self._on_reconnecting = on_reconnecting
        self._on_reconnected = on_reconnected

    async def _reconnect_on_disconnect(self):
        """Reconnect after an unexpected WebSocket disconnection.

        Opens a new WebSocket session and signals the pipeline to retry
        the current turn. Conversation history is enriched into the system
        instructions by the on_reconnecting callback before the new session
        starts.
        """
        if self._on_reconnecting:
            try:
                self._on_reconnecting()
            except Exception as e:
                logger.warning(f"Error in on_reconnecting callback: {e}")

        self._api_session_ready = False
        old_ws = self._websocket
        self._websocket = None

        try:
            if old_ws:
                await old_ws.close()
        except Exception:
            pass

        logger.info("Establishing new WebSocket connection...")
        await self._connect()

        for _ in range(100):
            if self._api_session_ready:
                break
            await asyncio.sleep(0.1)

        if self._api_session_ready:
            logger.info("Reconnection successful, session ready")
            if self._on_reconnected:
                try:
                    self._on_reconnected()
                except Exception as e:
                    logger.warning(f"Error in on_reconnected callback: {e}")
        else:
            logger.error("Reconnection failed — session not ready after 10s timeout")

    async def _handle_ws_close(self):
        """Check WebSocket close code after the receive loop exits.

        Call this at the end of ``_receive_task_handler()`` to detect
        unexpected disconnections and trigger automatic reconnection.
        """
        if self._disconnecting:
            return

        close_code = self._websocket.close_code if self._websocket else None
        close_reason = self._websocket.close_reason if self._websocket else ""

        if close_code is not None and close_code != 1000:
            logger.warning(
                f"WebSocket closed unexpectedly "
                f"(code={close_code}, reason={close_reason}), reconnecting..."
            )
            await self._reconnect_on_disconnect()
        elif close_code is not None:
            logger.info(f"WebSocket closed normally (code={close_code})")


class OpenAIRealtimeLLMServiceExplicitToolResult(ReconnectOnDisconnectMixin, OpenAIRealtimeLLMService):
    """OpenAI Realtime service that explicitly sends tool results to the API.

    Completely takes over function call handling to avoid the
    "conversation_already_has_active_response" error that occurs when
    response.create is sent during an active response.

    Also detects unexpected WebSocket disconnections and reconnects
    automatically via ``ReconnectOnDisconnectMixin``.

    Flow:
    1. response.function_call_arguments.done fires (during active response)
    2. We run the tool handler ourselves (run_function_calls)
    3. We pre-mark the call as completed to prevent base class auto-send
    4. We send conversation.item.create (function_call_output) with our result
    5. The server acknowledges the created function_call_output item
    6. response.done fires (response is now complete)
    7. We send response.create to trigger the model to continue
    """

    def __init__(
        self,
        get_last_tool_result: Optional[Callable[[], dict]] = None,
        capture_tool_phase: Optional[Callable[[str, dict, dict], Any]] = None,
        should_abort_after_tool_call: Optional[Callable[[], bool]] = None,
        abort_after_tool_call: Optional[Callable[[], Any]] = None,
        on_reconnecting: Optional[Callable[[], None]] = None,
        on_reconnected: Optional[Callable[[], None]] = None,
        rehydration_history_items: Optional[list[rt_events.ConversationItem]] = None,
        session_backdoor_options: Optional[dict[str, int]] = None,
        websocket_event_log_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._get_last_tool_result = get_last_tool_result
        self._capture_tool_phase = capture_tool_phase
        self._should_abort_after_tool_call = should_abort_after_tool_call
        self._abort_after_tool_call = abort_after_tool_call
        self._init_reconnection_callbacks(on_reconnecting, on_reconnected)
        self._pending_response_create = False
        self._waiting_for_response_done_before_response_create = False
        self._pending_tool_output_item_ids: set[str] = set()
        self._rehydration_history_items = list(rehydration_history_items or [])
        self._rehydration_history_seeded = False
        self._awaiting_manual_audio_commit = False
        self._manual_turn_input_committed = False
        self._manual_response_in_flight = False
        self._last_manual_commit_monotonic = 0.0
        self._session_backdoor_options = dict(session_backdoor_options or {})
        self._websocket_event_log_path = (
            Path(websocket_event_log_path).expanduser().resolve()
            if websocket_event_log_path
            else None
        )

    @staticmethod
    def _sanitize_ws_event_payload(payload):
        """Keep the event stream readable by truncating large audio/base64 fields."""
        if payload is None or isinstance(payload, (str, int, float, bool)):
            return payload
        if isinstance(payload, Path):
            return str(payload)
        if hasattr(payload, "model_dump"):
            try:
                return OpenAIRealtimeLLMServiceExplicitToolResult._sanitize_ws_event_payload(
                    payload.model_dump(exclude_none=True)
                )
            except TypeError:
                return OpenAIRealtimeLLMServiceExplicitToolResult._sanitize_ws_event_payload(
                    payload.model_dump()
                )
        if is_dataclass(payload):
            return OpenAIRealtimeLLMServiceExplicitToolResult._sanitize_ws_event_payload(
                asdict(payload)
            )
        if isinstance(payload, dict):
            sanitized = {}
            for key, value in payload.items():
                if (
                    key in {"audio", "delta"}
                    and isinstance(value, str)
                    and len(value) > 160
                ):
                    sanitized[key] = {
                        "__truncated__": True,
                        "length": len(value),
                        "preview": value[:32],
                    }
                    continue
                sanitized[key] = OpenAIRealtimeLLMServiceExplicitToolResult._sanitize_ws_event_payload(
                    value
                )
            return sanitized
        if isinstance(payload, (list, tuple, set)):
            return [
                OpenAIRealtimeLLMServiceExplicitToolResult._sanitize_ws_event_payload(item)
                for item in payload
            ]
        if hasattr(payload, "__dict__"):
            try:
                return {
                    "__type__": type(payload).__name__,
                    "fields": OpenAIRealtimeLLMServiceExplicitToolResult._sanitize_ws_event_payload(
                        vars(payload)
                    ),
                }
            except TypeError:
                pass
        return {
            "__type__": type(payload).__name__,
            "repr": repr(payload),
        }

    def _record_ws_event(self, *, direction: str, payload, raw_text: Optional[str] = None) -> None:
        if self._websocket_event_log_path is None:
            return

        event_type = payload.get("type") if isinstance(payload, dict) else None
        record = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "direction": direction,
            "event_type": event_type,
            "payload": self._sanitize_ws_event_payload(payload),
        }
        if raw_text is not None:
            record["raw_length"] = len(raw_text)

        self._websocket_event_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._websocket_event_log_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def _ws_send(self, realtime_message):
        self._record_ws_event(direction="outgoing", payload=realtime_message)
        await super()._ws_send(realtime_message)

    @staticmethod
    def _turn_detection_disabled(turn_detection) -> bool:
        """Treat both legacy False and explicit null as manual turn handling."""
        return turn_detection is False or turn_detection is None

    def _build_session_update_payload(self, settings: rt_events.SessionProperties) -> dict:
        """Serialize session settings and merge optional backdoor fields."""
        session_payload = settings.model_dump(exclude_none=True)
        if (
            settings.audio
            and settings.audio.input
            and self._turn_detection_disabled(settings.audio.input.turn_detection)
        ):
            session_payload.setdefault("audio", {}).setdefault("input", {})["turn_detection"] = None
        if self._session_backdoor_options:
            session_payload["backdoor_options"] = self._session_backdoor_options
        return {
            "type": "session.update",
            "session": session_payload,
        }

    async def _update_settings(self):
        """Send session.update, preserving Pipecat defaults plus backdoor options."""
        settings = self._session_properties
        adapter = self.get_llm_adapter()

        if self._context:
            llm_invocation_params = adapter.get_llm_invocation_params(self._context)

            if llm_invocation_params["tools"]:
                settings.tools = llm_invocation_params["tools"]

            if llm_invocation_params["system_instruction"]:
                settings.instructions = llm_invocation_params["system_instruction"]

        if settings.tools and isinstance(settings.tools, ToolsSchema):
            settings.tools = adapter.from_standard_tools(settings.tools)

        payload = self._build_session_update_payload(settings)
        await self._ws_send(payload)
        if self._session_backdoor_options:
            logger.info(
                "[OpenAI Realtime] Sent session.update with backdoor_options="
                f"{self._session_backdoor_options}"
            )

    async def _maybe_send_deferred_response_create(self, trigger: str) -> None:
        if not self._pending_response_create:
            return

        if self._waiting_for_response_done_before_response_create:
            logger.debug(
                "[OpenAI Realtime] Deferred response.create still waiting for response.done "
                f"(trigger={trigger})"
            )
            return

        if self._pending_tool_output_item_ids:
            logger.debug(
                "[OpenAI Realtime] Deferred response.create still waiting for tool-output ack(s) "
                f"{sorted(self._pending_tool_output_item_ids)} (trigger={trigger})"
            )
            return

        self._pending_response_create = False
        await self.send_client_event(rt_events.ResponseCreateEvent())
        logger.info(
            "[OpenAI Realtime] Sent deferred response.create after response.done and "
            f"tool-output ack (trigger={trigger})"
        )

    async def _handle_tool_output_item_event(self, evt, phase: str) -> None:
        item = evt.item
        if item.type != "function_call_output" or item.id not in self._pending_tool_output_item_ids:
            return

        self._pending_tool_output_item_ids.remove(item.id)
        logger.info(
            "[OpenAI Realtime] Tool output item acknowledged by server "
            f"(phase={phase}, item_id={item.id}, call_id={item.call_id})"
        )
        await self._maybe_send_deferred_response_create(
            trigger=f"conversation.item.{phase}:{item.id}"
        )

    def _manual_turn_handling_active(self) -> bool:
        return bool(
            self._session_properties.audio
            and self._session_properties.audio.input
            and self._turn_detection_disabled(self._session_properties.audio.input.turn_detection)
        )

    def reset_manual_turn_state(self) -> None:
        """Clear per-turn debounce state before queuing fresh user audio."""
        self._awaiting_manual_audio_commit = False
        self._manual_turn_input_committed = False
        self._manual_response_in_flight = False
        self._last_manual_commit_monotonic = 0.0

    async def seed_rehydration_history(self) -> None:
        """Seed prior turns into the live session with conversation.item.create."""
        if self._rehydration_history_seeded or not self._rehydration_history_items:
            return

        for _ in range(100):
            if self._api_session_ready:
                break
            await asyncio.sleep(0.1)

        if not self._api_session_ready:
            raise RuntimeError("OpenAI Realtime session was not ready before rehydration seeding")

        for item in self._rehydration_history_items:
            self._messages_added_manually[item.id] = True
            await self.send_client_event(rt_events.ConversationItemCreateEvent(item=item))

        self._rehydration_history_seeded = True
        logger.info(
            f"[OpenAI Realtime] Seeded {len(self._rehydration_history_items)} rehydration "
            "history items via conversation.item.create"
        )

    async def _handle_user_stopped_speaking(self, frame):
        """Debounce manual commit/response.create when server-side VAD is disabled."""
        if not self._manual_turn_handling_active():
            await super()._handle_user_stopped_speaking(frame)
            return

        if self._awaiting_manual_audio_commit:
            logger.warning("[OpenAI Realtime] Already waiting for input_audio_buffer.committed; ignoring duplicate stop event")
            return

        if self._manual_turn_input_committed:
            logger.debug(
                "[OpenAI Realtime] Ignoring duplicate user stop event after turn audio was already committed"
            )
            return

        if self._manual_response_in_flight:
            logger.debug("[OpenAI Realtime] Ignoring user stop event while response is still in flight")
            return

        now = time.monotonic()
        if now - self._last_manual_commit_monotonic < 0.75:
            logger.debug("[OpenAI Realtime] Debouncing duplicate user stop event")
            return

        # With turn_detection=null the server will not auto-commit buffered audio
        # or start a response for us. We have to do both explicitly here.
        self._awaiting_manual_audio_commit = True
        await self.send_client_event(rt_events.InputAudioBufferCommitEvent())
        await self.send_client_event(rt_events.ResponseCreateEvent())
        self._manual_response_in_flight = True
        self._last_manual_commit_monotonic = now
        logger.info("[OpenAI Realtime] Sent input_audio_buffer.commit and response.create")

    async def _handle_evt_input_audio_buffer_committed(self, evt):
        if not self._manual_turn_handling_active():
            return
        if not self._awaiting_manual_audio_commit:
            logger.debug(
                f"[OpenAI Realtime] input_audio_buffer.committed received without pending manual commit (item_id={evt.item_id})"
            )
            return

        self._awaiting_manual_audio_commit = False
        self._manual_turn_input_committed = True
        logger.info(f"[OpenAI Realtime] Input audio committed (item_id={evt.item_id})")

    async def _handle_evt_function_call_arguments_done(self, evt):
        """Run the tool, push function_call_output, and continue after response.done."""
        call_id = evt.call_id
        try:
            args = json.loads(evt.arguments)
            function_call_item = self._pending_function_calls.get(call_id)
            if function_call_item:
                del self._pending_function_calls[call_id]
                # Mark the call as completed before run_function_calls() returns so
                # the base class does not try to auto-send the tool result itself.
                self._completed_tool_calls.add(call_id)

                function_calls = [
                    FunctionCallFromLLM(
                        context=self._context,
                        tool_call_id=call_id,
                        function_name=function_call_item.name,
                        arguments=args,
                    )
                ]
                await self.run_function_calls(function_calls)
                logger.debug(f"[OpenAI Realtime] Processed function call: {function_call_item.name}")
            else:
                logger.warning(f"[OpenAI Realtime] No tracked function call for call_id: {call_id}")
                return
        except Exception as e:
            logger.error(f"[OpenAI Realtime] Failed to process function call: {e}")
            return

        tool_result = (
            self._get_last_tool_result()
            if self._get_last_tool_result
            else {"status": "success"}
        )
        capture_tool_phase = getattr(self, "_capture_tool_phase", None)
        if capture_tool_phase:
            capture_tool_phase(function_call_item.name, args, tool_result)
        should_abort_after_tool_call = getattr(
            self,
            "_should_abort_after_tool_call",
            None,
        )
        if (
            should_abort_after_tool_call
            and should_abort_after_tool_call()
        ):
            logger.info(
                "[OpenAI Realtime] Tool capture phase complete; skipping "
                "function_call_output and ending the current session"
            )
            abort_after_tool_call = getattr(
                self,
                "_abort_after_tool_call",
                None,
            )
            if abort_after_tool_call:
                maybe_awaitable = abort_after_tool_call()
                if asyncio.iscoroutine(maybe_awaitable):
                    await maybe_awaitable
            return

        output_json = json.dumps(
            self._sanitize_ws_event_payload(tool_result),
            ensure_ascii=False,
        )
        tool_output_item_id = uuid.uuid4().hex
        self._pending_tool_output_item_ids.add(tool_output_item_id)
        create_ev = rt_events.ConversationItemCreateEvent(
            item=rt_events.ConversationItem(
                id=tool_output_item_id,
                type="function_call_output",
                call_id=call_id,
                output=output_json,
            )
        )
        # response.function_call_arguments.done arrives while the original response
        # is still active. We need two separate gates before continuing:
        # 1. response.done, otherwise response.create can hit
        #    conversation_already_has_active_response
        # 2. a server-side ack that the function_call_output item was committed
        #    to conversation state
        self._pending_response_create = True
        self._waiting_for_response_done_before_response_create = True

        try:
            await self.send_client_event(create_ev)
        except Exception:
            self._pending_tool_output_item_ids.discard(tool_output_item_id)
            if not self._pending_tool_output_item_ids:
                self._pending_response_create = False
                self._waiting_for_response_done_before_response_create = False
            raise

        logger.info(f"[OpenAI Realtime] Sent function_call_output for call_id={call_id}")
        logger.info(
            "[OpenAI Realtime] Deferred response.create until response.done and "
            f"tool-output ack (item_id={tool_output_item_id})"
        )

    async def _handle_evt_response_done(self, evt):
        """Handle response.done: call super, then send deferred response.create if needed.

        This fires after the active response is complete, so it is safe to
        send response.create to trigger the model to continue with the tool
        result in context.
        """
        await super()._handle_evt_response_done(evt)
        self._manual_response_in_flight = False

        if self._pending_response_create:
            self._waiting_for_response_done_before_response_create = False
            await self._maybe_send_deferred_response_create(trigger="response.done")

    async def _handle_evt_conversation_item_added(self, evt):
        await super()._handle_evt_conversation_item_added(evt)
        await self._handle_tool_output_item_event(evt, phase="added")

    async def _handle_evt_conversation_item_done(self, evt):
        await super()._handle_evt_conversation_item_done(evt)
        await self._handle_tool_output_item_event(evt, phase="done")

    async def _handle_evt_error(self, evt):
        error = evt.error
        if (
            self._manual_turn_handling_active()
            and error is not None
            and error.code == "conversation_already_has_active_response"
        ):
            self._awaiting_manual_audio_commit = False
            self._manual_response_in_flight = True
            logger.warning(
                "[OpenAI Realtime] Ignoring conversation_already_has_active_response in manual mode; "
                "waiting for current response to finish"
            )
            return True
        await super()._handle_evt_error(evt)
        return False

    async def _receive_task_handler(self):
        """Extend the base receive loop with input_audio_buffer.committed handling."""
        async for message in self._websocket:
            try:
                raw_payload = json.loads(message)
            except json.JSONDecodeError:
                raw_payload = {"type": "__unparsed__", "raw_preview": message[:200]}
            self._record_ws_event(
                direction="incoming",
                payload=raw_payload,
                raw_text=message,
            )
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
            elif evt.type == "input_audio_buffer.committed":
                await self._handle_evt_input_audio_buffer_committed(evt)
            elif evt.type == "response.output_text.delta":
                await self._handle_evt_text_delta(evt)
            elif evt.type == "response.output_audio_transcript.delta":
                await self._handle_evt_audio_transcript_delta(evt)
            elif evt.type == "response.function_call_arguments.done":
                await self._handle_evt_function_call_arguments_done(evt)
            elif evt.type == "error":
                if not await self._maybe_handle_evt_retrieve_conversation_item_error(evt):
                    handled = await self._handle_evt_error(evt)
                    if not handled:
                        return
        await self._handle_ws_close()
