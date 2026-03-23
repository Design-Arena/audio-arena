"""Realtime pipeline for OpenAI Realtime and Gemini Live models.

This pipeline works with speech-to-speech models that use audio input/output:
- OpenAI Realtime (gpt-realtime)
- Gemini Live (gemini-*-native-audio-*)
- Ultravox (ultravox-v0.7)

Pipeline:
    paced_input → context_aggregator.user() → transcript.user() →
    llm → ToolCallRecorder → assistant_shim → audio_buffer → context_aggregator.assistant()
"""

import asyncio
import json
import os
import re
import time
import wave
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import soundfile as sf
from loguru import logger
from pipecat.frames.frames import (
    BotSpeakingFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMTextFrame,
    LLMMessagesAppendFrame,
    LLMRunFrame,
    MetricsFrame,
    OutputAudioRawFrame,
    TranscriptionMessage,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from audio_arena.processors.audio_buffer import WallClockAlignedAudioBufferProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.services.openai.realtime import events as rt_events
from pipecat.services.ultravox.llm import OneShotInputParams
from pipecat.transports.base_transport import TransportParams
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams

from audio_arena.pipelines.base import BasePipeline


class ResamplingSileroVAD(SileroVADAnalyzer):
    """SileroVADAnalyzer that resamples audio from input rate to 16kHz.

    Silero VAD only supports 16kHz or 8kHz. This subclass resamples incoming
    audio (e.g., 24kHz) to 16kHz before VAD processing.
    """

    def __init__(self, **kwargs):
        super().__init__(sample_rate=16000, **kwargs)
        self._input_sample_rate: int = 16000

    def set_sample_rate(self, sample_rate: int):
        # Store the input sample rate for resampling, but tell parent we're using 16kHz
        self._input_sample_rate = sample_rate
        super().set_sample_rate(16000)

    async def analyze_audio(self, buffer: bytes):
        # Resample before buffering/analysis if input rate differs from 16kHz
        if self._input_sample_rate != 16000:
            audio_int16 = np.frombuffer(buffer, np.int16).astype(np.float32)
            # Simple linear interpolation resampling
            ratio = 16000 / self._input_sample_rate
            new_length = int(len(audio_int16) * ratio)
            if new_length > 0:
                indices = np.linspace(0, len(audio_int16) - 1, new_length)
                resampled = np.interp(indices, np.arange(len(audio_int16)), audio_int16)
                buffer = resampled.astype(np.int16).tobytes()
        return await super().analyze_audio(buffer)


from audio_arena.processors.tool_call_recorder import ToolCallRecorder
from audio_arena.processors.tts_transcript import (
    TTSStoppedAssistantTranscriptProcessor,
)
from audio_arena.transports.null_audio_output import NullAudioOutputTransport
from audio_arena.transports.paced_input import PacedInputTransport


class TurnGate(FrameProcessor):
    """Gates turn advancement until bot finishes speaking.

    This processor coordinates between transcript completion and audio playback:
    1. Stores the pending assistant transcript when received (on TTSStoppedFrame)
    2. Waits for BotStoppedSpeakingFrame (from NullAudioOutputTransport)
    3. Adds a small delay to ensure all audio has been processed
    4. Only then triggers the turn-end callback

    This prevents the next turn's audio from being sent while the bot is
    still "speaking" (outputting audio frames).

    Additionally detects "empty responses" where the model returns only control
    tokens with no actual audio, which would otherwise cause the conversation to stall.
    """

    def __init__(
        self,
        on_turn_ready: Callable[[str], Any],
        audio_drain_delay: float = 0.5,
        no_response_timeout: float = 30.0,
        on_greeting_done: Optional[Callable[[], None]] = None,
        **kwargs,
    ):
        """Initialize the turn gate.

        Args:
            on_turn_ready: Async callback to invoke when turn is ready to advance.
                          Called with the assistant's response text.
            audio_drain_delay: Seconds to wait after BotStoppedSpeakingFrame before
                              triggering turn end. This allows remaining audio frames
                              to drain through the pipeline. Default 0.5s works well
                              when BOT_VAD_STOP_SECS is increased to 2s.
            no_response_timeout: Seconds to wait after UserStoppedSpeakingFrame before
                                declaring no response if no TTSStartedFrame arrived.
            on_greeting_done: Optional callback to invoke when the initial greeting
                             completes (first BotStoppedSpeakingFrame). Used to signal
                             that user audio can start playing.
        """
        super().__init__(**kwargs)
        self._on_turn_ready = on_turn_ready
        self._audio_drain_delay = audio_drain_delay
        self._no_response_timeout = no_response_timeout
        self._pending_transcript: Optional[str] = None
        self._turn_transcript_parts: list[str] = []
        self._bot_speaking = False
        self._turn_end_task: Optional[asyncio.Task] = None

        # TTS state tracking for response detection
        self._tts_started = False
        self._no_response_check_task: Optional[asyncio.Task] = None
        self._on_empty_response: Optional[Callable[[str], None]] = None
        # Recent raw output activity is a stronger signal than turn state alone.
        # This lets the watchdog treat in-flight text/audio as active even if the
        # higher-level TTS/Bot state has not caught up yet.
        self._last_output_activity_monotonic: float = 0.0
        self._recent_output_activity_window_secs: float = 3.0

        # Initial greeting detection
        self._on_greeting_done = on_greeting_done
        self._on_greeting_started: Optional[Callable[[], None]] = None
        self._greeting_signaled = False
        self._greeting_started_signaled = False

    def set_greeting_started_callback(self, callback: Callable[[], None]):
        """Set callback for when initial greeting starts (first BotStartedSpeakingFrame).

        Args:
            callback: Called when bot starts speaking for the first time.
        """
        self._on_greeting_started = callback

    def set_empty_response_callback(self, callback: Callable[[str], None]):
        """Set callback for when empty/no response is detected.

        Args:
            callback: Called with reason string ("empty_response" or "no_response").
        """
        self._on_empty_response = callback

    def set_pending_transcript(self, text: str):
        """Store transcript received from assistant_shim.

        Called by the transcript handler when assistant response is complete.
        The turn won't advance until BotStoppedSpeakingFrame is received.

        All segments within a single turn are accumulated in
        ``_turn_transcript_parts`` so that multi-segment responses
        (e.g. pre-tool preamble + post-tool answer) are fully preserved.
        """
        self._turn_transcript_parts.append(text)
        logger.info(
            f"[TurnGate] Storing pending transcript ({len(text)} chars, "
            f"parts={len(self._turn_transcript_parts)}): "
            f"{text[:80]}{'...' if len(text) > 80 else ''} | "
            f"bot_speaking={self._bot_speaking}"
        )
        self._pending_transcript = text

    def clear_pending(self):
        """Clear any pending transcript and TTS state (e.g., on reconnection or empty response)."""
        self._pending_transcript = None
        self._turn_transcript_parts = []
        self._tts_started = False
        self._bot_speaking = False
        self._last_output_activity_monotonic = 0.0
        if self._turn_end_task and not self._turn_end_task.done():
            self._turn_end_task.cancel()
            self._turn_end_task = None
        if self._no_response_check_task and not self._no_response_check_task.done():
            self._no_response_check_task.cancel()
            self._no_response_check_task = None

    def has_response_activity(self) -> bool:
        """Return True when the model has started producing output for this turn."""
        recent_output_activity = (
            self._last_output_activity_monotonic > 0
            and (time.monotonic() - self._last_output_activity_monotonic)
            <= self._recent_output_activity_window_secs
        )
        return bool(
            self._bot_speaking
            or self._tts_started
            or self._pending_transcript is not None
            or self._turn_transcript_parts
            or recent_output_activity
        )

    def _mark_output_activity(self):
        """Record recent downstream output and stop the no-response timer."""
        self._last_output_activity_monotonic = time.monotonic()
        if self._no_response_check_task and not self._no_response_check_task.done():
            self._no_response_check_task.cancel()
            self._no_response_check_task = None

    def _is_empty_transcript(self, text: str) -> bool:
        """Check if transcript is empty or only contains control tokens."""
        # Remove control tokens like <ctrl46>
        stripped = re.sub(r"<ctrl\d+>", "", text)
        return len(stripped.strip()) == 0

    async def _delayed_turn_end(self, text: str):
        """Wait for audio to drain, then trigger turn end.

        Uses the accumulated ``_turn_transcript_parts`` (joined with newline)
        so that multi-segment responses (pre-tool preamble + post-tool answer)
        are fully captured. Falls back to *text* if no parts were collected.
        """
        try:
            logger.info(f"[TurnGate] Waiting {self._audio_drain_delay}s for audio to drain...")
            await asyncio.sleep(self._audio_drain_delay)
            if self._turn_transcript_parts:
                final = "\n".join(self._turn_transcript_parts)
                if len(self._turn_transcript_parts) > 1:
                    logger.info(
                        f"[TurnGate] Joined {len(self._turn_transcript_parts)} transcript parts "
                        f"({len(final)} chars total)"
                    )
            else:
                final = text
            self._turn_transcript_parts = []
            logger.info(f"[TurnGate] Triggering turn end with transcript ({len(final)} chars)")
            await self._on_turn_ready(final)
        except asyncio.CancelledError:
            logger.info("[TurnGate] Turn end cancelled (likely bot started speaking again)")

    async def _check_no_response(self):
        """Check if model never responded after user stopped speaking."""
        try:
            await asyncio.sleep(self._no_response_timeout)
            # Treat any downstream text/audio activity as a valid response, even if
            # the coarser TTS/bot state has not updated yet.
            if not self.has_response_activity():
                logger.warning(
                    f"[NO_RESPONSE] No TTS response after {self._no_response_timeout}s"
                )
                if self._on_empty_response:
                    self._on_empty_response("no_response")
        except asyncio.CancelledError:
            pass

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Watch for TTS and Bot speaking frames to manage turn advancement."""
        await super().process_frame(frame, direction)

        # Raw downstream output is the most reliable "model is actively responding"
        # signal. Track it directly so watchdogs do not race against delayed
        # BotStopped/TTS bookkeeping.
        if isinstance(
            frame,
            (
                BotSpeakingFrame,
                BotStartedSpeakingFrame,
                LLMTextFrame,
                TTSAudioRawFrame,
                TTSTextFrame,
            ),
        ):
            self._mark_output_activity()

        # allow_interruptions=False should prevent this, but log if it still
        # arrives via a Pipecat internal edge case.
        if isinstance(frame, InterruptionFrame):
            pending_len = len(self._pending_transcript) if self._pending_transcript else 0
            logger.warning(
                f"[TurnGate] InterruptionFrame (unexpected — interruptions disabled) | "
                f"bot_speaking={self._bot_speaking} tts_started={self._tts_started} "
                f"pending_transcript={pending_len} chars "
                f"has_turn_end_task={self._turn_end_task is not None and not self._turn_end_task.done()}"
            )

        # Start no-response timeout when user stops speaking.
        # Check both VADUserStoppedSpeakingFrame (client-side VAD) and
        # UserStoppedSpeakingFrame (server-side VAD, e.g. xAI/OpenAI Realtime)
        # so the timeout fires regardless of which VAD detected speech end.
        if isinstance(frame, (VADUserStoppedSpeakingFrame, UserStoppedSpeakingFrame)):
            # Cancel any existing no-response check and start a new one
            if self._no_response_check_task and not self._no_response_check_task.done():
                self._no_response_check_task.cancel()
            self._no_response_check_task = asyncio.create_task(self._check_no_response())

        # Track TTS state for empty response detection
        if isinstance(frame, TTSStartedFrame):
            self._tts_started = True
            # Cancel no-response check - model is responding
            if self._no_response_check_task and not self._no_response_check_task.done():
                self._no_response_check_task.cancel()
                self._no_response_check_task = None

        elif isinstance(frame, TTSStoppedFrame):
            self._tts_started = False
            # If TTS stopped but bot never started speaking, this is an empty response
            # We can detect immediately - no timeout needed
            if self._pending_transcript is not None and not self._bot_speaking:
                if self._is_empty_transcript(self._pending_transcript):
                    logger.warning(
                        f"[EMPTY_RESPONSE] No bot audio generated, "
                        f"transcript='{self._pending_transcript[:50]}...'"
                    )
                    if self._on_empty_response:
                        self._on_empty_response("empty_response")
                    self._pending_transcript = None
                    self._turn_transcript_parts = []

        # If bot starts speaking, cancel pending checks and turn end
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True

            # Signal greeting started on first BotStartedSpeakingFrame
            # This allows the pipeline to know the bot is greeting
            if not self._greeting_started_signaled and self._on_greeting_started:
                self._greeting_started_signaled = True
                logger.info("[TurnGate] Initial greeting started, signaling greeting started")
                self._on_greeting_started()

            # Cancel no-response check - bot is speaking
            if self._no_response_check_task and not self._no_response_check_task.done():
                self._no_response_check_task.cancel()
                self._no_response_check_task = None
            # Cancel pending turn end if bot starts speaking again
            if self._turn_end_task and not self._turn_end_task.done():
                logger.info("[TurnGate] BotStartedSpeakingFrame received - cancelling pending turn end")
                self._turn_end_task.cancel()
                self._turn_end_task = None

        if isinstance(frame, BotStoppedSpeakingFrame):
            logger.info("[TurnGate] BotStoppedSpeakingFrame received")
            self._bot_speaking = False

            # Signal greeting done on first BotStoppedSpeakingFrame
            # This allows user audio to start playing after the initial greeting
            if not self._greeting_signaled and self._on_greeting_done:
                self._greeting_signaled = True
                logger.info("[TurnGate] Initial greeting complete, signaling greeting done")
                self._on_greeting_done()

            # If we have a pending transcript, schedule turn end after delay
            if self._pending_transcript is not None:
                text = self._pending_transcript
                self._pending_transcript = None
                # Cancel any existing turn end task
                if self._turn_end_task and not self._turn_end_task.done():
                    self._turn_end_task.cancel()
                # Cancel no-response check since we're advancing normally
                if self._no_response_check_task and not self._no_response_check_task.done():
                    self._no_response_check_task.cancel()
                    self._no_response_check_task = None
                # Schedule delayed turn end
                self._turn_end_task = asyncio.create_task(self._delayed_turn_end(text))

        await self.push_frame(frame, direction)


class GeminiLiveLLMServiceWithReconnection(GeminiLiveLLMService):
    """Extended Gemini Live service that exposes reconnection events.

    The base GeminiLiveLLMService handles reconnection internally when the
    10-minute session timeout occurs, but doesn't expose events for external
    coordination. This subclass:

    1. Calls on_reconnecting callback before disconnecting
    2. Calls on_reconnected callback after reconnecting
    3. Tracks whether we were in the middle of receiving a response

    This allows the test harness to:
    - Pause audio input during reconnection
    - Re-queue the interrupted turn's audio after reconnection
    - Reset turn tracking state
    """

    def __init__(
        self,
        on_reconnecting: Optional[Callable[[], None]] = None,
        on_reconnected: Optional[Callable[[], None]] = None,
        **kwargs,
    ):
        """Initialize with optional reconnection callbacks.

        Args:
            on_reconnecting: Called before disconnecting during reconnection.
                            Use this to pause audio input and save state.
            on_reconnected: Called after reconnection completes.
                           Use this to resume audio input and re-queue interrupted turn.
        """
        super().__init__(**kwargs)
        self._on_reconnecting = on_reconnecting
        self._on_reconnected = on_reconnected
        self._reconnecting = False
        self._context_ready = asyncio.Event()

    def is_reconnecting(self) -> bool:
        """Check if currently in the middle of a reconnection."""
        return self._reconnecting

    async def _reconnect(self):
        """Override to call callbacks before/after reconnection.

        Note: _reconnect() is called both for initial context setup (when setting
        system_instruction or tools) and for actual session timeout reconnections.
        We only want to trigger the callbacks for real reconnections, not initial
        setup. We detect this by checking if _session exists - if it's None, this
        is the initial connection, not a reconnection.
        """
        # Only trigger callbacks if we had an existing session (real reconnection)
        # Not for initial context setup which also calls _reconnect
        is_real_reconnection = self._session is not None

        self._reconnecting = True

        # Call on_reconnecting callback only for real reconnections
        if is_real_reconnection and self._on_reconnecting:
            try:
                logger.info("GeminiLiveWithReconnection: Calling on_reconnecting callback")
                self._on_reconnecting()
            except Exception as e:
                logger.warning(f"Error in on_reconnecting callback: {e}")

        # Call parent reconnect implementation
        try:
            await super()._reconnect()
        finally:
            self._reconnecting = False
            self._context_ready.set()

        # Call on_reconnected callback only for real reconnections
        if is_real_reconnection and self._on_reconnected:
            try:
                logger.info("GeminiLiveWithReconnection: Calling on_reconnected callback")
                self._on_reconnected()
            except Exception as e:
                logger.warning(f"Error in on_reconnected callback: {e}")


class LLMFrameLogger(FrameProcessor):
    """Logs every frame emitted by the LLM stage and captures TTFB metrics."""

    def __init__(self, recorder_accessor, vad_params: Optional[VADParams] = None):
        super().__init__()
        self._recorder_accessor = recorder_accessor
        self._vad_params = vad_params
        self._recording_start_time: Optional[float] = None

    def set_recording_start_time(self, t: float):
        """Set the recording start time for relative timestamp logging."""
        self._recording_start_time = t

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Log VAD frames with timing offset information
        if isinstance(frame, VADUserStartedSpeakingFrame):
            now = time.time()
            offset = self._vad_params.start_secs if self._vad_params else 0.2
            actual_start = now - offset
            rel_time = (now - self._recording_start_time) * 1000 if self._recording_start_time else 0
            actual_rel = (actual_start - self._recording_start_time) * 1000 if self._recording_start_time else 0
            logger.info(
                f"[VAD] UserStartedSpeaking at T+{rel_time:.0f}ms "
                f"(actual start: T+{actual_rel:.0f}ms, offset={offset*1000:.0f}ms)"
            )
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            now = time.time()
            offset = self._vad_params.stop_secs if self._vad_params else 0.8
            actual_end = now - offset
            rel_time = (now - self._recording_start_time) * 1000 if self._recording_start_time else 0
            actual_rel = (actual_end - self._recording_start_time) * 1000 if self._recording_start_time else 0
            logger.info(
                f"[VAD] UserStoppedSpeaking at T+{rel_time:.0f}ms "
                f"(actual end: T+{actual_rel:.0f}ms, offset={offset*1000:.0f}ms)"
            )
        elif not isinstance(frame, InputAudioRawFrame):
            logger.debug(f"[LLM→] {frame.__class__.__name__} ({direction})")

        # Capture TTFB from MetricsFrame for realtime/live models
        if isinstance(frame, MetricsFrame):
            for md in frame.data:
                if isinstance(md, TTFBMetricsData):
                    recorder = self._recorder_accessor()
                    if recorder:
                        recorder.record_ttfb(md.value)
        await self.push_frame(frame, direction)


class RealtimePipeline(BasePipeline):
    """Pipeline for OpenAI Realtime and Gemini Live models.

    This pipeline handles speech-to-speech models with:
    - Paced audio input at realtime pace
    - Server-side VAD for turn detection
    - Transcript-based end-of-turn detection
    - Reconnection handling for Gemini Live 10-minute timeout
    """

    requires_service = True

    def __init__(self, benchmark):
        super().__init__(benchmark)
        self.context_aggregator = None
        self.paced_input = None
        self.transcript = None
        self.assistant_shim = None
        self.audio_buffer: Optional[WallClockAlignedAudioBufferProcessor] = None
        self.turn_gate: Optional[TurnGate] = None
        self.output_transport: Optional[NullAudioOutputTransport] = None
        self.current_turn_audio_path: Optional[str] = None
        self.needs_turn_retry: bool = False
        self.reconnection_grace_until: float = 0
        # Track when current user audio will finish playing (monotonic time)
        # This prevents queuing the next turn before current audio finishes
        self._current_audio_end_time: float = 0
        # Watchdog timer: fires _on_empty_response if the turn doesn't advance
        # within audio_duration + buffer. Catches cases where VAD fails to
        # detect very short audio (e.g. "Yep."), so no VADUserStoppedSpeakingFrame
        # is emitted and the TurnGate's no-response timer never starts.
        self._turn_watchdog_task: Optional[asyncio.Task] = None
        self._turn_watchdog_timeout: float = 30.0  # seconds after audio ends
        self._turn_watchdog_extension_secs: float = 20.0
        self._turn_watchdog_max_extensions: int = 6
        # Event to signal when pipeline is ready (StartFrame has reached end)
        self._pipeline_ready_event: asyncio.Event = asyncio.Event()
        # Event to signal when initial greeting starts (first BotStartedSpeakingFrame)
        self._greeting_started: asyncio.Event = asyncio.Event()
        # Event to signal when initial greeting is complete (first BotStoppedSpeakingFrame)
        self._greeting_done: asyncio.Event = asyncio.Event()
        # Empty response retry tracking
        self._turn_retry_count: int = 0
        self._max_turn_retries: int = 3
        # Track first user end time per turn (monotonic time) for accurate V2V metrics
        self._first_user_end_time: Optional[float] = None
        # Conversation history for context preservation across reconnections.
        # Captured for all models so that any pipeline with system-prompt-based
        # reconnection (OpenAI Realtime, Grok) has consistent context.
        # Gemini uses session resumption handles instead, but we still record
        # history as a fallback.
        self._conversation_history: list[dict] = []
        self.MAX_CONTEXT_TURNS = 20

    def _is_gemini_live(self) -> bool:
        """Check if current model is Gemini Live."""
        if not self.model_name:
            return False
        m = self.model_name.lower()
        return (m.startswith("gemini") or m.startswith("models/gemini")) and (
            "live" in m or "native-audio" in m
        )

    def _is_openai_realtime(self) -> bool:
        """Check if current model is OpenAI Realtime."""
        if not self.model_name:
            return False
        m = self.model_name.lower()
        return "realtime" in m and m.startswith("gpt")

    def _is_ultravox_realtime(self) -> bool:
        """Check if current model is Ultravox Realtime."""
        if not self.model_name:
            return False
        m = self.model_name.lower()
        return "ultravox" in m

    def _is_grok_realtime(self) -> bool:
        """Check if current model is Grok/xAI Realtime."""
        if not self.model_name:
            return False
        m = self.model_name.lower()
        return "grok" in m and "realtime" in m

    def _uses_openai_realtime_history_seeding(self) -> bool:
        """Seed prior turns into the OpenAI Realtime session with conversation items."""
        return bool(
            self._rehydration_turns
            and self._is_openai_realtime()
            and (self.service_name or "").lower() == "openai-realtime"
        )

    def _build_openai_realtime_backdoor_options(self) -> dict[str, int]:
        """Build optional Realtime-only backdoor session settings."""
        payload: dict[str, int] = {}
        if self._juice is not None:
            payload["juice"] = self._juice
        if self._verbosity is not None:
            payload["verbosity"] = self._verbosity
        return payload

    @staticmethod
    def _build_openai_rehydration_history_items(
        golden_turns: list[dict[str, Any]]
    ) -> list[rt_events.ConversationItem]:
        """Build conversation items for OpenAI Realtime session-history seeding."""
        items: list[rt_events.ConversationItem] = []

        for turn_index, turn in enumerate(golden_turns):
            user_input = turn.get("input", "")
            golden_text = turn.get("golden_text", "")
            required_calls = turn.get("required_function_call")
            tool_responses = turn.get("function_call_response")

            items.append(
                rt_events.ConversationItem(
                    id=f"rehydrate_t{turn_index}_user",
                    type="message",
                    role="user",
                    status="completed",
                    content=[rt_events.ItemContent(type="input_text", text=user_input)],
                )
            )

            calls = required_calls if isinstance(required_calls, list) else [required_calls] if required_calls else []
            responses = (
                tool_responses
                if isinstance(tool_responses, list)
                else [tool_responses] if tool_responses is not None else []
            )

            for call_index, call in enumerate(calls):
                call_id = f"rehydrate_call_t{turn_index}_{call_index}"
                items.append(
                    rt_events.ConversationItem(
                        id=f"rehydrate_t{turn_index}_call_{call_index}",
                        type="function_call",
                        status="completed",
                        call_id=call_id,
                        name=call["name"],
                        arguments=json.dumps(call.get("args", {})),
                    )
                )
                if call_index < len(responses):
                    items.append(
                        rt_events.ConversationItem(
                            id=f"rehydrate_t{turn_index}_output_{call_index}",
                            type="function_call_output",
                            status="completed",
                            call_id=call_id,
                            output=json.dumps(responses[call_index]),
                        )
                    )

            if golden_text:
                items.append(
                    rt_events.ConversationItem(
                        id=f"rehydrate_t{turn_index}_assistant",
                        type="message",
                        role="assistant",
                        status="completed",
                        content=[rt_events.ItemContent(type="output_text", text=golden_text)],
                    )
                )

        return items

    def _seed_rehydration_history_for_reconnects(self) -> None:
        """Mirror rehydrated golden turns into reconnect fallback history."""
        if not self._rehydration_turns or self._conversation_history:
            return

        for turn in self._rehydration_turns:
            entry: dict[str, Any] = {
                "user": turn.get("input", ""),
                "assistant": turn.get("golden_text", ""),
            }
            required_calls = turn.get("required_function_call")
            tool_responses = turn.get("function_call_response")
            if required_calls:
                calls = required_calls if isinstance(required_calls, list) else [required_calls]
                entry["tool_calls"] = [
                    {"name": call["name"], "args": call.get("args", {})}
                    for call in calls
                ]
                responses = (
                    tool_responses
                    if isinstance(tool_responses, list)
                    else [tool_responses] if tool_responses is not None else []
                )
                if responses:
                    entry["tool_results"] = [
                        {
                            "name": call["name"],
                            "response": responses[idx],
                        }
                        for idx, call in enumerate(calls)
                        if idx < len(responses)
                    ]
            self._conversation_history.append(entry)

    def _reset_openai_manual_turn_state(self) -> None:
        """Clear per-turn manual turn-handling state before queuing audio."""
        llm = self._get_openai_realtime_llm()
        if not (
            self._disable_vad
            and self._is_openai_realtime()
            and (self.service_name or "").lower() == "openai-realtime"
            and llm is not None
        ):
            return
        llm.reset_manual_turn_state()

    async def _seed_openai_rehydration_history(self) -> None:
        """Populate prior turns into the live OpenAI Realtime session before audio."""
        llm = self._get_openai_realtime_llm()
        if not (self._uses_openai_realtime_history_seeding() and llm is not None):
            return
        await llm.seed_rehydration_history()

    def _get_openai_realtime_llm(self):
        from audio_arena.pipelines.openai_realtime import (
            OpenAIRealtimeLLMServiceExplicitToolResult,
        )

        if isinstance(self.llm, OpenAIRealtimeLLMServiceExplicitToolResult):
            return self.llm
        return None

    def _get_last_tool_result(self) -> dict[str, Any]:
        # OpenAI/xAI realtime services need a pull-based accessor because the tool
        # handler runs on the pipeline, but the websocket protocol work happens in
        # the service object.
        if self._last_tool_result is None:
            return {"status": "success"}
        return self._last_tool_result

    def _get_audio_duration(self, audio_path: str) -> float:
        """Get duration of an audio file in seconds.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Duration in seconds, or 0 if unable to determine.
        """
        try:
            info = sf.info(audio_path)
            return info.duration
        except Exception as e:
            logger.warning(f"Could not get duration for {audio_path}: {e}")
            return 0

    def _get_audio_path_for_turn(self, turn_index: int) -> Optional[str]:
        """Get the audio file path for a turn.

        Prefers benchmark.get_audio_path() if available, falls back to
        the turn's audio_file field.

        Args:
            turn_index: The effective turn index (index into effective_turns).

        Returns:
            Path to audio file as string, or None if not available.
        """
        # Try benchmark's get_audio_path method first (uses audio_dir)
        if hasattr(self.benchmark, "get_audio_path"):
            actual_index = self._get_actual_turn_index(turn_index)
            path = self.benchmark.get_audio_path(actual_index)
            if path and path.exists():
                return str(path)

        # Fall back to turn's audio_file field
        turn = self.effective_turns[turn_index]
        return turn.get("audio_file")

    def _create_llm(self, service_class: Optional[type], model: str) -> FrameProcessor:
        """Create LLM service with proper configuration for realtime models.

        For OpenAI Realtime, we must pass session_properties with turn_detection
        config at construction time. The server-side VAD settings prevent
        client-side interruptions from truncating responses.

        For Gemini Live, we use GeminiLiveLLMServiceWithReconnection and pass
        VAD parameters through the input params.
        """
        if service_class is None:
            raise ValueError("--service is required for this pipeline")

        class_name = service_class.__name__
        system_instruction = getattr(self.benchmark, "system_instruction", "")
        tools = getattr(self.benchmark, "tools_schema", None)

        if "OpenAIRealtime" in class_name:
            # OpenAI Realtime: Configure server-side VAD to prevent interruptions
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("OPENAI_API_KEY environment variable is required")

            turn_detection: rt_events.TurnDetection | None = rt_events.TurnDetection(
                type="server_vad",
                threshold=0.7,
                prefix_padding_ms=500,
                silence_duration_ms=800,
            )
            if self._disable_vad:
                turn_detection = None

            audio_config = rt_events.AudioConfiguration(
                input=rt_events.AudioInput(
                    turn_detection=turn_detection
                )
            )
            session_props = rt_events.SessionProperties(
                instructions=system_instruction,
                tools=tools,
                audio=audio_config,
            )
            kwargs = dict(
                api_key=api_key,
                model=model,
                system_instruction=system_instruction,
                session_properties=session_props,
            )
            if self.recorder and hasattr(self.recorder, "run_dir"):
                kwargs["websocket_event_log_path"] = str(
                    self.recorder.run_dir / "ws_events.jsonl"
                )
            backdoor_options = self._build_openai_realtime_backdoor_options()
            if backdoor_options:
                kwargs["session_backdoor_options"] = backdoor_options
            if "ExplicitToolResult" in class_name or "XAIRealtime" in class_name:
                # The service sends function_call_output over the websocket after
                # the pipeline's tool handler has already produced the payload.
                kwargs["get_last_tool_result"] = self._get_last_tool_result
            if "ExplicitToolResult" in class_name:
                kwargs["capture_tool_phase"] = self.capture_tool_phase
                kwargs["should_abort_after_tool_call"] = self.should_abort_after_tool_capture
                kwargs["abort_after_tool_call"] = self.abort_after_tool_capture
                kwargs["on_reconnecting"] = self._on_ws_reconnecting
                kwargs["on_reconnected"] = self._on_ws_reconnected
                if self._uses_openai_realtime_history_seeding():
                    # OpenAI Realtime can ingest prior turns directly as conversation
                    # items, which is closer to real session state than stuffing the
                    # same history back into the system prompt.
                    rehydration_history_items = self._build_openai_rehydration_history_items(
                        self._rehydration_turns or []
                    )
                    kwargs["rehydration_history_items"] = rehydration_history_items
                    logger.info(
                        f"[Rehydration] Using conversation.item.create history seeding with "
                        f"{len(rehydration_history_items)} prior items"
                    )
            return service_class(**kwargs)
        elif "UltravoxRealtime" in class_name:
            # Ultravox Realtime: Use OneShotInputParams
            api_key = os.getenv("ULTRAVOX_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "ULTRAVOX_API_KEY environment variable is required"
                )

            # Ultravox reads its system prompt only from OneShotInputParams at
            # construction time — it does NOT pick up later LLMContext updates.
            # Enrich the prompt with rehydration history here so the model sees
            # the golden conversation context before the target turn's audio.
            effective_prompt = system_instruction
            if self._rehydration_turns:
                _, instruction_suffix = self.build_rehydration_history(
                    self._rehydration_turns
                )
                effective_prompt = system_instruction + instruction_suffix
                logger.info(
                    f"[Rehydration/Ultravox] Enriched OneShotInputParams system_prompt "
                    f"with {len(self._rehydration_turns)} golden turns "
                    f"({len(effective_prompt)} chars total)"
                )

            params = OneShotInputParams(
                api_key=api_key,
                system_prompt=effective_prompt,
                temperature=1.0,
                model=model,
            )
            return service_class(
                params=params,
                one_shot_selected_tools=tools,
            )
        elif "GeminiLive" in class_name:
            # Gemini Live: Enable auto-response on context initialization
            # When context is set with a user message like "Greet the user briefly",
            # Gemini will automatically produce a greeting response.
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise EnvironmentError("GOOGLE_API_KEY environment variable is required")

            return GeminiLiveLLMServiceWithReconnection(
                api_key=api_key,
                model=model,
                system_instruction=system_instruction,
                tools=tools,
                inference_on_context_initialization=not self._rehydration_turns,
                on_reconnecting=self._on_gemini_reconnecting,
                on_reconnected=self._on_gemini_reconnected,
            )
        else:
            # For other services, use base class implementation
            return super()._create_llm(service_class, model)

    def _setup_context(self) -> None:
        """Create LLMContext with system prompt and tools.

        For OpenAI Realtime and Grok Realtime, we also add an initial user
        message to trigger the greeting when LLMRunFrame is queued.

        In rehydration mode, golden conversation history is appended to the
        system instruction so the model has perfect prior context.
        """
        system_instruction = getattr(self.benchmark, "system_instruction", "")
        tools = getattr(self.benchmark, "tools_schema", None)

        # In rehydration mode, OpenAI Realtime seeds prior turns directly into the
        # session. Other providers fall back to system-instruction hydration.
        if self._rehydration_turns and not self._uses_openai_realtime_history_seeding():
            _, instruction_suffix = self.build_rehydration_history(self._rehydration_turns)
            system_instruction = system_instruction + instruction_suffix
            logger.info(
                f"[Rehydration] Enriched system instruction with {len(self._rehydration_turns)} "
                f"golden turns ({len(system_instruction)} chars total)"
            )
        elif self._uses_openai_realtime_history_seeding():
            logger.info(
                f"[Rehydration] Using conversation.item.create session-history seeding; "
                f"system instruction remains base prompt ({len(system_instruction)} chars)"
            )

        # Both OpenAI Realtime and Gemini Live read the system instruction from
        # an LLMContextFrame. The pipecat service extracts the system message
        # and applies it via session properties (OpenAI) or context (Gemini).
        messages = [{"role": "system", "content": system_instruction}]

        # Add initial greeting trigger for models that need it:
        # - OpenAI Realtime and Grok Realtime: need user message + LLMRunFrame
        # - Gemini Live: needs user message with inference_on_context_initialization=True
        # - Ultravox: auto-greets, no trigger needed
        # Skip greeting in rehydration mode — the model's instructions say it's
        # mid-conversation, so a fresh greeting creates a semantic conflict that
        # can cause subsequent user audio to be misinterpreted.
        if not self._rehydration_turns and (
            self._is_openai_realtime() or self._is_grok_realtime() or self._is_gemini_live()
        ):
            messages.append({"role": "user", "content": "Greet the user briefly."})

        self.context = LLMContext(messages, tools=tools)
        self.context_aggregator = LLMContextAggregatorPair(self.context)

    def _setup_llm(self) -> None:
        """Configure LLM and set up reconnection callbacks for Gemini Live."""
        self.llm.register_function(None, self._function_catchall)

        # Set up reconnection callbacks for Gemini Live
        if self._is_gemini_live() and isinstance(self.llm, GeminiLiveLLMServiceWithReconnection):
            self.llm._on_reconnecting = self._on_gemini_reconnecting
            self.llm._on_reconnected = self._on_gemini_reconnected

    def _on_gemini_reconnecting(self):
        """Called when Gemini Live starts reconnecting due to session timeout."""
        logger.info(f"Gemini reconnecting: pausing audio, turn {self.turn_idx} will be retried")
        self.needs_turn_retry = True
        # Pause audio input to avoid sending audio during reconnection
        self.paced_input.pause()
        # Clear the transcript buffer to discard partial responses from before reconnection
        self.assistant_shim.clear_buffer()
        # Clear TurnGate state to prevent stale transcripts from interfering
        if self.turn_gate:
            self.turn_gate.clear_pending()
        # Set grace period to ignore stale TTSStoppedFrame events that arrive after reconnection
        self.reconnection_grace_until = time.monotonic() + 10.0
        logger.info(
            f"Set reconnection grace period until {self.reconnection_grace_until}"
        )

    def _on_gemini_reconnected(self):
        """Called when Gemini Live reconnection completes."""
        logger.info(f"Gemini reconnected: scheduling turn {self.turn_idx} retry")
        # Resume audio input
        self.paced_input.signal_ready()
        # Schedule a task to re-queue the current turn's audio after a short delay
        asyncio.create_task(self._retry_current_turn_after_reconnection())

    # --- Shared WebSocket reconnection callbacks (OpenAI Realtime + Grok) ---

    def _on_ws_reconnecting(self):
        """Called when OpenAI/Grok closes the WebSocket unexpectedly, before reconnecting.

        Pauses audio, clears partial state, enriches system instructions with
        conversation history so the model retains context across the session boundary.
        """
        logger.info(f"Reconnecting: pausing audio, turn {self.turn_idx} will be retried")
        self.needs_turn_retry = True
        self.paced_input.pause()
        self.assistant_shim.clear_buffer()
        if self.turn_gate:
            self.turn_gate.clear_pending()
        self.reconnection_grace_until = time.monotonic() + 10.0

        self._update_instructions_with_history()

    def _on_ws_reconnected(self):
        """Called after a successful OpenAI/Grok WebSocket reconnection."""
        logger.info(f"Reconnected: scheduling turn {self.turn_idx} retry")
        self.paced_input.signal_ready()
        asyncio.create_task(self._retry_current_turn_after_reconnection())

    def _update_instructions_with_history(self):
        """Append recent conversation history to the session instructions.

        Modifies self.llm._session_properties.instructions so that when the
        new session calls _update_settings(), the enriched prompt is sent
        via session.update. The model sees prior turns as part of its
        instructions, preserving context without affecting audio behavior.

        Used by both OpenAI Realtime (GPT) and Grok Realtime pipelines.
        """
        if not self._conversation_history:
            return

        history = self._conversation_history[-self.MAX_CONTEXT_TURNS:]
        original = getattr(self.benchmark, "system_instruction", "")

        lines = [
            "\n\n--- CONVERSATION HISTORY ---",
            "The following is the conversation so far. The user's name, "
            "preferences, and any actions you have taken (tool calls, "
            "registrations, schedule changes, etc.) are still in effect. "
            "Continue naturally.",
            "",
        ]
        for turn in history:
            lines.append(f"User: {turn['user']}")
            if turn.get("tool_calls"):
                for tc in turn["tool_calls"]:
                    lines.append(f"  [Tool call: {tc['name']}({tc['args']})]")
            if turn.get("tool_results"):
                for tool_result in turn["tool_results"]:
                    lines.append(
                        f"  [Tool result: {json.dumps(tool_result.get('response', {}))}]"
                    )
            lines.append(f"Assistant: {turn['assistant']}")
            lines.append("")

        enriched = original + "\n".join(lines)

        self.llm._session_properties.instructions = enriched
        logger.info(
            f"[Reconnection] Enriched system instructions with {len(history)} of "
            f"{len(self._conversation_history)} conversation turns "
            f"({len(enriched)} chars total)"
        )

    async def _retry_current_turn_after_reconnection(self):
        """Re-queue the current turn's audio after reconnection.

        This wraps _retry_current_turn with reconnection-specific grace period handling.
        """
        # Increment retry count for reconnection as well
        self._turn_retry_count += 1

        # Use unified retry logic
        await self._retry_current_turn()

        # After retry, clear the reconnection grace period
        if not self.needs_turn_retry:
            # Wait for audio to finish then clear grace period
            await asyncio.sleep(5.0)
            self.reconnection_grace_until = 0
            logger.info("Cleared reconnection grace period - accepting new transcript updates")

    def _on_empty_response(self, reason: str = "empty_response"):
        """Called when TurnGate detects an empty or no response.

        Args:
            reason: Either "empty_response" (TTS with no audio) or "no_response" (no TTS at all).
        """
        tag = "EMPTY_RESPONSE" if reason == "empty_response" else "NO_RESPONSE"
        logger.info(f"[{tag}] turn={self.turn_idx} retry_count={self._turn_retry_count}")

        if self._turn_retry_count >= self._max_turn_retries:
            logger.error(
                f"[EMPTY_RESPONSE] Max retries ({self._max_turn_retries}) reached for turn {self.turn_idx}"
            )
            # Advance to next turn to avoid infinite stall
            asyncio.create_task(self._force_advance_turn())
            return

        self._turn_retry_count += 1
        self.needs_turn_retry = True
        # Clear TurnGate state
        if self.turn_gate:
            self.turn_gate.clear_pending()
        # Clear transcript buffer
        if self.assistant_shim:
            self.assistant_shim.clear_buffer()

        # Reuse the same retry logic as reconnection
        asyncio.create_task(self._retry_current_turn())

    async def _retry_current_turn(self):
        """Unified retry logic for empty responses (called from reconnection retry as well)."""
        if not self.needs_turn_retry:
            logger.info("No turn retry needed")
            return

        logger.info(f"Waiting 2s before retrying turn {self.turn_idx} (attempt {self._turn_retry_count})")
        await asyncio.sleep(2.0)

        if not self.needs_turn_retry:
            logger.info("Turn retry cancelled (turn completed normally)")
            return

        audio_path = self.current_turn_audio_path or self._get_audio_path_for_turn(self.turn_idx)
        if audio_path:
            logger.info(f"Re-queuing audio for turn {self.turn_idx}: {audio_path}")
            try:
                # Manual no-VAD mode latches commit state per turn; retries must
                # clear that state before replaying audio or the next stop event
                # will be ignored by the OpenAI Realtime service.
                self._reset_openai_manual_turn_state()
                self.paced_input.enqueue_wav_file(audio_path)
                self.needs_turn_retry = False
                audio_duration = self._get_audio_duration(audio_path)
                self._start_turn_watchdog(audio_duration)
                logger.info(f"Successfully re-queued audio for turn {self.turn_idx}")
            except Exception as e:
                logger.exception(f"Failed to re-queue audio for turn {self.turn_idx}: {e}")
        else:
            logger.warning(f"No audio path available for turn {self.turn_idx}")
            self.needs_turn_retry = False

    async def _force_advance_turn(self):
        """Force turn advancement after max retries exceeded."""
        logger.warning(f"[EMPTY_RESPONSE] Force advancing past turn {self.turn_idx}")
        await self._on_turn_end("[EMPTY_RESPONSE: No valid response after max retries]")

    def _start_turn_watchdog(self, audio_duration: float):
        """Start a watchdog timer for the current turn.

        If the turn doesn't advance within audio_duration + _turn_watchdog_timeout,
        trigger _on_empty_response. This catches VAD failures where SileroVAD
        doesn't detect very short audio, so no VADUserStoppedSpeakingFrame fires.
        """
        self._cancel_turn_watchdog()
        timeout = audio_duration + self._turn_watchdog_timeout
        self._turn_watchdog_task = asyncio.create_task(
            self._turn_watchdog(timeout, self.turn_idx)
        )

    def _cancel_turn_watchdog(self):
        """Cancel any running watchdog timer."""
        if self._turn_watchdog_task and not self._turn_watchdog_task.done():
            self._turn_watchdog_task.cancel()
            self._turn_watchdog_task = None

    async def _turn_watchdog(self, timeout: float, expected_turn: int):
        """Watchdog: if turn stalls without model output, trigger empty response.

        Some realtime turns legitimately run long (streamed text/audio and
        delayed BotStoppedSpeakingFrame). When we detect active output, extend
        instead of force-retrying immediately.
        """
        extension_interval = self._turn_watchdog_extension_secs
        max_extensions = self._turn_watchdog_max_extensions
        try:
            await asyncio.sleep(timeout)
            extensions = 0
            while self.turn_idx == expected_turn and not self.done:
                has_activity = bool(self.turn_gate and self.turn_gate.has_response_activity())
                if has_activity and extensions < max_extensions:
                    extensions += 1
                    logger.info(
                        f"[TURN_WATCHDOG] Turn {expected_turn}: output still active; "
                        f"extending watchdog ({extensions}/{max_extensions})"
                    )
                    await asyncio.sleep(extension_interval)
                    continue
                logger.warning(
                    f"[TURN_WATCHDOG] Turn {expected_turn} did not advance within "
                    f"{timeout + extensions * extension_interval:.1f}s — triggering empty response"
                )
                self._on_empty_response("no_response")
                break
        except asyncio.CancelledError:
            pass

    def _build_task(self) -> None:
        """Build the pipeline with paced input and transcript processors."""

        def recorder_accessor():
            return self.recorder

        def duplicate_ids_accessor():
            return self._duplicate_tool_call_ids

        # Determine sample rate from first audio file
        default_sr = 24000
        t0_audio = self._get_audio_path_for_turn(0)
        if t0_audio:
            try:
                _, t0_sr = sf.read(t0_audio, dtype="int16", always_2d=True)
                default_sr = int(t0_sr)
            except Exception as e:
                logger.warning(f"Could not read sample rate from {t0_audio}: {e}")

        # Create local VAD analyzer for user speech detection
        # This emits VADUserStartedSpeakingFrame/VADUserStoppedSpeakingFrame
        # which we can compare against WAV-based VAD for timing analysis
        vad_params = VADParams(
            start_secs=0.2,  # Emit VADUserStartedSpeaking 0.2s after speech starts
            stop_secs=0.8,   # Emit VADUserStoppedSpeaking 0.8s after speech ends
        )
        # Silero VAD only supports 16kHz or 8kHz - use our resampling subclass
        # to handle 24kHz audio from OpenAI/Ultravox
        user_vad = ResamplingSileroVAD(params=vad_params)
        logger.info(
            f"[VAD] User VAD config: start_secs={vad_params.start_secs}, "
            f"stop_secs={vad_params.stop_secs}"
        )

        # Create paced input transport with VAD
        input_params = TransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=default_sr,
            audio_in_channels=1,
            audio_in_passthrough=True,
            vad_analyzer=user_vad,
        )
        emit_user_stopped_speaking = bool(
            self._disable_vad
            and self._is_openai_realtime()
            and (self.service_name or "").lower() == "openai-realtime"
        )
        self.paced_input = PacedInputTransport(
            input_params,
            pre_roll_ms=100,
            continuous_silence=True,
            emit_user_stopped_speaking=emit_user_stopped_speaking,
        )
        if emit_user_stopped_speaking:
            logger.info(
                "[PacedInput] emit_user_stopped_speaking enabled for OpenAI Realtime with --disable-vad"
            )

        # Create transcript processors
        self.transcript = TranscriptProcessor()
        self.assistant_shim = TTSStoppedAssistantTranscriptProcessor()

        # Create audio buffer processor for recording both user and bot audio
        # NullAudioOutputTransport is the "source of truth" for wall-clock aligned recording.
        # It inserts silence for any gaps > 10ms in BOTH user and bot audio tracks.
        # AudioBufferProcessor just accumulates the continuous streams from output_transport.
        logger.info(f"[AudioRecording] Creating AudioBufferProcessor with sample_rate={default_sr}, num_channels=2")
        self.audio_buffer = WallClockAlignedAudioBufferProcessor(
            sample_rate=default_sr,
            num_channels=2,  # Stereo: user on left channel, bot on right channel
        )

        # Register event handler to save audio when track data is ready
        @self.audio_buffer.event_handler("on_track_audio_data")
        async def on_track_audio_data(
            processor, user_audio: bytes, bot_audio: bytes, sample_rate: int, num_channels: int
        ):
            """Save conversation audio with user and bot on separate channels."""
            if os.environ.get("SKIP_AUDIO_RECORDING"):
                logger.info("[AudioRecording] Skipping audio save (SKIP_AUDIO_RECORDING set)")
                return

            logger.info(
                f"[AudioRecording] on_track_audio_data triggered: "
                f"user={len(user_audio)} bytes, bot={len(bot_audio)} bytes, "
                f"{sample_rate}Hz, {num_channels}ch"
            )

            # Get run directory from recorder
            if not self.recorder or not hasattr(self.recorder, "run_dir"):
                logger.error("[AudioRecording] Cannot save audio: no recorder or run_dir available")
                return

            # Convert to numpy for processing
            user_np = np.frombuffer(user_audio, dtype=np.int16)
            bot_np = np.frombuffer(bot_audio, dtype=np.int16)

            # Pad shorter track to match longer
            max_len = max(len(user_np), len(bot_np))
            if len(user_np) < max_len:
                user_np = np.concatenate([user_np, np.zeros(max_len - len(user_np), dtype=np.int16)])
            if len(bot_np) < max_len:
                bot_np = np.concatenate([bot_np, np.zeros(max_len - len(bot_np), dtype=np.int16)])

            # Interleave for stereo: user=left, bot=right
            stereo = np.zeros(max_len * 2, dtype=np.int16)
            stereo[0::2] = user_np
            stereo[1::2] = bot_np

            output_path = self.recorder.run_dir / "conversation.wav"
            logger.info(f"[AudioRecording] Saving conversation audio to {output_path}")

            try:
                with wave.open(str(output_path), "wb") as wf:
                    wf.setnchannels(2)  # Stereo
                    wf.setsampwidth(2)  # 16-bit audio = 2 bytes per sample
                    wf.setframerate(sample_rate)
                    wf.writeframes(stereo.tobytes())

                # Calculate duration for logging
                duration_secs = max_len / sample_rate
                file_size_mb = (max_len * 2 * 2) / (1024 * 1024)
                logger.info(
                    f"[AudioRecording] Saved conversation audio: {output_path} "
                    f"({duration_secs:.1f}s, {file_size_mb:.2f}MB)"
                )

                # Log silence insertion statistics from output transport
                if self.output_transport is not None and hasattr(self.output_transport, 'log_recording_summary'):
                    self.output_transport.log_recording_summary()
            except Exception as e:
                logger.exception(f"[AudioRecording] Failed to save audio: {e}")

        # Register event handler for transcript updates
        # Note: We store the transcript but wait for BotStoppedSpeakingFrame before advancing turn
        @self.assistant_shim.event_handler("on_transcript_update")
        async def on_transcript_update(processor, frame):
            # Check grace period
            if time.monotonic() < self.reconnection_grace_until:
                logger.warning(
                    f"Ignoring transcript update during reconnection grace period "
                    f"(until {self.reconnection_grace_until})"
                )
                return

            for msg in frame.messages:
                if isinstance(msg, TranscriptionMessage) and getattr(msg, "role", None) == "assistant":
                    timestamp = f"[{msg.timestamp}] " if msg.timestamp else ""
                    line = f"{timestamp}{msg.role}: {msg.content}"
                    logger.info(f"Transcript: {line}")
                    # Clear retry flag - turn completed successfully
                    self.needs_turn_retry = False
                    # Store transcript in turn_gate; it will trigger _on_turn_end
                    # when BotStoppedSpeakingFrame is received
                    self.turn_gate.set_pending_transcript(msg.content)

        # Create TurnGate to coordinate transcript with audio playback completion
        # Pass greeting callbacks to signal when initial bot greeting starts/completes
        self.turn_gate = TurnGate(
            on_turn_ready=self._on_turn_end,
            on_greeting_done=lambda: self._greeting_done.set(),
        )
        self.turn_gate.set_greeting_started_callback(lambda: self._greeting_started.set())
        # Set up empty response callback for all models so the pipeline
        # can retry or force-advance when the model fails to respond
        self.turn_gate.set_empty_response_callback(self._on_empty_response)

        # Create null output transport to generate BotStoppedSpeakingFrame
        # This tracks when the bot finishes "speaking" (outputting audio)
        # Increase the silence threshold from 0.35s to 2s to handle LLM pauses during generation
        import pipecat.transports.base_output as base_output_module

        base_output_module.BOT_VAD_STOP_SECS = 2.0
        logger.info("[AudioRecording] Set BOT_VAD_STOP_SECS to 2.0s for more reliable turn detection")

        self.output_transport = NullAudioOutputTransport(
            TransportParams(
                audio_out_enabled=True,
                audio_out_sample_rate=default_sr,
            )
        )

        self.llm_logger = LLMFrameLogger(recorder_accessor, vad_params=vad_params)

        pipeline = Pipeline(
            [
                self.paced_input,
                self.context_aggregator.user(),
                self.transcript.user(),
                self.llm,
                self.llm_logger,
                ToolCallRecorder(recorder_accessor, duplicate_ids_accessor),
                self.assistant_shim,
                self.turn_gate,  # Wait for BotStoppedSpeakingFrame before advancing turn
                self.context_aggregator.assistant(),
                self.output_transport,  # Paces bot audio & inserts silence for both tracks
                self.audio_buffer,  # Record continuous wall-clock aligned audio
            ]
        )

        self.task = PipelineTask(
            pipeline,
            idle_timeout_secs=45,
            idle_timeout_frames=(InputAudioRawFrame, OutputAudioRawFrame, MetricsFrame),
            params=PipelineParams(
                allow_interruptions=False,
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
        )

        # Register event handler to detect when pipeline is ready
        # This fires when StartFrame reaches the end of the pipeline,
        # meaning all services (including OpenAI WebSocket) are connected
        @self.task.event_handler("on_pipeline_started")
        async def on_pipeline_started(task, frame):
            logger.info("[Pipeline] StartFrame reached end - pipeline is ready, initializing recording")
            await self._initialize_recording_and_start_audio()

    async def _queue_first_turn(self) -> None:
        """Queue the startup context frame for Gemini Live."""
        if self._is_gemini_live():
            await self.task.queue_frames([LLMContextFrame(self.context)])

    async def _initialize_recording_and_start_audio(self) -> None:
        """Initialize recording baselines and queue the first turn."""
        if self.output_transport is not None:
            self.output_transport.reset_recording_baseline(
                recording_sample_rate=self.audio_buffer._init_sample_rate
            )
            self.output_transport.enable_greeting_tag()
            logger.info("[AudioRecording] NullAudioOutputTransport recording baseline set")

        if self.paced_input is not None:
            self.paced_input.set_recording_baseline()
            logger.info("[AudioRecording] PacedInputTransport recording baseline set")

        logger.info("[AudioRecording] Starting audio recording")
        await self.audio_buffer.start_recording()

        if hasattr(self, 'llm_logger') and self.llm_logger is not None and self.output_transport is not None:
            self.llm_logger.set_recording_start_time(self.output_transport._recording_start_time)

        self._seed_rehydration_history_for_reconnects()
        await self._seed_openai_rehydration_history()

        if self._oracle_continuation_only:
            logger.info(
                "[OracleContinuation] Starting assistant generation from GT-seeded "
                "current-turn tool state"
            )
            await self.task.queue_frames([LLMRunFrame()])
            return

        # In rehydration mode, skip greeting entirely — no user message was sent
        # (see _setup_context) and no inference was triggered.  The model will
        # process its (potentially large) system instruction lazily when the
        # first real user audio arrives.
        # For Gemini, wait for the initial context reconnect to finish so
        # audio frames aren't dropped during the reconnect window.
        if self._rehydration_turns:
            if self._is_gemini_live() and isinstance(self.llm, GeminiLiveLLMServiceWithReconnection):
                logger.info("[Pipeline] Rehydration mode — waiting for Gemini context reconnect to complete...")
                try:
                    await asyncio.wait_for(self.llm._context_ready.wait(), timeout=30.0)
                    logger.info("[Pipeline] Gemini context ready, proceeding to audio")
                except asyncio.TimeoutError:
                    logger.warning("[Pipeline] Gemini context ready timed out after 30s, proceeding anyway")
            else:
                logger.info(
                    f"[Pipeline] Rehydration mode — skipping greeting, proceeding to audio "
                    + (
                        f"({len(self._rehydration_turns)} golden turns via conversation.item.create)"
                        if self._uses_openai_realtime_history_seeding()
                        else f"({len(self._rehydration_turns)} golden turns in system prompt)"
                    )
                )
        else:
            # Trigger initial greeting for models that need explicit ResponseCreateEvent.
            # - Ultravox: auto-greets when websocket connects (no trigger needed)
            # - OpenAI/Grok Realtime: need LLMRunFrame to trigger _create_response()
            # - Gemini Live: auto-greets via inference_on_context_initialization=True
            if self._is_openai_realtime() or self._is_grok_realtime():
                logger.info("[Pipeline] Triggering initial greeting via LLMRunFrame for OpenAI/Grok Realtime")
                await self.task.queue_frames([LLMRunFrame()])

            # Two-phase wait for greeting to finish before sending user audio:
            # 1. Wait up to 8s for bot to START speaking (BotStartedSpeakingFrame)
            # 2. If bot started, wait up to 30s for bot to STOP speaking (BotStoppedSpeakingFrame)
            # 3. If no bot speech within 8s, proceed immediately (model doesn't greet)
            greeting_start_timeout = 8.0
            greeting_complete_timeout = 30.0

            logger.info(f"[Pipeline] Waiting up to {greeting_start_timeout}s for initial greeting to start...")
            greeting_occurred = False
            try:
                await asyncio.wait_for(self._greeting_started.wait(), timeout=greeting_start_timeout)
                logger.info(f"[Pipeline] Bot started greeting, waiting up to {greeting_complete_timeout}s for completion...")
                try:
                    await asyncio.wait_for(self._greeting_done.wait(), timeout=greeting_complete_timeout)
                    logger.info("[Pipeline] Initial greeting complete, proceeding with user audio")
                    greeting_occurred = True
                except asyncio.TimeoutError:
                    logger.error(
                        f"[TURN_FAILURE] Greeting did not complete within {greeting_complete_timeout}s timeout. "
                        "Bot started speaking but never stopped. This may indicate a hung connection or model issue."
                    )
                    greeting_occurred = True
            except asyncio.TimeoutError:
                logger.info("[Pipeline] No greeting started within timeout, model doesn't greet — proceeding with user audio")

            if greeting_occurred:
                logger.info("[Pipeline] Clearing TurnGate state after greeting")
                self.turn_gate.clear_pending()

        # Queue first turn audio
        turn = self._get_current_turn()
        audio_path = self._get_audio_path_for_turn(self.turn_idx)
        self.current_turn_audio_path = audio_path

        if audio_path:
            try:
                self._reset_openai_manual_turn_state()
                # Track when this audio will finish playing
                audio_duration = self._get_audio_duration(audio_path)
                self._current_audio_end_time = time.monotonic() + audio_duration
                # Track first user end time for V2V metrics (only set once per turn)
                if self._first_user_end_time is None:
                    self._first_user_end_time = self._current_audio_end_time
                    logger.info(
                        f"[USER_AUDIO_QUEUED] turn={self.turn_idx} "
                        f"predicted_end={self._first_user_end_time:.3f} duration={audio_duration:.3f}"
                    )
                logger.info(
                    f"Queued paced audio for first turn: {audio_path} "
                    f"(duration: {audio_duration:.1f}s)"
                )
                self.paced_input.enqueue_wav_file(audio_path)
                self._start_turn_watchdog(audio_duration)
            except Exception as e:
                logger.exception(f"Failed to queue audio from {audio_path}: {e}")
                self.current_turn_audio_path = None
                self._current_audio_end_time = 0
                # Fall back to text
                if self._is_gemini_live():
                    await self.task.queue_frames(
                        [
                            LLMMessagesAppendFrame(
                                messages=[{"role": "user", "content": turn["input"]}]
                            )
                        ]
                    )
                else:
                    await self.task.queue_frames([LLMRunFrame()])
        else:
            # No audio file, use text
            self._current_audio_end_time = 0
            if self._is_gemini_live():
                await self.task.queue_frames(
                    [
                        LLMMessagesAppendFrame(
                            messages=[{"role": "user", "content": turn["input"]}]
                        )
                    ]
                )
            else:
                await self.task.queue_frames([LLMRunFrame()])

    async def _queue_next_turn(self) -> None:
        """Queue audio or text for the next turn."""
        turn = self._get_current_turn()
        audio_path = self._get_audio_path_for_turn(self.turn_idx)
        self.current_turn_audio_path = audio_path

        if audio_path:
            try:
                self._reset_openai_manual_turn_state()
                # Track when this audio will finish playing
                audio_duration = self._get_audio_duration(audio_path)
                self._current_audio_end_time = time.monotonic() + audio_duration
                # Track first user end time for V2V metrics (only set once per turn)
                if self._first_user_end_time is None:
                    self._first_user_end_time = self._current_audio_end_time
                    logger.info(
                        f"[USER_AUDIO_QUEUED] turn={self.turn_idx} "
                        f"predicted_end={self._first_user_end_time:.3f} duration={audio_duration:.3f}"
                    )
                logger.info(
                    f"Queued paced audio for turn {self.turn_idx}: {audio_path} "
                    f"(duration: {audio_duration:.1f}s)"
                )
                self.paced_input.enqueue_wav_file(audio_path)
                self._start_turn_watchdog(audio_duration)
            except Exception as e:
                logger.exception(f"Failed to queue audio for turn {self.turn_idx}: {e}")
                audio_path = None

        if not audio_path:
            self.current_turn_audio_path = None
            self._current_audio_end_time = 0
            # Fall back to text
            if self._is_gemini_live():
                await self.task.queue_frames(
                    [
                        LLMMessagesAppendFrame(
                            messages=[{"role": "user", "content": turn["input"]}],
                            run_llm=False,
                        )
                    ]
                )
            else:
                # OpenAI Realtime fallback
                self.context.add_messages([{"role": "user", "content": turn["input"]}])
                await self.task.queue_frames([LLMRunFrame()])

    async def _cleanup_after_run(self) -> None:
        """Stop watchdogs and feeder threads for one-turn rehydrated runs."""
        self._cancel_turn_watchdog()
        if self.turn_gate is not None:
            self.turn_gate.clear_pending()
        if self.paced_input is not None:
            self.paced_input.shutdown_feeder()

    async def _on_turn_end(self, assistant_text: str) -> None:
        """Override to wait for user audio to finish before advancing turn.

        This prevents the next turn's audio from being queued while the current
        turn's audio is still playing. Without this, models with aggressive VAD
        (like gpt-realtime with default settings) may respond prematurely, causing
        the benchmark to queue the next turn before the current audio finishes.
        This results in overlapping audio in the recording.

        Args:
            assistant_text: The assistant's response text.
        """
        if self.done:
            return

        self._cancel_turn_watchdog()

        truncated = len(assistant_text) < 20
        if truncated:
            logger.warning(
                f"[TRUNCATED_RESPONSE] turn={self.turn_idx} "
                f"text='{assistant_text}' ({len(assistant_text)} chars) — "
                f"response appears truncated"
            )

        # Wait for current user audio to finish playing before advancing
        # This ensures clean separation between turns in the recording
        if self._current_audio_end_time > 0:
            remaining = self._current_audio_end_time - time.monotonic()
            if remaining > 0:
                logger.info(
                    f"[TurnSync] Waiting {remaining:.1f}s for user audio to finish "
                    f"before advancing to next turn"
                )
                await asyncio.sleep(remaining)
            # Clear the end time after waiting
            self._current_audio_end_time = 0

        # Reset retry tracking for next turn
        self._turn_retry_count = 0
        self._first_user_end_time = None

        # Capture conversation history for context preservation across reconnections.
        # Recorded for all models so any pipeline can use it on disconnect.
        if self.turn_idx < len(self.effective_turns):
            user_text = self.effective_turns[self.turn_idx].get("input", "")
            tool_calls = list(self._seen_tool_calls) if self._seen_tool_calls else []
            entry: dict = {"user": user_text, "assistant": assistant_text}
            if tool_calls:
                entry["tool_calls"] = [
                    {"name": name, "args": args} for name, args in tool_calls
                ]
            self._conversation_history.append(entry)

        # Call base class implementation for common turn handling
        await super()._on_turn_end(assistant_text)
