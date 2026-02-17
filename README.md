# ConversationBench: Multi-Turn Speech-to-Speech Evaluation Benchmark

ConversationBench is a benchmark for evaluating speech-to-speech voice AI models through spoken audio. It tests knowledge retrieval, tool use, error recovery, adversarial attacks, long-range memory, and numerical reasoning across 75 turns of a single continuous conversation.

ConversationBench is part of [Audio Arena](https://audioarena.ai), built by [Arcada Labs](https://arcada.dev).

- **Leaderboard & results**: [audioarena.ai](https://audioarena.ai)
- **Dataset on Hugging Face**: [arcada-labs/audio-arena](https://huggingface.co/datasets/arcada-labs/audio-arena)
- **Source code**: [github.com/Design-Arena/audio-arena](https://github.com/Design-Arena/audio-arena)

## What makes this different from text benchmarks

- **Audio input**: Each turn is a `.wav` file generated with TTS (OpenAI `tts-1`, `alloy` voice), not text. Models must process speech, not read.
- **Continuous conversation**: All 75 turns form a single continuous conversation. Later turns reference earlier ones. The model must track registrations, cancellations, corrections, and prior answers across the full session.
- **Tool use over speech**: The model has 9 functions it can call (register for sessions, cancel actions, check conflicts, etc.) and must decide when and how to call them based on spoken instructions.
- **Adversarial and edge-case turns**: Prompt injection, sycophancy traps, false presuppositions, distractor injection, and implicit corrections — all delivered via voice.

## Benchmark scenario

The conversation simulates a voice assistant for the **AI Engineer World's Fair 2025** conference. The user ("Jennifer Smith") asks about sessions, registers for talks, submits suggestions, deals with errors, and tests the model's limits over 75 turns.

The model is grounded in a **946-line knowledge base** containing the full conference schedule, speaker bios, venue logistics, ticket pricing, and more. It also has access to **9 tool functions** for actions like registering for sessions, voting, and submitting dietary requests.

## Quick Start

```bash
# Install dependencies
uv sync

# List available benchmarks
uv run audio-arena list-benchmarks

# Run a benchmark. Results will be saved to runs/conversation_bench/<timestamp>_<model_name>
uv run audio-arena run conversation_bench --model claude-sonnet-4-5 --service anthropic

# Judge the results
uv run audio-arena judge runs/conversation_bench/<timestamp>_claude-sonnet-4-5
```

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Design-Arena/audio-arena.git
cd audio-arena
uv sync
```

## Environment Variables

Set the API keys for the services you want to use. You only need the keys for the models you plan to test.

```bash
# Required for judging (Claude evaluates all benchmark results)
export ANTHROPIC_API_KEY=sk-ant-...

# Model provider keys (set whichever you need)
export OPENAI_API_KEY=sk-...          # OpenAI (text and realtime)
export GOOGLE_API_KEY=...             # Google (Gemini text and Gemini Live)
export ULTRAVOX_API_KEY=...           # Ultravox
export XAI_API_KEY=...                # xAI (Grok Realtime)

# AWS Nova models (text and speech-to-speech)
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...          # Optional: for temporary credentials
export AWS_REGION=us-east-1           # Optional: defaults to us-east-1
```

You can also create a `.env` file in the project root with these variables.

## CLI Commands

### Running Benchmarks

```bash
# Basic usage with text model
uv run audio-arena run <benchmark> --model <model> --service <service>

# Examples:
uv run audio-arena run conversation_bench --model claude-sonnet-4-5 --service anthropic
uv run audio-arena run conversation_bench --model gpt-4o --service openai
uv run audio-arena run conversation_bench --model gemini-2.5-flash --service google

# Realtime audio models 
uv run audio-arena run conversation_bench --model gpt-realtime --service openai-realtime
uv run audio-arena run conversation_bench --model gemini-2.5-flash-native-audio-preview-12-2025 --service gemini-live
uv run audio-arena run conversation_bench --model ultravox-v0.7 --service ultravox-realtime

# Nova Sonic (no --service needed, pipeline creates its own LLM)
uv run audio-arena run conversation_bench --model amazon.nova-2-sonic-v1:0 --pipeline nova-sonic

# Grok (xAI) Realtime
uv run audio-arena run conversation_bench --model grok-realtime

# Debug with limited turns
uv run audio-arena run conversation_bench --model gpt-4o --service openai --only-turns 0,1,2

# Verbose logging
uv run audio-arena run conversation_bench --model gpt-4o --service openai --verbose
```

### Judging Runs

After a benchmark run completes, judge the results using Claude:

```bash
# Judge a specific run
uv run audio-arena judge runs/conversation_bench/20251213T123456_claude-sonnet-4-5

# Judge with specific turns
uv run audio-arena judge runs/conversation_bench/20251213T123456_claude-sonnet-4-5 --only-turns 0,1,2

# Use a different judge model
uv run audio-arena judge runs/conversation_bench/20251213T123456_claude-sonnet-4-5 --judge-model claude-sonnet-4-5
```

The Claude judge evaluates each turn on up to 5 dimensions:

| Dimension | Scored on | Description |
|-----------|-----------|-------------|
| `tool_use_correct` | Every turn | Did the model call the expected function with correct arguments? |
| `instruction_following` | Every turn | Did the model answer the question or advance the task? |
| `kb_grounding` | Every turn | Is the response factually consistent with the knowledge base? |
| `state_tracking` | Turns tagged with long-range memory, cancellation flow, or implicit correction | Does the model correctly track information from earlier turns? |
| `ambiguity_handling` | Turns tagged with ambiguous entity or compound ambiguity | Does the model correctly disambiguate entities and constraints? |

For speech-to-speech runs, a 6th dimension is added automatically when a `conversation.wav` file is present:

| Dimension | Description |
|-----------|-------------|
| `turn_taking` | Audio timing correctness — detects missing responses, negative TTFB, empty responses (control tokens only), alignment drift, and audio overlap |

When turn-taking failures occur, the judge is more lenient on `instruction_following` since garbled audio may cause transcription issues.

```bash
# Judge a speech-to-speech run (turn-taking analysis runs automatically)
uv run audio-arena judge runs/conversation_bench/20260111T123456_gpt-realtime_abc123

# Skip turn-taking analysis
uv run audio-arena judge runs/conversation_bench/20260111T123456_gpt-realtime_abc123 --skip-turn-taking
```

Judge outputs (saved to the run directory):
- `claude_summary.json` — score metrics (includes `turn_taking_failures` for S2S runs)
- `claude_analysis.md` — human-readable report with failures
- `claude_judged.jsonl` — per-turn judgments with reasoning

See the [Methodology](#methodology) section for details on two-phase evaluation, penalty absorption, and category-aware scoring.

### Listing Options

```bash
# List available benchmarks
uv run audio-arena list-benchmarks

# List available pipelines
uv run audio-arena list-pipelines

# List service aliases
uv run audio-arena list-aliases
```

## Service Aliases

For convenience, common service classes have short aliases:

| Alias | Provider |
|-------|----------|
| `openai` | OpenAI (text models) |
| `openai-realtime` | OpenAI Realtime (speech-to-speech) |
| `anthropic` | Anthropic (Claude models) |
| `google` | Google (Gemini text models) |
| `gemini-live` | Google Gemini Live (speech-to-speech) |
| `bedrock` | AWS Bedrock (Nova text models) |
| `ultravox-realtime` | Ultravox (speech-to-speech) |

Additional providers (OpenRouter, Groq, Cerebras) are also supported — run `uv run audio-arena list-aliases` to see all options.

You can also use fully-qualified class names:

```bash
uv run audio-arena run conversation_bench \
    --model gpt-4o \
    --service pipecat.services.openai.llm.OpenAILLMService
```

## Benchmarks

Benchmarks are located in `benchmarks/`. Each benchmark is a self-contained Python package with:
- `config.py` - Benchmark configuration (turns, tools, system instruction)
- `turns.py` - Turn definitions with golden answers
- `tools.py` - Tool/function schema definitions
- `system.py` - System prompt with knowledge base
- `data/knowledge_base.txt` - Knowledge base content
- `audio/` - Pre-recorded audio files for each turn

### Available Benchmarks

| Benchmark | Description | Knowledge Base |
|-----------|-------------|----------------|
| `conversation_bench` | 75-turn hard benchmark | ~12K tokens |

## Pipelines

| Pipeline | Use Case | Auto-Detection Pattern |
|----------|----------|------------------------|
| `text` | Synchronous text LLMs | Default for all models |
| `realtime` | OpenAI Realtime, Gemini Live, Ultravox | `*realtime*`, `*native-audio*`, `*live*`, `*ultravox*` |
| `grok-realtime` | xAI Grok Realtime | `grok*realtime*` |
| `nova-sonic` | AWS Nova Sonic | `*nova-sonic*`, `*nova_sonic*` |

## Output Structure

Runs are saved to `runs/<benchmark>/<timestamp>_<model>/`:

```
runs/
└── conversation_bench/
    └── 20251213T123456_claude-sonnet-4-5/
        ├── transcript.jsonl        # Turn-by-turn results
        ├── runtime.json            # Run metadata and metrics
        ├── run.log                 # Debug logs
        ├── claude_summary.json     # Judge summary (after judging)
        ├── claude_judged.jsonl     # Per-turn judgments (after judging)
        └── claude_analysis.md      # Human-readable analysis (after judging)
```

## Project Structure

```
audio-arena/
├── src/audio_arena/               # Main package
│   ├── cli.py                     # CLI entry point
│   ├── pipelines/                 # Pipeline implementations
│   │   ├── base.py                # Abstract base pipeline
│   │   ├── text.py                # Text pipeline
│   │   ├── realtime.py            # Realtime pipeline (Gemini/Ultravox)
│   │   ├── openai_realtime.py     # OpenAI Realtime pipeline
│   │   ├── grok_realtime.py       # Grok Realtime pipeline
│   │   └── nova_sonic.py          # Nova Sonic pipeline
│   ├── processors/                # Frame processors
│   │   ├── tool_call_recorder.py  # Records tool calls
│   │   ├── tts_transcript.py      # TTS transcript handling
│   │   └── audio_buffer.py        # Audio buffer processing
│   ├── transports/                # Input/output transports
│   │   ├── paced_input.py         # Paced audio input
│   │   └── null_audio_output.py   # Null audio sink
│   ├── recording/                 # Transcript recording
│   │   └── transcript_recorder.py # Records transcripts
│   └── judging/                   # Judge implementations
│       ├── claude_judge.py        # Claude-based judging
│       └── turn_taking.py         # Turn-taking analysis
│
├── benchmarks/
│   └── conversation_bench/      # 75-turn hard benchmark
│       ├── config.py
│       ├── turns.py
│       ├── tools.py
│       ├── system.py
│       ├── audio/
│       └── data/
│
├── scripts/
│   └── analyze_turn_metrics.py    # Per-turn timing analysis
├── runs/                          # Output directory (gitignored)
├── LICENSE
└── pyproject.toml
```

## Comprehensive Turn Metrics Analysis

For detailed per-turn timing analysis of speech-to-speech models, use the comprehensive metrics script:

```bash
# Analyze a run with summary statistics
uv run python scripts/analyze_turn_metrics.py runs/conversation_bench/<timestamp>_<model>

# Show per-turn breakdown table
uv run python scripts/analyze_turn_metrics.py runs/conversation_bench/<timestamp>_<model> -v

# Output as JSON (for programmatic use)
uv run python scripts/analyze_turn_metrics.py runs/conversation_bench/<timestamp>_<model> --json
```

### Metrics Explained

The script consolidates timing data from multiple sources and calculates the following metrics:

| Metric | Description | Calculation |
|--------|-------------|-------------|
| **Server TTFB** | Time from request to first byte from model | Read from `transcript.jsonl` (reported by Pipecat) |
| **Pipeline TTFB** | Time from user speech end to bot audio tag | `bot_tag_log_ms - user_end_ms` (Silero VAD) |
| **WAV V2V** | Voice-to-voice latency measured from audio | `bot_silero_start_ms - user_end_ms` (Silero VAD) |
| **Silent Pad (RMS)** | Silent padding before speech (RMS detection) | `bot_rms_onset_ms - bot_tag_log_ms` |
| **Silent Pad (VAD)** | Silent padding before speech (Silero VAD) | `bot_silero_start_ms - bot_tag_wav_ms` |
| **Tag Alignment** | Drift between log position and WAV detection | `bot_tag_log_ms - bot_tag_wav_ms` |

**Key metric relationships:**
- **WAV V2V = Pipeline TTFB + Silent Pad (VAD)** - The total voice-to-voice latency includes both the time waiting for audio to arrive and any initial silence in the audio stream
- **Pipeline TTFB** measures when audio starts arriving at the pipeline
- **Silent Pad** measures how much silence is at the beginning of the audio (most models send 40-120ms of silence before speech)

### Alignment Sanity Check

The script verifies that log-based timestamps match actual audio positions by detecting audio tags (2kHz tones) embedded in the WAV file:

- **Bot tags**: Inserted when bot audio arrives at the pipeline
- **Alignment OK**: Log and WAV positions match within ±20ms tolerance
- **Issues detected**: Missing tags, extra tags, or drift outside tolerance

### Output Files

When run with `--json`, the script outputs structured data that can be saved:

```bash
# Save metrics to JSON file
uv run python scripts/analyze_turn_metrics.py runs/conversation_bench/<timestamp>_<model> --json > turn_metrics.json
```

## Methodology

### Original benchmark

ConversationBench builds on the original [30-turn multi-turn evaluation](https://github.com/kwindla/aiewf-eval) created by [Kwindla Kramer](https://github.com/kwindla) at [Daily](https://www.daily.co/) ([blog post](https://www.daily.co/blog/benchmarking-llms-for-voice-agent-use-cases/)). That benchmark tested both text and speech-to-speech models on tool use, instruction following, and knowledge base grounding in an AI Engineer World's Fair conference assistant scenario. It used a [Pipecat](https://github.com/pipecat-ai/pipecat)-based evaluation pipeline to drive multi-turn conversations against models from OpenAI, Google, Anthropic, and others, with Claude as an automated judge.

The original 30-turn benchmark was an important proof of concept — it demonstrated that multi-turn conversation evaluation over audio was both feasible and revealing. However, during development of ConversationBench we found that 30 turns were not sufficiently challenging: most frontier models scored above 90% on nearly every category, making it difficult to differentiate between models or identify meaningful failure modes.

### What changed in ConversationBench

We replaced the majority of the original turns and rebuilt the benchmark from scratch as a **75-turn static hard benchmark**. Only a small number of basic QA and tool-use turns from the original were retained, and even those were revised.

Key changes:

- **Most original questions were removed.** Only a handful of basic QA and tool-use turns were retained (and revised). The remaining turns are entirely new.
- **2.5x more turns.** The benchmark grew from 30 to 75 turns, enabling deeper stateful conversation testing and longer-range memory challenges.
- **Harder categories across the board:**
  - *Adversarial traps* — authority appeals, plausible hallucinations, subtle prompt injection, near-miss entities, false recall
  - *Multi-step tool use & long-range memory* — conditional logic, parallel chains, implicit requirements, rollbacks, recall across many turns
  - *Error recovery* — cascading failures, partial success states, ambiguous error messages
  - *Cancellation flow & state tracking* — changes of mind, correct handling of cancelled actions across turns
  - *Ambiguity handling* — same-name entities, compound ambiguity, dependent or contradictory constraints
  - *Implicit correction* — nested misconceptions, partial truths, false attributions
  - *Distractor injection* — buried questions, emotional manipulation, technical tangents
- **Expanded knowledge base.** The grounding document grew to 946 lines to support the more complex queries.
- **New evaluation dimensions.** ConversationBench adds `state_tracking` and `ambiguity_handling` as scored dimensions, in addition to the original three (`tool_use_correct`, `instruction_following`, `kb_grounding`).

### Scoring Rubric

**Category-aware dimensions.** Core dimensions (tool use, instruction following, KB grounding) are scored on every turn. `state_tracking` and `ambiguity_handling` are scored only on turns tagged with the relevant categories, so models are never penalized on out-of-scope dimensions.

**Two-phase evaluation.** An initial turn-by-turn pass is followed by a realignment pass that detects early or late function calls and cascading effects. If a required call was made a turn early, later turns are not penalized for the "missing" call; if made late, the turn where it actually happened gets credit.

**Penalty absorption.** When a missed tool call has a more specific root cause, the penalty lands on that dimension instead of `tool_use_correct` — e.g., unnecessary clarification penalizes `ambiguity_handling`, forgotten state penalizes `state_tracking`. This avoids double-penalizing while ensuring every failure is counted exactly once.

**Strict dimension separation.** Failing to call a tool is scored only under `tool_use_correct` (or absorbed by a more specific dimension). `instruction_following` fails only when the assistant's words and actions contradict each other in a non-tool sense.

**Turn-taking leniency.** For speech-to-speech runs, a `turn_taking` dimension captures audio timing issues (overlaps, interruptions, missing responses). When turn-taking fails, the judge is more lenient on `instruction_following` to account for transcription artifacts.

The benchmark is **static**: the same 75 user inputs (and corresponding audio) are used for every run, with golden expectations defined in `benchmarks/conversation_bench/turns.py`, so results are comparable across models and runs.

## Acknowledgments

Audio Arena is built on [Pipecat](https://github.com/pipecat-ai/pipecat), the open-source framework for voice and multimodal AI. The original 30-turn evaluation was created by [Kwindla Hultman Kramer](https://github.com/kwindla) at [Daily](https://www.daily.co/) — see the [original blog post](https://www.daily.co/blog/benchmarking-llms-for-voice-agent-use-cases/) and [repo](https://github.com/kwindla/aiewf-eval).

Judging is powered by [Claude](https://www.anthropic.com/) via the [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk). Voice activity detection uses [Silero VAD](https://github.com/snakers4/silero-vad).

## Citation

If you use this benchmark, please cite:

```bibtex
@misc{audioarena2026,
  title={AudioArena: A Multi-Turn Speech-to-Speech Evaluation Benchmark},
  author={Arcada Labs},
  year={2026},
  url={https://huggingface.co/datasets/arcada-labs/audio-arena}
}
```

## License

MIT — see [LICENSE](LICENSE) for details.
