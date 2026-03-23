# Audio Arena: Multi-Turn Speech-to-Speech Evaluation Framework

Audio Arena is a framework for evaluating speech-to-speech voice AI models through spoken audio. It includes 6 benchmarks spanning 221 total turns across different domains, testing knowledge retrieval, tool use, error recovery, adversarial attacks, long-range memory, state tracking, and numerical reasoning.

Built by [Arcada Labs](https://arcada.dev).

- **Leaderboard & results**: [audioarena.ai/leaderboard](https://audioarena.ai/leaderboard)
- **Datasets on Hugging Face**: [arcada-labs](https://huggingface.co/arcada-labs)
- **Source code**: [github.com/Design-Arena/audio-arena](https://github.com/Design-Arena/audio-arena)

## What makes this different from text benchmarks

- **Audio input**: Each turn is a `.wav` file (TTS-generated or human-recorded), not text. Models must process speech, not read.
- **Continuous conversation**: Turns form a single continuous conversation per benchmark. Later turns reference earlier ones. The model must track state, corrections, and prior answers across the full session.
- **Tool use over speech**: Models have domain-specific functions they can call and must decide when and how to call them based on spoken instructions.
- **Adversarial and edge-case turns**: Prompt injection, sycophancy traps, false presuppositions, false memory traps, distractor injection, and implicit corrections — all delivered via voice.

## Benchmarks

| Benchmark | Turns | Scenario | HF Dataset |
|-----------|-------|----------|------------|
| `conversation_bench` | 75 | Conference assistant for AI Engineer World's Fair — session registration, schedule queries, 9 tool functions, ~946-line knowledge base | [arcada-labs/conversation-bench](https://huggingface.co/datasets/arcada-labs/conversation-bench) |
| `appointment_bench` | 25 | Dental office appointment scheduling — two patients (Daniel/Danielle Nolan), two doctors, phone number swap+revert, 4 false memory traps, slot-taken error recovery | [arcada-labs/appointment-bench](https://huggingface.co/datasets/arcada-labs/appointment-bench) |
| `assistant_bench` | 31 | Personal assistant — flight booking, email, calendar, reminders, dual requests in single turns, topic switching, late references, correction-chain recall | [arcada-labs/assistant-bench](https://huggingface.co/datasets/arcada-labs/assistant-bench) |
| `event_bench` | 29 | Event planning — venue+catering+guest count changes, mid-sentence self-corrections, vague pronoun resolution, multi-request reversals, retroactive date changes | [arcada-labs/event-bench](https://huggingface.co/datasets/arcada-labs/event-bench) |
| `grocery_bench` | 30 | Grocery ordering — chained corrections, homophone collisions, conditional additions/removals, order reconciliation, quantity math, swap operations | [arcada-labs/grocery-bench](https://huggingface.co/datasets/arcada-labs/grocery-bench) |
| `product_bench` | 31 | Laptop comparison shopping — multi-intent turns, retroactive corrections, conditional arithmetic chains, discount stacking edge cases, 3-step order modification chains | [arcada-labs/product-bench](https://huggingface.co/datasets/arcada-labs/product-bench) |

Each benchmark is a self-contained Python package under `benchmarks/` with:
- `config.py` — Benchmark configuration (turns, tools, system instruction, HF repo)
- `turns.py` — Turn definitions with golden answers
- `tools.py` — Tool/function schema definitions
- `system.py` — System prompt with knowledge base
- `data/knowledge_base.txt` — Knowledge base content
- `audio/` — TTS audio, downloaded automatically from Hugging Face on first run (gitignored)
- `real_audio/` — Human-recorded audio per speaker, downloaded from HF when `--real-audio` is used (gitignored)

## Quick Start

```bash
# Install dependencies
uv sync

# List available benchmarks
uv run audio-arena list-benchmarks

# Run a benchmark (audio files download automatically from HF on first run)
uv run audio-arena run appointment_bench --model claude-sonnet-4-5 --service anthropic

# Judge the results
uv run audio-arena judge runs/appointment_bench/<timestamp>_claude-sonnet-4-5
```

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Design-Arena/audio-arena.git
cd audio-arena
uv sync
```

Audio files are hosted on Hugging Face and downloaded automatically when you first run a benchmark. To pre-download manually:

```bash
huggingface-cli download arcada-labs/appointment-bench --local-dir benchmarks/appointment_bench --include "audio/*.wav"
```

## Environment Variables

Set the API keys for the services you want to use. You only need the keys for the models you plan to test.

```bash
# Judging key (Claude is the default judge for all benchmarks)
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...         # Required for OpenAI judge override and OpenAI models

# Model provider keys (set whichever you need)
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
uv run audio-arena run appointment_bench --model gpt-4o --service openai
uv run audio-arena run grocery_bench --model gemini-2.5-flash --service google

# Realtime audio models 
uv run audio-arena run conversation_bench --model gpt-realtime --service openai-realtime
uv run audio-arena run event_bench --model gemini-2.5-flash-native-audio-preview-12-2025 --service gemini-live
uv run audio-arena run product_bench --model ultravox-v0.7 --service ultravox-realtime

# Nova Sonic (no --service needed, pipeline creates its own LLM)
uv run audio-arena run conversation_bench --model amazon.nova-2-sonic-v1:0 --pipeline nova-sonic

# Grok (xAI) Realtime
uv run audio-arena run conversation_bench --model grok-realtime

# Debug with limited turns
uv run audio-arena run appointment_bench --model gpt-4o --service openai --only-turns 0,1,2

# Rehydrated single-turn replay against prior golden context
uv run audio-arena run event_bench --model gpt-realtime-1.5 --service openai-realtime --rehydrate

# Rehydrated OpenAI Realtime with manual turn commits (VAD disabled)
uv run audio-arena run event_bench --model gpt-realtime-1.5 --service openai-realtime --rehydrate --disable-vad

# Verbose logging
uv run audio-arena run conversation_bench --model gpt-4o --service openai --verbose
```

`--rehydrate` runs each target turn in a fresh session with golden prior context. For OpenAI Realtime, prior turns are seeded into the live session with `conversation.item.create`; the benchmark does not use `response.create.input` to replay history. With `--disable-vad`, the target turn still uses live audio plus manual `input_audio_buffer.commit` and `response.create` events to close the user turn.

### Judging Runs

After a benchmark run completes, judge the results:

```bash
# Judge a specific run (defaults to Claude judge)
uv run audio-arena judge runs/appointment_bench/20251213T123456_gpt-4o

# Judge with specific turns
uv run audio-arena judge runs/conversation_bench/20251213T123456_claude-sonnet-4-5 --only-turns 0,1,2

# Use a different judge model
uv run audio-arena judge runs/conversation_bench/20251213T123456_claude-sonnet-4-5 --judge-model claude-sonnet-4-5

# Use the OpenAI judge backend instead
uv run audio-arena judge runs/event_bench/20260310T011417_gpt-realtime-1.5_52ea9df5 --judge openai
```

All benchmarks default to the Claude judge (`claude-opus-4-5`). Use `--judge openai` to switch to the OpenAI judge (`gpt-5.2`).

The judge evaluates each turn on up to 5 dimensions:

| Dimension | Scored on | Description |
|-----------|-----------|-------------|
| `tool_use_correct` | Every turn | Did the model call the expected function with correct arguments? |
| `instruction_following` | Every turn | Did the model answer the question or advance the task? |
| `kb_grounding` | Every turn | Is the response factually consistent with the knowledge base? |
| `state_tracking` | Turns tagged with long-range memory, cancellation flow, or implicit correction | Does the model correctly track information from earlier turns? |
| `ambiguity_handling` | Turns tagged with ambiguous entity or compound ambiguity | Does the model correctly disambiguate entities and constraints? |

In plain language, these grader metrics mean:

- `tool_use_correct`: The assistant used the right tool at the right time, with the right arguments, and did not skip a required tool call.
- `instruction_following`: The assistant actually completed the user's request for that turn. This is about answering the question or advancing the task, not just sounding plausible.
- `kb_grounding`: The assistant's factual claims are supported by the benchmark knowledge base or by tool results returned during the conversation.
- `state_tracking`: The assistant stayed consistent with earlier turns, including prior registrations, cancellations, corrections, and other conversation state.
- `ambiguity_handling`: The assistant handled under-specified or ambiguous requests correctly, such as asking for clarification when needed or resolving the ambiguity from available context.

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
- `<judge>_summary.json` — score metrics (includes `turn_taking_failures` for S2S runs)
- `<judge>_analysis.md` — human-readable report with failures
- `<judge>_judged.jsonl` — per-turn judgments with reasoning

For example, Claude judging writes `claude_summary.json`, `claude_analysis.md`, and `claude_judged.jsonl`.

See the [Methodology](#methodology) section for details on two-phase evaluation, penalty absorption, and category-aware scoring.

### Real Audio (Human-Recorded)

Benchmarks support real (human-recorded) audio alongside the default TTS-generated audio. Each speaker's recordings live in a subdirectory under `real_audio/`:

```
benchmarks/appointment_bench/real_audio/
├── person1/
│   ├── turn_000.wav
│   ├── turn_001.wav
│   └── ...
└── person2/
    └── ...
```

Real audio is hosted on the same HF dataset repos and downloaded automatically when needed.

```bash
# Run with a specific speaker's real audio
uv run audio-arena run appointment_bench --model gpt-realtime --real-audio person1

# Run all available speakers sequentially
uv run audio-arena run appointment_bench --model gpt-realtime --real-audio all

# List available speakers for a benchmark
uv run audio-arena list-speakers appointment_bench
```

To pre-download real audio manually:

```bash
huggingface-cli download arcada-labs/appointment-bench --local-dir benchmarks/appointment_bench --include "real_audio/**/*.wav"
```

### Listing Options

```bash
# List available benchmarks
uv run audio-arena list-benchmarks

# List available speakers for a benchmark
uv run audio-arena list-speakers appointment_bench

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

## Pipelines

| Pipeline | Use Case | Auto-Detection Pattern |
|----------|----------|------------------------|
| `text` | Synchronous text LLMs | Default for all models |
| `realtime` | OpenAI Realtime, Gemini Live, Ultravox | `*realtime*`, `*native-audio*`, `*live*`, `*ultravox*` |
| `grok-realtime` | xAI Grok Realtime | `grok*realtime*` |
| `nova-sonic` | AWS Nova Sonic | `*nova-sonic*`, `*nova_sonic*` |

## Output Structure

Runs are saved to `runs/<benchmark>/<timestamp>_<model>/` (gitignored):

```
runs/
└── appointment_bench/
    └── 20251213T123456_gpt-4o/
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
│   │   ├── realtime.py            # Shared realtime pipeline orchestration
│   │   ├── openai_realtime.py     # OpenAI Realtime explicit-tool-result service
│   │   ├── grok_realtime.py       # Grok Realtime pipeline
│   │   └── nova_sonic.py          # Nova Sonic pipeline
│   ├── processors/                # Frame processors
│   ├── transports/                # Input/output transports
│   ├── recording/                 # Transcript recording
│   └── judging/                   # Judge implementations
│       ├── llm_judge.py           # Claude judge
│       └── openai_judge.py        # OpenAI judge
│
├── benchmarks/                    # Benchmark packages
│   ├── conversation_bench/        # 75-turn conference assistant
│   ├── appointment_bench/         # 25-turn dental appointment scheduling
│   ├── assistant_bench/           # 31-turn personal assistant
│   ├── event_bench/               # 29-turn event planning
│   ├── grocery_bench/             # 30-turn grocery ordering
│   └── product_bench/             # 31-turn laptop comparison
│       ├── config.py              # HF repo, description, turn count
│       ├── turns.py               # Turn definitions with golden answers
│       ├── tools.py               # Tool/function schemas
│       ├── system.py              # System prompt + knowledge base
│       ├── data/knowledge_base.txt
│       ├── audio/                 # TTS audio (gitignored, downloaded from HF)
│       └── real_audio/            # Human-recorded audio (gitignored, downloaded from HF)
│
├── scripts/
│   ├── run_all_benchmarks.py      # Run all S2S models on all benchmarks
│   ├── analyze_turn_metrics.py    # Per-turn timing analysis
│   ├── compare_model_runs.py      # Multi-model comparison with CSV + plots
│   ├── build_experiment_review.py # Self-contained HTML review for a judged run
│   ├── generate_audio.py          # TTS WAV generation for benchmark turns
│   └── ...                        # Additional analysis and batch scripts
│
├── runs/                          # Output directory (gitignored)
├── LICENSE
└── pyproject.toml
```

Audio files are stored on Hugging Face, not in this repo. They are downloaded automatically on first run.

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

The methodology below describes the scoring rubric used across all benchmarks. The detailed history section is specific to `conversation_bench`, which was the first and largest benchmark in the suite.

### Scoring Rubric

**Category-aware dimensions.** Core dimensions (tool use, instruction following, KB grounding) are scored on every turn. `state_tracking` and `ambiguity_handling` are scored only on turns tagged with the relevant categories, so models are never penalized on out-of-scope dimensions.

**Two-phase evaluation.** An initial turn-by-turn pass is followed by a realignment pass that detects early or late function calls and cascading effects. If a required call was made a turn early, later turns are not penalized for the "missing" call; if made late, the turn where it actually happened gets credit.

**Penalty absorption.** When a missed tool call has a more specific root cause, the penalty lands on that dimension instead of `tool_use_correct` — e.g., unnecessary clarification penalizes `ambiguity_handling`, forgotten state penalizes `state_tracking`. This avoids double-penalizing while ensuring every failure is counted exactly once.

**Strict dimension separation.** Failing to call a tool is scored only under `tool_use_correct` (or absorbed by a more specific dimension). `instruction_following` fails only when the assistant's words and actions contradict each other in a non-tool sense.

**Turn-taking leniency.** For speech-to-speech runs, a `turn_taking` dimension captures audio timing issues (overlaps, interruptions, missing responses). When turn-taking fails, the judge is more lenient on `instruction_following` to account for transcription artifacts.

Each benchmark is **static**: the same user inputs (and corresponding audio) are used for every run, with golden expectations defined in each benchmark's `turns.py`, so results are comparable across models and runs.

### ConversationBench history

ConversationBench builds on the original [30-turn multi-turn evaluation](https://github.com/kwindla/aiewf-eval) created by [Kwindla Hultman Kramer](https://github.com/kwindla) at [Daily](https://www.daily.co/) ([blog post](https://www.daily.co/blog/benchmarking-llms-for-voice-agent-use-cases/)). That benchmark tested both text and speech-to-speech models on tool use, instruction following, and knowledge base grounding in an AI Engineer World's Fair conference assistant scenario. It used a [Pipecat](https://github.com/pipecat-ai/pipecat)-based evaluation pipeline to drive multi-turn conversations against models from OpenAI, Google, Anthropic, and others, with Claude as an automated judge.

The original 30-turn benchmark was an important proof of concept — it demonstrated that multi-turn conversation evaluation over audio was both feasible and revealing. However, during development of ConversationBench we found that 30 turns were not sufficiently challenging: most frontier models scored above 90% on nearly every category, making it difficult to differentiate between models or identify meaningful failure modes.

We replaced the majority of the original turns and rebuilt the benchmark from scratch as a **75-turn static hard benchmark**. Only a small number of basic QA and tool-use turns from the original were retained, and even those were revised. The remaining 5 benchmarks (`appointment_bench`, `assistant_bench`, `event_bench`, `grocery_bench`, `product_bench`) were built from scratch to test different domains and failure modes.

## Acknowledgments

Audio Arena is built on [Pipecat](https://github.com/pipecat-ai/pipecat), the open-source framework for voice and multimodal AI. The original 30-turn evaluation was created by [Kwindla Hultman Kramer](https://github.com/kwindla) at [Daily](https://www.daily.co/) — see the [original blog post](https://www.daily.co/blog/benchmarking-llms-for-voice-agent-use-cases/) and [repo](https://github.com/kwindla/aiewf-eval).

Judging is powered by [Claude](https://www.anthropic.com/) via the [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk). Voice activity detection uses [Silero VAD](https://github.com/snakers4/silero-vad).

## Citation

If you use these benchmarks, please cite:

```bibtex
@misc{audioarena2026,
  title={Audio Arena: Multi-Turn Speech-to-Speech Evaluation Benchmarks},
  author={Arcada Labs},
  year={2026},
  url={https://audioarena.ai}
}
```

## License

MIT — see [LICENSE](LICENSE) for details.
