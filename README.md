# Lume

Lume is a macOS voice assistant. It uses the DeepSeek API for reasoning and
downloads the FunASR `iic/SenseVoiceSmall` speech-recognition model locally on
first use.

## Quick start on a new Mac

Requirements: macOS, Xcode Command Line Tools, Python 3.13, and microphone and
Accessibility permissions. Install [uv](https://docs.astral.sh/uv/) if it is
not already available.

```sh
git clone git@github.com:brookwillow/lume.git
cd lume
export DEEPSEEK_API_KEY='your-deepseek-api-key'
uv sync --locked --project py
uv run --project py lume-prefetch-model
uv run --project py lume
```

`lume-prefetch-model` downloads and checks the ASR model once, so the first
interactive launch does not wait for model download. Model files are cached by
ModelScope (normally under `~/.cache/modelscope`) and are deliberately not in
Git: they are large generated/downloaded artifacts. To reuse them across Macs,
copy that cache directory or set `MODELSCOPE_CACHE` to a shared path before
running the prefetch command.

The application will ask macOS for microphone and Accessibility permissions.
Grant both in **System Settings → Privacy & Security**. The API key is read only
from `DEEPSEEK_API_KEY`; use `.env.example` as a reference, but do not commit a
real key.

## Development

```sh
uv run --project py pytest py/tests
```

Dependency versions are pinned in `py/uv.lock`. Refresh them intentionally with
`uv lock --project py`, then commit the updated lock file.
