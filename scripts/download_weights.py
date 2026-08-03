"""Prefetch the model weights into the local Hugging Face cache.

    python scripts/download_weights.py --wavlm

Same work as ``kova-tts download``, reachable without the console script: a Dockerfile wants the
download in a layer of its own so a rebuild does not repeat it, and a machine that is about to
lose its network wants it now rather than at the first synthesis.

The flags and the implementation both live in :mod:`kova_tts.cli`, so the two entry points cannot
drift apart. Anything already pointed at a local checkpoint by ``.env`` is reported and skipped.
"""

from __future__ import annotations

import argparse
import sys

from kova_tts.cli import CommandError, add_download_arguments, run_download


def main(argv: list[str] | None = None) -> int:
    parser = add_download_arguments(
        argparse.ArgumentParser(
            prog="download_weights.py",
            description="Prefetch the LM, codec and optionally WavLM for offline use.",
        )
    )
    try:
        return run_download(parser.parse_args(argv))
    except CommandError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
