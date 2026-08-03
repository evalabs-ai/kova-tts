"""``python -m kova_tts.server`` -- the server without going through the top-level CLI."""

from __future__ import annotations

import sys

from kova_tts.server.app import main

if __name__ == "__main__":
    sys.exit(main())
