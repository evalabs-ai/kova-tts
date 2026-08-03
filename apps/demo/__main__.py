"""``python -m apps.demo``, equivalent to ``python apps/demo/app.py``."""

from __future__ import annotations

import sys

from .app import main

if __name__ == "__main__":
    sys.exit(main())
