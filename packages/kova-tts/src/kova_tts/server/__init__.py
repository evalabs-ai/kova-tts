"""A local HTTP + WebSocket server around :class:`~kova_tts.engine.tts.KovaTTS`.

Three ways to ask for the same audio:

* ``POST /v1/tts`` -- wait for the whole utterance, get a wav (or raw PCM) back;
* ``POST /v1/tts/stream`` -- Server-Sent Events, base64 PCM chunks as the codec produces them;
* ``WS /v1/ws`` -- an incremental session: send text as you have it, flush when you want audio.

This is a **single-user local server by design**. The LM generator holds one static KV cache and
one set of CUDA graph buffers, so exactly one generation can be in flight; the server serialises
on a lock and refuses a genuinely concurrent second caller with ``409`` rather than queueing it.
See :mod:`kova_tts.server.engine` for why the refusal is a short wait rather than an instant no.

    from kova_tts.server import create_app
    app = create_app(device="cuda")

or from a terminal::

    python -m kova_tts.server --port 8000
"""

from __future__ import annotations

from kova_tts.server.app import create_app, main

__all__ = ["create_app", "main"]
