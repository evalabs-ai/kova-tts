"""The Gradio demo: for most people, the first time they hear this model.

Run it with ``kova-tts demo``, or directly::

    uv run python apps/demo/app.py --port 7860

This file composes the rest of the directory and launches it. :mod:`content` holds what the
page says, :mod:`session` the engine and the state around it, :mod:`streaming` the endpoint the
browser pulls audio from, and :mod:`ui` the Blocks themselves.

A Gradio app is a FastAPI app underneath, which is what :func:`build_app` trades on: the page
and the streaming endpoint go in one process holding one model, so the browser talks to the
model directly without a second server, a second copy of the weights, or a cross-origin
request.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Any

import gradio as gr
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from kova_tts import CLONE_SAMPLING, TTS_SAMPLING
from kova_tts.server import errors as server_errors
from kova_tts.server.protocol import ErrorResponse

# Every way in reaches this file first, and only some of them give it a package to import its
# siblings from: ``kova-tts demo`` loads it straight from its path. Putting this directory on
# the path makes the plain imports below work in all of them.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from content import BUSY, CSS, EXAMPLES, TITLE  # noqa: E402
from session import (  # noqa: E402
    BASE_LABEL,
    BASE_VOICE,
    BUSY_TIMEOUT,
    DemoError,
    DemoSession,
    build_transcriber,
    load_engine,
    warm_up,
)
from streaming import MAX_CHARS, STREAM_PATH, add_stream_route  # noqa: E402
from ui import build_ui, player_js  # noqa: E402

log = logging.getLogger("kova_tts.demo")

# Re-exported, so that app.py stays the one file a caller has to load to get the whole demo.
__all__ = [
    "BASE_LABEL",
    "BASE_VOICE",
    "BUSY",
    "BUSY_TIMEOUT",
    "CLONE_SAMPLING",
    "CSS",
    "EXAMPLES",
    "MAX_CHARS",
    "STREAM_PATH",
    "TITLE",
    "TTS_SAMPLING",
    "DemoError",
    "DemoSession",
    "add_stream_route",
    "build_app",
    "build_transcriber",
    "build_ui",
    "load_engine",
    "main",
    "player_js",
    "warm_up",
]


def build_app(
    session: DemoSession | None = None,
    *,
    title: str = TITLE,
    stream_path: str = STREAM_PATH,
) -> FastAPI:
    """The whole demo as one ASGI application: the page, and the audio it plays.

    Mounting the Blocks inside a FastAPI app of our own -- rather than letting ``launch()``
    build one -- is what makes room for the streaming endpoint, and it is also what lets a test
    drive the page and the stream through one client.
    """
    session = session or DemoSession(loader=load_engine)
    app = FastAPI(title=title, docs_url="/docs", redoc_url=None)

    # The server package's handlers: `Busy` becomes a 409 and a validation failure a 422, both
    # in the {"error", "message"} envelope the player already knows how to read.
    server_errors.install(app)

    @app.exception_handler(DemoError)
    async def _unavailable(_request: Any, exc: DemoError) -> JSONResponse:
        # A machine with no weights is not a bad request and not a bug: it is a server that
        # cannot serve yet, and the message says which file to point where.
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(error="unavailable", message=str(exc)).model_dump(),
        )

    add_stream_route(app, session, path=stream_path)
    return gr.mount_gradio_app(
        app,
        build_ui(session, title=title, stream_path=stream_path),
        path="/",
        theme=gr.themes.Soft(),
        css=CSS,
        show_error=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kova-tts demo",
        description="Launch the Kova TTS demo in a browser.",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="interface to bind (default: %(default)s)"
    )
    parser.add_argument("--port", type=int, default=7860, help="port (default: %(default)s)")
    parser.add_argument("--share", action="store_true", help="expose a public gradio.live link")
    parser.add_argument("--open", action="store_true", help="open a browser window on startup")
    parser.add_argument(
        "--preload",
        action="store_true",
        help="load and warm up the model at startup, so the first visitor waits for none of it",
    )
    parser.add_argument("--model", default=None, help="model directory or Hub repo id")
    parser.add_argument("--codec", default=None, help="codec checkpoint")
    parser.add_argument("--wavlm", default=None, help="WavLM directory or Hub repo id")
    parser.add_argument("--lora-dir", default=None, help="directory of LoRA voices")
    parser.add_argument("--device", default=None, help="torch device, e.g. cuda:1 or mps")
    parser.add_argument(
        "--decode-window",
        type=int,
        default=None,
        help="codec frames per streamed chunk, and so how far apart they are; raising it "
        "trades latency for throughput, which is worth doing on Apple Silicon",
    )
    parser.add_argument(
        "--backend",
        default=None,
        choices=("auto", "torch", "mlx"),
        help="decode loop; the default reads it off the checkpoint",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="log what the engine is doing")
    return parser


def _banner_backend(args: argparse.Namespace) -> str | None:
    """Which backend the banner should name, resolved without loading anything.

    The banner paints before the weights are read, so this has to answer from the checkpoint
    on disk. A checkpoint that cannot be resolved at all is not this function's problem --
    the banner has its own line for that -- so it answers ``None`` and says nothing.
    """
    from kova_tts.engine import backends

    try:
        return backends.resolve(args.model, args.backend)
    except Exception:  # noqa: BLE001 - a banner must never be the thing that fails to load
        return None


def share_link(host: str, port: int) -> str | None:
    """A temporary public URL for a locally bound server, using Gradio's tunnel.

    ``launch()`` would have done this; the demo runs its own server instead, so it asks for the
    tunnel directly. A failure here is worth a sentence, not a refusal to serve: the local URL
    still works.
    """
    import secrets

    from gradio import networking

    try:
        url = networking.setup_tunnel(host, port, secrets.token_urlsafe(32), None, None)
    except Exception as exc:  # noqa: BLE001 - no tunnel is a degraded demo, not a broken one
        log.warning("Could not create a share link: %s", exc)
        return None
    return url.replace("http://", "https://", 1)


def main(argv: list[str] | None = None) -> int:
    """Launch the demo. Blocks until the server is stopped."""
    import uvicorn

    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    session = DemoSession(
        loader=lambda: load_engine(
            model=args.model,
            codec=args.codec,
            wavlm=args.wavlm,
            lora_dir=args.lora_dir,
            device=args.device,
            backend=args.backend,
            decode_window=args.decode_window,
        ),
        lora_root=args.lora_dir,
        device=args.device,
        backend=_banner_backend(args),
    )
    if args.preload:
        try:
            warm_up(session.engine())
        except DemoError as exc:
            # Not fatal: the page is still worth serving, and it will say the same thing.
            print(f"warning: {exc}", file=sys.stderr)

    app = build_app(session)
    local_url = f"http://{args.host}:{args.port}"
    print(f"{TITLE} on {local_url}", file=sys.stderr)
    if args.share and (public := share_link(args.host, args.port)):
        print(f"Public URL: {public}", file=sys.stderr)
    if args.open:
        # After the bind, not before: a browser that arrives first sees a connection refused.
        threading.Timer(1.5, webbrowser.open, args=(local_url,)).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info" if args.verbose else "error")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
