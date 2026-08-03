"""Building the application, and running it.

:func:`create_app` is the seam everything else hangs off. It takes either a model to load or a
model already loaded, so the test suite drives every endpoint against a stub without a GPU, a
checkpoint, or torch, and the serving path differs only in which object ends up in
``app.state``.

The model is loaded **once, in the lifespan**, off the event loop. Nothing else in this package
constructs a :class:`~kova_tts.engine.tts.KovaTTS`; a per-request load would be seconds of
latency and several gigabytes of duplicated weights.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from kova_tts import __version__
from kova_tts.server import errors, routes, ws
from kova_tts.server.engine import DEFAULT_BUSY_TIMEOUT, Engine

log = logging.getLogger(__name__)

#: Loopback by default. This server has no authentication and it does not want any: the answer
#: to "who may use my GPU" is "whoever is on this machine". Pass ``--host 0.0.0.0`` knowingly.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

#: Reference codes pushed through the codec before a cloned generation. One second, rather than
#: the whole clip, because the preroll is decoded before a single word of the target text is:
#: it is time-to-first-audio spent re-rendering audio that is thrown away again.
DEFAULT_CLONE_PREROLL = 80

#: Spoken once at startup when ``warmup`` is on. Its content does not matter; what matters is
#: that it is a whole short sentence, so the run touches every stage.
WARMUP_TEXT = "The model is ready."


def create_app(
    *,
    tts: Any | None = None,
    model: str | os.PathLike[str] | None = None,
    codec: str | os.PathLike[str] | None = None,
    wavlm: str | os.PathLike[str] | None = None,
    device: str | None = None,
    lora_dir: str | os.PathLike[str] | None = None,
    clone_preroll: int | None = DEFAULT_CLONE_PREROLL,
    busy_timeout: float = DEFAULT_BUSY_TIMEOUT,
    warmup: bool = True,
) -> FastAPI:
    """Build the application.

    Args:
        tts: An already-loaded model. When given, nothing is loaded and `model`, `codec`,
            `wavlm`, `device`, `lora_dir`, `clone_preroll` and `warmup` are all ignored -- this
            is how the tests drive the endpoints against a stub.
        model: LM directory or Hub repo id. ``None`` resolves through :mod:`kova_tts.paths`.
        codec: Codec checkpoint. ``None`` resolves through :mod:`kova_tts.paths`.
        wavlm: WavLM directory, needed only to encode reference audio.
        device: Torch device for the model, e.g. ``"cuda"`` or ``"cuda:1"``.
        lora_dir: Directory of LoRA voices, one subdirectory each.
        clone_preroll: Reference codes used to warm the codec before a cloned generation.
        busy_timeout: Seconds a second caller waits for the model before getting a 409.
        warmup: Synthesize one short sentence at startup. Worth it: it forces the codec load
            and the CUDA graph capture that the first real request would otherwise pay for.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        injected = app.state.tts
        loaded = (
            injected
            if injected is not None
            else await asyncio.to_thread(
                _load,
                model=model,
                codec=codec,
                wavlm=wavlm,
                device=device,
                lora_dir=lora_dir,
                clone_preroll=clone_preroll,
            )
        )
        app.state.engine = Engine(loaded, busy_timeout=busy_timeout)
        if injected is None and warmup:
            await asyncio.to_thread(_warm, loaded)
        try:
            yield
        finally:
            app.state.engine = None

    app = FastAPI(
        title="Kova TTS",
        summary="Local text-to-speech: full synthesis, server-sent events, and a streaming "
        "WebSocket session.",
        version=__version__,
        lifespan=lifespan,
    )
    # Set before the lifespan runs so `create_app(tts=...)` needs no other wiring.
    app.state.tts = tts
    app.state.engine = None

    errors.install(app)
    app.include_router(routes.router)
    app.include_router(ws.router)
    return app


def _load(
    *,
    model: str | os.PathLike[str] | None,
    codec: str | os.PathLike[str] | None,
    wavlm: str | os.PathLike[str] | None,
    device: str | None,
    lora_dir: str | os.PathLike[str] | None,
    clone_preroll: int | None,
) -> Any:
    """Load the model. Imported here so ``import kova_tts.server`` costs no torch."""
    from kova_tts.engine.tts import KovaTTS

    log.info("Loading the model; this takes a few seconds the first time.")
    return KovaTTS.from_pretrained(
        model,
        codec=codec,
        wavlm=wavlm,
        device=device,
        lora_root=lora_dir,
        clone_preroll=clone_preroll,
    )


def _warm(tts: Any) -> None:
    """Run one throwaway generation so the first real request does not pay for the setup."""
    try:
        tts.generate(WARMUP_TEXT)
    except Exception:  # noqa: BLE001 - a warmup that fails must not stop the server booting
        log.warning("Warmup generation failed; the first request will be slower.", exc_info=True)


# ---------------------------------------------------------------------------------------- CLI


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach this command's flags to `parser`, so the top-level CLI can host it too."""
    parser.add_argument("--host", default=DEFAULT_HOST, help="interface to bind")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="port to bind")

    group = parser.add_argument_group("model")
    group.add_argument("--model", default=None, help="LM directory or Hub repo id")
    group.add_argument("--codec", default=None, help="codec checkpoint")
    group.add_argument("--wavlm", default=None, help="WavLM-large directory or repo id")
    group.add_argument("--device", default=None, help="torch device, e.g. cuda or cuda:1")
    group.add_argument("--lora-dir", default=None, help="directory of LoRA voices")
    group.add_argument(
        "--clone-preroll",
        type=int,
        default=DEFAULT_CLONE_PREROLL,
        help=f"reference codes decoded to warm a cloned generation (default: "
        f"{DEFAULT_CLONE_PREROLL}, one second)",
    )

    group = parser.add_argument_group("serving")
    group.add_argument(
        "--busy-timeout",
        type=float,
        default=DEFAULT_BUSY_TIMEOUT,
        help="seconds a second caller waits for the model before a 409 (0 refuses at once)",
    )
    group.add_argument(
        "--no-warmup",
        action="store_true",
        help="skip the startup generation; the first request pays for it instead",
    )
    group.add_argument(
        "--log-level",
        default="info",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kova-tts serve",
        description="Serve Kova TTS over HTTP and WebSocket on this machine.",
    )
    return add_arguments(parser)


def run(args: argparse.Namespace) -> int:
    """Execute a parsed command. Split out so the top-level CLI can call it directly."""
    import uvicorn

    app = create_app(
        model=args.model,
        codec=args.codec,
        wavlm=args.wavlm,
        device=args.device,
        lora_dir=args.lora_dir,
        clone_preroll=args.clone_preroll,
        busy_timeout=args.busy_timeout,
        warmup=not args.no_warmup,
    )
    print(f"kova-tts {__version__} serving on http://{args.host}:{args.port}", file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def main(argv: list[str] | None = None) -> int:
    """``python -m kova_tts.server`` and ``kova-tts serve`` both end up here."""
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
