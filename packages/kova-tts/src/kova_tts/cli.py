"""Command line entry point: ``kova-tts <command>``.

One rule shapes this file: **nothing heavy is imported at module scope.** ``kova-tts paths`` is
the first thing a user runs when something is misconfigured, so it has to answer without dragging
in transformers, peft, gradio or faster-whisper. Every subcommand imports what it needs inside
its own handler, and even ``prepare-data``'s flags -- which are grafted on from
:mod:`kova_tts.data.prepare` rather than restated -- are only grafted when that command is the
one being run.

Torch does not arrive either, which is only true because ``kova_codec.KovaCodec`` is lazy in
that package's ``__init__``: :mod:`kova_tts` reaches ``kova_codec`` for its constants, so an
eager import there would tax every entry point in this project. ``TestLaziness`` in the CLI
tests asserts it.

Commands that already have a working entry point elsewhere are grafted on rather than
reimplemented. ``prepare-data`` borrows its flags from
:func:`kova_tts.data.prepare.add_arguments`; ``finetune``, ``merge``, ``serve`` and ``demo``
forward their arguments untouched to the ``main(argv)`` they belong to, which is why those
subparsers do not redeclare a single flag -- the target parsers already set ``prog`` to the name
used here, so their ``--help`` and their errors read correctly from under this CLI.

Not all of those targets are always there. The server lives behind the ``server`` extra, the
demo behind ``demo`` and in ``apps/`` rather than in the installed package, and either may be
absent on a given machine. Both are reached through :func:`_entry_point` / :func:`_load_demo`,
which turn a missing module or a missing optional dependency into one sentence naming what to
install, instead of an ImportError traceback.

:func:`asr_transcriber` is the other seam worth knowing about. :class:`~kova_tts.engine.tts.KovaTTS`
deliberately ships no ASR; it takes a ``callable(path) -> str``. This module builds that callable
out of :mod:`kova_tts.data.asr`, which is what lets ``--clone-audio`` transcribe a reference clip
by itself, and what ``examples/voice_clone.py`` uses to do the same from Python.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kova_tts import CLONE_SAMPLING, TTS_SAMPLING, __version__, paths

#: Files a Hub snapshot never needs: the same weights in frameworks this project does not use.
HUB_IGNORE = ("*.h5", "*.msgpack", "*.onnx", "*.tflite")

#: Sampling knobs ``generate`` can override. ``None`` means "leave the voice's preset alone".
_SAMPLING_FLAGS = ("temperature", "top_p", "top_k", "repetition_penalty", "max_tokens")


class CommandError(RuntimeError):
    """Something the user can fix, reported as ``error: <message>`` with exit code 2.

    Every raise site owes the reader the fix, not just the fault: which file is missing, which
    extra to install, which flag to drop.
    """


# --------------------------------------------------------------------------------- lazy loading


def _entry_point(
    module: str,
    attribute: str,
    *,
    label: str,
    extra: str | None = None,
) -> Any:
    """Import ``module.attribute`` on first use, or explain what is missing.

    The subcommands behind an optional extra are unimportable on an install that did not ask
    for them, and the fix is always the same sentence: install the extra.
    """
    try:
        loaded = importlib.import_module(module)
    except ImportError as exc:
        raise CommandError(_unavailable(label, exc, extra=extra)) from exc
    try:
        return getattr(loaded, attribute)
    except AttributeError as exc:
        raise CommandError(
            f"{label} is installed but has no {attribute}() to call ({module}.{attribute} is "
            f"missing). This is a version mismatch: reinstall the package."
        ) from exc


def _unavailable(label: str, exc: ImportError, *, extra: str | None) -> str:
    """The message for a subcommand whose implementation could not be imported."""
    install = (
        f" Install it with `uv sync --extra {extra}` (or `pip install 'kova-tts[{extra}]'`)."
        if extra
        else ""
    )
    return f"{label} is not available: {exc}.{install}"


def _call_entry(entry: Callable[[list[str]], Any], argv: list[str]) -> int:
    """Call a forwarding entry point's ``main(argv)`` and normalise its exit code."""
    result = entry(argv)
    return int(result) if isinstance(result, int) else 0


def _demo_app_path() -> Path:
    """Where ``apps/demo/app.py`` lives in a source checkout.

    The demo is an application in this repository, not part of the installed package, so it is
    loaded from the checkout by path. A wheel install has no ``apps/`` directory and is told so.
    """
    here = Path(__file__).resolve()
    root = here.parents[4] if len(here.parents) > 4 else here.parent
    return root / "apps" / "demo" / "app.py"


def _load_demo() -> Callable[..., Any]:
    """Load ``apps/demo/app.py`` and return its ``main``."""
    path = _demo_app_path()
    if not path.is_file():
        raise CommandError(
            f"The demo app was not found at {path}. It ships with the source checkout rather "
            f"than with the installed package: clone the repository and run `uv run kova-tts "
            f"demo` from it."
        )
    spec = importlib.util.spec_from_file_location("kova_tts_demo_app", path)
    if spec is None or spec.loader is None:  # pragma: no cover - only a corrupt .py gets here
        raise CommandError(f"{path} could not be loaded as a Python module.")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        raise CommandError(_unavailable("The demo app", exc, extra="demo")) from exc
    entry = getattr(module, "main", None)
    if entry is None:
        raise CommandError(f"{path} has no main() to call.")
    return entry


# ------------------------------------------------------------------------------------ the seams


def asr_transcriber(
    model: str | None = None,
    *,
    language: str | None = None,
    device: str | None = None,
    compute_type: str | None = None,
) -> Callable[[str], str]:
    """A ``callable(path) -> str`` for :class:`~kova_tts.engine.tts.KovaTTS`, backed by ASR.

    The engine takes a transcriber rather than shipping one, so this is where the ``data`` extra
    gets wired in from the outside. Also used by ``examples/voice_clone.py``, which is the whole
    reason it is public.

    The model is built on the first call, not here: loading faster-whisper costs half a gigabyte
    and several seconds, and a run that turns out to have a transcript should never pay for it.
    """
    options = {
        key: value
        for key, value in (
            ("model", model),
            ("language", language),
            ("device", device),
            ("compute_type", compute_type),
        )
        if value is not None
    }
    loaded: list[Any] = []

    def transcribe(path: str) -> str:
        from kova_tts.audio import load_audio
        from kova_tts.data.asr import MissingDependency, load_transcriber

        if not loaded:
            print(f"transcribing {path} ...", file=sys.stderr, flush=True)
            try:
                loaded.append(load_transcriber(**options))
            except MissingDependency as exc:
                raise CommandError(str(exc)) from exc
        return loaded[0].transcribe(load_audio(path))

    return transcribe


def download_weights(
    *,
    repo: str | None = None,
    wavlm: bool = False,
    cache_dir: str | Path | None = None,
    force: bool = False,
    token: str | None = None,
) -> list[tuple[str, Path | str, int]]:
    """Prefetch the LM, the codec and optionally WavLM into the local Hub cache.

    So that a later run works with no network: a container image, an air-gapped box, or simply a
    first synthesis that should not stall on a multi-gigabyte download. Anything already pointed
    at a local path by ``.env`` is reported and skipped -- there is nothing to fetch for it, and
    downloading a second copy would be a surprise measured in gigabytes.

    Returns ``(label, location, bytes)`` per artifact; ``bytes`` is 0 for an artifact that was
    already local.
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    repo_id = repo or paths.hub_repo()
    options: dict[str, Any] = {"cache_dir": cache_dir, "token": token, "force_download": force}
    fetched: list[tuple[str, Path | str, int]] = []

    def snapshot(label: str, source: str) -> None:
        if _is_local(source):
            fetched.append((label, source, 0))
            return
        directory = Path(
            _hub_call(
                lambda: snapshot_download(source, ignore_patterns=list(HUB_IGNORE), **options),
                what=f"the {label} repository {source!r}",
            )
        )
        fetched.append((label, directory, _size_of(directory)))

    snapshot("model", repo_id if repo else paths.model_path())

    # The codec is one file inside the model repository, so a snapshot has already brought it
    # down; asking for it by name is what pins the exact file a later run will open.
    configured_codec = None if repo else _configured(paths.ENV_CODEC)
    if configured_codec is not None:
        fetched.append(("codec", configured_codec, 0))
    else:
        file = Path(
            _hub_call(
                lambda: hf_hub_download(repo_id, paths.CODEC_HUB_FILENAME, **options),
                what=f"{paths.CODEC_HUB_FILENAME} from {repo_id!r}",
            )
        )
        fetched.append(("codec", file, _size_of(file)))

    if wavlm:
        snapshot("wavlm", paths.wavlm_path())
    return fetched


def _hub_call(call: Callable[[], str], *, what: str) -> str:
    """Run one Hub download, turning any failure into an actionable message.

    Every failure mode here -- repository not published, no network, gated repo, expired token --
    reaches the user as the same kind of problem: this machine cannot get these files right now,
    and the way out is either credentials, a different repository, or a local checkpoint.
    """
    try:
        return call()
    except Exception as exc:  # noqa: BLE001 - the fix is the same whatever the Hub raised
        raise CommandError(
            f"Could not download {what}: {type(exc).__name__}: {exc}\n"
            f"  - the released weights may not be public yet; set KOVA_HUB_REPO to the "
            f"repository you have access to,\n"
            f"  - or run `huggingface-cli login` if it is gated,\n"
            f"  - or point KOVA_MODEL_PATH / KOVA_CODEC_PATH at local checkpoints and skip the "
            f"download entirely (`kova-tts paths` shows what resolved)."
        ) from exc


def _configured(variable: str) -> str | None:
    """The value of a ``KOVA_*`` variable, ``.env`` included, or ``None`` when it is unset."""
    import os

    paths.load_dotenv()
    return os.environ.get(variable, "").strip() or None


def _is_local(value: str | Path) -> bool:
    """True when an artifact resolved to a path on this machine rather than a Hub repo id."""
    return Path(value).expanduser().exists()


def _size_of(path: Path) -> int:
    """Bytes on disk, following the symlinks a Hub snapshot is made of."""
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024


# ------------------------------------------------------------------------------------- commands


def _cmd_paths(_args: argparse.Namespace) -> int:
    """Show where each model artifact resolves to, so a bad .env is obvious."""
    dotenv = paths.load_dotenv()
    print(f"config    {dotenv if dotenv else 'no .env found'}")

    ok = True
    for label, resolve in (
        ("model", paths.model_path),
        ("wavlm", paths.wavlm_path),
        ("codec", paths.codec_path),
        ("loras", paths.lora_dir),
    ):
        try:
            value = resolve()
        except paths.MissingArtifact as exc:
            print(f"{label:<9} ERROR: {exc}")
            ok = False
            continue
        except Exception as exc:  # noqa: BLE001 - hub failures are reported, not raised
            print(f"{label:<9} unavailable: {exc}")
            ok = False
            continue
        print(f"{label:<9} {value if value is not None else 'not configured'}")

    voices = paths.available_loras()
    print(f"voices    {', '.join(voices) if voices else 'none'}")
    return 0 if ok else 1


def _cmd_generate(args: argparse.Namespace) -> int:
    """Synthesize text to a WAV file. The everyday command."""
    text = _resolve_text(args)
    reference = _existing(args.clone_audio, "Reference audio") if args.clone_audio else None
    if reference is not None and args.voice:
        raise CommandError("Pass --voice or --clone-audio, not both: they are two ways to say who.")
    if reference is None and args.clone_text:
        raise CommandError("--clone-text describes a reference clip; pass --clone-audio too.")
    # Built before the model is loaded, so a bad --temperature costs nothing.
    params = _sampling(args, cloning=reference is not None)

    tts_class = _entry_point("kova_tts.engine.tts", "KovaTTS", label="The TTS engine")
    tts = tts_class.from_pretrained(
        args.model,
        codec=args.codec,
        wavlm=args.wavlm,
        device=args.device,
        lora_root=args.lora_dir,
        transcriber=asr_transcriber(
            args.asr_model, language=args.asr_language, device=args.asr_device
        ),
    )

    voice: Any = args.voice
    if reference is not None:
        voice = tts.clone(str(reference), args.clone_text)
        if args.clone_text is None:
            # A wrong transcript is the usual cause of garbled cloning, so show what was heard.
            print(f'reference heard as: "{voice.ref_text}"', file=sys.stderr)

    started = time.perf_counter()
    if args.stream:
        wav, first_audio = _stream(tts, text, voice, params=params, seed=args.seed)
    else:
        wav, first_audio = tts.generate(text, voice, params=params, seed=args.seed), None
    elapsed = time.perf_counter() - started

    if wav.size == 0:
        raise CommandError(
            "The model produced no audio for that text. Check that it contains speakable "
            "characters, and try a different --seed."
        )
    written = tts.save(wav, args.out)
    seconds = wav.size / tts.sample_rate
    timing = f"{elapsed:.1f} s"
    if first_audio is not None:
        timing += f", first audio after {first_audio:.2f} s"
    print(f"{written}  {seconds:.2f} s of audio in {timing} ({seconds / max(elapsed, 1e-9):.1f}x)")
    return 0


def _cmd_prepare_data(args: argparse.Namespace) -> int:
    """Run the dataset pipeline, whose flags this subparser hosts verbatim."""
    return int(_entry_point("kova_tts.data.prepare", "run", label="Dataset preparation")(args))


def _cmd_finetune(args: argparse.Namespace) -> int:
    entry = _entry_point("kova_tts.finetune.train", "main", label="Finetuning", extra="finetune")
    return int(entry(args.forwarded))


def _cmd_merge(args: argparse.Namespace) -> int:
    entry = _entry_point("kova_tts.finetune.merge", "main", label="Merging", extra="finetune")
    return int(entry(args.forwarded))


def _cmd_serve(args: argparse.Namespace) -> int:
    entry = _entry_point("kova_tts.server.app", "main", label="The server", extra="server")
    return _call_entry(entry, args.forwarded)


def _cmd_demo(args: argparse.Namespace) -> int:
    return _call_entry(_load_demo(), args.forwarded)


def _cmd_download(args: argparse.Namespace) -> int:
    return run_download(args)


# ---------------------------------------------------------------------------- generate helpers


def _resolve_text(args: argparse.Namespace) -> str:
    """The text to speak, from the positional argument, a file, or stdin (``--text-file -``)."""
    if args.text is not None and args.text_file is not None:
        raise CommandError("Give the text as an argument or with --text-file, not both.")
    if args.text_file is None:
        text = args.text or ""
    elif str(args.text_file) == "-":
        text = sys.stdin.read()
    else:
        text = _existing(args.text_file, "Text file").read_text(encoding="utf-8")
    if not text.strip():
        raise CommandError(
            "Nothing to speak. Pass the text as an argument, or point --text-file at a file "
            "(or at - to read stdin)."
        )
    return text.strip()


def _sampling(args: argparse.Namespace, *, cloning: bool) -> Any:
    """The caller's sampling overrides on top of the preset this voice calls for.

    ``None`` when nothing was overridden, which lets the engine choose the preset itself -- the
    two presets were tuned separately and picking the wrong one is audible.
    """
    overrides = {
        name: getattr(args, name)
        for name in _SAMPLING_FLAGS
        if getattr(args, name, None) is not None
    }
    if not overrides:
        return None
    return (CLONE_SAMPLING if cloning else TTS_SAMPLING).replace(**overrides)


def _stream(tts: Any, text: str, voice: Any, *, params: Any, seed: int | None):
    """Collect a streamed generation, timing the first frame that carries audio.

    Streaming to a file is not faster overall; it is here because time-to-first-audio is the
    number a streaming deployment lives on, and this is the cheapest way to measure it.
    """
    import numpy as np

    started = time.perf_counter()
    first: float | None = None
    chunks: list[Any] = []
    for frame in tts.stream(text, voice, params=params, seed=seed):
        if frame.samples.size and first is None:
            first = time.perf_counter() - started
        chunks.append(frame.samples)
    return (
        np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32),
        first if first is not None else time.perf_counter() - started,
    )


def _existing(value: str | Path, what: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_file():
        raise CommandError(f"{what} not found: {path}")
    return path


# ------------------------------------------------------------------------------------- download


def add_download_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach the download flags, so ``scripts/download_weights.py`` can host them too."""
    parser.add_argument(
        "--repo",
        default=None,
        help=f"Hub repo holding the LM and codec (default: {paths.DEFAULT_HUB_REPO}, "
        f"or KOVA_HUB_REPO)",
    )
    parser.add_argument(
        "--wavlm",
        action="store_true",
        help="also fetch WavLM, which is only needed to encode audio (cloning, dataset prep)",
    )
    parser.add_argument("--cache-dir", default=None, help="Hub cache directory to fill")
    parser.add_argument("--token", default=None, help="Hub token, for a gated repository")
    parser.add_argument(
        "--force", action="store_true", help="re-download even when the cache already has it"
    )
    return parser


def run_download(args: argparse.Namespace) -> int:
    """Execute a parsed download, reporting what landed where."""
    fetched = download_weights(
        repo=args.repo,
        wavlm=args.wavlm,
        cache_dir=args.cache_dir,
        force=args.force,
        token=args.token,
    )
    for label, location, size in fetched:
        detail = f"{_human(size)}  {location}" if size else f"already local  {location}"
        print(f"{label:<9} {detail}")
    total = sum(size for _, _, size in fetched)
    print(f"{'total':<9} {_human(total)} downloaded")
    return 0


# --------------------------------------------------------------------------------------- parser


def build_parser(argv: list[str] | None = None) -> argparse.ArgumentParser:
    """The whole CLI.

    `argv` is peeked at, never parsed here: ``prepare-data``'s flags are grafted on from
    :mod:`kova_tts.data.prepare`, and importing that module is work no other command should pay
    for. Called with no arguments -- from tests, or from documentation tooling -- every
    subcommand is built.
    """
    parser = argparse.ArgumentParser(
        prog="kova-tts", description="Kova TTS: synthesize, clone, prepare data, finetune, serve."
    )
    parser.add_argument("--version", action="version", version=f"kova-tts {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_paths = sub.add_parser("paths", help="show resolved model artifact locations")
    p_paths.set_defaults(func=_cmd_paths)

    _add_generate(sub)

    p_prepare = sub.add_parser(
        "prepare-data",
        help="turn a folder of recordings into a JSONL finetuning corpus",
        description="Turn a folder of recordings into a JSONL corpus for LoRA finetuning.",
    )
    p_prepare.set_defaults(func=_cmd_prepare_data)
    if argv is None or "prepare-data" in argv:
        from kova_tts.data.prepare import add_arguments

        add_arguments(p_prepare)

    _add_forwarding(
        sub,
        "finetune",
        help="train a per-voice LoRA adapter (--config CONFIG.yaml)",
        func=_cmd_finetune,
    )
    _add_forwarding(
        sub,
        "merge",
        help="merge a LoRA adapter into its base model (ADAPTER OUTPUT)",
        func=_cmd_merge,
    )

    _add_forwarding(sub, "serve", help="run the HTTP + streaming server", func=_cmd_serve)
    _add_forwarding(sub, "demo", help="run the browser demo", func=_cmd_demo)

    p_download = sub.add_parser(
        "download",
        help="prefetch weights from the Hub for offline use",
        description="Prefetch the LM, codec and optionally WavLM into the local Hub cache.",
    )
    add_download_arguments(p_download)
    p_download.set_defaults(func=_cmd_download)

    return parser


def _add_generate(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "generate",
        help="synthesize text to a WAV file",
        description="Synthesize speech. Give the text as an argument, or with --text-file.",
    )
    parser.add_argument("text", nargs="?", help="the text to speak")
    parser.add_argument(
        "--text-file", default=None, help="read the text from a file, or from - for stdin"
    )
    parser.add_argument("-o", "--out", default="out.wav", help="output WAV (default: out.wav)")
    parser.add_argument(
        "--voice", default=None, help="LoRA voice name; `kova-tts paths` lists what is installed"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="decode as the model generates, and report time-to-first-audio",
    )
    parser.add_argument("--seed", type=int, default=None, help="make the sampling reproducible")

    group = parser.add_argument_group("cloning")
    group.add_argument("--clone-audio", default=None, help="reference recording to clone")
    group.add_argument(
        "--clone-text",
        default=None,
        help="what the reference says, word for word; transcribed with ASR when omitted, "
        "which needs the data extra",
    )

    group = parser.add_argument_group("sampling", "unset flags keep the preset the voice implies")
    group.add_argument("--temperature", type=float, default=None)
    group.add_argument("--top-p", type=float, default=None)
    group.add_argument("--top-k", type=int, default=None)
    group.add_argument("--repetition-penalty", type=float, default=None)
    group.add_argument("--max-tokens", type=int, default=None, help="generation budget in codes")

    group = parser.add_argument_group("artifacts", "each defaults to the configured KOVA_* path")
    group.add_argument("--model", default=None, help="LM directory or Hub repo id")
    group.add_argument("--codec", default=None, help="codec checkpoint")
    group.add_argument("--wavlm", default=None, help="WavLM directory or repo id (cloning only)")
    group.add_argument("--lora-dir", default=None, help="directory of LoRA voices")
    group.add_argument("--device", default=None, help="torch device, e.g. cuda:1")

    group = parser.add_argument_group(
        "transcription", "used only when cloning without --clone-text"
    )
    group.add_argument("--asr-model", default=None, help="faster-whisper model (default: small)")
    group.add_argument("--asr-language", default=None, help="ISO code; detected per clip if unset")
    group.add_argument("--asr-device", default=None, choices=("cuda", "cpu"))

    parser.set_defaults(func=_cmd_generate)


def _add_forwarding(
    sub: argparse._SubParsersAction, name: str, *, help: str, func: Callable[..., int]
) -> None:
    """Add a subcommand that parses nothing and forwards everything.

    ``add_help=False`` is the point: ``kova-tts merge --help`` should print the real parser's
    help, not a paraphrase of it that drifts. Those parsers already name themselves
    ``kova-tts <name>``, so their usage lines come out right.

    The cost is that ``--help`` for one of these reports a missing extra instead of printing
    help, which is the right answer anyway: options you cannot run are not the problem to fix.
    """
    parser = sub.add_parser(name, add_help=False, help=help)
    parser.set_defaults(func=func, passthrough=True)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser(argv)
    args, forwarded = parser.parse_known_args(argv)
    if forwarded and not getattr(args, "passthrough", False):
        parser.error(f"unrecognized arguments: {' '.join(forwarded)}")
    args.forwarded = forwarded

    try:
        return args.func(args)
    except CommandError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except paths.MissingArtifact as exc:
        print(f"error: {exc}\nRun `kova-tts paths` to see what resolved.", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        # OSError covers the whole family of "this machine cannot get at that": a missing file, a
        # permission, and the error transformers raises for a checkpoint it cannot find. None of
        # them is a bug in this program, so none of them earns a traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover - depends on a signal arriving
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
