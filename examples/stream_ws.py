"""Drive an incremental WebSocket session and write the result to a file.

    python examples/stream_ws.py "Hello. This is the second sentence." --out speech.wav
    python examples/stream_ws.py "Sixteen kilohertz, for a voice agent." --rate 16000
    python examples/stream_ws.py "A container, streamed." --format flac --out speech.flac
    python examples/stream_ws.py "Now in a cloned voice." \\
        --clone-audio reference.wav --clone-text "exactly what the clip says"

Needs only ``websockets``, which the ``server`` extra already installs.

The session exists for callers that do not have the whole text yet -- an LLM writing a reply a
token at a time, say. ``send_text`` hands text over as it appears and the session speaks as far
into it as the text allows; a flush is how you say *that is the end of the turn*, at which point
the model runs to its own natural end.

This script has its text up front, so it treats each sentence as a turn. A client that does not
should call ``send_text`` as often as it likes -- that costs nothing -- and flush **once**, when
the reply is finished. Flushing more often does not make the audio start sooner, because the
session is already speaking; it only asks for more endings than the reply has.

Frames, in the order they occur. Client to server::

    {"start_context": {"voice": null, "reference": null, "seed": null, "sampling": null,
                       "response_format": {"encoding": "pcm", "sample_rate": 32000}}}
    {"send_text": "some text "}          repeat as often as you like
    {"flush": true, "flush_id": "s0"}    end the turn: finish everything sent so far
    {"close_context": true, "flush_id": "end"}   finish the rest, then end the session

Server to client::

    {"context_started": {...the accepted configuration...}}
    {"audio_chunk": "<base64 of the next bytes of the stream>"}   from as soon as there is
                                                                 enough text to speak
    {"flush_completed": true, "flush_id": "s0"}            terminates every flush
    {"context_closed": true}                               the last frame of the session
    {"error": "...", "flush_id": "s0"}                     flush_id only when one flush failed

``flush_completed`` always arrives, even when a flush produced no audio or failed, so a client
that waits for it after every flush can never hang. Frames for two turns never interleave.

``reference`` clones a voice from a recording: ``--clone-audio`` sends the file itself, base64
with its transcript, and ``--clone-codes`` sends a clip you have already encoded, which is what a
client speaking as the same voice all day should do. The transcript is required either way and
has to match the clip word for word.

``response_format`` picks the container and the rate. Every ``audio_chunk`` of a session is a
slice of **one** stream, so concatenating them in order gives one file: a ``wav`` session sends
its header on the first chunk and its trailer on the last, and a chunk on its own is not a file.
``context_started`` echoes the format that is really in effect, which is what this script reads
to know what to write.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
import wave
from pathlib import Path

from websockets.sync.client import connect

#: Where this script chooses to flush, which is where it says a turn has ended. The server does
#: its own splitting inside a generation; this one is about where the speech is allowed to come
#: to a stop, not about how the text is read.
SENTENCE = re.compile(r"(?<=[.!?])\s+")

#: Containers the session can send. ``pcm`` is headerless samples, which this script wraps in a
#: wav on the way to disk; the rest are already files and are written through unchanged.
FORMATS = ("pcm", "wav", "mp3", "flac")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("text", help="what to say")
    parser.add_argument("--server", default="ws://127.0.0.1:8000", help="base WebSocket URL")
    parser.add_argument("--voice", default=None, help="a voice name from GET /v1/voices")
    parser.add_argument(
        "--clone-audio",
        default=None,
        help="a recording to speak as, sent with the session; needs --clone-text",
    )
    parser.add_argument(
        "--clone-codes",
        default=None,
        help="a JSON file holding that recording's codes, for a voice you reuse; needs "
        "--clone-text",
    )
    parser.add_argument(
        "--clone-text",
        default=None,
        help="what the reference recording says, word for word. Required when cloning",
    )
    parser.add_argument("--seed", type=int, default=None, help="fix the sampler")
    parser.add_argument(
        "--format",
        default="pcm",
        choices=FORMATS,
        help="container for the audio chunks (default: pcm)",
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=None,
        help="output sample rate; the model's own 32000 by default. 16000 for a voice agent",
    )
    parser.add_argument("--out", default=None, help="where to write the audio")
    return parser.parse_args(argv)


def reference(args: argparse.Namespace) -> dict | None:
    """The reference clip to clone from, in whichever form was asked for.

    The transcript is not optional: the model reads it as the words already spoken in the clip
    and carries on from them, so a wrong one leaves it saying words the recording does not
    contain. There is no speech recognition in the server to fall back on.
    """
    if args.clone_audio and args.clone_codes:
        raise SystemExit("pass --clone-audio or --clone-codes, not both: they are one clip.")
    if not args.clone_audio and not args.clone_codes:
        if args.clone_text:
            raise SystemExit("--clone-text describes a clip; pass --clone-audio or --clone-codes.")
        return None
    if not args.clone_text:
        raise SystemExit("cloning needs --clone-text: what the reference recording says.")
    if args.clone_audio:
        clip = base64.b64encode(Path(args.clone_audio).read_bytes()).decode("ascii")
        return {"transcript": args.clone_text, "audio": clip}
    return {"transcript": args.clone_text, "codes": json.loads(Path(args.clone_codes).read_text())}


def run(args: argparse.Namespace) -> tuple[bytes, dict]:
    """Speak `args.text` a sentence at a time; return the bytes and the format they are in."""
    sentences = [part for part in SENTENCE.split(args.text.strip()) if part] or [args.text]
    clip = reference(args)
    response_format = {"encoding": args.format}
    if args.rate is not None:
        response_format["sample_rate"] = args.rate

    audio = bytearray()
    started = time.perf_counter()
    first: float | None = None

    # The clip goes over the socket, so the frame is as large as the recording is; the default
    # limit is a megabyte, which a few seconds of wav is well inside but not indefinitely.
    with connect(f"{args.server.rstrip('/')}/v1/ws", max_size=None) as ws:
        ws.send(
            json.dumps(
                {
                    "start_context": {
                        "voice": args.voice,
                        "reference": clip,
                        "seed": args.seed,
                        "response_format": response_format,
                    }
                }
            )
        )
        opening = json.loads(ws.recv())
        if "error" in opening:
            raise SystemExit(f"error: {opening['error']}")
        effective = opening["context_started"]["response_format"]
        print(f"context started: {opening['context_started']}")

        for index, sentence in enumerate(sentences):
            ws.send(json.dumps({"send_text": sentence}))
            # The last sentence closes the session in the same round trip, so the server never
            # waits on a client that has nothing left to say.
            last = index == len(sentences) - 1
            key = "close_context" if last else "flush"
            ws.send(json.dumps({key: True, "flush_id": f"s{index}"}))

            while True:
                frame = json.loads(ws.recv())
                if "audio_chunk" in frame:
                    if first is None:
                        first = time.perf_counter() - started
                        print(f"first audio after {first * 1000:.0f} ms")
                    audio += base64.b64decode(frame["audio_chunk"])
                elif "error" in frame:
                    print(f"error: {frame['error']}", file=sys.stderr)
                elif "flush_completed" in frame:
                    print(f"flush {frame['flush_id']} done, {len(audio)} bytes so far")
                    if not last:
                        break
                elif "context_closed" in frame:
                    return bytes(audio), effective
    return bytes(audio), effective


def write(audio: bytes, response_format: dict, path: str) -> str:
    """Save the session's bytes, wrapping raw pcm in a wav so a player can open it."""
    if response_format["encoding"] != "pcm":
        # Already a file: the session opened the container on its first chunk and closed it on
        # its last, so these bytes are the whole thing and nothing needs adding.
        with open(path, "wb") as out:
            out.write(audio)
        return f"{len(audio)} bytes"
    rate = response_format["sample_rate"]
    with wave.open(path, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)  # the session sends 16-bit samples
        out.setframerate(rate)
        out.writeframes(audio)
    return f"{len(audio) // 2 / rate:.2f} s at {rate} Hz"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        audio, response_format = run(args)
    except OSError as exc:
        print(f"cannot reach {args.server}: {exc}", file=sys.stderr)
        return 1

    if not audio:
        print("the server produced no audio", file=sys.stderr)
        return 1

    suffix = "wav" if args.format == "pcm" else args.format
    path = args.out or f"stream_ws_output.{suffix}"
    print(f"wrote {path}: {write(audio, response_format, path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
