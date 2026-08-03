"""Stream a sentence over Server-Sent Events and write it to a wav file.

    python examples/stream_sse.py "Hello from the streaming endpoint." --out speech.wav

Nothing but the standard library, so this doubles as the protocol specification. The whole
exchange is one HTTP request:

    POST /v1/tts/stream
    {"text": "...", "voice": null, "seed": null, "sampling": {"temperature": 0.9}}

and a ``text/event-stream`` response of blocks separated by a blank line::

    event: chunk
    data: {"index": 0, "audio": "<base64>", "sample_rate": 32000}

    event: done
    data: {"chunks": 7, "samples": 86800, "duration_seconds": 2.712, "sample_rate": 32000}

``audio`` is base64 of raw 16-bit little-endian mono PCM at ``sample_rate``. Concatenating
every chunk's bytes gives the whole utterance -- which is exactly what this script does, and
why the wav header can only be written at the end. A stream always finishes with one terminal
event: ``done``, or ``error`` carrying ``{"error": ..., "message": ...}``.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request
import wave


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("text", help="what to say")
    parser.add_argument("--server", default="http://127.0.0.1:8000", help="base URL")
    parser.add_argument("--voice", default=None, help="a voice name from GET /v1/voices")
    parser.add_argument("--seed", type=int, default=None, help="fix the sampler")
    parser.add_argument("--out", default="stream_sse_output.wav", help="where to write the wav")
    return parser.parse_args(argv)


def stream(server: str, body: dict):
    """Yield ``(event name, data)`` pairs from the SSE response, as they arrive.

    Server-Sent Events are simpler than they look: UTF-8 text, one ``field: value`` per line,
    and a blank line ends an event. Only ``event`` and ``data`` are used here.
    """
    request = urllib.request.Request(
        f"{server.rstrip('/')}/v1/tts/stream",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        name = ""
        for raw in response:
            line = raw.decode("utf-8").rstrip("\n")
            if not line:  # blank line: the event is complete
                name = ""
            elif line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                yield name, json.loads(line[len("data: ") :])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    body = {"text": args.text, "voice": args.voice, "seed": args.seed}

    audio = bytearray()
    sample_rate = 32000
    started = time.perf_counter()
    first: float | None = None

    try:
        for name, data in stream(args.server, body):
            if name == "chunk":
                if first is None:
                    first = time.perf_counter() - started
                    print(f"first audio after {first * 1000:.0f} ms")
                audio += base64.b64decode(data["audio"])
                sample_rate = data["sample_rate"]
            elif name == "done":
                print(f"{data['chunks']} chunks, {data['duration_seconds']:.2f} s of audio")
            elif name == "error":
                print(f"error: {data['message']}", file=sys.stderr)
                return 1
    except urllib.error.HTTPError as exc:
        # Anything the server refused outright -- a bad request, an unknown voice, or a 409
        # because it is already generating -- arrives as the JSON error envelope.
        print(f"error: {exc.read().decode('utf-8', 'replace')}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"cannot reach {args.server}: {exc.reason}", file=sys.stderr)
        return 1

    if not audio:
        print("the server produced no audio", file=sys.stderr)
        return 1

    with wave.open(args.out, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)  # the stream is 16-bit
        out.setframerate(sample_rate)
        out.writeframes(bytes(audio))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
