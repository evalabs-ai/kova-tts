#!/usr/bin/env python3
"""Call the OpenAI-compatible endpoint with nothing but the standard library.

    python examples/openai_client.py "Hello from a local server." --out speech.mp3

The point of the endpoint is that clients you did not write already speak it, so this example
is deliberately not written with the ``openai`` SDK: it shows the request those clients send,
byte for byte, and how the response comes back. With the SDK it is three lines::

    from openai import OpenAI

    client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")
    client.audio.speech.create(model="tts-1", voice="default", input="...").stream_to_file(out)

Audio is read as it arrives rather than in one lump, which is why the time to first byte printed
below is a fraction of the time the whole utterance takes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8000"


def synthesize(url: str, body: dict, out: str) -> None:
    """POST the request and write the audio, reporting when the first byte arrived."""
    request = urllib.request.Request(  # noqa: S310 - an http(s) URL the caller chose
        f"{url.rstrip('/')}/v1/audio/speech",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/octet-stream"},
    )

    started = time.perf_counter()
    first: float | None = None
    written = 0
    with urllib.request.urlopen(request) as response, open(out, "wb") as sink:  # noqa: S310
        media_type = response.headers.get("Content-Type", "?")
        while True:
            # read1, not read: read() blocks until it has the whole 8 KB, which for a
            # compressed container is a second of audio and would hide the head start
            # entirely. read1 hands over whatever has arrived.
            chunk = response.read1(8192)
            if not chunk:
                break
            if first is None:
                first = time.perf_counter() - started
                print(f"first byte after {first * 1000:.0f} ms")
            sink.write(chunk)
            written += len(chunk)

    print(f"wrote {out}: {written} bytes, {media_type}")
    print(f"whole utterance in {time.perf_counter() - started:.2f} s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("text", help="what to say")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"server (default: {DEFAULT_URL})")
    parser.add_argument("--out", default="speech.mp3", help="file to write")
    parser.add_argument(
        "--voice",
        default="alloy",
        help="a name from GET /v1/voices. OpenAI's stock names are accepted too and speak in "
        "the base voice unless the server maps them (default: alloy, as a stock client sends)",
    )
    parser.add_argument(
        "--format",
        default="mp3",
        choices=("mp3", "opus", "flac", "wav", "pcm"),
        help="response_format (default: mp3, as OpenAI's own default)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="extension: output rate, e.g. 16000 for a voice agent. Default is the model's",
    )
    args = parser.parse_args(argv)

    # Exactly the body openai-python sends for audio.speech.create(...), plus this server's
    # optional sample_rate. Fields the SDK was not given are simply absent.
    body = {
        "model": "tts-1",
        "input": args.text,
        "voice": args.voice,
        "response_format": args.format,
    }
    if args.sample_rate is not None:
        body["sample_rate"] = args.sample_rate

    print(f"POST /v1/audio/speech  response_format={args.format}")
    try:
        synthesize(args.url, body, args.out)
    except urllib.error.HTTPError as failure:
        # Every failure is the server's usual envelope: {"error": ..., "message": ...}, and the
        # message is expected to say what to change.
        detail = json.loads(failure.read().decode() or "{}")
        print(f"{failure.code}: {detail.get('message', failure.reason)}", file=sys.stderr)
        return 1
    except urllib.error.URLError as failure:
        print(f"could not reach {args.url}: {failure.reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
