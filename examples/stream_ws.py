"""Drive an incremental WebSocket session and write the result to a wav file.

    python examples/stream_ws.py "Hello. This is the second sentence." --out speech.wav

Needs only ``websockets``, which the ``server`` extra already installs.

The session exists for callers that do not have the whole text yet -- an LLM writing a reply a
token at a time, say. Text is *buffered* by ``send_text`` and only spoken when you ask for it,
so you choose where the sentence boundaries are, and therefore where the latency goes. This
script sends one sentence per ``send_text`` and flushes after each, which is the pattern worth
copying: flush per sentence, never per word.

Frames, in the order they occur. Client to server::

    {"start_context": {"voice": null, "seed": null, "sampling": null,
                       "response_format": {"encoding": "pcm", "sample_rate": 32000}}}
    {"send_text": "some text "}          repeat as often as you like
    {"flush": true, "flush_id": "s0"}    speak everything buffered, stay open
    {"close_context": true, "flush_id": "end"}   speak the rest, then end

Server to client::

    {"context_started": {...the accepted configuration...}}
    {"audio_chunk": "<base64 16-bit little-endian PCM>"}   repeated
    {"flush_completed": true, "flush_id": "s0"}            terminates every flush
    {"context_closed": true}                               the last frame of the session
    {"error": "...", "flush_id": "s0"}                     flush_id only when one flush failed

``flush_completed`` always arrives, even when a flush produced no audio or failed, so a client
that waits for it after every flush can never hang. Frames for two flushes never interleave.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
import wave

from websockets.sync.client import connect

#: Where this script chooses to flush. The server does its own sentence splitting inside one
#: generation; this split is about *when audio starts*, not about how the text is read.
SENTENCE = re.compile(r"(?<=[.!?])\s+")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("text", help="what to say")
    parser.add_argument("--server", default="ws://127.0.0.1:8000", help="base WebSocket URL")
    parser.add_argument("--voice", default=None, help="a voice name from GET /v1/voices")
    parser.add_argument("--seed", type=int, default=None, help="fix the sampler")
    parser.add_argument("--out", default="stream_ws_output.wav", help="where to write the wav")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> bytes:
    """Speak `args.text` a sentence at a time and return the concatenated PCM."""
    sentences = [part for part in SENTENCE.split(args.text.strip()) if part] or [args.text]
    audio = bytearray()
    started = time.perf_counter()
    first: float | None = None

    with connect(f"{args.server.rstrip('/')}/v1/ws") as ws:
        ws.send(json.dumps({"start_context": {"voice": args.voice, "seed": args.seed}}))
        opening = json.loads(ws.recv())
        if "error" in opening:
            raise SystemExit(f"error: {opening['error']}")
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
                    return bytes(audio)
    return bytes(audio)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        audio = run(args)
    except OSError as exc:
        print(f"cannot reach {args.server}: {exc}", file=sys.stderr)
        return 1

    if not audio:
        print("the server produced no audio", file=sys.stderr)
        return 1

    with wave.open(args.out, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)  # the session streams 16-bit
        out.setframerate(32000)  # response_format.sample_rate, which this server fixes
        out.writeframes(audio)
    print(f"wrote {args.out}: {len(audio) // 2 / 32000:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
