"""What flushing a WebSocket session more often actually costs.

    python scripts/ws_latency.py --server ws://127.0.0.1:8000

A session speaks as text arrives, generating as far into it as the text allows. A flush does not
start the speech: it says **the turn is over**, and the model runs to its own natural end. So
there is only one place a client has to flush, and this script measures what happens to the
audio when it flushes anywhere else.

The recommended usage is the first row of the table. The other three are the same paragraph
delivered by a client that declares the turn over 3, 11 and 32 times, and they are here because
the cost is worth seeing rather than asserting:

``end``
    One flush, when the text is finished. **This is the usage to copy.**
``sentence``
    A flush at every sentence boundary. Sentence ends are real turn ends, so this costs little.
``timer``
    A flush every ``--flush-every`` milliseconds, boundary or not.
``word``
    A flush after every word. The extreme: thirty-two turns for one paragraph.

Reported per strategy:

* **turns** -- how many times the client said the turn was over. One flush is one turn.
* **first audio** -- milliseconds from the first word being sent to the first audio byte. Low is
  not automatically good here: read it next to the speech column.
* **speech** -- seconds of audio produced, and the ratio against the one-turn run. Each turn
  gets its own ending, so a paragraph cut into thirty-two turns is thirty-two endings and the
  same words take several times as long to say.
* **delivery** -- how fast audio arrives once it has started, against the speed it is played at.
  Above 1.0 the session produces faster than a listener consumes.
* **buffer floor** -- the least audio a player would have had in hand, in seconds, from the
  moment it started playing. Negative means it ran dry and the speech stuttered.
* **worst join** -- the largest sample-to-sample step at a flush boundary, against the 99.9th
  percentile of every step in the waveform. A boundary that restarted the voice would stand out
  from the signal's own distribution; one that carried it through does not.

On an RTX 3090 at 16 kHz out, delivery runs 1.4-1.8x realtime and the buffer floor goes slightly
negative -- -0.06 s on ``end``, -0.01 s on ``word`` -- so a player that starts on the very first
chunk can briefly run dry. On an RTX 5090 the floor stayed at or above 0.33 s on every strategy.
Buffer a few hundred milliseconds before starting playback if you are near the 3090's figure.

Audio is asked for as ``pcm`` so a byte count is a sample count and arrival times mean what they
say. Nothing is written to disk.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import statistics
import sys
import time

import numpy as np
from websockets.sync.client import connect

#: What a fast LLM emits, and therefore what a client can forward. Speech runs at about half
#: this, which is the headroom the session has to work in.
DEFAULT_WORDS_PER_SECOND = 5.0

#: A paragraph, not a sentence: several boundaries, and long enough that a strategy which waits
#: for the end is visibly waiting.
DEFAULT_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "It landed softly on the far bank and looked around for a while. "
    "Then it went on its way, following the river north."
)

#: The one to copy, and the baseline the speech column is measured against.
RECOMMENDED = "end"

#: What each strategy tells the server, in the words the table prints.
STRATEGIES = {
    "end": "the end of the turn",
    "sentence": "a sentence",
    "timer": "a timer",
    "word": "every word",
}

#: 16-bit samples, so two bytes is one sample and one byte count is one duration.
BYTES_PER_SAMPLE = 2

_SENTENCE_END = re.compile(r"[.!?][\"')\]]*$")


class Session:
    """One run of one strategy: feeds words on a clock and records what comes back."""

    def __init__(self, ws, rate: int) -> None:
        self.ws = ws
        self.rate = rate
        #: (arrival time, bytes) for every audio chunk, in order.
        self.chunks: list[tuple[float, int]] = []
        #: Byte offset at which each flush was acknowledged.
        self.boundaries: list[int] = []
        self.audio = bytearray()
        self.started = 0.0
        self.closed = 0.0
        self.flushes = 0

    def poll(self, until: float) -> None:
        """Read frames until `until`, so receiving never delays the next word."""
        while True:
            remaining = until - time.perf_counter()
            if remaining <= 0:
                return
            try:
                frame = json.loads(self.ws.recv(timeout=remaining))
            except TimeoutError:
                return
            self.handle(frame)

    def drain(self) -> None:
        """Read until the session closes."""
        while not self.closed:
            self.handle(json.loads(self.ws.recv()))

    def handle(self, frame: dict) -> None:
        if "audio_chunk" in frame:
            payload = base64.b64decode(frame["audio_chunk"])
            self.chunks.append((time.perf_counter() - self.started, len(payload)))
            self.audio += payload
        elif "flush_completed" in frame:
            self.boundaries.append(len(self.audio))
        elif "context_closed" in frame:
            self.closed = time.perf_counter() - self.started
        elif "error" in frame:
            raise SystemExit(f"session failed: {frame['error']}")

    def send(self, frame: dict) -> None:
        self.ws.send(json.dumps(frame))

    def flush(self, *, close: bool = False) -> None:
        self.send({"close_context" if close else "flush": True, "flush_id": str(self.flushes)})
        self.flushes += 1


def should_flush(strategy: str, word: str, since_flush: float, every: float) -> bool:
    """Whether this client declares the turn over here, having just sent `word`."""
    if strategy == "word":
        return True
    if strategy == "sentence":
        return bool(_SENTENCE_END.search(word))
    if strategy == "timer":
        return since_flush >= every
    return False


def run(args: argparse.Namespace, strategy: str) -> dict:
    """Drive one session and return its measurements."""
    words = args.text.split()
    interval = 1.0 / args.words_per_second
    every = args.flush_every / 1000.0
    response_format = {"encoding": "pcm", "sample_rate": args.rate}

    with connect(args.server.rstrip("/") + "/v1/ws", max_size=None) as ws:
        session = Session(ws, args.rate)
        session.send({"start_context": {"seed": args.seed, "response_format": response_format}})
        opening = json.loads(ws.recv())
        if "error" in opening:
            raise SystemExit(f"error: {opening['error']}")
        rate = opening["context_started"]["response_format"]["sample_rate"]

        session.started = time.perf_counter()
        last_flush = session.started
        for index, word in enumerate(words):
            # Words arrive on a clock, not as fast as the socket will take them: that clock is
            # the whole point of the measurement.
            session.poll(session.started + index * interval)
            session.send({"send_text": f"{word} "})
            now = time.perf_counter()
            if index < len(words) - 1 and should_flush(strategy, word, now - last_flush, every):
                session.flush()
                last_flush = now
        session.flush(close=True)
        session.drain()

    return measure(session, rate, strategy)


def measure(session: Session, rate: int, strategy: str) -> dict:
    """Turn one session's arrivals into the numbers the table reports."""
    samples = np.frombuffer(bytes(session.audio), dtype="<i2").astype(np.float32) / 32767.0
    seconds = samples.size / rate
    first = session.chunks[0][0] if session.chunks else float("nan")

    # A player starts on the first chunk and consumes in real time from there, so the margin
    # before each *later* chunk is what it still had in hand while it waited for that one.
    delivered = 0.0
    floor = float("inf")
    for index, (arrival, size) in enumerate(session.chunks):
        if index:
            floor = min(floor, delivered - (arrival - first))
        delivered += size / BYTES_PER_SAMPLE / rate
    # The speed audio arrives at once it has started, which is the number a player cares about:
    # the wait for the first chunk is time-to-first-audio and is reported on its own.
    window = session.closed - first

    joins = np.array([offset // BYTES_PER_SAMPLE for offset in session.boundaries[:-1]])
    joins = joins[(joins > 0) & (joins < samples.size)]
    steps = np.abs(np.diff(samples)) if samples.size > 1 else np.zeros(1, dtype=np.float32)
    return {
        "strategy": strategy,
        "turns": session.flushes,
        "first_audio_ms": first * 1000.0,
        "wall_seconds": session.closed,
        "audio_seconds": seconds,
        "realtime": seconds / window if window > 0 else float("nan"),
        "buffer_floor": floor if len(session.chunks) > 1 else float("nan"),
        "worst_join": float(steps[joins - 1].max()) if joins.size else float("nan"),
        "step_p999": float(np.percentile(steps, 99.9)),
    }


def _number(value: float, spec: str) -> str:
    """`value` formatted, or a dash where there was nothing to measure.

    A single-turn run has no interior flush boundary, so its worst join is not a small number --
    it does not exist, and printing ``nan`` invites reading it as one.
    """
    return format(value, spec) if value == value else "-".rjust(len(format(0.0, spec)))


def report(rows: list[dict], args: argparse.Namespace) -> None:
    """Print the table, then say what it means, so neither can be read without the other."""
    baseline = next((row["audio_seconds"] for row in rows if row["strategy"] == RECOMMENDED), None)
    header = (
        f"{'flush on':21s} {'turns':>5s} {'first audio':>12s} {'speech':>9s} {'vs 1 turn':>10s} "
        f"{'delivery':>9s} {'buffer floor':>13s} {'worst join':>11s} {'99.9% step':>11s}"
    )
    print(f"\n{len(args.text.split())} words at {args.words_per_second}/s, {args.rate} Hz out")
    print(header)
    print("-" * len(header))
    for row in rows:
        label = STRATEGIES[row["strategy"]]
        if row["strategy"] == "timer":
            label = f"a {args.flush_every:.0f} ms timer"
        ratio = f"{row['audio_seconds'] / baseline:9.1f}x" if baseline else " " * 10
        print(
            f"{label:21s} {row['turns']:5d} {row['first_audio_ms']:9.0f} ms "
            f"{row['audio_seconds']:7.2f} s {ratio} "
            f"{row['realtime']:8.2f}x {_number(row['buffer_floor'], '12.2f')}s "
            f"{_number(row['worst_join'], '11.5f')} {row['step_p999']:11.5f}"
        )

    print(
        f"\nFlush on {STRATEGIES[RECOMMENDED]}. That is the first row, and it is the one to copy."
    )
    print(
        "A flush says the turn is over, not 'start now' -- the session is already speaking. The\n"
        "low first-audio figures below it are not a win: each extra flush is another turn, with\n"
        "its own ending, which is why the speech column grows while the words stay the same.\n"
        "Sentence ends are real turn ends, so that row costs nothing; a timer and a word are not."
    )
    print(
        "\nworst join is the largest step at a flush boundary; it stays under the 99.9th "
        "percentile\nof the signal's own steps when the session carried the voice through the "
        "boundary.\n"
        "A negative buffer floor means a player starting on the first chunk would have run dry."
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ws_latency.py", description=__doc__.splitlines()[0], allow_abbrev=False
    )
    parser.add_argument("--server", default="ws://127.0.0.1:8000", help="base WebSocket URL")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="what to say")
    parser.add_argument(
        "--strategy",
        default="all",
        choices=("all", *STRATEGIES),
        help="when the client says the turn is over (default: run all four)",
    )
    parser.add_argument(
        "--words-per-second",
        type=float,
        default=DEFAULT_WORDS_PER_SECOND,
        help=f"rate the text arrives at (default: {DEFAULT_WORDS_PER_SECOND})",
    )
    parser.add_argument(
        "--flush-every",
        type=float,
        default=500.0,
        help="milliseconds between flushes for the timer strategy (default: 500)",
    )
    parser.add_argument("--rate", type=int, default=16000, help="output sample rate")
    parser.add_argument("--seed", type=int, default=None, help="fix the sampler")
    parser.add_argument(
        "--repeat", type=int, default=1, help="runs per strategy; the median is reported"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    strategies = tuple(STRATEGIES) if args.strategy == "all" else (args.strategy,)
    rows = []
    for strategy in strategies:
        runs = [run(args, strategy) for _ in range(max(1, args.repeat))]
        rows.append(
            {
                key: statistics.median([r[key] for r in runs])
                if isinstance(runs[0][key], float)
                else runs[0][key]
                for key in runs[0]
            }
        )
    report(rows, args)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except OSError as exc:
        print(f"cannot reach the server: {exc}", file=sys.stderr)
        sys.exit(1)
