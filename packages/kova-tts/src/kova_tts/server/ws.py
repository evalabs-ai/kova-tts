"""``WS /v1/ws`` -- an incremental synthesis session.

One connection drives one stream. The client opens a context, pushes text as it becomes
available, and says when a turn is over::

    -> {"start_context": {"voice": "some-voice"}}
    <- {"context_started": {...}}
    -> {"send_text": "The first half of a sentence "}
    -> {"send_text": "and the second half."}
    <- {"audio_chunk": "<base64 audio>"}    (as soon as there is enough text to speak)
    -> {"flush": true, "flush_id": "a"}
    <- {"audio_chunk": "<base64 audio>"}    (the rest of the turn)
    <- {"flush_completed": true, "flush_id": "a"}
    -> {"close_context": true, "flush_id": "b"}
    <- {"audio_chunk": "<base64 audio>"}    (the container's own tail)
    <- {"flush_completed": true, "flush_id": "b"}
    <- {"context_closed": true}

**Text is spoken while it is still arriving.** A flush does not start the speech; it ends it.
That is the point of the session: a caller producing text a token at a time -- an LLM, a chat UI
-- hands it over as it arrives and the session speaks as far into it as the text allows,
extending the same utterance every time more turns up.

**How far, and when.** The session holds a growing chunk: the text it has been given and the
codes generated for that text so far. A burst renders up to :data:`CODES_PER_CHAR` codes per
character of new text and stops there. The rate deliberately undershoots what speech really
costs, because the cap is what keeps the model reading rather than writing: allowed to run past
the text it was given, it invents the words that come next. Undershooting costs nothing, since
the next burst continues the same utterance from exactly where this one stopped.
:data:`MIN_BUFFER_CHARS` keeps the loop from bursting on a handful of characters, where there is
too little context to say anything well.

**A flush finishes the turn, not the utterance.** Nothing caps the bursts it runs and nothing
stops them but the model's own end of speech, so the last words of a turn are whole words however
far the estimate above undershot -- and ``flush_completed`` then means what it says: every sample
of that turn has been encoded and sent. The chunk stays open across it where it can, and the
next turn goes on extending the same generation from where the last one left off -- so no
boundary is audible, whatever the client chose.

**A turn still costs an ending, though.** A turn that runs out of words mid-phrase leaves the
chunk holding more codes than its text accounts for, which rotates it: the model finishes what it
was saying and the next turn opens a fresh utterance. Flushing per word therefore buys the first
syllable sooner and pays for it in duration, several times over. Flush when the turn is genuinely
over.

**Every burst continues the last one.** Its prompt is the chunk's text with the codes generated
for that text as the start of the continuation, so the model resumes an utterance instead of
beginning one -- there is no leading silence, no fresh breath, and no seam at a burst boundary,
whether or not a flush happened to fall there. A chunk is only closed when it is full: its text
and its codes then become the carry in front of the next one, which is the same pair the model
uses to thread one generation into the next.

**A cloned voice leads every prompt.** ``start_context`` takes a reference clip -- audio to
encode here, or codes encoded once somewhere else -- and its transcript then sits in front of the
chunk's text with its codes in front of the continuation, on every burst. The first decoder of a
cloned session is primed from the tail of that reference, since there is no earlier audio for it
to resume from.

**Generation and decoding overlap.** Two tasks share a queue of codes: one drives the model, the
other decodes whatever has arrived and puts it on the wire. Audio for the start of a turn leaves
while its end is still being generated.

**Ordering is guaranteed.** Inbound frames are read by one task; generation runs in a second,
one burst at a time, in arrival order; every outgoing frame
goes through a single queue drained by a third. So ``flush_completed`` can never overtake the
audio it terminates, and two turns can never interleave their chunks, however fast the client
sends.

**The client picks the format.** ``response_format`` names a container from
:data:`kova_tts.audio.STREAMING_FORMATS` and a sample rate; the session encodes into one
container for its whole life and ``context_started`` reports what it settled on. A realtime
pipeline fixed at 16 kHz asks for 16 kHz and gets it from the same filter every other path here
uses, its state crossing burst and turn boundaries alike.

**Text is spoken as words, normalized.** Only complete words leave the buffer -- a word cut in
half between two ``send_text`` frames is never read as two -- and with ``normalize`` on (the
default) only complete sentences, since "$1" and ",000" normalize differently apart than
together. A flush takes whatever is left.

**With the aligner loaded, chunks are larger.** A chunk closes at a sentence
end once it holds :data:`ALIGNED_SOFT_CHARS`, or at a word boundary at
:data:`ALIGNED_MAX_CHARS`, and hands on only its aligned last few seconds
(:func:`~kova_tts.engine.tts._aligned_carry`); a flush closes it too, so every word of a turn
has been timed when ``flush_completed`` goes out. ``timestamps`` frames then report when each
word is spoken. Without the aligner, chunks are :data:`~kova_tts.engine.tts.MAX_SEGMENT_CHARS`
and carried whole, as described above.

One thing this session deliberately does not do: it holds the model for a burst at a time, not
for the life of the connection, so a session waiting on its client costs nothing and a burst that
collides with an HTTP request gets an ``error`` frame rather than deadlocking.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import soundfile as sf
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from kova_codec.constants import TOKEN_RATE, codes_to_seconds
from kova_tts import audio as audio_io
from kova_tts import voices as voices_module
from kova_tts.engine.types import CLONE_SAMPLING, Voice, WordTimestamp
from kova_tts.normalization import Word, preprocess
from kova_tts.normalization.split import last_sentence_end, stable_sentence_end, tags_closed
from kova_tts.server import formats
from kova_tts.server import protocol as wire
from kova_tts.server.engine import (
    Engine,
    close_on_engine_thread,
    engine_thread,
    on_engine_thread,
)
from kova_tts.server.errors import InvalidRequest

if TYPE_CHECKING:  # pragma: no cover - typing only; the aligner imports torch
    from kova_tts.alignment import AlignmentStream

log = logging.getLogger(__name__)

router = APIRouter()

#: Sentinel put on the outbound queue to retire the writer task.
_STOP = None

#: Sentinel put on a turn's code queue when its last burst has finished generating.
_END = None

#: Codes a character of text is allowed to account for. Speech runs at about eight characters a
#: second against eighty codes a second, so ten would be the honest rate -- and this is a third
#: of it on purpose. The cap decides how far past its text a burst may generate, and past that
#: text there is nothing to read: the model starts writing the continuation itself, in a voice
#: the client never asked it to invent. Undershooting only means a burst stops early and the next
#: one carries on, which is free; overshooting is words nobody sent.
CODES_PER_CHAR = 3

#: Characters that have to arrive before the session speaks unbidden. Below this there is too
#: little to render well -- a fragment of a sentence gives the model no phrase to shape -- and the
#: audio it would produce is a few hundred milliseconds that the next burst has to sound like a
#: continuation of anyway. A flush ignores the threshold, so a short reply that is complete is
#: spoken as soon as it is complete.
MIN_BUFFER_CHARS = 50

#: With the aligner, a chunk closes at the first sentence end past this many characters...
ALIGNED_SOFT_CHARS = 200

#: ...or, with no sentence end to close at, at the last word boundary before this many.
ALIGNED_MAX_CHARS = 400

#: Reference codes pushed through the decoder before a cloned session's first audio. Every one of
#: them is decoded and then thrown away, so this is time-to-first-audio spent on audio nobody
#: hears; one second is enough for the decoder's LSTM and first convolution to start from real
#: context instead of from silence.
PREROLL_CODES = 80

#: Name the voice built from a client's reference carries. A session speaks as exactly one voice
#: and never refers to it by name, so it needs one only because every voice has one.
REFERENCE_NAME = "reference"


@dataclass(frozen=True, slots=True)
class _Work:
    """The end of a turn: the text buffered when it was asked for, and whether it ends the
    session."""

    text: str
    flush_id: str
    close: bool


@dataclass(slots=True)
class _Chunk:
    """The generation in progress: the words it is speaking, and the codes it has produced.

    The two halves grow together and are only ever used together -- the text is what the next
    burst asks the model to go on reading, the codes are how far into it the model has got.

    ``level`` is the last point at which the two were known to be level with each other: the
    model brought an utterance to its own end there, so everything up to it has been said. How
    far ahead of the text the next burst may run is measured from that point rather than from the
    start of the chunk, which is what lets a chunk carry on being extended after a turn has ended
    on it.

    ``alignment`` times the chunk's words against its codes when the aligner is loaded.
    """

    words: list[Word] = field(default_factory=list)
    codes: list[int] = field(default_factory=list)
    level: tuple[int, int] = (0, 0)
    alignment: AlignmentStream | None = None

    @property
    def text(self) -> str:
        """What the model reads: the words in their spoken form."""
        return " ".join(word.normalized for word in self.words)

    @property
    def settled(self) -> bool:
        """True when the model has finished everything this chunk has been given."""
        return self.level[0] == len(self.text)

    @property
    def target(self) -> int:
        """The most codes this chunk may hold, for the text it has been given so far."""
        chars, codes = self.level
        return codes + CODES_PER_CHAR * (len(self.text) - chars)

    def settle(self) -> None:
        """Record that the model has just finished reading everything the chunk holds."""
        self.level = (len(self.text), len(self.codes))


class _Model:
    """The model, in the shape a continuous session uses it.

    :class:`~kova_tts.engine.tts.KovaTTS` renders one text per call and owns the continuity
    inside it: the chunking, the chunk-to-chunk carry, the codec decoder. A session spreads a
    single utterance over many calls arriving minutes apart, so it holds that state itself and
    needs the pieces underneath rather than the whole. They are named here, in one place, so the
    session talks to something session-shaped and the reach into the model is a short list:

    * :func:`~kova_tts.engine.tts.split_sentences` and
      :data:`~kova_tts.engine.tts.MAX_SEGMENT_CHARS` for how much text one chunk holds;
    * ``KovaTTS._prepare`` and ``KovaTTS._sampling`` to resolve a voice and its preset;
    * ``KovaTTS.clone`` and ``KovaTTS._preroll`` for a reference clip: encoding one, and the tail
      of it the first decoder starts warm from;
    * ``KovaTTS._prompt_ids``, which puts the reference and the carry in front of a chunk and
      drops the carry when the two together would not fit the KV cache;
    * ``KovaTTS._Carry`` and :data:`~kova_tts.engine.tts.MAX_CARRY_CODES`, which are the rule for
      what may be carried at all -- text and codes together, or nothing;
    * ``KovaTTS.aligner`` and ``_aligned_carry`` for the aligned chunking, when it is loaded;
    * ``Generator.stream_ids`` for the codes, and the codec for the audio.

    The imports happen here rather than at module scope so ``import kova_tts.server`` still costs
    no torch: the endpoints, their validators and the whole HTTP surface load without one.
    """

    def __init__(self, tts: Any) -> None:
        from kova_tts.alignment import AlignmentStream
        from kova_tts.engine.decoder import CONV_PADDING, LOOKAHEAD, WINDOW, StreamingDecoder
        from kova_tts.engine.tts import (
            CARRY_SECONDS,
            MAX_CARRY_CODES,
            MAX_SEGMENT_CHARS,
            _aligned_carry,
            _Carry,
            _codes_for_chars,
            split_sentences,
        )

        self._tts = tts
        self._decoder = StreamingDecoder
        self._carry = _Carry
        self._aligned_carry = _aligned_carry
        self._alignment_stream = AlignmentStream
        self._split = split_sentences
        self._max_carry_codes = MAX_CARRY_CODES
        self._carry_codes = round(CARRY_SECONDS * TOKEN_RATE)
        self._codes_for_chars = _codes_for_chars
        #: The word aligner, or ``None``; it decides which chunking the session uses.
        self.aligner = getattr(tts, "aligner", None)
        #: Text one chunk holds: what the model renders in a single generation.
        self.max_chars = ALIGNED_MAX_CHARS if self.aligner is not None else MAX_SEGMENT_CHARS
        #: Codes of the previous turn replayed into a new decoder before any audio is kept.
        #: One whole window's worth -- what a window emits, its lookahead on both sides and the
        #: convolution's reach on both ends -- which is everything the decoder reads to produce
        #: a single window, so the first window of a turn is computed from real audio history
        #: instead of from silence.
        self.prime_codes = WINDOW + 2 * LOOKAHEAD + 2 * CONV_PADDING

    def chunks(self, text: str) -> list[str]:
        """Split text into pieces no larger than one generation, at its own boundaries."""
        if self.aligner is not None:
            return _aligned_chunks(text)
        return self._split(text)

    def alignment(self, *, live: bool) -> Any:
        """A new chunk's word alignment, or ``None`` without the aligner."""
        if self.aligner is None:
            return None
        return self._alignment_stream(self.aligner, live=live)

    def prepare(self, voice: Any) -> Any:
        """Resolve a voice and put the LM into its weights. Blocking; call inside a thread."""
        return self._tts._prepare(voice)

    def sampling(self, voice: Any, params: Any) -> Any:
        """The sampler for this voice, with the session's overrides applied."""
        return self._tts._sampling(voice, params)

    def prompt_ids(self, text: str, voice: Any, carry: Any, params: Any) -> list[int]:
        """Tokens for one burst, continuing `carry`. Blocking; call inside a thread."""
        return self._tts._prompt_ids(text, voice, carry, params)

    def codes(self, ids: list[int], params: Any) -> Any:
        """The blocking iterator of codes for one prompt."""
        return self._tts.generator.stream_ids(ids, params)

    def continuation(self, carry: Any, codes: list[int]) -> Any:
        """What leads a burst's prompt: the chunk before this one, then this one so far.

        Both are the same pair -- some text, and the codes it was rendered as -- so they simply
        concatenate: the previous chunk's text and codes, and then the codes this chunk has
        produced for the text the prompt is about to state again. Handing the model its own
        codes back is what makes a burst resume an utterance rather than begin one.
        """
        if carry is None and not codes:
            return None
        text = carry.text if carry is not None else ""
        prior = carry.codes if carry is not None else ()
        return self._carry(text, tuple(prior) + tuple(codes))

    def codes_for(self, text: str) -> int:
        """The most codes the model's own estimate allows this much text to be spoken as.

        The same number :data:`~kova_tts.engine.tts.MAX_CARRY_CODES` is derived from, so a chunk
        held inside it is a chunk that can always be carried.
        """
        return self._codes_for_chars(len(text))

    def carry(self, text: str, codes: list[int]) -> Any:
        """What a finished chunk hands on, or ``None`` when it is too long to hand anything on.

        Text and codes travel together or not at all: codes on their own are several seconds of
        speech for a sentence the model has not begun, and it answers that by ending the
        utterance on the first step. A chunk whose codes overflow the carry is therefore dropped
        whole, and the next chunk starts from its own text alone.
        """
        return self._carry(text, tuple(codes)) if len(codes) <= self._max_carry_codes else None

    def aligned_carry(self, timed: list[Any], codes: list[int], *, complete: bool) -> Any:
        """What a finished chunk hands on when the aligner is loaded: its last few seconds, cut
        at a word the aligner placed, so it fits however long the chunk ran."""
        return self._aligned_carry(timed, codes, complete=complete)

    def decoder(self) -> Any:
        """A decoder for one turn. Blocking on first use -- the codec loads then."""
        return self._decoder(self._tts.codec)

    def clone(self, wav: np.ndarray, transcript: str, sample_rate: int) -> Any:
        """Encode a reference waveform, recorded at `sample_rate`, into a voice to speak as.

        Blocking, and the only call here that loads WavLM: encoding needs the half of the codec
        that decoding never touches, which is a second codec built on first use.
        """
        return self._tts.clone(wav, transcript, name=REFERENCE_NAME, sample_rate=sample_rate)

    def preroll(self, voice: Any) -> tuple[int, ...]:
        """The tail of a reference the first decoder of a cloned session starts warm from."""
        return tuple(self._tts._preroll(voice))[-PREROLL_CODES:]

    def reference_headroom(self, voice: Any) -> tuple[int, int]:
        """Tokens the worst burst of a cloned session needs, against what the KV cache holds.

        The reference is not a start-up cost but part of every prompt the session builds: its
        codes and its transcript lead each burst, in front of a chunk's text and of the codes
        that chunk is up to. The worst case is a full chunk that has already been spoken, with
        the carry in front of it dropped -- ``_prompt_ids`` drops it precisely when it would not
        fit -- and a whole chunk's worth of generation still reserved on top, since the reserve
        is what the chunk might need rather than what is left of it. Text is counted a token per
        character, which no tokenizer ever reaches, so the estimate errs towards accepting.

        With the aligner the carry is bounded by time rather than dropped, so it is counted in,
        and the chunk's codes are counted once: its larger chunks would not fit twice, and the
        generator clamps a final burst to the room that is left rather than failing.
        """
        chunk = self.max_chars + 2 * self._codes_for_chars(self.max_chars)
        if self.aligner is not None:
            chunk = self.max_chars + self._codes_for_chars(self.max_chars) + self._carry_codes
        return len(voice.ref_codes) + len(voice.ref_text) + chunk, int(
            self._tts.generator.max_cache_len
        )


class Session:
    """The state behind one connection: the text it has been given, the utterance that text is
    becoming, and the continuity that runs through all of it.

    Args:
        engine: The shared model. Reserved for a burst at a time, never for the whole session.
        start: The accepted configuration.
        out: The connection's outbound queue. The endpoint owns the task draining it, so this
            class never touches the socket and cannot emit frames out of order.
    """

    def __init__(self, *, engine: Engine, start: wire.StartConfig, out: asyncio.Queue) -> None:
        self.engine = engine
        self.out = out
        self.start = start
        self.params = _session_sampling(engine, start)
        # One encoder for the session: the chunks are slices of a single container, and a
        # container restarted between turns is a second file spliced into the middle of the
        # first, which no decoder reads as one stream. Its tail goes out at close_context.
        self.encoder = formats.stream_encoder(
            start.response_format.encoding,
            start.response_format.sample_rate or engine.sample_rate,
            engine.sample_rate,
        )
        # Taken from the encoder rather than from the request, because the encoder is what
        # writes the bytes: mp3 and opus each accept a fixed set of rates and snap anything else
        # up to the nearest one, and that snap is the one thing a client would have to guess.
        self.response_format = start.response_format.model_copy(
            update={"sample_rate": self.encoder.rate}
        )
        self.model = _Model(engine.tts)
        #: The voice built from ``start_context``'s reference, once it has been encoded.
        self.voice: Any = None

        self._buffer: list[str] = []
        #: Buffered characters when the last look for a sentence end found none, so the producer
        #: waits for more text rather than looking again at the same text.
        self._stalled = -1
        #: Words taken from the buffer that have not fitted into the chunk yet.
        self._pending: list[Word] = []
        #: Codes of the chunks already closed: where the open chunk's timestamps start from.
        self._spoken_codes = 0
        self._work: asyncio.Queue[_Work] = asyncio.Queue()
        #: Set whenever a frame arrives, so the producer wakes for text as well as for a flush.
        self._ready = asyncio.Event()
        self._producer: asyncio.Task | None = None
        #: The generation in progress: what it is speaking, and how far it has got.
        self._chunk = _Chunk()
        #: The chunk before it, as text and codes together: continuation context for this one.
        self._carry: Any = None
        #: Codes the last decoder ended on, replayed into the next one as its context.
        self._decoded: tuple[int, ...] = ()
        #: The open turn: codes on their way to the decoder, and the task draining them.
        self._queue: asyncio.Queue[int | None] | None = None
        self._decoding: asyncio.Task | None = None

    @property
    def speaker(self) -> Any:
        """Who to speak as: the cloned voice if the client sent a reference, else the name it
        chose."""
        return self.voice if self.voice is not None else self.start.voice

    # ------------------------------------------------------------------ inbound frame handlers

    async def open(self) -> None:
        """Encode the reference if there is one, start generating, and acknowledge the context.

        The reference is dealt with here, before the acknowledgement and before a word of text:
        it is the one part of a session's setup that can fail for a reason the client can fix,
        and a client that has streamed a paragraph before being told its clip is unusable has
        wasted the session.
        """
        if self.start.reference is not None:
            self.voice = await self._clone(self.start.reference)
            # There is no earlier audio for the first decoder to resume from, so the reference
            # plays that part: its tail is real speech in this voice, immediately before the
            # first sample the session will send.
            self._decoded = self.model.preroll(self.voice)
        self._producer = asyncio.create_task(self._run())
        await self._emit(wire.ContextStarted(context_started=self._accepted()))

    def send_text(self, text: str) -> None:
        self._buffer.append(text)
        self._ready.set()

    def flush(self, flush_id: str | None) -> None:
        self._enqueue(flush_id or "flush", close=False)

    def close_context(self, flush_id: str | None) -> None:
        self._enqueue(flush_id or "close", close=True)

    def _enqueue(self, flush_id: str, *, close: bool) -> None:
        # Snapshot the buffer at the moment the turn was ended, not when the work runs: text
        # that arrives while the session is still speaking belongs to the turn after this one.
        text = "".join(self._buffer)
        self._buffer.clear()
        self._work.put_nowait(_Work(text=text, flush_id=flush_id, close=close))
        self._ready.set()

    async def aclose(self) -> None:
        """Tear down after a disconnect, without waiting for work still queued."""
        if self._producer is not None:
            self._producer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._producer
            self._producer = None
        await self._discard()
        await self.out.put(_STOP)

    # ------------------------------------------------------------------ the producer

    async def _run(self) -> None:
        """Speak for as long as the session lives: bursts while text arrives, turns as they end."""
        try:
            while True:
                work = await self._next()
                try:
                    if work is None:
                        await self._extend()
                    else:
                        await self._finish(work)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - reported to the client, session lives
                    log.exception("synthesis failed")
                    await self._discard()
                    flush_id = work.flush_id if work is not None else None
                    await self._emit(wire.Error(error=str(exc), flush_id=flush_id))
                if work is None:
                    continue
                # Acknowledged however it went, so a client that waits for it never hangs.
                if work.close:
                    await self._close_stream(work.flush_id)
                await self._emit(wire.FlushCompleted(flush_id=work.flush_id))
                if work.close:
                    await self._emit(wire.ContextClosed())
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a session failure is a frame, not a crash
            log.exception("websocket session failed")
            await self._emit(wire.Error(error=str(exc)))
        finally:
            await self.out.put(_STOP)

    async def _next(self) -> _Work | None:
        """Wait until there is something to do, and say which of the two it is.

        ``None`` means "speak more of what has arrived". A queued turn always wins, so text that
        turned up after a flush stays buffered for the turn after it rather than being spoken as
        part of the one the client has already ended.
        """
        while True:
            self._ready.clear()
            if not self._work.empty():
                return self._work.get_nowait()
            if self._pending:
                return None
            buffered = sum(len(part) for part in self._buffer)
            if buffered >= MIN_BUFFER_CHARS and buffered != self._stalled:
                return None
            await self._ready.wait()

    async def _extend(self) -> None:
        """Speak further into the turn in progress, without ending it."""
        taken = await asyncio.to_thread(self._words, self._take())
        if not taken and not self._pending:
            # Nothing ready to speak yet: wait for more text before looking again.
            self._stalled = sum(len(part) for part in self._buffer)
        self._pending = self._fill(self._pending + taken)
        if not self._pending:
            await self._burst(final=False)
            return
        # The chunk is full. Finish what it holds and start the next one behind it, rather than
        # growing a prompt that would eventually outgrow the KV cache.
        await self._burst(final=True)
        await self._rotate()

    def _take(self) -> str:
        """The text in the buffer that is ready to speak before the turn ends.

        Complete words only. With normalization on, complete sentences only -- the first one the
        buffer holds -- or, when a sentence runs past what one chunk holds, everything up to its
        last word boundary there. The rest stays buffered for later or for the flush.
        """
        text = "".join(self._buffer)
        if self.start.normalize:
            end = stable_sentence_end(text)
            if not end and len(text) >= self.model.max_chars:
                end = _word_end(text, self.model.max_chars) or self.model.max_chars
        else:
            end = _word_end(text, len(text))
        self._buffer = [text[end:]] if end < len(text) else []
        return text[:end]

    def _words(self, text: str) -> list[Word]:
        """`text` as words, each with the spoken form the model will read. Blocking: normalizing
        is CPU work, and the first call loads the grammars."""
        if not text.strip():
            return []
        sentences = preprocess(text, self.model.max_chars, self.start.normalize)
        return [word for sentence in sentences for word in sentence.words]

    async def _finish(self, work: _Work) -> None:
        """Speak everything the turn holds, ending the utterance where its text ends.

        Nothing caps these bursts: the client has said no more text is coming, so the model runs
        to its own stop and finishes the word it is on. Text longer than one chunk is spoken as
        several, exactly as it would have been had it arrived a piece at a time.

        Without the aligner the chunk is not closed here. A turn ending is the client saying it
        has run out of words, not the session saying the utterance is over, so where the turn
        lands on a boundary the next one goes on extending the same generation. A turn that ends
        mid-phrase leaves the chunk over-generated against its text, and :meth:`_rotate` starts a
        fresh one. With the aligner it is always closed: its aligned carry keeps the next turn
        continuous, and closing is what times its last words.
        """
        pending = self._pending + await asyncio.to_thread(self._words, work.text)
        self._pending = []
        while True:
            pending = self._fill(pending)
            await self._burst(final=True)
            if not pending:
                break
            await self._rotate()
        if self.model.aligner is not None:
            await self._rotate()
        await self._end_turn()

    def _fill(self, pending: list[Word]) -> list[Word]:
        """Move as much of `pending` into the growing chunk as it has room for; return the rest.

        The chunk and the new text are split together, at the model's own boundaries and to its
        own size: a chunk that ended mid-sentence and the words that continue it are one
        sentence again, which is what the model should be reading, and where a chunk ends is
        where the voice will come to a stop.

        A chunk only ever grows. On the rare text where splitting the two together would move a
        boundary back into words the chunk has already begun speaking, the new text waits for
        the chunk after this one instead.
        """
        if not pending:
            return []
        chunk = self._chunk
        pieces = self.model.chunks(" ".join(w.normalized for w in chunk.words + pending))
        room = len(pieces[0].split()) - len(chunk.text.split())
        if room < 0:
            return pending
        taken = 0
        for word in pending:
            room -= len(word.normalized.split())
            if room < 0:
                break
            taken += 1
        if not chunk.words:
            taken = max(taken, 1)  # one word longer than a whole chunk still has to be said
        if chunk.alignment is None:
            chunk.alignment = self.model.alignment(live=self.start.timestamps)
        if chunk.alignment is not None:
            chunk.alignment.add_words(pending[:taken])
        chunk.words.extend(pending[:taken])
        return pending[taken:]

    async def _burst(self, *, final: bool) -> None:
        """Generate the next stretch of the chunk, streaming its codes to the decoder.

        The model is reserved for exactly this: a voice has to be loaded into the LM's weights
        before a prompt can be built against it, and the burst has to finish before another
        request may change them. It is released again in between, so a session that is waiting
        on its client is not holding the GPU.

        Args:
            final: Let the model finish. Nothing caps the generation and nothing stops it but
                the model's own end of speech, which is what makes the last words of a turn
                whole words. An unfinished burst is capped instead to the codes the text it has
                been given can account for, and stops there.
        """
        chunk = self._chunk
        room = chunk.target - len(chunk.codes)
        if not chunk.text.strip() or chunk.settled or (not final and room <= 0):
            return
        queue = await self._open_turn()
        async with self.engine.reserve():
            voice = await on_engine_thread(self.model.prepare, self.speaker)
            params = self.model.sampling(voice, self.params)
            if not final:
                params = params.replace(max_tokens=max(room, 1))
            ids = await self._prompt_ids(chunk, voice, params)
            produced = await self._stream_codes(ids, params, queue, None if final else chunk.target)
        if final or produced < room:
            # Either the client said the turn was over and the model ran to its own end, or it
            # got there first. Text and speech are level again, and the next burst measures how
            # far it may run from here -- the chunk goes on being extended, because a turn
            # ending is not the utterance ending.
            chunk.settle()
            if len(chunk.codes) > self.model.codes_for(chunk.text):
                # Unless the chunk now holds more speech than its words can account for, which
                # is what a turn ended on a fragment leaves behind: the stop and the pause the
                # model renders at the end of an utterance, paid for by no text. Extending it
                # further would take the pair further from any prompt the model was trained on,
                # and it answers that by wandering, so the chunk is closed here instead and the
                # next words start one of their own behind it.
                await self._rotate()

    async def _prompt_ids(self, chunk: _Chunk, voice: Any, params: Any) -> list[int]:
        """Tokens for this burst, keeping the codes the chunk has already produced.

        What leads a prompt is two things at once: the chunk before this one, and how far this
        one has got. A continuation that will not fit the KV cache is dropped, and it is dropped
        whole -- which is right for the first of those and ruinous for the second, since a chunk
        whose own codes are missing from its prompt is a chunk the model starts again from the
        beginning and a listener hears twice. So when the pair does not fit, the prompt is built
        again around the half that cannot be given up, and the join with the previous chunk is
        what gives way instead.
        """
        ids = await asyncio.to_thread(
            self.model.prompt_ids,
            chunk.text,
            voice,
            self.model.continuation(self._carry, chunk.codes),
            params,
        )
        if self._carry is None or not chunk.codes:
            return ids
        kept = await asyncio.to_thread(
            self.model.prompt_ids,
            chunk.text,
            voice,
            self.model.continuation(None, chunk.codes),
            params,
        )
        # The prompt that kept something is the longer one, and a dropped continuation leaves
        # nothing at all in front of the text.
        return kept if len(kept) > len(ids) else ids

    async def _stream_codes(
        self,
        ids: list[int],
        params: Any,
        codes: asyncio.Queue[int | None],
        limit: int | None,
    ) -> int:
        """Drive the LM for one burst, one thread hop per code, and return how many it produced.

        `limit` is the code count the chunk stops at, counted over the whole chunk rather than
        over this burst: a burst is a continuation, and where the previous one stopped is where
        this one starts counting from.

        A hop costs microseconds against a decode step's milliseconds; what it buys is an event
        loop free to send the audio of earlier codes while these are still being generated. Every
        hop lands on the engine thread, which on MLX is not optional -- see
        :func:`~kova_tts.server.engine.engine_thread`.
        """
        produced = 0
        loop = asyncio.get_running_loop()
        stream = self.model.codes(ids, params)
        try:
            while limit is None or len(self._chunk.codes) < limit:
                code = await loop.run_in_executor(engine_thread(), next, stream, None)
                if code is None:
                    break
                produced += 1
                self._chunk.codes.append(code)
                codes.put_nowait(code)
                if self._chunk.alignment is not None:
                    self._chunk.alignment.add_codes((code,))
                    await self._timestamps(self._chunk.alignment.take())
            return produced
        finally:
            # A burst given up on -- one that hit its cap, a cancelled turn, a client that hung
            # up mid-sentence -- leaves the LM marked in flight until its iterator is closed.
            close_on_engine_thread(stream)

    async def _rotate(self) -> None:
        """Close the growing chunk and start the next one behind it.

        The finished chunk becomes the continuation context for the one that follows -- its text
        and the codes it was rendered as, together or not at all -- so a chunk boundary threads
        through the prompt exactly as a burst boundary threads through the codes. With the
        aligner it is the chunk's aligned tail instead, and its last words' timings go out.
        """
        chunk, self._chunk = self._chunk, _Chunk()
        if chunk.alignment is None:
            if chunk.codes:
                self._carry = self.model.carry(chunk.text, chunk.codes)
            return
        timed = await asyncio.to_thread(chunk.alignment.finish)
        await self._timestamps(chunk.alignment.take())
        if chunk.codes:
            complete = len(timed) == len(chunk.words)
            self._carry = self.model.aligned_carry(timed, chunk.codes, complete=complete)
        self._spoken_codes += len(chunk.codes)

    async def _timestamps(self, timed: list[Any]) -> None:
        """Send the open chunk's newly timed words, if the client asked for timestamps."""
        if timed and self.start.timestamps:
            offset = self._spoken_codes / TOKEN_RATE
            words = [
                WordTimestamp(t.word.original, offset + t.start_ms / 1000, offset + t.end_ms / 1000)
                for t in timed
            ]
            await self._emit(wire.Timestamps(timestamps=wire.WordTimings.of(words)))

    # ------------------------------------------------------------------ the turn's decoder

    async def _open_turn(self) -> asyncio.Queue[int | None]:
        """The queue this turn's codes go on, starting the decoder if the turn is a new one.

        The decoder spans the turn rather than the burst: a burst boundary is not a boundary in
        the audio at all, and handing every burst its own decoder would put a seam at each one.
        """
        if self._decoding is not None and self._queue is not None:
            if not self._decoding.done():
                return self._queue
            # Decoding only ends when the turn does, so a task already finished here has failed.
            # Surfacing it now stops the next burst generating into a queue nobody is draining.
            await self._decoding
        # Off the engine thread: building a decoder is bookkeeping, and this runs before the turn
        # holds the model, so it must not queue behind a request that does.
        decoder = await asyncio.to_thread(self.model.decoder)
        self._queue = asyncio.Queue()
        self._decoding = asyncio.create_task(self._decode(decoder, self._queue))
        return self._queue

    async def _end_turn(self) -> None:
        """Wait for every code of the turn to be decoded, encoded and sent."""
        if self._decoding is None or self._queue is None:
            return
        self._queue.put_nowait(_END)
        decoding, self._decoding, self._queue = self._decoding, None, None
        await decoding

    async def _discard(self) -> None:
        """Drop the turn after a failure.

        The codes generated for the chunk went to a decoder whose output never reached the
        client, so keeping them as continuation context would have the next chunk carry on from
        speech nobody heard.
        """
        if self._decoding is not None:
            self._decoding.cancel()
            await asyncio.gather(self._decoding, return_exceptions=True)
            self._decoding = None
            self._queue = None
        if self._chunk.alignment is not None:
            self._chunk.alignment.close()
        self._chunk = _Chunk()

    async def _decode(self, decoder: Any, codes: asyncio.Queue[int | None]) -> None:
        """Turn queued codes into audio and put it on the wire as it completes.

        The decoder starts primed with the codes the previous turn ended on -- or, at the start
        of a cloned session, with the tail of the reference -- so its LSTM and its first
        convolution resume from that audio rather than from silence, and the samples they
        produce a second time are dropped rather than sent twice.
        """
        recent: deque[int] = deque(self._decoded, maxlen=self.model.prime_codes)
        if self._decoded:
            await on_engine_thread(decoder.prime, self._decoded)
        try:
            while True:
                batch, done = await _next_codes(codes)
                if batch:
                    payload = await on_engine_thread(self._encode, decoder.push, batch)
                    # Counted as decoded only once they are, so a turn that dies halfway does
                    # not leave the next one priming from audio nobody ever heard.
                    recent.extend(batch)
                    await self._send(payload)
                if done:
                    break
            await self._send(await on_engine_thread(self._encode, decoder.finish))
        finally:
            self._decoded = tuple(recent)

    def _encode(self, produce: Any, *args: Any) -> bytes:
        """Decode audio and encode it into the session's container, in one hop off the loop.

        Both halves are C code holding the GIL, and the loop has a socket to keep answering.
        """
        return self.encoder.encode(produce(*args))

    async def _close_stream(self, flush_id: str) -> None:
        """Close the container and send whatever trailer it has.

        The encoder spans the session, so this runs once, after the last turn's audio and before
        it is acknowledged.
        """
        try:
            await self._send(await asyncio.to_thread(self.encoder.finish))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - reported to the client, the session still ends
            log.exception("could not close the %s stream", self.start.response_format.encoding)
            await self._emit(wire.Error(error=str(exc), flush_id=flush_id))

    # ------------------------------------------------------------------ the cloned voice

    async def _clone(self, reference: wire.VoiceReference) -> Any:
        """The voice this session speaks as, from the reference on its ``start_context``.

        Codes are taken as they stand; audio is decoded and encoded here, which is GPU work and
        so goes through the same reservation a burst does. That is the whole difference between
        the two forms at run time -- one of them starts a session, the other loads WavLM first.
        """
        if reference.codes is not None:
            voice = _voice_from_codes(reference)
        else:
            async with self.engine.reserve():
                # Off the engine thread: this runs before the session holds the model, and must
                # not wait out another client's utterance to encode a reference.
                voice = await asyncio.to_thread(_encode_reference, self.model, reference)
        needed, capacity = self.model.reference_headroom(voice)
        if needed > capacity:
            room = codes_to_seconds(max(len(voice.ref_codes) - (needed - capacity), 0))
            raise ValueError(
                f"the reference is {voice.ref_seconds:.1f} s ({len(voice.ref_codes)} codes) and "
                f"a session puts it in front of every prompt it builds, which leaves {needed} "
                f"tokens against a KV cache of {capacity}. Clone from at most {room:.1f} s of "
                f"audio, or shorten the transcript"
            )
        return voice

    def _accepted(self) -> wire.StartedConfig:
        """The configuration as the session will really run it, for ``context_started``."""
        reference = (
            wire.ReferenceInEffect(
                seconds=round(self.voice.ref_seconds, 3), codes=len(self.voice.ref_codes)
            )
            if self.voice is not None
            else None
        )
        return wire.StartedConfig(
            voice=self.start.voice,
            reference=reference,
            sampling=self.start.sampling,
            response_format=self.response_format,
            normalize=self.start.normalize,
            timestamps=self.start.timestamps,
        )

    # ------------------------------------------------------------------ outbound frames

    async def _send(self, payload: bytes) -> None:
        """Put encoded bytes on the wire, if there are any.

        Nothing coming back is ordinary: a frame-based container emits no bytes until it holds a
        whole frame's worth, and an empty return never means the end of anything.
        """
        if payload:
            chunk = base64.b64encode(payload).decode("ascii")
            await self._emit(wire.AudioChunk(audio_chunk=chunk))

    async def _emit(self, frame: wire.OutgoingFrame) -> None:
        await self.out.put(frame)


def _session_sampling(engine: Engine, start: wire.StartConfig) -> Any:
    """The sampler for the session, or ``None`` to leave the preset to the model.

    Overrides are applied to the preset the session's voice calls for, which for a cloned voice
    is not the plain-TTS one: sending a single knob has to leave the rest of them where the
    model would have put them anyway.
    """
    if start.reference is None or start.sampling is None:
        return engine.sampling(start.sampling)
    try:
        return start.sampling.apply(CLONE_SAMPLING)
    except ValueError as exc:
        raise InvalidRequest(f"sampling: {exc}") from exc


def _encode_reference(model: _Model, reference: wire.VoiceReference) -> Any:
    """Turn a base64 recording into a voice. Blocking: decoding, resampling and WavLM."""
    wav, rate = _reference_waveform(reference.audio or "")
    _check_reference_length(wav.size / rate)
    return model.clone(wav, reference.transcript, rate)


def _voice_from_codes(reference: wire.VoiceReference) -> Voice:
    """A voice from a clip that was encoded somewhere else.

    Nothing is loaded and nothing is decoded here -- the codes *are* the reference. They are
    checked by :class:`~kova_tts.engine.types.Voice` itself, so a code outside the codebook is
    refused by the same rule that would refuse it on any other path into the model.
    """
    codes = tuple(reference.codes or ())
    _check_reference_length(codes_to_seconds(len(codes)))
    return Voice(name=REFERENCE_NAME, ref_codes=codes, ref_text=reference.transcript)


def _reference_waveform(encoded: str) -> tuple[np.ndarray, int]:
    """Base64 of an audio file -> ``(mono waveform, its rate)``, or a refusal naming what went
    wrong.

    Left at the rate it was recorded at: the clone path decides what the encoder gets, and a
    16 kHz phone recording is encoded natively where the checkpoint allows it.

    The two failures are told apart because they are different mistakes: one is the frame, the
    other is what was put in it.
    """
    try:
        data = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise ValueError(
            f"the reference audio is not valid base64 ({exc}); send the bytes of an audio file, "
            f"base64-encoded, with no data: prefix in front of them"
        ) from exc
    try:
        samples, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(
            f"the reference audio could not be decoded ({exc}); send a whole audio file -- wav, "
            f"flac, ogg, mp3, anything soundfile reads -- rather than raw samples"
        ) from exc
    return audio_io.as_waveform(samples), int(rate)


def _check_reference_length(seconds: float) -> None:
    """Refuse a clip a session cannot clone from, before anything expensive happens to it.

    The bounds are :mod:`kova_tts.voices`'s own, so a clip refused here is one no path into the
    model would have taken either. The maximum is enforced by refusing rather than by trimming:
    over a session the reference is not a one-off cost but part of every prompt, and a client
    that sent a minute of audio should hear that it did rather than be given twenty seconds of
    it and left to wonder which twenty.
    """
    if seconds < voices_module.MIN_REFERENCE_SECONDS:
        raise ValueError(
            f"the reference is {seconds:.2f} s of audio; cloning needs at least "
            f"{voices_module.MIN_REFERENCE_SECONDS:.0f} s of clean speech to work from"
        )
    if seconds > voices_module.MAX_REFERENCE_SECONDS:
        raise ValueError(
            f"the reference is {seconds:.1f} s of audio, and a session clones from at most "
            f"{voices_module.MAX_REFERENCE_SECONDS:.0f} s: every prompt it builds carries the "
            f"whole clip, so a longer one costs cache and time on every burst it generates"
        )


def _word_end(text: str, limit: int) -> int:
    """Length of the longest prefix of ``text[:limit]`` that ends a word outside any ``[tag]``,
    or 0 when there is none."""
    for index in range(min(limit, len(text)) - 1, -1, -1):
        if text[index].isspace() and tags_closed(text[: index + 1]):
            return index + 1
    return 0


def _aligned_chunks(text: str) -> list[str]:
    """The aligned chunk boundary, as a split: the first piece ends at the last
    sentence end between :data:`ALIGNED_SOFT_CHARS` and :data:`ALIGNED_MAX_CHARS`, or failing
    that at the last word boundary before the maximum."""
    text = text.strip()
    if len(text) < ALIGNED_SOFT_CHARS:
        return [text]
    end = last_sentence_end(text[:ALIGNED_MAX_CHARS])
    if end < ALIGNED_SOFT_CHARS:
        if len(text) <= ALIGNED_MAX_CHARS:
            return [text]
        end = _word_end(text, ALIGNED_MAX_CHARS) or ALIGNED_MAX_CHARS
    head, tail = text[:end].strip(), text[end:].strip()
    return [head, tail] if tail else [head]


async def _next_codes(codes: asyncio.Queue[int | None]) -> tuple[list[int], bool]:
    """Everything waiting on `codes`, and whether the turn has finished generating.

    Waits for one code and then takes the rest without waiting. Batching costs no latency -- the
    decoder emits a window only once every code in it has arrived, so handing it ten codes at
    once produces exactly what handing it ten codes in turn would -- and it saves a hop and a
    round trip through the codec for each one.
    """
    batch: list[int] = []
    code = await codes.get()
    while code is not _END:
        batch.append(code)
        try:
            code = codes.get_nowait()
        except asyncio.QueueEmpty:
            return batch, False
    return batch, True


# ---------------------------------------------------------------------------------- the endpoint


async def _writer(ws: WebSocket, out: asyncio.Queue) -> None:
    """Drain the outbound queue to the socket. The only task that sends."""
    while True:
        frame = await out.get()
        if frame is _STOP:
            return
        await ws.send_json(wire.to_wire(frame))


@router.websocket("/v1/ws")
async def stream(ws: WebSocket) -> None:
    """Read frames, dispatch them, and close cleanly once the session is done."""
    engine: Engine = ws.app.state.engine
    await ws.accept()

    out: asyncio.Queue = asyncio.Queue()
    writer = asyncio.create_task(_writer(ws, out))
    session: Session | None = None
    closing = False

    try:
        while not closing:
            try:
                raw = await ws.receive_json()
            except WebSocketDisconnect:
                break
            except (ValueError, KeyError, TypeError) as exc:  # not JSON, or not text
                await out.put(wire.Error(error=f"bad frame: {exc}"))
                continue

            try:
                frame = wire.parse_incoming(raw)
            except ValueError as exc:  # unknown key, or a field the model rejects
                await out.put(wire.Error(error=f"bad frame: {exc}"))
                continue

            if isinstance(frame, wire.StartContext):
                if session is not None:
                    await out.put(wire.Error(error="context already started"))
                    continue
                try:
                    engine.check_voice(frame.start_context.voice)
                    engine.check_timestamps(frame.start_context.timestamps)
                    session = Session(engine=engine, start=frame.start_context, out=out)
                    await session.open()
                except Exception as exc:  # noqa: BLE001 - a rejected start is recoverable
                    # The session never opened, so the client can correct the configuration and
                    # send start_context again on the same connection.
                    session = None
                    await out.put(wire.Error(error=str(exc)))
                    continue
            elif session is None:
                await out.put(wire.Error(error="no active context; send start_context first"))
            elif isinstance(frame, wire.SendText):
                session.send_text(frame.send_text)
            elif isinstance(frame, wire.Flush):
                session.flush(frame.flush_id)
            elif isinstance(frame, wire.CloseContext):
                session.close_context(frame.flush_id)
                # Stop reading, but do not close the socket: the producer is still speaking the
                # last of the text, and the writer has to get those frames out first.
                closing = True
    except WebSocketDisconnect:
        log.debug("websocket client disconnected")
    finally:
        if not closing:
            # The reader stopped for some other reason -- a disconnect, an error -- so nothing
            # is going to produce the frames the writer is waiting for. Retire both.
            if session is not None:
                await session.aclose()
            else:
                await out.put(_STOP)
        try:
            # When closing, this ends by itself: the producer emits context_closed and stops.
            await writer
        except Exception:  # noqa: BLE001 - the socket died under the writer, nothing to send
            log.debug("websocket writer stopped early", exc_info=True)
            if session is not None:
                await session.aclose()
        with contextlib.suppress(RuntimeError):
            await ws.close()
