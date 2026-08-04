"""What the page says: its prose, its example prompts, and the player's own markup."""

from __future__ import annotations

TITLE = "Kova TTS"
TAGLINE = "Expressive speech that starts playing before it has finished generating."

READY = "Ready when you are."

BUSY = (
    "The model is already speaking. It generates one clip at a time -- give it a moment and "
    "press Speak again."
)

#: Prompts written for this demo, each showing something: a held pause, a change of register,
#: digits and units read aloud, and a long passage whose later sentences are still being
#: generated while the first are playing.
EXAMPLES = [
    "The kettle clicked off, and for a moment the whole kitchen was completely quiet.",
    "Wait. You're telling me the entire thing runs on one graphics card? That cannot be right.",
    "Take the second left, carry on for about four hundred metres, and if you reach the bridge, "
    "you have gone too far.",
    "Speech comes back in small pieces, eighty of them a second, so the first words are already "
    "playing while the last ones are still being written.",
    "Here is the part I find strange. It does not plan the sentence before it starts talking. "
    "It writes the sound one fragment at a time, left to right, and somehow the pauses still "
    "land where a person would put them, and the question at the end still rises.",
]

CLONE_HELP = """\
**Record yourself reading the script below**, then press Clone. The transcript is already filled
in to match it, so there is nothing else to type and no transcription step. It becomes a voice
you can use on the Speak tab straight away. Nothing is saved to disk: the clone lives in this
session only.

Bringing your own recording instead? Replace the transcript with exactly what it says. Cloning
*continues* the reference, so a transcript that does not match the audio garbles the output.
Leaving it empty transcribes the recording automatically, which needs the `data` extra
(`uv sync --extra data`).
"""

#: Passages to read aloud for a cloning reference, so the transcript is known in advance and no
#: transcription is needed. Each is 10-15 seconds at a normal pace -- long enough to carry a
#: voice, short enough to read without stumbling -- and between them they cover a broad spread
#: of vowels and consonants. Deliberately free of proper nouns and numerals, which are the two
#: things a reader is most likely to say differently from how they are written.
CLONE_SCRIPTS = (
    "The harbour lights came on just after six, and the whole bay turned a shade of orange I "
    "had never seen before. I stood there a while, watching the boats swing round on their "
    "moorings.",
    "She asked me twice whether the package had arrived, and both times I had to say no. It "
    "turned up on Thursday, soaked through, with the label peeling off one corner.",
    "Every question you ask changes the answer a little. That is the strange thing about it "
    "-- you cannot measure something this small without nudging it somewhere else first.",
)

CSS = """
footer { display: none !important; }
"""

#: The player's own markup and style. It travels inside the component rather than through
#: ``launch(css=...)`` so that :func:`~ui.build_ui` embedded in someone else's server still
#: looks like a player. ``<style>`` set through innerHTML applies; ``<script>`` would not run,
#: which is why the JavaScript arrives through a load event instead.
PLAYER_HTML = f"""
<style>
.kova-player {{ margin-top: 0.5rem; }}
.kova-track {{
  display: flex; align-items: center; gap: 0.75rem;
}}
.kova-bar {{
  flex: 1; height: 6px; border-radius: 3px;
  background: var(--neutral-200, #e5e7eb); overflow: hidden;
}}
.kova-fill {{
  height: 100%; width: 0%; border-radius: 3px;
  background: var(--color-accent, #f97316); transition: width 80ms linear;
}}
.kova-elapsed {{ font-variant-numeric: tabular-nums; font-size: 0.85rem; opacity: 0.75; }}
.kova-clip-row {{ display: flex; align-items: center; gap: 0.75rem; margin-top: 0.75rem; }}
.kova-clip-row audio {{ flex: 1; height: 40px; }}
.kova-clip-row a {{ font-size: 0.85rem; white-space: nowrap; }}
.kova-status {{ min-height: 1.6em; margin-top: 0.6rem; font-variant-numeric: tabular-nums; }}
</style>
<div class="kova-player">
  <div class="kova-track">
    <div class="kova-bar"><div class="kova-fill" id="kova-fill"></div></div>
    <span class="kova-elapsed" id="kova-elapsed">0:00 / 0:00</span>
  </div>
  <div class="kova-clip-row" id="kova-clip-row" hidden>
    <audio id="kova-clip" controls preload="metadata"></audio>
    <a id="kova-download" href="#" download="kova-speech.wav" hidden>Download .wav</a>
  </div>
  <p class="kova-status" id="kova-status">{READY}</p>
</div>
"""

#: What the Speak button runs. Gradio hands a browser-side handler the values of `inputs`, in
#: order, so this signature is the ``controls`` list in :func:`~ui.build_ui`.
SPEAK_JS = """
(text, voice, temperature, top_p, top_k, repetition_penalty, max_tokens, seed) =>
  window.kovaDemo.speak({
    text, voice, temperature, top_p, top_k, repetition_penalty, max_tokens, seed
  })
"""

STOP_JS = "() => window.kovaDemo.stop()"
