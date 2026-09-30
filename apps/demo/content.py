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
**Record yourself reading the script below**, then press Clone. It becomes a voice you can use
on the Speak tab straight away (under **Your recording**). Nothing is saved to disk: the clone
lives in this session only.

Bringing your own recording instead? As soon as it is recorded or uploaded, it is transcribed
automatically (NVIDIA Parakeet) and the transcript box fills in with what was said. Check it
before cloning: cloning *continues* the reference, so a transcript that does not match the
audio word for word garbles the output.
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
/* Everything is scoped under #kova-player: an id outranks Gradio's own button, link and
   .prose rules, which would otherwise repaint the controls grey and squash the icons. */
#kova-player {{ margin-top: 0.5rem; }}
#kova-player .kova-track {{ display: flex; align-items: center; gap: 0.75rem; }}
#kova-player .kova-toggle, #kova-player .kova-download {{
  flex: none; display: inline-flex; align-items: center; justify-content: center;
  width: 2rem; height: 2rem; min-width: 0; margin: 0; padding: 0;
  border: none; border-radius: 50%; box-shadow: none; filter: none; opacity: 1;
  text-decoration: none; cursor: pointer; transition: background 120ms, opacity 120ms;
}}
#kova-player .kova-toggle {{ background: var(--color-accent, #3b82f6); }}
#kova-player .kova-toggle:not(:disabled):hover {{ filter: brightness(1.1); }}
#kova-player .kova-toggle:disabled {{ opacity: 0.35; cursor: default; }}
#kova-player .kova-download {{ background: transparent; }}
#kova-player .kova-download:hover {{ background: var(--neutral-100, #f3f4f6); }}
#kova-player .kova-download[hidden] {{ display: none; }}
#kova-player .kova-toggle svg, #kova-player .kova-download svg {{
  flex: none; width: 1rem; height: 1rem;
}}
#kova-player .kova-toggle svg * {{ fill: #fff; }}
#kova-player .kova-download svg * {{ fill: var(--color-accent, #3b82f6); }}
#kova-player .kova-toggle .kova-icon-pause,
#kova-player[data-playing] .kova-toggle .kova-icon-play {{ display: none; }}
#kova-player[data-playing] .kova-toggle .kova-icon-pause {{ display: block; }}
#kova-player .kova-seek {{ flex: 1; padding: 0.5rem 0; touch-action: none; outline: none; }}
#kova-player[data-seekable] .kova-seek {{ cursor: pointer; }}
#kova-player .kova-bar {{
  height: 6px; border-radius: 3px;
  background: var(--neutral-200, #e5e7eb); overflow: hidden;
}}
#kova-player[data-seekable] .kova-seek:hover .kova-bar,
#kova-player .kova-seek:focus-visible .kova-bar {{
  height: 8px; margin: -1px 0; border-radius: 4px;
}}
#kova-player .kova-seek:focus-visible .kova-bar {{
  box-shadow: 0 0 0 2px var(--color-accent, #3b82f6);
}}
#kova-player .kova-fill {{
  height: 100%; width: 0%; border-radius: inherit;
  background: var(--color-accent, #3b82f6); transition: width 80ms linear;
}}
#kova-player[data-dragging] .kova-fill {{ transition: none; }}
#kova-player .kova-elapsed {{
  font-variant-numeric: tabular-nums; font-size: 0.85rem; opacity: 0.75;
}}
#kova-player .kova-status {{
  min-height: 1.6em; margin-top: 0.6rem; font-variant-numeric: tabular-nums;
}}
</style>
<div class="kova-player" id="kova-player">
  <div class="kova-track">
    <button class="kova-toggle" id="kova-toggle" type="button" aria-label="Play" disabled>
      <svg class="kova-icon-play" viewBox="0 0 16 16" aria-hidden="true">
        <path d="M4.5 2.8v10.4a.6.6 0 0 0 .9.5l8.3-5.2a.6.6 0 0 0 0-1L5.4 2.3a.6.6 0 0 0-.9.5z"/>
      </svg>
      <svg class="kova-icon-pause" viewBox="0 0 16 16" aria-hidden="true">
        <rect x="3.5" y="2.5" width="3" height="11" rx="0.8"/>
        <rect x="9.5" y="2.5" width="3" height="11" rx="0.8"/>
      </svg>
    </button>
    <div class="kova-seek" id="kova-seek" role="slider" tabindex="0" aria-label="Seek"
         aria-valuemin="0" aria-valuemax="0" aria-valuenow="0">
      <div class="kova-bar"><div class="kova-fill" id="kova-fill"></div></div>
    </div>
    <span class="kova-elapsed" id="kova-elapsed">0:00 / 0:00</span>
    <a class="kova-download" id="kova-download" href="#" download="kova-speech.wav"
       title="Download .wav" aria-label="Download .wav" hidden>
      <svg viewBox="0 0 16 16" aria-hidden="true">
        <path d="M8 1.5a.75.75 0 0 1 .75.75v6.69l2.22-2.22a.75.75 0 1 1 1.06 1.06l-3.5 3.5a.75.75
          0 0 1-1.06 0l-3.5-3.5a.75.75 0 1 1 1.06-1.06l2.22 2.22V2.25A.75.75 0 0 1 8 1.5zM2.75
          12.5a.75.75 0 0 0 0 1.5h10.5a.75.75 0 0 0 0-1.5H2.75z"/>
      </svg>
    </a>
  </div>
  <!-- The finished clip plays through this element; the transport above is its only face. -->
  <audio id="kova-clip" preload="auto" hidden></audio>
  <p class="kova-status" id="kova-status">{READY}</p>
</div>
"""

#: What the Speak button runs. Gradio hands a browser-side handler the values of `inputs`, in
#: order, so this signature is the ``controls`` list in :func:`~ui.build_ui`.
SPEAK_JS = """
(text, voice, temperature, top_p, top_k, repetition_penalty, max_tokens) =>
  window.kovaDemo.speak({
    text, voice, temperature, top_p, top_k, repetition_penalty, max_tokens
  })
"""

STOP_JS = "() => window.kovaDemo.stop()"
