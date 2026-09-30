"""What the page says: its prose, its example prompts, its stylesheet, and the player's markup.

The look follows kova.ai: a warm cream canvas, near-black ink, one teal accent, pill buttons and
lowercase headings. The palette lives in :data:`CSS` as custom properties, so a colour is
changed in one place.
"""

from __future__ import annotations

from pathlib import Path

TITLE = "Kova TTS"
HEADING = "what should it say?"
TAGLINE = "Sound starts in under half a second — the rest keeps generating while it plays."

READY = "Ready when you are."

BUSY = (
    "The model is already speaking. It generates one clip at a time -- give it a moment and "
    "press Generate again."
)

#: Prompts written for this demo, each showing something: a held pause, a change of register,
#: digits and units read aloud, and a long passage whose later sentences are still being
#: generated while the first are playing -- then everyday lines, the kind anyone might say. The
#: Random button deals them out.
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
    "Hey, are you still up for lunch tomorrow, or should we move it to Friday?",
    "I just realised I've been wearing my jumper inside out all morning.",
    "Honestly, that was the best pizza I've had in ages.",
    "Can you grab some milk on your way home? We're completely out.",
    "No rush at all, just let me know whenever you get a chance.",
    "I was going to go for a run, but then it started raining, so here we are.",
    "Did you see the game last night? I can't believe how that ended!",
    "Sorry I'm late, the traffic was absolutely ridiculous.",
    "We should really plan a trip somewhere warm this winter.",
    "Okay, I'm heading out now. Text me if you need anything.",
]

CLONE_INTRO = (
    "About 15 seconds of your voice is enough. The new voice is selected as soon as it is ready."
)

#: The two ways to give the clone its reference, as the clone panel's switch offers them. Reading
#: the passage needs no transcription: the passage is the transcript.
CLONE_READ = "Read aloud"
CLONE_FREE = "Freestyle or upload"

TRANSCRIPT_PLACEHOLDER = "Record or upload a clip, and what it says appears here."
TRANSCRIPT_HINT = (
    "Transcribed automatically (NVIDIA Parakeet). Check it matches the clip word for word: "
    "cloning continues the reference, so a mismatch garbles the output."
)

PRIVACY = "Clones stay in memory for this session only and are never written to disk."

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

#: The brand's typeface. It is not on Google Fonts, so it arrives through ``<head>``; anyone
#: offline gets Hanken Grotesk from the theme, or the system sans after that.
HEAD = (
    '<link rel="stylesheet" '
    'href="https://api.fontshare.com/v2/css?f[]=general-sans@400,500,600&display=swap">'
)

LOGO_SVG = Path(__file__).with_name("kova-logo.svg").read_text(encoding="utf-8")

#: Stroke icons, drawn at 24 px and scaled by CSS. Buttons that Gradio renders take theirs as a
#: CSS mask (see ``--icon`` in :data:`CSS`), so these are only for markup written here.
ICON_ARROW = (
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 7h10v10"/><path d="M7 17 17 7"/></svg>'
)


def _mask(path: str) -> str:
    """An inline SVG as a CSS ``url()``, for ``mask-image``."""
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' "
        f"stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>{path}</svg>"
    )
    return f'url("data:image/svg+xml;utf8,{svg}")'


_SHUFFLE = _mask(
    "<path d='m18 14 4 4-4 4'/><path d='m18 2 4 4-4 4'/>"
    "<path d='M2 18h1.973a4 4 0 0 0 3.3-1.7l5.454-8.6a4 4 0 0 1 3.3-1.7H22'/>"
    "<path d='M2 6h1.972a4 4 0 0 1 3.6 2.2'/><path d='M22 18h-6.041a4 4 0 0 1-3.3-1.8l-.359-.45'/>"
)
_MIC_PLUS = _mask(
    "<rect x='7' y='3' width='6' height='11' rx='3'/>"
    "<path d='M3.5 11a6.5 6.5 0 0 0 11 4.7M10 18v3M19 5v6M16 8h6'/>"
)
_MIC = _mask(
    "<rect x='9' y='3' width='6' height='11' rx='3'/><path d='M5.5 11a6.5 6.5 0 0 0 13 0M12 18v3'/>"
)
_SQUARE = _mask("<rect x='4' y='4' width='16' height='16' rx='3' fill='black'/>")
_SLIDERS = _mask(
    "<path d='M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12'/><circle cx='16' cy='6' r='2'/>"
    "<circle cx='10' cy='12' r='2'/><circle cx='18' cy='18' r='2'/>"
)
_PLAY = _mask("<path d='M7 4.5v15l12-7.5z' fill='black'/>")
_REFRESH = _mask("<path d='M20 11a8 8 0 1 0-2.3 5.7M20 4v7h-7'/>")
_CLOSE = _mask("<path d='M6 6l12 12M18 6 6 18'/>")
_LOCK = _mask(
    "<rect x='5' y='11' width='14' height='10' rx='2'/><path d='M8 11V7a4 4 0 0 1 8 0v4'/>"
)

CSS = f"""
:root, .dark {{
  --kova-canvas: #f5f1ea;
  --kova-surface: #faf8f4;
  --kova-sunken: #f1ece3;
  --kova-ink: #2a2826;
  --kova-ink-2: #4a4844;
  --kova-muted: #6b6862;
  --kova-line: #d9d2c7;
  --kova-line-soft: #e4ddd2;
  --kova-taupe: #c4bdb0;
  --kova-teal: #0f8f86;
  --kova-teal-strong: #0b746c;
  --kova-teal-deep: #08453f;
  --kova-mint: #e6eee8;
  --kova-mint-line: #bcd2cc;
  --kova-red: #d52b1e;
  --kova-shadow: 0 18px 50px -20px rgba(42, 40, 38, 0.18);
  --kova-sans: "General Sans", "Hanken Grotesk", system-ui, -apple-system, sans-serif;
  --kova-mono: "JetBrains Mono", ui-monospace, SFMono-Regular, monospace;
}}

footer {{ display: none !important; }}
body, gradio-app, .gradio-container, .main, .contain {{
  background: var(--kova-canvas) !important; color: var(--kova-ink); font-family: var(--kova-sans);
}}
.gradio-container {{ max-width: 100% !important; padding: 0 !important; }}
.gradio-container .main, .gradio-container .main > .wrap, .gradio-container .contain {{
  padding: 0 !important;
}}
/* gr.HTML pads its content even with padding=False; our markup does its own spacing. */
.gradio-container .html-container {{ padding: 0 !important; }}

/* ------------------------------------------------------------------------- header */
#kova-header {{
  display: flex; align-items: center; gap: 16px; padding: 22px 40px; flex-wrap: wrap;
}}
#kova-header .kova-logo svg {{ display: block; height: 24px; width: auto; }}
#kova-header .kova-badge {{
  border: 1px solid var(--kova-line); border-radius: 999px; padding: 2px 10px;
  font-size: 11px; font-weight: 600; letter-spacing: 0.04em; color: var(--kova-muted);
}}
#kova-header .kova-spacer {{ flex: 1; }}
#kova-header .kova-machine {{
  display: inline-flex; align-items: center; gap: 8px; padding: 6px 12px;
  border: 1px solid var(--kova-line); border-radius: 999px; font-size: 13px;
  color: var(--kova-muted); font-family: var(--kova-mono);
}}
#kova-header .kova-machine::before {{
  content: ""; width: 8px; height: 8px; border-radius: 50%; background: var(--kova-teal);
}}
#kova-header .kova-docs {{
  display: inline-flex; align-items: center; gap: 6px; min-height: 36px; padding: 0 16px;
  border-radius: 999px; background: var(--kova-mint); border: 1px solid var(--kova-mint-line);
  color: var(--kova-teal-deep); font-size: 14px; font-weight: 500; text-decoration: none;
}}
#kova-header .kova-docs svg {{
  width: 14px; height: 14px; fill: none; stroke: currentColor; stroke-width: 2.5;
  stroke-linecap: round; stroke-linejoin: round;
}}

/* ------------------------------------------------------------------------ layout */
#kova-main {{
  width: 100%; max-width: 808px; margin: 0 auto; padding: 24px 24px 64px;
  box-sizing: border-box; gap: 24px !important;
}}
#kova-intro h1 {{
  margin: 0 0 6px; font-size: clamp(1.75rem, 4vw, 2.25rem); line-height: 1.1; font-weight: 600;
  letter-spacing: -0.028em; color: var(--kova-ink); text-transform: lowercase;
}}
#kova-intro p {{ margin: 0; font-size: 16px; line-height: 1.6; color: var(--kova-muted); }}
#kova-banner {{
  background: #fbeee4; border: 1px solid #efcfb8; border-radius: 16px; padding: 12px 18px;
  color: var(--kova-ink-2); font-size: 14px;
}}
#kova-banner p {{ margin: 4px 0; }}

/* Every Gradio block inside our cards goes flat: the cards draw the only borders. */
#kova-main .block, #kova-main .form, #kova-main .gr-group, #kova-main .styler {{
  background: transparent !important; border: none !important; box-shadow: none !important;
  padding: 0 !important; border-radius: 0 !important;
}}

/* ---------------------------------------------------------------------- composer */
#kova-composer {{
  background: var(--kova-surface) !important; border: 1px solid var(--kova-line) !important;
  border-radius: 20px !important; box-shadow: var(--kova-shadow) !important;
  overflow: hidden; gap: 0 !important;
}}
#kova-text textarea {{
  background: transparent !important; border: none !important; box-shadow: none !important;
  padding: 20px 22px 8px !important; font-family: var(--kova-sans) !important;
  font-size: 18px !important; line-height: 1.55 !important; color: var(--kova-ink) !important;
  min-height: 150px;
}}
#kova-text textarea::placeholder {{ color: var(--kova-muted); opacity: 0.7; }}
#kova-toolbar {{
  display: flex; align-items: center; flex-wrap: wrap; gap: 10px !important;
  padding: 12px 14px 14px 12px; border-top: 1px solid var(--kova-line-soft);
}}
#kova-meta {{
  display: flex; align-items: center; justify-content: space-between; gap: 8px !important;
  padding: 0 22px 6px 12px;
}}
#kova-meta > *, #kova-toolbar > * {{
  flex: none !important; min-width: 0 !important; width: auto !important;
}}
#kova-toolbar > #kova-settings {{ margin-left: auto; }}
#kova-count {{ font-size: 13px; color: var(--kova-muted); font-variant-numeric: tabular-nums; }}
#kova-count[data-over] {{ color: var(--kova-red); }}

/* Buttons: pills, with icons drawn by a mask so they take the text colour. */
#kova-main button.kova-btn {{
  display: inline-flex; align-items: center; justify-content: center; gap: 6px;
  min-height: 44px; min-width: 0; padding: 0 16px; border-radius: 999px !important;
  font-family: var(--kova-sans); font-size: 14px; font-weight: 500; box-shadow: none !important;
  border: 1px solid var(--kova-line) !important; background: var(--kova-surface) !important;
  color: var(--kova-ink-2) !important; cursor: pointer; transition: background 150ms, color 150ms;
  flex: none !important; flex-direction: row !important; white-space: nowrap;
  width: auto !important; max-width: none !important; height: 44px; align-self: center;
}}
#kova-main button.kova-btn:hover {{ background: #ede6dc !important; }}
#kova-main button.kova-btn::before {{
  content: ""; flex: none; width: 16px; height: 16px; background: currentColor;
  -webkit-mask: var(--icon) center / contain no-repeat;
  mask: var(--icon) center / contain no-repeat;
}}
#kova-main button.kova-btn.kova-plain {{
  border-color: transparent !important; background: transparent !important;
  color: var(--kova-muted) !important; font-size: 13px; padding: 0 10px;
}}
#kova-main button.kova-btn.kova-plain:hover {{ color: var(--kova-teal) !important; }}
/* The label stays for screen readers at zero size; no gap, or it still pushes the icon aside. */
#kova-main button.kova-btn.kova-icon-only {{
  width: 44px !important; min-width: 44px !important; padding: 0; font-size: 0; gap: 0;
}}
#kova-main button.kova-btn.kova-icon-only::before {{ width: 18px; height: 18px; }}
#kova-main button.kova-btn.kova-on {{
  background: var(--kova-mint) !important; border-color: var(--kova-mint-line) !important;
  color: var(--kova-teal-deep) !important;
}}
#kova-main button.kova-btn.kova-primary {{
  background: var(--kova-teal-strong) !important; border-color: var(--kova-teal-strong) !important;
  color: #fff !important; font-size: 15px; font-weight: 600; padding: 0 22px;
}}
#kova-main button.kova-btn.kova-primary:hover {{ background: var(--kova-teal-deep) !important; }}
#kova-main button.kova-btn.kova-dark {{
  background: var(--kova-ink) !important; border-color: var(--kova-ink) !important;
  color: #fff !important; font-size: 15px; font-weight: 600; padding: 0 20px;
}}
#kova-main button.kova-btn.kova-no-icon::before {{ display: none; }}
#kova-random {{ --icon: {_SHUFFLE}; }}
#kova-record-more {{ --icon: {_MIC_PLUS}; }}
#kova-settings {{ --icon: {_SLIDERS}; }}
#kova-generate {{ --icon: {_PLAY}; }}
#kova-another {{ --icon: {_REFRESH}; }}
#kova-clone-close {{ --icon: {_CLOSE}; }}
#kova-generate::before {{ width: 12px !important; height: 12px !important; }}

/* Segmented switches -- the voice source, the clone panel's mode: one pill per option, the
   picked one raised. The options' .wrap is the one holding the labels; Gradio's status tracker
   is a .wrap too. Gradio centres a container-less block with auto margins, which in the toolbar
   would split the free space with the settings button's and float the switch off the left. */
#kova-main .kova-switch {{ margin: 0 !important; padding: 0 !important; }}
#kova-main .kova-switch .wrap:has(> label) {{
  position: relative;
  display: flex; flex-wrap: nowrap; gap: 2px !important; height: 44px; box-sizing: border-box;
  padding: 3px; border: 1px solid var(--kova-line); border-radius: 999px;
  background: var(--kova-sunken);
}}
#kova-main .kova-switch label {{
  display: flex; align-items: center; justify-content: center; position: relative;
  margin: 0 !important; padding: 0 14px !important; border: none !important;
  border-radius: 999px !important; background: transparent !important; box-shadow: none !important;
  font-family: var(--kova-sans); font-size: 13px; font-weight: 500; white-space: nowrap;
  color: var(--kova-muted) !important; cursor: pointer; transition: background 150ms, color 150ms;
}}
#kova-main .kova-switch label:hover {{ color: var(--kova-ink) !important; }}
#kova-main .kova-switch label.selected,
#kova-main .kova-switch .kova-thumb {{
  background: var(--kova-surface) !important; color: var(--kova-ink) !important;
  box-shadow: 0 1px 3px rgba(42, 40, 38, 0.14) !important;
}}
/* switch.js slides one thumb behind the labels onto the picked one; once it is in, the picked
   label leaves the raised look to it. */
#kova-main .kova-switch .kova-thumb {{
  position: absolute; top: 0; left: 0; z-index: 0; pointer-events: none;
  border-radius: 999px; opacity: 0;
  transition-property: transform, width, height; transition-duration: 280ms;
  transition-timing-function: cubic-bezier(0.3, 0.7, 0.3, 1);
}}
#kova-main .kova-switch .wrap:has(> label)[data-sliding] label.selected {{
  background: transparent !important; box-shadow: none !important;
}}
#kova-main .kova-switch label {{ z-index: 1; }}
@media (prefers-reduced-motion: reduce) {{
  #kova-main .kova-switch .kova-thumb {{ transition: none; }}
}}
/* The radio itself stays in the page for keyboards and screen readers, just not on it. */
#kova-main .kova-switch input[type=radio] {{
  position: absolute; opacity: 0; width: 1px; height: 1px; margin: 0;
}}
#kova-main .kova-switch label:has(input:focus-visible) {{ outline: 2px solid var(--kova-teal); }}
#kova-main .kova-switch label span {{ margin: 0 !important; }}

#kova-voice {{ width: 320px !important; margin: 0 !important; }}
#kova-voice .wrap {{
  height: 44px; min-height: 0; box-sizing: border-box; padding: 0 8px 0 14px;
  border: 1px solid var(--kova-line) !important; border-radius: 999px !important;
  background: var(--kova-surface) !important;
}}
#kova-voice .wrap-inner {{ padding: 0 !important; height: 100%; }}
/* The arrow is laid over the input's right end: keep the text, and its ellipsis, clear of it. */
#kova-voice .icon-wrap {{ right: 0 !important; }}
#kova-voice input {{
  text-overflow: ellipsis; padding-right: 26px !important;
  background: transparent !important; font-family: var(--kova-sans); font-size: 14px;
  color: var(--kova-ink);
}}

/* Under the toolbar: which voice within the source. Its own line, so the toolbar above keeps
   one shape whatever the source. */
#kova-voice-row {{
  border-top: 1px solid var(--kova-line-soft) !important; padding: 12px 14px 12px 22px !important;
  gap: 14px !important; align-items: center; flex-wrap: nowrap;
}}
#kova-voice-row > * {{ flex: none !important; min-width: 0 !important; width: auto !important; }}
.kova-row-label {{ font-size: 13px; font-weight: 500; color: var(--kova-muted); }}

/* Then a preset's reference clip, or the nudge to record a first voice. */
#kova-preview, #kova-empty-clones, #kova-empty-loras {{
  border-top: 1px solid var(--kova-line-soft) !important; padding: 12px 22px 14px !important;
  gap: 16px !important; align-items: center;
}}
#kova-preset-audio {{ flex: 0 0 280px !important; }}
/* The transcript gets a card as tall as the player beside it: a heading in its top-left corner
   and the quote under it. The card is the markdown's own wrapper, because Gradio's
   hide-container wins on the block itself. */
#kova-preset-text {{
  flex: 1 1 0 !important; align-self: stretch; display: flex !important; flex-direction: column;
  font-size: 14px; color: var(--kova-ink-2);
}}
#kova-preset-text [data-testid="markdown-wrapper"] {{
  flex: 1 1 auto; padding: 14px 18px; border-radius: 14px; background: var(--kova-sunken);
}}
#kova-preset-text p {{ margin: 0; line-height: 1.55; }}
/* The heading, in the clone panel's "Read aloud" style. */
#kova-preset-text p:first-child {{
  margin-bottom: 8px; font-size: 12px; font-weight: 600; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--kova-muted);
}}
.kova-hint {{ margin: 0; font-size: 14px; color: var(--kova-muted); }}

/* --------------------------------------------------------------- clone + settings */
#kova-clone {{
  background: var(--kova-sunken) !important; border-top: 1px solid var(--kova-line-soft) !important;
  padding: 20px 22px 22px !important; gap: 16px !important;
}}
.kova-panel-head {{ display: flex; align-items: flex-start; gap: 12px; }}
.kova-panel-head h2 {{
  margin: 0; font-size: 15px; font-weight: 600; color: var(--kova-ink); text-transform: lowercase;
}}
.kova-panel-head p {{
  margin: 4px 0 0; font-size: 13px; line-height: 1.5; color: var(--kova-muted);
}}
#kova-clone-row {{ gap: 16px !important; align-items: stretch; }}
/* The mode switch hugs its two options rather than spanning the panel. */
#kova-clone-mode {{ align-self: flex-start; width: fit-content !important; }}
#kova-script-card, #kova-transcript-card, #kova-record-card {{
  background: var(--kova-surface) !important; border: 1px solid var(--kova-line) !important;
  border-radius: 14px !important; padding: 14px 16px !important; gap: 10px !important;
}}
/* The script card stretches to the whole right-hand side: the recording card and the name row. */
#kova-record-side {{ gap: 12px !important; }}
#kova-record-side > #kova-record-card {{ flex: 1 1 auto !important; }}
#kova-name-row {{ gap: 10px !important; align-items: center; flex-wrap: nowrap; }}
#kova-name-row > #kova-clone-name {{ flex: 1 1 0 !important; min-width: 0 !important; }}
/* Nothing to clear before anything is recorded: Gradio shows its Clear button regardless. A
   clip brings a download link into the same corner, which is what tells the two states apart. */
#kova-reference .icon-button-wrapper:not(:has([data-testid="download-link"])) {{
  display: none !important;
}}
/* One stage for every state -- ready to record, recording, recorded, or waiting for a file -- so
   the card keeps its size as it moves between them: the body takes whatever height the tallest
   state needs, centred, with the source switch pinned beneath. Gradio clips the block, which cut
   the switch's bottom edge off; nothing here needs clipping. */
#kova-reference {{ overflow: visible !important; flex: none !important; }}
/* Upload mode floats the "Your recording" label over the drop area; keep it above, as elsewhere. */
#kova-reference > label.float {{ position: static !important; }}
#kova-reference .audio-container {{
  display: flex; flex-direction: column; min-height: 232px; gap: 10px;
}}
#kova-reference .audio-container > .component-wrapper,
#kova-reference .audio-container > button.center {{
  flex: 1 1 auto; display: flex; flex-direction: column; justify-content: center;
  height: auto !important; min-height: 0 !important;
}}
#kova-reference .audio-container > button.center > .wrap {{
  min-height: 0 !important; height: auto !important;
}}
#kova-reference .audio-container > button.center {{ color: var(--kova-muted); font-size: 14px; }}
/* While recording, wavesurfer's cursor sits at the start and the left-hand clock at 0:00 for the
   whole take, so it reads as frozen. Both go, and the running length becomes a recording light:
   a pulsing red dot and the time, centred. It holds still while paused. */
#kova-reference .microphone > div::part(cursor) {{ display: none; }}
/* On playback the cursor does move; it is the teal of the heard part of the waveform. */
#kova-reference div::part(cursor), #kova-preset-audio div::part(cursor) {{
  background: var(--kova-teal);
}}
#kova-reference .component-wrapper:has(.microphone) .timestamps {{ justify-content: center; }}
#kova-reference .component-wrapper:has(.microphone) .timestamps .time {{ display: none; }}
#kova-reference .component-wrapper:has(.microphone) .timestamps .duration {{
  display: inline-flex; align-items: center; gap: 8px;
  font-family: var(--kova-mono); font-size: 14px; color: var(--kova-ink);
}}
#kova-reference .component-wrapper:has(.microphone) .timestamps .duration::before {{
  content: ""; width: 8px; height: 8px; border-radius: 50%; background: var(--kova-red);
  animation: kova-recording 1.2s ease-in-out infinite;
}}
/* Paused is when Gradio hides the live Stop -- the paused one is always in the page. */
#kova-reference .component-wrapper:has(.stop-button[style*="none"]) .timestamps .duration::before {{
  animation: none; opacity: 0.35;
}}
@keyframes kova-recording {{ 50% {{ opacity: 0.2; }} }}
@media (prefers-reduced-motion: reduce) {{
  #kova-reference .timestamps .duration::before {{ animation: none !important; }}
}}
/* Recording uses the browser's default microphone: no device picker. */
#kova-reference .mic-select {{ display: none !important; }}
#kova-reference .controls:has(.record-button) {{ justify-content: center !important; }}
/* Record, Stop, Pause and Resume as the page's own pills, not Gradio's buttons with a teal dot.
   Gradio shows and hides them itself, so nothing here sets their display -- which is also why
   Resume, not a flex box, stays text alone. Pausing swaps Stop for a differently-classed twin,
   which looks exactly the same here. */
#kova-reference .record-button, #kova-reference .stop-button,
#kova-reference .stop-button-paused, #kova-reference .pause-button,
#kova-reference .resume-button {{
  align-items: center; justify-content: center; gap: 8px; width: auto; min-width: 0;
  height: 40px; padding: 0 18px !important; border-radius: 999px !important;
  border: 1px solid var(--kova-line) !important; background: var(--kova-surface) !important;
  font-family: var(--kova-sans); font-size: 14px; font-weight: 500; color: var(--kova-ink);
}}
#kova-reference .record-button:hover, #kova-reference .stop-button:hover,
#kova-reference .stop-button-paused:hover, #kova-reference .pause-button:hover,
#kova-reference .resume-button:hover {{ background: #ede6dc !important; }}
#kova-reference .pause-button {{ width: 40px; padding: 0 !important; color: var(--kova-ink-2); }}
/* Centred on its own: the pause button is not a flex box, so the icon would sit at the left. */
#kova-reference .pause-button svg {{
  display: block; width: 14px; height: 14px; margin: 0 auto; fill: currentColor;
}}
#kova-reference .record-button::before, #kova-reference .stop-button::before,
#kova-reference .stop-button-paused::before {{
  flex: none; width: 16px !important; height: 16px !important; margin: 0 !important;
  border-radius: 0 !important; background: currentColor !important; animation: none !important;
  -webkit-mask: {_MIC} center / contain no-repeat; mask: {_MIC} center / contain no-repeat;
}}
#kova-reference .stop-button, #kova-reference .stop-button-paused {{
  color: var(--kova-red); border-color: var(--kova-red) !important;
}}
#kova-reference .stop-button::before, #kova-reference .stop-button-paused::before {{
  width: 12px !important; height: 12px !important;
  -webkit-mask-image: {_SQUARE}; mask-image: {_SQUARE};
}}
/* Gradio's two bare source icons become a labelled switch, in the voice switch's look, so both
   ways in -- a file or the microphone -- are plain from the start. */
#kova-reference .source-selection {{
  display: flex !important; flex: none; gap: 2px; width: fit-content; margin: 0 auto; padding: 3px;
  border: 1px solid var(--kova-line); border-radius: 999px; background: var(--kova-sunken);
}}
#kova-reference .source-selection button {{
  display: flex; align-items: center; gap: 6px; width: auto !important; height: 32px;
  margin: 0; padding: 0 12px; border-radius: 999px; background: transparent;
  font-family: var(--kova-sans); font-size: 13px; font-weight: 500; color: var(--kova-muted);
}}
#kova-reference .source-selection button::after {{ content: attr(aria-label); }}
#kova-reference .source-selection button svg {{ width: 14px; height: 14px; }}
#kova-reference .source-selection button:hover {{ color: var(--kova-ink); }}
#kova-reference .source-selection button.selected {{
  background: var(--kova-surface); color: var(--kova-ink);
  box-shadow: 0 1px 3px rgba(42, 40, 38, 0.14);
}}
.kova-eyebrow {{
  font-size: 12px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--kova-muted);
}}
#kova-script textarea {{
  background: transparent !important; border: none !important; box-shadow: none !important;
  padding: 0 !important; font-family: var(--kova-sans) !important; font-size: 16px !important;
  line-height: 1.55 !important; color: var(--kova-ink) !important; resize: none;
}}
#kova-clone-name input, #kova-clone-name textarea, #kova-transcript textarea {{
  background: #fff !important; border: 1px solid var(--kova-line) !important;
  border-radius: 12px !important; font-family: var(--kova-sans); font-size: 15px;
}}
/* The freestyle transcript fills its card, like the passage it stands in for: every wrapper
   Gradio puts between the card and the textarea stretches with it. */
#kova-transcript-card > #kova-transcript, #kova-transcript *:has(textarea) {{
  display: flex !important; flex-direction: column; flex: 1 1 auto !important;
}}
#kova-transcript textarea {{ flex: 1 1 auto; line-height: 1.55 !important; resize: none; }}
.kova-note {{ margin: 0; font-size: 12px; line-height: 1.5; color: var(--kova-muted); }}
/* The name sits beside Clone as a pill of the same height. */
#kova-clone-name input, #kova-clone-name textarea {{
  box-sizing: border-box; height: 44px !important; min-height: 44px; padding: 11px 18px !important;
  border-radius: 999px !important; resize: none; overflow: hidden; line-height: 20px;
}}
#kova-clone-row label span {{
  color: var(--kova-ink-2); font-size: 13px;
}}
#kova-clone-status {{ font-size: 14px; color: var(--kova-ink-2); }}
.kova-privacy {{
  display: flex; align-items: center; gap: 8px; margin: 0;
  font-size: 12px; color: var(--kova-muted);
}}
.kova-privacy::before {{
  content: ""; width: 13px; height: 13px; background: currentColor;
  -webkit-mask: {_LOCK} center / contain no-repeat; mask: {_LOCK} center / contain no-repeat;
}}

#kova-settings-panel {{
  background: var(--kova-surface) !important; border: 1px solid var(--kova-line) !important;
  border-radius: 20px !important; padding: 20px 22px !important; gap: 16px !important;
}}
/* Gradio packs each row's sliders into one gapless .form; without room between them, the left
   slider's max runs into the right one's name, and stacked on a phone they touch. */
#kova-settings-panel .form {{ gap: 16px 40px !important; }}
#kova-settings-panel input[type=range] {{ accent-color: var(--kova-teal); }}
#kova-settings-panel label span, #kova-settings-panel .info {{ font-size: 13px; }}
#kova-reset {{ margin-left: auto; }}

/* ------------------------------------------------------------------------- player */
#kova-player {{
  background: var(--kova-surface); border: 1px solid var(--kova-line); border-radius: 20px;
  padding: 18px 22px; display: flex; flex-direction: column; gap: 14px;
}}
#kova-player .kova-track {{ display: flex; align-items: center; gap: 16px; }}
#kova-player .kova-toggle, #kova-player .kova-download {{
  flex: none; display: inline-flex; align-items: center; justify-content: center;
  width: 44px; height: 44px; min-width: 0; margin: 0; padding: 0; border-radius: 50%;
  box-shadow: none; filter: none; opacity: 1; text-decoration: none; cursor: pointer;
  transition: background 120ms, opacity 120ms;
}}
#kova-player .kova-toggle {{ border: none; background: var(--kova-teal-strong); }}
#kova-player .kova-toggle:not(:disabled):hover {{ background: var(--kova-teal-deep); }}
#kova-player .kova-toggle:disabled {{ opacity: 0.35; cursor: default; }}
#kova-player .kova-download {{ border: 1px solid var(--kova-line); background: transparent; }}
#kova-player .kova-download:hover {{ background: #ede6dc; }}
#kova-player .kova-download[hidden] {{ display: none; }}
#kova-player .kova-toggle svg, #kova-player .kova-download svg {{
  flex: none; width: 16px; height: 16px;
}}
#kova-player .kova-toggle svg * {{ fill: #fff; }}
#kova-player .kova-download svg * {{ fill: var(--kova-ink-2); }}
#kova-player .kova-toggle .kova-icon-pause,
#kova-player[data-playing] .kova-toggle .kova-icon-play {{ display: none; }}
#kova-player[data-playing] .kova-toggle .kova-icon-pause {{ display: block; }}
#kova-player .kova-seek {{
  flex: 1; min-width: 0; height: 44px; display: flex; align-items: center; gap: 3px;
  touch-action: none; outline: none; border-radius: 8px;
}}
#kova-player[data-seekable] .kova-seek {{ cursor: pointer; }}
#kova-player .kova-seek:focus-visible {{ box-shadow: 0 0 0 2px rgba(15, 143, 134, 0.4); }}
#kova-player .kova-bar {{
  flex: 1; min-width: 1px; height: 4px; border-radius: 2px; background: var(--kova-line-soft);
  transition: height 120ms ease-out, background 80ms linear;
}}
#kova-player .kova-bar[data-state="ready"] {{ background: var(--kova-taupe); }}
#kova-player .kova-bar[data-state="played"] {{ background: var(--kova-teal); }}
#kova-player[data-dragging] .kova-bar {{ transition: none; }}
#kova-player .kova-elapsed {{
  flex: none; font-family: var(--kova-mono); font-size: 13px; color: var(--kova-muted);
  font-variant-numeric: tabular-nums;
}}
#kova-player .kova-foot {{
  display: flex; align-items: center; gap: 32px; flex-wrap: wrap;
  padding-top: 14px; border-top: 1px solid var(--kova-line-soft);
}}
#kova-player .kova-stats {{ display: flex; gap: 32px; }}
#kova-player .kova-stats[hidden] {{ display: none; }}
#kova-player .kova-stat {{ display: flex; flex-direction: column; gap: 2px; }}
#kova-player .kova-stat span:first-child {{ font-size: 12px; color: var(--kova-muted); }}
#kova-player .kova-stat span:last-child {{
  font-family: var(--kova-mono); font-size: 15px; font-weight: 500; color: var(--kova-ink);
}}
#kova-player .kova-status {{
  margin: 0 0 0 auto; font-size: 13px; color: var(--kova-muted); text-align: right;
}}

@media (max-width: 640px) {{
  #kova-header {{ padding: 16px; gap: 10px; }}
  #kova-header .kova-docs {{ display: none; }}
  #kova-main {{ padding: 8px 16px 40px; }}
  #kova-text textarea {{ padding: 16px 16px 8px !important; font-size: 17px !important; }}
  #kova-meta {{ padding: 0 16px 4px 6px; }}
  #kova-toolbar {{ padding: 12px; }}
  /* Four pills do not fit a phone's width on one line: two by two. */
  #kova-toolbar > #kova-source {{ flex: 1 1 100% !important; width: 100% !important; }}
  #kova-main #kova-source .wrap:has(> label) {{
    display: grid; grid-template-columns: 1fr 1fr; height: auto; border-radius: 22px;
  }}
  #kova-main #kova-source label {{
    height: 36px; padding: 0 8px !important; border-radius: 18px !important;
  }}
  #kova-main #kova-source .kova-thumb {{ border-radius: 18px; }}
  #kova-voice-row {{ padding: 12px 12px 12px 16px !important; }}
  #kova-preview, #kova-empty-clones, #kova-empty-loras {{
    flex-direction: column; align-items: stretch;
  }}
  #kova-preset-audio, #kova-preset-text {{ flex: none !important; }}
  #kova-voice-row {{ flex-wrap: wrap; }}
  #kova-voice-row > #kova-voice {{
    flex: 1 1 0 !important; width: auto !important; min-width: 200px !important;
  }}
  #kova-clone {{ padding: 16px !important; }}
  #kova-clone-row {{ flex-direction: column; }}
  #kova-player {{ padding: 14px 16px; }}
  #kova-player .kova-track {{ gap: 10px; }}
  #kova-player .kova-seek {{ gap: 2px; }}
  #kova-player .kova-bar {{ min-width: 0; }}
  /* Half the bars: at phone width, 72 of them do not fit beside the controls. */
  #kova-player .kova-bar:nth-child(2n) {{ display: none; }}
  #kova-player .kova-elapsed {{ font-size: 12px; }}
  #kova-player .kova-stats {{ gap: 18px; }}
  #kova-player .kova-stat span:last-child {{ font-size: 13px; white-space: nowrap; }}
  #kova-player .kova-status {{ margin: 0; text-align: left; }}
}}
"""


def header_html(summary: str, docs_url: str | None = "/docs") -> str:
    """The top bar: the wordmark, what this machine is running, and a link to the API."""
    docs = (
        f'<a class="kova-docs" href="{docs_url}" target="_blank" rel="noopener">API docs'
        f"{ICON_ARROW}</a>"
        if docs_url
        else ""
    )
    return (
        '<header id="kova-header">'
        f'<a class="kova-logo" href="https://kova.ai" aria-label="Kova">{LOGO_SVG}</a>'
        '<span class="kova-badge">tts demo</span><span class="kova-spacer"></span>'
        f'<span class="kova-machine" title="Device, backend and installed voices">{summary}</span>'
        f"{docs}</header>"
    )


INTRO_HTML = f'<div id="kova-intro"><h1>{HEADING}</h1><p>{TAGLINE}</p></div>'

CLONE_HEAD_HTML = (
    f'<div class="kova-panel-head"><div><h2>clone a voice</h2><p>{CLONE_INTRO}</p></div></div>'
)

SETTINGS_HEAD_HTML = (
    '<div class="kova-panel-head"><div><h2>generation settings</h2>'
    "<p>These start at the preset for the selected voice, which is what the model was tuned "
    "with; switching voice resets them. Most people never need to touch them.</p></div></div>"
)

PRIVACY_HTML = f'<p class="kova-privacy">{PRIVACY}</p>'

#: How many bars the waveform is drawn with. Each covers an equal slice of the clip's length.
WAVE_BARS = 72

#: The player's own markup. Its style is in :data:`CSS`; ``<script>`` set through innerHTML
#: would not run, which is why the JavaScript arrives through a load event instead.
PLAYER_HTML = f"""
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
      {'<span class="kova-bar"></span>' * WAVE_BARS}
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
  <div class="kova-foot">
    <div class="kova-stats" id="kova-stats" hidden>
      <div class="kova-stat"><span>First audio</span><span id="kova-stat-first">–</span></div>
      <div class="kova-stat"><span>Speech</span><span id="kova-stat-speech">–</span></div>
      <div class="kova-stat"><span>Speed</span><span id="kova-stat-speed">–</span></div>
    </div>
    <p class="kova-status" id="kova-status" role="status">{READY}</p>
  </div>
</div>
"""

#: What the Generate button runs. Gradio hands a browser-side handler the values of `inputs`, in
#: order, so this signature is the ``controls`` list in :func:`~ui.build_ui`.
GENERATE_JS = """
(text, voice, temperature, top_p, top_k, repetition_penalty, max_tokens) =>
  window.kovaDemo.speak({
    text, voice, temperature, top_p, top_k, repetition_penalty, max_tokens
  })
"""

#: Keeps the character count under the text box in step, however the text changed.
COUNT_JS = "(text) => { window.kovaDemo && window.kovaDemo.count(text); }"
