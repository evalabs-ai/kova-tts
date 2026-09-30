"""The page itself: one prompt box, its panels and callbacks, and the player wired to the stream.

Everything happens from the prompt box. Random fills it with an example, the voice picker and
New voice sit in its toolbar -- New voice opens the cloning panel right beneath the text -- and
the settings button opens the sampling controls. There are no tabs to go looking in.

Speaking never reaches Python. Gradio's own streaming audio component would take the frames
:meth:`KovaTTS.stream` yields, re-encode each one to AAC with ffmpeg and serve them as HLS
segments -- a lossy codec applied 2.5 times a second, with an encoder priming gap at every
segment join. It is audible, and it is not in the finished clip, so one generation comes out
damaged in the streaming player and clean in the other. The page therefore does not use that
component: the Speak button hands the control values to ``player.js``, which pulls raw
16-bit PCM from :data:`~streaming.STREAM_PATH` and schedules it on a Web Audio clock. Nothing is
re-encoded between the codec and the speakers.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import gradio as gr

from kova_tts import TTS_SAMPLING

from content import (
    CLONE_HEAD_HTML,
    CLONE_SCRIPTS,
    COUNT_JS,
    EXAMPLES,
    INTRO_HTML,
    PLAYER_HTML,
    PRIVACY_HTML,
    SETTINGS_HEAD_HTML,
    SPEAK_JS,
    STOP_JS,
    TITLE,
    TRANSCRIPT_HELP,
    header_html,
)
from session import BASE_VOICE, SOURCE_BASE, SOURCE_CLONE, DemoSession, load_engine
from streaming import MAX_CHARS, STREAM_PATH

#: Classes for the toolbar buttons that open a panel, closed and open.
_TOGGLE = ["kova-btn"]
_TOGGLE_ON = ["kova-btn", "kova-on"]


def build_theme() -> gr.themes.Base:
    """Gradio's side of the kova.ai look; :data:`~content.CSS` does the rest.

    The fonts are the part only a theme can set everywhere, including inside the components the
    stylesheet never names.
    """
    return gr.themes.Base(
        primary_hue=gr.themes.colors.teal,
        neutral_hue=gr.themes.colors.stone,
        radius_size=gr.themes.sizes.radius_lg,
        font=["General Sans", gr.themes.GoogleFont("Hanken Grotesk"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    ).set(
        body_background_fill="#f5f1ea",
        body_background_fill_dark="#f5f1ea",
        body_text_color="#2a2826",
        body_text_color_dark="#2a2826",
        body_text_color_subdued="#6b6862",
        body_text_color_subdued_dark="#6b6862",
        background_fill_primary="#faf8f4",
        background_fill_primary_dark="#faf8f4",
        background_fill_secondary="#f1ece3",
        background_fill_secondary_dark="#f1ece3",
        border_color_primary="#d9d2c7",
        border_color_primary_dark="#d9d2c7",
        block_background_fill="transparent",
        block_background_fill_dark="transparent",
        input_background_fill="#ffffff",
        input_background_fill_dark="#ffffff",
        color_accent="#0f8f86",
        color_accent_soft="#e6eee8",
        color_accent_soft_dark="#e6eee8",
        slider_color="#0f8f86",
        slider_color_dark="#0f8f86",
        button_primary_background_fill="#0b746c",
        button_primary_background_fill_dark="#0b746c",
        button_primary_background_fill_hover="#08453f",
        button_primary_background_fill_hover_dark="#08453f",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        link_text_color="#0b746c",
        link_text_color_dark="#0b746c",
    )


def player_js(*, stream_path: str = STREAM_PATH, model_loaded: bool = False) -> str:
    """``player.js``, wrapped as the browser-side handler for a Gradio load event.

    The settings it reads are written just before it, rather than templated into it, so the file
    stays a plain script that a person can read and a linter could check.
    """
    source = Path(__file__).with_name("player.js").read_text(encoding="utf-8")
    return (
        "() => {\n"
        f"  window.KOVA_STREAM_PATH = {json.dumps(stream_path)};\n"
        f"  window.KOVA_MAX_CHARS = {MAX_CHARS};\n"
        f"  window.KOVA_MODEL_LOADED = {json.dumps(bool(model_loaded))};\n"
        # Gradio re-runs a load handler on every connection; the player is a singleton.
        "  if (window.kovaDemo) return;\n"
        f"{source}\n"
        "}"
    )


def build_ui(
    session: DemoSession | None = None,
    *,
    title: str = TITLE,
    stream_path: str = STREAM_PATH,
    docs_url: str | None = "/docs",
) -> gr.Blocks:
    """Construct the interface, without launching it.

    Args:
        session: The state the callbacks run against. ``None`` builds one that loads the real
            engine from the environment on first use.
        title: Browser tab title.
        stream_path: Where the browser should POST for audio. Only worth changing if these
            Blocks are mounted somewhere other than :func:`~app.build_app` puts them.
        docs_url: Where the header's "API docs" link points, or ``None`` for no link.
            :func:`~app.build_app` serves the endpoint's OpenAPI page at ``/docs``.

    Returns:
        A :class:`gradio.Blocks` ready for ``.launch()``, or for a test to inspect. It is only
        half of a working demo on its own: something has to serve `stream_path`, which is what
        :func:`~app.build_app` is for.

    The theme and stylesheet are not set here: Gradio 6 moved both to ``launch()``, so
    :func:`~app.build_app` applies them (:func:`build_theme` and :data:`~content.CSS`) and a
    caller mounting these Blocks elsewhere picks its own.
    """
    session = session or DemoSession(loader=load_engine)

    # Analytics off: this runs on someone else's machine, against their weights, and Gradio's
    # default is to phone home on launch and on error.
    with gr.Blocks(title=title, fill_width=True, analytics_enabled=False) as ui:
        gr.HTML(header_html(session.summary(), docs_url), padding=False)

        with gr.Column(elem_id="kova-main"):
            gr.HTML(INTRO_HTML, padding=False)
            warnings = session.warnings()
            if warnings:
                gr.Markdown("\n\n".join(warnings), elem_id="kova-banner")

            with gr.Column(elem_id="kova-composer"):
                text = gr.Textbox(
                    show_label=False,
                    container=False,
                    lines=5,
                    max_lines=14,
                    autofocus=True,
                    placeholder=f"Type or paste up to {MAX_CHARS:,} characters…",
                    elem_id="kova-text",
                )
                # Under the text, as on kova.ai: an example to try, and how much room is left.
                with gr.Row(elem_id="kova-meta"):
                    random_button = gr.Button(
                        "Random", elem_id="kova-random", elem_classes=["kova-btn", "kova-plain"]
                    )
                    gr.HTML(
                        f'<span id="kova-count">0 / {MAX_CHARS:,}</span>',
                        padding=False,
                        elem_id="kova-count-box",
                    )
                with gr.Row(elem_id="kova-toolbar"):
                    # The voice is picked in two steps: where it comes from, then which one.
                    source = gr.Dropdown(
                        choices=session.sources(),
                        value=SOURCE_BASE,
                        show_label=False,
                        container=False,
                        label="Voice source",
                        elem_id="kova-source",
                    )
                    # The value the player sends; its choices follow the source. Hidden for
                    # the base model, which has exactly one voice.
                    voice = gr.Dropdown(
                        choices=session.choices(SOURCE_BASE),
                        value=BASE_VOICE,
                        show_label=False,
                        container=False,
                        label="Voice",
                        visible=False,
                        elem_id="kova-voice",
                    )
                    new_voice = gr.Button(
                        "New voice", elem_id="kova-new-voice", elem_classes=_TOGGLE
                    )
                    settings = gr.Button(
                        "Settings",
                        elem_id="kova-settings",
                        elem_classes=[*_TOGGLE, "kova-icon-only"],
                    )
                    speak_button = gr.Button(
                        "Speak", elem_id="kova-speak", elem_classes=["kova-btn", "kova-primary"]
                    )
                    stop_button = gr.Button(
                        "Stop", elem_id="kova-stop", elem_classes=["kova-btn", "kova-dark"]
                    )

                # What the picked voice sounds like, for a zero-shot preset: its reference clip
                # and the transcript it is encoded from.
                with gr.Row(visible=False, elem_id="kova-preview") as preview_row:
                    preset_preview = gr.Audio(
                        label="Reference clip",
                        interactive=False,
                        elem_id="kova-preset-audio",
                    )
                    preset_text = gr.Markdown(elem_id="kova-preset-text")
                # "Your recording" before anything has been recorded.
                with gr.Row(visible=False, elem_id="kova-empty-clones") as clone_hint:
                    gr.HTML(
                        '<p class="kova-hint">No recordings yet — clone one and it will '
                        "appear here.</p>",
                        padding=False,
                    )
                    go_clone = gr.Button(
                        "Record a voice",
                        elem_id="kova-go-clone",
                        elem_classes=["kova-btn", "kova-no-icon"],
                        scale=0,
                    )

                with gr.Column(visible=False, elem_id="kova-clone") as clone_panel:
                    with gr.Row(equal_height=False):
                        gr.HTML(CLONE_HEAD_HTML, padding=False)
                        close_clone = gr.Button(
                            "Close",
                            elem_id="kova-clone-close",
                            elem_classes=["kova-btn", "kova-plain", "kova-icon-only"],
                            scale=0,
                        )
                    with gr.Row(elem_id="kova-clone-row"):
                        with gr.Column(elem_id="kova-script-card"):
                            with gr.Row(equal_height=True):
                                gr.HTML(
                                    '<span class="kova-eyebrow">Read aloud</span>', padding=False
                                )
                                another_script = gr.Button(
                                    "Another",
                                    elem_id="kova-another",
                                    elem_classes=["kova-btn", "kova-plain"],
                                    scale=0,
                                )
                            clone_script = gr.Textbox(
                                value=CLONE_SCRIPTS[0],
                                show_label=False,
                                container=False,
                                lines=4,
                                interactive=False,
                                elem_id="kova-script",
                            )
                        with gr.Column(elem_id="kova-record-card"):
                            reference = gr.Audio(
                                sources=["microphone", "upload"],
                                type="filepath",
                                label="Your recording",
                                elem_id="kova-reference",
                            )
                            with gr.Row(equal_height=True):
                                clone_name = gr.Textbox(
                                    label="Name",
                                    placeholder="My voice",
                                    elem_id="kova-clone-name",
                                )
                                clone_button = gr.Button(
                                    "Clone",
                                    elem_id="kova-clone-button",
                                    elem_classes=["kova-btn", "kova-primary", "kova-no-icon"],
                                    scale=0,
                                )
                    with gr.Accordion(
                        "Using your own recording? Check the transcript",
                        open=False,
                        elem_id="kova-transcript-box",
                    ) as transcript_box:
                        gr.Markdown(TRANSCRIPT_HELP)
                        # Prefilled with the script and still editable, so reading the script
                        # needs no typing while an uploaded clip can have its own transcript.
                        clone_transcript = gr.Textbox(
                            label="Transcript",
                            value=CLONE_SCRIPTS[0],
                            lines=3,
                            info="What the recording says, word for word.",
                            elem_id="kova-transcript",
                        )
                    clone_status = gr.Markdown("", elem_id="kova-clone-status")
                    gr.HTML(PRIVACY_HTML, padding=False)

            with gr.Column(visible=False, elem_id="kova-settings-panel") as settings_panel:
                with gr.Row(equal_height=False):
                    gr.HTML(SETTINGS_HEAD_HTML, padding=False)
                    reset = gr.Button(
                        "Reset to preset",
                        elem_id="kova-reset",
                        elem_classes=["kova-btn", "kova-plain", "kova-no-icon"],
                        scale=0,
                    )
                with gr.Row():
                    temperature = gr.Slider(
                        0.1,
                        1.5,
                        TTS_SAMPLING.temperature,
                        step=0.05,
                        label="Temperature",
                        info="Higher is livelier, lower is steadier.",
                    )
                    top_p = gr.Slider(
                        0.05,
                        1.0,
                        TTS_SAMPLING.top_p,
                        step=0.01,
                        label="Top-p",
                        info="Share of likely sounds considered.",
                    )
                with gr.Row():
                    top_k = gr.Slider(
                        0, 200, TTS_SAMPLING.top_k, step=1, label="Top-k", info="0 turns it off."
                    )
                    repetition_penalty = gr.Slider(
                        1.0,
                        2.0,
                        TTS_SAMPLING.repetition_penalty,
                        step=0.05,
                        label="Repetition penalty",
                        info="Discourages stutters and loops.",
                    )
                max_tokens = gr.Slider(
                    256,
                    4096,
                    TTS_SAMPLING.max_tokens,
                    # Fine enough a step to land on both presets exactly: a coarser one
                    # would snap the cloning budget to a number nothing was tuned with.
                    step=4,
                    label="Token budget",
                    info="80 tokens is one second of audio; this caps a single sentence.",
                )

            gr.HTML(PLAYER_HTML, padding=False)

        clone_open = gr.State(False)
        settings_open = gr.State(False)

        # ----------------------------------------------------------------------- behaviour

        def preset_values(name: str) -> tuple[float, float, int, float, int]:
            preset = session.preset(name)
            return (
                preset.temperature,
                preset.top_p,
                preset.top_k,
                preset.repetition_penalty,
                preset.max_tokens,
            )

        def on_voice_change(name: str) -> tuple[Any, ...]:
            """The sliders follow the voice's preset; a zero-shot preset shows its reference."""
            ref = session.preset_reference(name)
            return (
                *preset_values(name),
                gr.Row(visible=ref is not None),
                gr.update(value=ref[0] if ref else None),
                f"*Reference transcript:* “{ref[1]}”" if ref else "",
            )

        def picker_for(chosen_source: str, value: str | None = None) -> tuple[Any, ...]:
            """The voice picker and the empty-recordings hint, for one voice source."""
            options = session.choices(chosen_source)
            values = [v for _, v in options]
            selected = value if value in values else (values[0] if values else None)
            empty_clones = chosen_source == SOURCE_CLONE and not options
            return (
                gr.update(
                    choices=options,
                    value=selected,
                    visible=chosen_source != SOURCE_BASE and bool(options),
                ),
                gr.Row(visible=empty_clones),
            )

        def on_random(current: str) -> str:
            """An example other than the one already in the box."""
            pool = [prompt for prompt in EXAMPLES if prompt != (current or "").strip()]
            return random.choice(pool or EXAMPLES)

        def toggle_panel(extra: list[str]) -> Any:
            """A click handler that opens a panel if it is closed, and closes it if it is open."""

            def toggle(is_open: bool) -> tuple[bool, Any, Any]:
                now = not is_open
                classes = [*(_TOGGLE_ON if now else _TOGGLE), *extra]
                return now, gr.Column(visible=now), gr.Button(elem_classes=classes)

            return toggle

        def close_clone_panel() -> tuple[bool, Any, Any]:
            return False, gr.Column(visible=False), gr.Button(elem_classes=_TOGGLE)

        def open_clone_panel() -> tuple[bool, Any, Any]:
            return True, gr.Column(visible=True), gr.Button(elem_classes=_TOGGLE_ON)

        def on_clone(*values: Any) -> tuple[Any, ...]:
            message, cloned = session.clone_voice(*values)
            if cloned is None:
                # The panel stays open, with the reason in it, for another try.
                unchanged = (gr.update(),) * 4
                return message, *unchanged, True, gr.update(), gr.update()
            # Land the user where the new voice is usable: panel closed, voice already selected.
            return (
                message,
                gr.update(value=SOURCE_CLONE),
                *picker_for(SOURCE_CLONE, cloned),
                gr.update(),
                False,
                gr.Column(visible=False),
                gr.Button(elem_classes=_TOGGLE),
            )

        def on_reference(path: str | None) -> tuple[Any, Any, Any]:
            """Transcribe a recording the moment it arrives, so the box holds what was said."""
            status, heard = session.transcribe_reference(path)
            if heard is None:
                return status, gr.update(), gr.update()
            # Open the transcript, so it gets checked before Clone is pressed.
            return status, heard, gr.Accordion(open=True)

        def on_another_script(shown: str, typed: str) -> tuple[Any, Any]:
            """Rotate to the next script, keeping the transcript in step with it.

            The transcript only follows along while it still matches the script on display.
            Someone who has pasted their own recording's transcript must not lose it to a
            misplaced click.
            """
            index = CLONE_SCRIPTS.index(shown) if shown in CLONE_SCRIPTS else -1
            nxt = CLONE_SCRIPTS[(index + 1) % len(CLONE_SCRIPTS)]
            follows = (typed or "").strip() == (shown or "").strip()
            return nxt, (nxt if follows else gr.update())

        # The Speak button hands the control values straight to the player; a Gradio event in
        # the middle could only re-encode the audio or hold it back.
        controls = [text, voice, temperature, top_p, top_k, repetition_penalty, max_tokens]
        speak_button.click(None, controls, None, js=SPEAK_JS)
        text.submit(None, controls, None, js=SPEAK_JS)
        stop_button.click(None, None, None, js=STOP_JS)
        text.change(None, text, None, js=COUNT_JS)

        # Opening a panel or dealing an example touches no model, so none of it queues behind
        # a clone in progress.
        instant: dict[str, Any] = {"queue": False, "show_progress": "hidden"}
        random_button.click(on_random, text, text, **instant)
        new_voice.click(
            toggle_panel([]), clone_open, [clone_open, clone_panel, new_voice], **instant
        )
        close_clone.click(close_clone_panel, None, [clone_open, clone_panel, new_voice], **instant)
        settings.click(
            toggle_panel(["kova-icon-only"]),
            settings_open,
            [settings_open, settings_panel, settings],
            **instant,
        )

        sliders = [temperature, top_p, top_k, repetition_penalty, max_tokens]
        voice.change(on_voice_change, voice, [*sliders, preview_row, preset_preview, preset_text])
        # Only a person picking changes the source; on_clone sets it programmatically and fills
        # the picker itself, which .input (unlike .change) leaves alone.
        source.input(picker_for, source, [voice, clone_hint], **instant)
        go_clone.click(open_clone_panel, None, [clone_open, clone_panel, new_voice], **instant)
        reset.click(preset_values, voice, sliders, **instant)
        another_script.click(
            on_another_script,
            [clone_script, clone_transcript],
            [clone_script, clone_transcript],
            **instant,
        )
        clone_button.click(
            on_clone,
            [reference, clone_transcript, clone_name],
            [
                clone_status,
                source,
                voice,
                clone_hint,
                transcript_box,
                clone_open,
                clone_panel,
                new_voice,
            ],
            concurrency_limit=1,
        )
        # Both ways a clip arrives. Not .change: that also fires when the clip is cleared.
        for arrival in (reference.stop_recording, reference.upload):
            arrival(
                on_reference,
                reference,
                [clone_status, clone_transcript, transcript_box],
                show_progress="minimal",
            )

        script = player_js(stream_path=stream_path, model_loaded=session.loaded)
        ui.load(None, None, None, js=script)

    # One clone at a time, and one queue position per visitor: the engine has a single KV cache,
    # so a higher limit would only convert waiting into failing.
    ui.queue(default_concurrency_limit=1)
    return ui
