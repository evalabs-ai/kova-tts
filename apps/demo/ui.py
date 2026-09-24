"""The page itself: two tabs, their callbacks, and the player wired to the streaming endpoint.

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
from pathlib import Path
from typing import Any

import gradio as gr

from kova_tts import TTS_SAMPLING

from content import (
    CLONE_HELP,
    CLONE_SCRIPTS,
    EXAMPLES,
    PLAYER_HTML,
    SPEAK_JS,
    STOP_JS,
    TAGLINE,
    TITLE,
)
from session import BASE_VOICE, DemoSession, load_engine
from streaming import MAX_CHARS, STREAM_PATH


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
) -> gr.Blocks:
    """Construct the interface, without launching it.

    Args:
        session: The state the callbacks run against. ``None`` builds one that loads the real
            engine from the environment on first use.
        title: Browser tab title and page heading.
        stream_path: Where the browser should POST for audio. Only worth changing if these
            Blocks are mounted somewhere other than :func:`~app.build_app` puts them.

    Returns:
        A :class:`gradio.Blocks` ready for ``.launch()``, or for a test to inspect. It is only
        half of a working demo on its own: something has to serve `stream_path`, which is what
        :func:`~app.build_app` is for.

    The theme and stylesheet are not set here: Gradio 6 moved both to ``launch()``, so
    :func:`~app.build_app` applies them and a caller mounting these Blocks elsewhere picks its
    own.
    """
    session = session or DemoSession(loader=load_engine)

    # Analytics off: this runs on someone else's machine, against their weights, and Gradio's
    # default is to phone home on launch and on error.
    with gr.Blocks(title=title, fill_width=False, analytics_enabled=False) as ui:
        gr.Markdown(f"# {title}\n{TAGLINE}")
        gr.Markdown("\n\n".join(f"- {line}" for line in session.notices()))

        with gr.Tabs() as tabs:
            with gr.Tab("Speak", id="speak"):
                with gr.Row():
                    with gr.Column(scale=3):
                        text = gr.Textbox(
                            label="Text",
                            lines=4,
                            max_lines=14,
                            autofocus=True,
                            placeholder="Say something...",
                        )
                    with gr.Column(scale=2):
                        voice = gr.Dropdown(
                            choices=session.choices(),
                            value=BASE_VOICE,
                            label="Voice",
                            info="LoRA voices installed here, plus anything you clone.",
                        )
                        with gr.Row():
                            speak_button = gr.Button("Speak", variant="primary", scale=3)
                            stop_button = gr.Button("Stop", variant="stop", scale=1)

                gr.Examples(
                    examples=[[prompt] for prompt in EXAMPLES],
                    inputs=[text],
                    label="Or try one of these",
                )

                with gr.Accordion("Advanced", open=False):
                    gr.Markdown(
                        "These start at the preset for the selected voice, which is what the "
                        "model was tuned with. The cloning preset differs only in its token "
                        "budget -- switching voice resets all five."
                    )
                    with gr.Row():
                        temperature = gr.Slider(
                            0.1, 1.5, TTS_SAMPLING.temperature, step=0.05, label="Temperature"
                        )
                        top_p = gr.Slider(0.05, 1.0, TTS_SAMPLING.top_p, step=0.01, label="Top-p")
                    with gr.Row():
                        top_k = gr.Slider(
                            0, 200, TTS_SAMPLING.top_k, step=1, label="Top-k", info="0 turns it off"
                        )
                        repetition_penalty = gr.Slider(
                            1.0,
                            2.0,
                            TTS_SAMPLING.repetition_penalty,
                            step=0.05,
                            label="Repetition penalty",
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

            with gr.Tab("Clone a voice", id="clone"):
                gr.Markdown(CLONE_HELP)
                with gr.Row():
                    with gr.Column():
                        reference = gr.Audio(
                            sources=["upload", "microphone"],
                            type="filepath",
                            label="Reference recording",
                        )
                        clone_name = gr.Textbox(
                            label="Name this voice",
                            placeholder="taken from the filename if you leave it empty",
                        )
                    with gr.Column():
                        clone_script = gr.Textbox(
                            label="Read this aloud",
                            value=CLONE_SCRIPTS[0],
                            lines=4,
                            interactive=False,
                        )
                        # Prefilled with the script and still editable, so reading the script
                        # needs no typing while an uploaded clip can have its own transcript.
                        clone_transcript = gr.Textbox(
                            label="Transcript",
                            value=CLONE_SCRIPTS[0],
                            lines=4,
                            info="What the recording says, word for word.",
                        )
                        with gr.Row():
                            another_script = gr.Button("Different script")
                            clone_button = gr.Button("Clone this voice", variant="primary")
                clone_status = gr.Markdown("")

        # ----------------------------------------------------------------------- behaviour

        def on_voice_change(name: str) -> tuple[float, float, int, float, int]:
            preset = session.preset(name)
            return (
                preset.temperature,
                preset.top_p,
                preset.top_k,
                preset.repetition_penalty,
                preset.max_tokens,
            )

        def on_clone(*values: Any) -> tuple[Any, ...]:
            message, cloned = session.clone_voice(*values)
            if cloned is None:
                return message, gr.update(), gr.update()
            # Land the user where the new voice is usable, already selected.
            return (
                message,
                gr.update(choices=session.choices(), value=cloned),
                gr.Tabs(selected="speak"),
            )

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

        voice.change(
            on_voice_change,
            voice,
            [temperature, top_p, top_k, repetition_penalty, max_tokens],
        )
        another_script.click(
            on_another_script,
            [clone_script, clone_transcript],
            [clone_script, clone_transcript],
        )
        clone_button.click(
            on_clone,
            [reference, clone_transcript, clone_name],
            [clone_status, voice, tabs],
            concurrency_limit=1,
        )

        script = player_js(stream_path=stream_path, model_loaded=session.loaded)
        ui.load(None, None, None, js=script)

    # One clone at a time, and one queue position per visitor: the engine has a single KV cache,
    # so a higher limit would only convert waiting into failing.
    ui.queue(default_concurrency_limit=1)
    return ui
