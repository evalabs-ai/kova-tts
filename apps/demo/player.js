/**
 * The demo's audio player: 16-bit PCM in, gap-free sound out.
 *
 * Why this exists. Gradio's streaming audio component does not send the samples it is given:
 * it re-encodes every yielded frame to AAC with ffmpeg and serves the result as HLS segments.
 * That is a lossy codec applied to ~390 ms of speech at a time, and each segment carries its
 * own encoder priming, so the joins click. The bytes the model produces are already exactly
 * what an AudioBuffer wants, so this file takes them straight from the server's documented SSE
 * stream and schedules them itself. Nothing is re-encoded anywhere, and consecutive frames abut
 * to the sample.
 *
 * The scheme is the one the production playground uses: decode base64 -> Int16 -> Float32, hand
 * each frame to an AudioBufferSourceNode started at a running cursor, and let the audio clock
 * -- not a timer, and not the network -- decide when each frame is heard.
 *
 * app.py loads this file and runs it once per page load, which installs `window.kovaDemo`. The
 * Speak and Stop buttons are ordinary Gradio buttons whose click handlers are pure JavaScript.
 */

(function () {
    "use strict";

    /** Endpoint the page streams from. app.py sets this; the default is where it mounts it. */
    const STREAM_PATH = window.KOVA_STREAM_PATH || "/v1/tts/stream";

    /** The codec's own rate. Every chunk states its rate; this is only the opening assumption. */
    const CODEC_SAMPLE_RATE = 48000;

    /**
     * Slack between "now" and the first frame's start time. Web Audio will silently drop a
     * source scheduled in the past, and the first frame is scheduled while the main thread is
     * still busy decoding it, so it needs a little of the audio clock to be still ahead of it.
     * 80 ms is inaudible as latency and is comfortably more than one render quantum.
     */
    const START_DELAY_SEC = 0.08;

    /** Longest text the demo accepts, mirrored from app.py only to fail fast in the browser. */
    const MAX_CHARS = window.KOVA_MAX_CHARS || 1200;

    /**
     * Length of speech the text will take, before generation has said otherwise:
     *
     *     seconds ~ 0.049 * characters + 0.70 * sentences + 0.31 * clause breaks - 0.15
     *
     * Fitted on the shipped model over 96 generations -- short replies to long paragraphs, the
     * base voice and two LoRAs, two seeds each. The median error is 6%, which is the model's own
     * seed-to-seed variation on identical text; the worst tenth are within 17%. Every pause is
     * worth more than its one character, which is what the two punctuation terms are for.
     */
    const PACE = { perChar: 0.049, perSentence: 0.70, perClause: 0.31, offset: -0.15 };

    /** Shortest length ever estimated: a word or two is still a second of audio with its pauses. */
    const MIN_ESTIMATE_SEC = 1.0;

    /**
     * While generating, received audio past this fraction of the estimate means the estimate is
     * too short, and it is raised to keep this much headroom. Close to 1 because the estimate is
     * rarely far off: the total then overshoots by at most ~5% before settling on the real one.
     */
    const ESTIMATE_HEADROOM = 0.95;

    /** How far a voice's learned correction may pull an estimate, either way. */
    const CORRECTION_RANGE = [0.6, 1.6];

    /** How quickly the displayed length glides to a new estimate: a time constant, in seconds. */
    const ESTIMATE_GLIDE_SEC = 0.35;

    /** Longest run of words from the text that goes into a download's filename. */
    const FILENAME_WORDS = 5;

    // ------------------------------------------------------------------ decoding and packaging

    /** base64 -> Int16Array. The payload is little-endian, which is what a DataView-free view
     *  of the bytes already gives us on every platform a browser runs on. */
    function base64ToInt16(base64) {
        if (!base64) return new Int16Array(0);
        const binary = atob(base64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
        // slice(0) so the Int16Array owns an aligned buffer of its own.
        return new Int16Array(bytes.buffer.slice(0));
    }

    function int16ToFloat32(int16) {
        const float32 = new Float32Array(int16.length);
        for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768;
        return float32;
    }

    /**
     * Linear resampling, used only when the browser refused to open a context at the codec's
     * rate. Per-frame interpolation error at a frame boundary is far below the noise floor;
     * a whole extra buffering layer to avoid it would not be.
     */
    function resampleLinear(input, fromRate, toRate) {
        if (fromRate === toRate || input.length === 0) return input;
        const ratio = fromRate / toRate;
        const output = new Float32Array(Math.max(1, Math.round(input.length / ratio)));
        for (let i = 0; i < output.length; i++) {
            const position = i * ratio;
            const left = Math.floor(position);
            const right = Math.min(left + 1, input.length - 1);
            const fraction = position - left;
            output[i] = input[left] + (input[right] - input[left]) * fraction;
        }
        return output;
    }

    /** A 44-byte canonical WAV header for `samples` mono 16-bit frames. */
    function wavHeader(samples, sampleRate) {
        const buffer = new ArrayBuffer(44);
        const view = new DataView(buffer);
        const dataSize = samples * 2;
        const ascii = (offset, text) => {
            for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
        };
        ascii(0, "RIFF");
        view.setUint32(4, 36 + dataSize, true);
        ascii(8, "WAVE");
        ascii(12, "fmt ");
        view.setUint32(16, 16, true); // PCM header length
        view.setUint16(20, 1, true); // format: uncompressed PCM
        view.setUint16(22, 1, true); // channels
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true); // byte rate
        view.setUint16(32, 2, true); // block align
        view.setUint16(34, 16, true); // bits per sample
        ascii(36, "data");
        view.setUint32(40, dataSize, true);
        return buffer;
    }

    /** The frames already received, as one playable, downloadable wav. No re-encoding. */
    function wavBlob(chunks, sampleRate) {
        const samples = chunks.reduce((total, chunk) => total + chunk.length, 0);
        const pcm = new Uint8Array(samples * 2);
        let offset = 0;
        for (const chunk of chunks) {
            pcm.set(new Uint8Array(chunk.buffer, chunk.byteOffset, chunk.byteLength), offset);
            offset += chunk.byteLength;
        }
        return new Blob([wavHeader(samples, sampleRate), pcm], { type: "audio/wav" });
    }

    // --------------------------------------------------------------------- the length estimate

    /** `PACE` applied to `text`: its estimated length in seconds, before any correction. */
    function estimateSeconds(text) {
        const sentences = (text.match(/[.!?]+(\s|$)/g) || []).length;
        const clauses = (text.match(/[,;:\u2014\u2013-]\s/g) || []).length;
        const seconds =
            PACE.perChar * text.length +
            PACE.perSentence * sentences +
            PACE.perClause * clauses +
            PACE.offset;
        return Math.max(MIN_ESTIMATE_SEC, seconds);
    }

    /**
     * Actual length over estimated length, learned per voice from the runs this page finished.
     * The installed voices all sit within a few percent of `PACE`; a clone can be well off it,
     * and its second generation should not repeat the first one's misjudgement.
     */
    const correctionByVoice = new Map();

    /**
     * The total length the transport shows: a guess from the text before any audio exists,
     * corrected as audio arrives, and exact once generation ends.
     *
     * The received length alone makes a poor total -- it grows by a frame at a time, so the bar
     * rescales every ~390 ms. Instead the target changes rarely (see ESTIMATE_HEADROOM) and the
     * displayed value glides to it, so a correction is a smooth drift rather than a jump.
     */
    class LengthEstimate {
        constructor() {
            this.reset();
        }

        reset() {
            this.target = 0;
            this.shown = 0;
            this.exact = true;
            this.painted = null;
        }

        /** Guess from the text, before generation starts. */
        begin(text, voice) {
            const correction = correctionByVoice.get(voice || "") || 1;
            this.target = Math.max(MIN_ESTIMATE_SEC, estimateSeconds(text) * correction);
            this.shown = this.target;
            this.exact = false;
            this.painted = null;
        }

        /** Audio has arrived: raise the target only if it is about to be overtaken. */
        received(seconds) {
            if (this.exact) return;
            if (seconds > this.target * ESTIMATE_HEADROOM) {
                this.target = seconds / ESTIMATE_HEADROOM;
            }
        }

        /** Generation is over and `seconds` is all there is; learn from it if it finished. */
        settle(seconds, learn) {
            this.target = seconds;
            this.exact = true;
            if (learn && learn.text && seconds > 0) {
                const key = learn.voice || "";
                const [low, high] = CORRECTION_RANGE;
                const observed = Math.min(high, Math.max(low, seconds / estimateSeconds(learn.text)));
                const previous = correctionByVoice.get(key);
                correctionByVoice.set(key, previous ? (previous + observed) / 2 : observed);
            }
        }

        /** The length to draw this frame, eased towards the target. */
        value() {
            const now = performance.now();
            const dt = this.painted === null ? 0 : (now - this.painted) / 1000;
            this.painted = now;
            this.shown += (this.target - this.shown) * (1 - Math.exp(-dt / ESTIMATE_GLIDE_SEC));
            if (Math.abs(this.target - this.shown) < 0.005) this.shown = this.target;
            return this.shown;
        }

        /** Still gliding, so painting has to continue after playback has stopped. */
        moving() {
            return this.shown !== this.target;
        }
    }

    // ---------------------------------------------------------------------------- the player

    /**
     * Streaming PCM playback over Web Audio.
     *
     * One AudioContext for the life of the page -- contexts are a limited resource and opening
     * one per generation is how a page ends up unable to make sound on the fifth press.
     * `clearPlayback` is what makes a second generation behave exactly like the first: it stops
     * every source still scheduled and puts the cursor back on the clock as it is *now*.
     */
    class StreamingPlayer {
        constructor() {
            this.context = null;
            this.sources = [];
            this.chunks = [];
            this.nextStartTime = 0;
            this.startedAt = null;
            this.sampleRate = CODEC_SAMPLE_RATE;
            this.length = new LengthEstimate();
            this.paused = false;
        }

        /** Open the context, asking for the codec's rate so nothing has to be resampled. */
        init() {
            if (this.context) return this.context;
            const Context = window.AudioContext || window.webkitAudioContext;
            try {
                this.context = new Context({
                    sampleRate: CODEC_SAMPLE_RATE,
                    latencyHint: "interactive",
                });
            } catch (error) {
                // Some browsers refuse a rate the output device cannot do natively. Fine: the
                // frames get resampled on the way in instead.
                this.context = new Context();
            }
            this.nextStartTime = this.context.currentTime;
            return this.context;
        }

        /**
         * Bring the context out of "suspended", which is where every browser starts it until a
         * user gesture says otherwise. Called from the click that starts a generation, and once
         * from the first pointer event anywhere on the page, so the hardware is already awake
         * by the time the first frame lands.
         */
        unlock() {
            const context = this.init();
            if (context.state === "running") return;
            try {
                // A moment of silence keeps the output device from going back to sleep during
                // the wait for the first frame, which on some platforms eats the start of it.
                const warm = context.createBufferSource();
                warm.buffer = context.createBuffer(1, Math.ceil(context.sampleRate / 2), context.sampleRate);
                warm.connect(context.destination);
                warm.start();
            } catch (error) {
                /* not fatal: resume() below is what actually matters */
            }
            const resumed = context.resume();
            if (resumed && resumed.catch) resumed.catch(() => undefined);
        }

        /** Schedule one decoded frame at the cursor, and move the cursor past it. */
        enqueue(base64, sampleRate) {
            const context = this.init();
            const int16 = base64ToInt16(base64);
            if (int16.length === 0) return;
            this.sampleRate = sampleRate || this.sampleRate;
            this.chunks.push(int16);

            const samples = resampleLinear(
                int16ToFloat32(int16),
                this.sampleRate,
                context.sampleRate,
            );
            const buffer = context.createBuffer(1, samples.length, context.sampleRate);
            buffer.getChannelData(0).set(samples);

            const source = context.createBufferSource();
            source.buffer = buffer;
            source.connect(context.destination);

            // max(): the first frame starts a little ahead of the clock, and every frame after
            // it starts exactly where the previous one ended -- unless generation fell behind
            // playback, in which case there is nothing to be done but start again from now.
            const startAt = Math.max(this.nextStartTime, context.currentTime + START_DELAY_SEC);
            source.start(startAt);
            if (this.startedAt === null) this.startedAt = startAt;
            this.nextStartTime = startAt + buffer.duration;
            this.sources.push(source);
            source.onended = () => {
                this.sources = this.sources.filter((queued) => queued !== source);
            };
        }

        /**
         * Hold playback by suspending the context. The audio clock stops with it, so frames that
         * keep arriving meanwhile still line up behind the ones already queued, and resuming
         * carries on from the same sample.
         */
        pause() {
            if (!this.context || this.paused) return;
            this.paused = true;
            this.context.suspend().catch(() => undefined);
        }

        resume() {
            if (!this.context || !this.paused) return;
            this.paused = false;
            this.context.resume().catch(() => undefined);
        }

        /** Stop everything still scheduled, keeping the audio received so far. */
        stopPlayback() {
            for (const source of this.sources) {
                try {
                    source.stop();
                } catch (error) {
                    /* already finished */
                }
            }
            this.sources = [];
        }

        /** Stop, forget, and put the cursor back on the clock: the state a new run starts in. */
        clearPlayback() {
            this.stopPlayback();
            this.paused = false;
            this.chunks = [];
            this.startedAt = null;
            this.nextStartTime = this.context ? this.context.currentTime : 0;
            this.length.reset();
        }

        /** Seconds of audio received so far. */
        duration() {
            const samples = this.chunks.reduce((total, chunk) => total + chunk.length, 0);
            return samples / this.sampleRate;
        }

        /** How far playback has got, in seconds, capped at what has actually been received. */
        position() {
            if (!this.context || this.startedAt === null) return 0;
            const elapsed = this.context.currentTime - this.startedAt;
            return Math.min(this.duration(), Math.max(0, elapsed));
        }

        blob() {
            return wavBlob(this.chunks, this.sampleRate);
        }
    }

    // ------------------------------------------------------------------------------ the page

    const player = new StreamingPlayer();

    /** The run in flight: its abort handle, and enough state for the status line. */
    let active = null;

    /** True once a generation has completed, so the first press can say the model is loading. */
    let modelLoaded = Boolean(window.KOVA_MODEL_LOADED);

    /** The object URL of the finished clip, revoked when the next one replaces it. */
    let clipUrl = null;

    /** The most recent run, kept past its end so a stopped clip can still be named. */
    let lastRun = null;

    /**
     * Which audio the transport is showing and driving. It follows the stream while a run is
     * being generated and heard, and moves to the finished clip -- a hidden <audio> element --
     * as soon as the listener pauses-and-seeks, replays, or stops. From then on the same bar
     * scrubs the clip, so there is only ever one player on the page.
     */
    let onClip = false;

    /** A drag along the bar in progress, and whether to carry on playing when it ends. */
    let dragging = null;

    /** A paint already requested for the next frame, so the loop never runs twice over. */
    let paintQueued = false;

    const element = (id) => document.getElementById(id);

    function setStatus(message) {
        const status = element("kova-status");
        if (status) status.textContent = message;
    }

    function clock(seconds) {
        const whole = Math.max(0, Math.floor(seconds));
        return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
    }

    /** The finished clip can be scrubbed once it exists and nothing is being generated. */
    function seekable() {
        return Boolean(clipUrl) && !active;
    }

    /** Whether the button should offer "pause": sound is coming out, or about to. */
    function playing() {
        if (dragging) return dragging.wasPlaying;
        if (onClip) {
            const clip = element("kova-clip");
            return Boolean(clip && !clip.paused);
        }
        return !player.paused && (Boolean(active) || player.sources.length > 0);
    }

    function schedulePaint() {
        if (paintQueued) return;
        paintQueued = true;
        window.requestAnimationFrame(() => {
            paintQueued = false;
            paint();
        });
    }

    /** Paint the transport: playback position within the (estimated, then exact) total. */
    function paint() {
        const total = Math.max(player.length.value(), player.duration());
        const clip = element("kova-clip");
        const position = onClip && clip ? Math.min(total, clip.currentTime) : player.position();
        const approximate = player.length.exact ? "" : "~";
        const label = `${clock(position)} / ${approximate}${clock(total)}`;
        const isPlaying = playing();

        const fill = element("kova-fill");
        const elapsed = element("kova-elapsed");
        const root = element("kova-player");
        const toggle = element("kova-toggle");
        const seek = element("kova-seek");
        if (fill) fill.style.width = total > 0 ? `${Math.min(100, (100 * position) / total)}%` : "0%";
        if (elapsed) elapsed.textContent = label;
        if (root) {
            root.toggleAttribute("data-playing", isPlaying);
            root.toggleAttribute("data-seekable", seekable());
        }
        if (toggle) {
            toggle.disabled = !(active || player.sources.length > 0 || clipUrl);
            toggle.setAttribute("aria-label", isPlaying ? "Pause" : "Play");
        }
        if (seek) {
            seek.setAttribute("aria-valuemax", total.toFixed(1));
            seek.setAttribute("aria-valuenow", position.toFixed(1));
            seek.setAttribute("aria-valuetext", label);
        }
        // Keep painting until the last scheduled frame has been heard, not until the last one
        // has been received: generation finishes well before playback does.
        const clipPlaying = onClip && clip && !clip.paused;
        if (active || player.sources.length > 0 || player.length.moving() || clipPlaying || dragging) {
            schedulePaint();
        }
    }

    /** Lowercase words joined by hyphens, safe in a filename on every platform. */
    function slug(text, words) {
        return text
            .normalize("NFKD")
            .replace(/[\u0300-\u036f]/g, "")
            .toLowerCase()
            .replace(/[^a-z0-9]+/g, " ")
            .trim()
            .split(" ")
            .filter(Boolean)
            .slice(0, words)
            .join("-");
    }

    /**
     * What a downloaded clip is called: the voice, when it was generated, how it starts, and the
     * seed that reproduces it -- `kova_erika_2026-09-24_14-05-32_the-kettle-clicked_seed42.wav`.
     */
    function clipFilename(run) {
        const pad = (n) => String(n).padStart(2, "0");
        const at = run.at;
        const date = `${at.getFullYear()}-${pad(at.getMonth() + 1)}-${pad(at.getDate())}`;
        const time = `${pad(at.getHours())}-${pad(at.getMinutes())}-${pad(at.getSeconds())}`;
        const parts = [
            "kova",
            slug(run.voice || "base", 4) || "voice",
            `${date}_${time}`,
            slug(run.text, FILENAME_WORDS),
            `seed${run.seed}`,
        ];
        return `${parts.filter(Boolean).join("_")}.wav`;
    }

    /** Hand the finished audio to the hidden <audio> element, for replaying and downloading. */
    function publishClip(run) {
        const clip = element("kova-clip");
        const download = element("kova-download");
        if (!clip || player.chunks.length === 0) return;
        if (clipUrl) URL.revokeObjectURL(clipUrl);
        clipUrl = URL.createObjectURL(player.blob());
        clip.src = clipUrl;
        if (download) {
            download.href = clipUrl;
            if (run) download.download = clipFilename(run);
            download.hidden = false;
        }
        schedulePaint();
    }

    /** Take the previous clip off the transport, so a new run starts with nothing to replay. */
    function retireClip() {
        const clip = element("kova-clip");
        if (clip) {
            clip.pause();
            clip.removeAttribute("src");
            clip.load();
        }
        if (clipUrl) URL.revokeObjectURL(clipUrl);
        clipUrl = null;
        onClip = false;
        const download = element("kova-download");
        if (download) download.hidden = true;
    }

    /**
     * Hand the transport from the stream to the finished clip, at the point the stream had
     * reached -- or from the top, if it had already played to the end.
     */
    function moveToClip(clip, at) {
        if (onClip) return;
        player.stopPlayback();
        onClip = true;
        clip.currentTime = at >= player.duration() - 0.05 ? 0 : at;
    }

    /** The play/pause button. While streaming it holds the stream; after, it drives the clip. */
    function toggle() {
        const clip = element("kova-clip");
        if (!onClip && (active || player.sources.length > 0)) {
            if (player.paused) player.resume();
            else player.pause();
        } else if (clip && clipUrl) {
            if (!clip.paused) {
                clip.pause();
            } else {
                moveToClip(clip, player.position());
                clip.play().catch(() => undefined);
            }
        }
        schedulePaint();
    }

    /** Jump the clip to `seconds`, keeping it playing if it was. */
    function seekTo(seconds) {
        const clip = element("kova-clip");
        if (!clip || !seekable()) return;
        const wasPlaying = playing();
        moveToClip(clip, player.position());
        clip.currentTime = Math.min(player.duration(), Math.max(0, seconds));
        if (wasPlaying && clip.paused) clip.play().catch(() => undefined);
        schedulePaint();
    }

    function seekFraction(event, seek) {
        const box = seek.getBoundingClientRect();
        const fraction = box.width > 0 ? (event.clientX - box.left) / box.width : 0;
        return Math.min(1, Math.max(0, fraction)) * player.duration();
    }

    /** Parse `data: {json}\n\n` events off a fetch body. Named events are ignored: the payload
     *  says what it is, and a `done` or an `error` is unmistakable. */
    async function* readEvents(body) {
        const reader = body.getReader();
        const decoder = new TextDecoder();
        let buffered = "";
        try {
            for (;;) {
                const { done, value } = await reader.read();
                if (done) break;
                buffered += decoder.decode(value, { stream: true });
                let boundary;
                while ((boundary = buffered.indexOf("\n\n")) !== -1) {
                    const block = buffered.slice(0, boundary);
                    buffered = buffered.slice(boundary + 2);
                    const data = block
                        .split("\n")
                        .filter((line) => line.startsWith("data:"))
                        .map((line) => line.slice(5).trim())
                        .join("\n");
                    if (!data) continue;
                    try {
                        yield JSON.parse(data);
                    } catch (error) {
                        /* a half-written event is not worth stopping for */
                    }
                }
            }
        } finally {
            reader.releaseLock();
        }
    }

    /** Read the error envelope the server sends, falling back to the status code. */
    async function describeFailure(response) {
        try {
            const body = await response.json();
            if (body && body.message) return body.message;
        } catch (error) {
            /* not JSON */
        }
        return `The server answered ${response.status}.`;
    }

    /**
     * Generate `options.text` and play it as it arrives.
     *
     * Every run starts by clearing the player, which is the whole of the fix for a second
     * generation not streaming: no source from the previous run is left scheduled, and the
     * cursor starts from the clock rather than from wherever the last run left it.
     */
    async function speak(options) {
        const text = (options.text || "").trim();
        if (!text) {
            setStatus("Type something for the model to say, then press Speak.");
            return;
        }
        if (text.length > MAX_CHARS) {
            setStatus(
                `That is ${text.length.toLocaleString()} characters; this demo generates up to ` +
                    `${MAX_CHARS.toLocaleString()} at a time. Trim it, or use the Python API.`,
            );
            return;
        }

        // A press while a run is in flight replaces it, rather than colliding with it: the
        // abort closes the server's generator, which releases the engine for this request.
        stop({ quiet: true });

        const controller = new AbortController();
        const seed = Number.isFinite(options.seed) && options.seed >= 0
            ? Math.floor(options.seed)
            : Math.floor(Math.random() * 2147483646);
        active = {
            controller,
            seed,
            text,
            voice: options.voice || "",
            at: new Date(),
            chunks: 0,
            firstAudio: null,
        };
        lastRun = active;

        player.unlock();
        player.clearPlayback();
        player.length.begin(text, active.voice);

        retireClip();
        setStatus(modelLoaded ? "Generating..." : "Loading the model, which takes a few seconds...");
        schedulePaint();

        const started = performance.now();
        try {
            const response = await fetch(STREAM_PATH, {
                method: "POST",
                headers: { "content-type": "application/json" },
                body: JSON.stringify({
                    text: text,
                    voice: options.voice || null,
                    seed: seed,
                    sampling: {
                        temperature: options.temperature,
                        top_p: options.top_p,
                        top_k: options.top_k,
                        repetition_penalty: options.repetition_penalty,
                        max_tokens: options.max_tokens,
                    },
                }),
                signal: controller.signal,
            });

            if (!response.ok) {
                setStatus(await describeFailure(response));
                player.length.settle(0);
                active = null;
                return;
            }

            for await (const event of readEvents(response.body)) {
                if (controller.signal.aborted) return;
                if (typeof event.audio === "string") {
                    if (active.firstAudio === null) {
                        active.firstAudio = (performance.now() - started) / 1000;
                        modelLoaded = true;
                    }
                    player.enqueue(event.audio, event.sample_rate);
                    player.length.received(player.duration());
                    active.chunks += 1;
                    setStatus(
                        `First audio in ${active.firstAudio.toFixed(2)} s · ` +
                            `${player.duration().toFixed(1)} s generated...`,
                    );
                } else if (typeof event.message === "string") {
                    setStatus(`That failed: ${event.message}`);
                    player.length.settle(player.duration());
                    active = null;
                    return;
                } else if (typeof event.chunks === "number") {
                    finish(event, started);
                    return;
                }
            }
            // The body ended without a terminal event: the connection dropped mid-generation.
            if (active) {
                setStatus("The connection closed before the model had finished.");
                player.length.settle(player.duration());
                publishClip(active);
                active = null;
            }
        } catch (error) {
            if (controller.signal.aborted) return;
            setStatus(`Generation failed: ${error && error.message ? error.message : error}`);
            player.length.settle(player.duration());
            active = null;
        }
    }

    /** The `done` event: report the run, and hand the finished clip to the transport. */
    function finish(event, started) {
        const run = active;
        active = null;
        if (!run) return;
        const elapsed = (performance.now() - started) / 1000;
        const spoken = event.duration_seconds || player.duration();
        player.length.settle(player.duration(), { text: run.text, voice: run.voice });
        schedulePaint();
        publishClip(run);
        if (run.chunks === 0) {
            setStatus(
                "The model produced no audio for that text. Try rephrasing it, or add some " +
                    "punctuation so it has a sentence to work with.",
            );
            return;
        }
        const speed = elapsed > 0 ? spoken / elapsed : 0;
        setStatus(
            `First audio in ${(run.firstAudio || 0).toFixed(2)} s · ${spoken.toFixed(1)} s of ` +
                `speech in ${elapsed.toFixed(1)} s (${speed.toFixed(1)}× real time) · seed ${run.seed}`,
        );
    }

    /**
     * Stop playing and stop generating. The engine is released by the server as soon as the
     * request is abandoned, so the next press works immediately.
     */
    function stop(options) {
        const quiet = Boolean(options && options.quiet);
        const interrupted = Boolean(active);
        const at = player.position();
        if (active) {
            active.controller.abort();
            active = null;
        }
        player.stopPlayback();
        const clip = element("kova-clip");
        if (clip) clip.pause();
        if (quiet) return;
        // Half a generation is still worth keeping; a finished one has already been published,
        // and re-publishing it would yank the clip out from under a listener who is scrubbing.
        if (interrupted) {
            player.length.settle(player.duration());
            publishClip(lastRun);
        }
        // The transport stays where it stopped, so play picks up from there.
        if (clip && clipUrl) moveToClip(clip, at);
        schedulePaint();
        setStatus("Stopped.");
    }

    // The transport is wired by delegation: Gradio owns the markup and may re-render it.
    document.addEventListener("click", (event) => {
        if (event.target.closest && event.target.closest("#kova-toggle")) toggle();
    });

    document.addEventListener("pointerdown", (event) => {
        const seek = event.target.closest && event.target.closest("#kova-seek");
        const clip = element("kova-clip");
        if (!seek || !clip || !seekable() || event.button !== 0) return;
        dragging = { wasPlaying: playing() };
        moveToClip(clip, player.position());
        clip.pause();
        seek.setPointerCapture(event.pointerId);
        const root = element("kova-player");
        if (root) root.toggleAttribute("data-dragging", true);
        clip.currentTime = seekFraction(event, seek);
        schedulePaint();
    });

    document.addEventListener("pointermove", (event) => {
        const seek = element("kova-seek");
        const clip = element("kova-clip");
        if (dragging && seek && clip) clip.currentTime = seekFraction(event, seek);
    });

    const endDrag = () => {
        if (!dragging) return;
        const clip = element("kova-clip");
        if (dragging.wasPlaying && clip) clip.play().catch(() => undefined);
        dragging = null;
        const root = element("kova-player");
        if (root) root.toggleAttribute("data-dragging", false);
        schedulePaint();
    };
    document.addEventListener("pointerup", endDrag);
    document.addEventListener("pointercancel", endDrag);

    document.addEventListener("keydown", (event) => {
        if (!event.target.closest || !event.target.closest("#kova-seek") || !seekable()) return;
        const clip = element("kova-clip");
        const now = onClip && clip ? clip.currentTime : player.position();
        const jumps = { ArrowLeft: now - 5, ArrowRight: now + 5, Home: 0, End: player.duration() };
        if (event.key in jumps) {
            event.preventDefault();
            seekTo(jumps[event.key]);
        } else if (event.key === " " || event.key === "Enter") {
            event.preventDefault();
            toggle();
        }
    });

    // The clip and the stream must never sound at once, whatever started the clip -- the button,
    // or a media key reaching the hidden element directly.
    for (const type of ["play", "pause", "ended", "seeked", "loadedmetadata"]) {
        document.addEventListener(
            type,
            (event) => {
                if (!event.target || event.target.id !== "kova-clip") return;
                if (type === "play" && !onClip) moveToClip(event.target, player.position());
                schedulePaint();
            },
            true,
        );
    }

    // Browsers keep an AudioContext suspended until a gesture. Speak is a gesture, but priming
    // on the first pointer event means the hardware is awake before the first frame arrives.
    document.addEventListener("pointerdown", () => player.unlock(), { once: true, capture: true });

    window.kovaDemo = { speak, stop, player };
})();
