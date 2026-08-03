#!/bin/sh
# Dispatch for the one image: `serve` (the default), `demo`, or any other kova-tts subcommand.
#
#   docker run ... kova-tts:latest                       # serve on 0.0.0.0:8000
#   docker run ... kova-tts:latest demo                  # Gradio on 0.0.0.0:7860
#   docker run ... kova-tts:latest generate "hi" -o /out/hi.wav
#   docker run ... kova-tts:latest paths                 # what resolved, and what didn't
#   docker run ... kova-tts:latest bash                  # anything else runs verbatim
#
# The one thing this script exists to do is bind to 0.0.0.0. Both servers default to
# 127.0.0.1, which is right on a workstation and useless in a container: a published port
# forwards to the container's external interface, and a process listening only on loopback
# never sees it. So `--host 0.0.0.0` is supplied unless the caller named a host themselves --
# `docker run ... serve --host 127.0.0.1` still means what it says, for anyone deliberately
# running with `--network host`.
#
# Everything is `exec`ed, so the server is PID 1 and `docker stop` reaches it as SIGTERM
# rather than having it swallowed by a shell that is then killed after ten seconds.

set -eu

# True when the arguments already contain `--flag` or `--flag=value`.
has_flag() {
    wanted=$1
    shift
    for arg in "$@"; do
        case "$arg" in
            "$wanted" | "$wanted"=*) return 0 ;;
        esac
    done
    return 1
}

# `docker run --entrypoint kova-entrypoint <image>` arrives here with nothing at all.
[ $# -gt 0 ] || set -- serve

cmd=$1

case "$cmd" in
    serve)
        shift
        has_flag --host "$@" || set -- "$@" --host 0.0.0.0
        # KOVA_PORT is a convenience for compose, where the published port is already written
        # once and repeating it in the command is how the two drift apart.
        has_flag --port "$@" || set -- "$@" --port "${KOVA_PORT:-8000}"
        exec kova-tts serve "$@"
        ;;

    demo)
        shift
        has_flag --host "$@" || set -- "$@" --host 0.0.0.0
        has_flag --port "$@" || set -- "$@" --port "${KOVA_DEMO_PORT:-7860}"
        exec kova-tts demo "$@"
        ;;

    generate | paths | download | prepare-data | finetune | merge)
        exec kova-tts "$@"
        ;;

    kova-tts)
        # Already spelled out in full; do not prefix it a second time.
        shift
        exec kova-tts "$@"
        ;;

    -h | --help | help)
        exec kova-tts --help
        ;;

    *)
        # A shell, a python, a pytest, a one-off script. If it is not on PATH, say which
        # commands exist rather than leaving the caller with /bin/sh's "exec: not found".
        if ! command -v "$cmd" >/dev/null 2>&1; then
            echo "kova-entrypoint: '$cmd' is neither a kova-tts subcommand nor a program in" \
                 "this image. Try one of: serve, demo, generate, paths, download," \
                 "prepare-data, finetune, merge -- or a shell command such as bash." >&2
            exit 127
        fi
        exec "$@"
        ;;
esac
