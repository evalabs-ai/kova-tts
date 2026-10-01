"""Locating model artifacts on disk or on the Hugging Face Hub.

Every artifact resolves the same way:

1. the value passed to the function,
2. the matching ``KOVA_*`` environment variable, read from the nearest ``.env`` if one exists,
3. the Hugging Face Hub.

A value that looks like a filesystem path must exist; anything else is handed to the Hub. This
keeps a typo in ``.env`` from silently turning into a multi-gigabyte download.

Set ``KOVA_DISABLE_DOTENV=1`` to skip the ``.env`` lookup.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Hub repository holding the LM, tokenizer and codec checkpoint. Voices are separate.
DEFAULT_HUB_REPO = "kova-ai/kova-tts-1"

#: Codec checkpoint filename within the Hub repository.
CODEC_HUB_FILENAME = "codec.pt"

#: Word-alignment checkpoint filename within the Hub repository.
ALIGNMENT_HUB_FILENAME = "alignment.pt"

#: WavLM is pulled straight from its upstream repository.
DEFAULT_WAVLM_REPO = "microsoft/wavlm-large"

ENV_MODEL = "KOVA_MODEL_PATH"
ENV_CODEC = "KOVA_CODEC_PATH"
ENV_ALIGNMENT = "KOVA_ALIGNMENT_PATH"
ENV_WAVLM = "KOVA_WAVLM_PATH"
ENV_LORA_DIR = "KOVA_LORA_DIR"
ENV_HUB_REPO = "KOVA_HUB_REPO"
ENV_DISABLE_DOTENV = "KOVA_DISABLE_DOTENV"

#: Values of ``KOVA_DISABLE_DOTENV`` that leave the ``.env`` lookup switched on.
_FALSY = frozenset({"", "0", "false", "no", "off"})

_dotenv_loaded = False


class MissingArtifact(FileNotFoundError):
    """A configured path does not exist, or a required artifact could not be located."""


def load_dotenv(*, force: bool = False) -> Path | None:
    """Load the nearest ``.env`` into the environment, once per process.

    Existing environment variables win, so an explicit ``KOVA_MODEL_PATH=... python ...``
    always overrides the file. Returns the file that was loaded, or ``None``.
    """
    global _dotenv_loaded
    if _dotenv_loaded and not force:
        return None
    _dotenv_loaded = True

    # Falsy *spellings* count as off, not just an unset variable: `KOVA_DISABLE_DOTENV=0`
    # plainly means "do not disable", and a truthiness test would silently do the opposite.
    if os.environ.get(ENV_DISABLE_DOTENV, "").strip().lower() not in _FALSY:
        return None

    from dotenv import find_dotenv
    from dotenv import load_dotenv as _load

    found = find_dotenv(usecwd=True)
    if not found:
        return None
    _load(found, override=False)
    return Path(found)


def reset_dotenv_cache() -> None:
    """Forget that ``.env`` was loaded. Intended for tests."""
    global _dotenv_loaded
    _dotenv_loaded = False


def _env(name: str) -> str | None:
    load_dotenv()
    value = os.environ.get(name, "").strip()
    return value or None


def _looks_local(value: str) -> bool:
    """True if `value` names a filesystem path rather than a Hub repository id."""
    return value.startswith(("/", "~", ".")) or Path(value).expanduser().exists()


def _require(value: str, *, what: str, kind: str, env_var: str) -> Path:
    """Validate a local path, with an error that says where the value came from."""
    path = Path(value).expanduser()
    if not path.exists():
        raise MissingArtifact(
            f"{what} not found at {path}. Set {env_var} in your .env to a valid path, "
            f"or unset it to download from the Hugging Face Hub."
        )
    if kind == "dir" and not path.is_dir():
        raise MissingArtifact(f"{what} must be a directory, but {path} is a file.")
    if kind == "file" and not path.is_file():
        raise MissingArtifact(f"{what} must be a file, but {path} is a directory.")
    return path


def hub_repo() -> str:
    """Hub repository to fall back on."""
    return _env(ENV_HUB_REPO) or DEFAULT_HUB_REPO


def model_path(explicit: str | os.PathLike[str] | None = None) -> str:
    """Directory holding the LM (``config.json``, weights, tokenizer), or a Hub repo id."""
    value = str(explicit) if explicit is not None else _env(ENV_MODEL)
    if value is None:
        return hub_repo()
    if _looks_local(value):
        return str(_require(value, what="Model directory", kind="dir", env_var=ENV_MODEL))
    return value


def wavlm_path(explicit: str | os.PathLike[str] | None = None) -> str:
    """Directory holding WavLM-large, or a Hub repo id. Only needed to encode audio."""
    value = str(explicit) if explicit is not None else _env(ENV_WAVLM)
    if value is None:
        return DEFAULT_WAVLM_REPO
    if _looks_local(value):
        return str(_require(value, what="WavLM directory", kind="dir", env_var=ENV_WAVLM))
    return value


def codec_path(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Codec checkpoint file, downloading it from the Hub if no local path is configured."""
    value = str(explicit) if explicit is not None else _env(ENV_CODEC)
    if value is None:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(repo_id=hub_repo(), filename=CODEC_HUB_FILENAME))
    return _require(value, what="Codec checkpoint", kind="file", env_var=ENV_CODEC)


def alignment_path(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Word-alignment checkpoint file, downloading it from the Hub if no local path is set."""
    value = str(explicit) if explicit is not None else _env(ENV_ALIGNMENT)
    if value is None:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(repo_id=hub_repo(), filename=ALIGNMENT_HUB_FILENAME))
    return _require(value, what="Alignment checkpoint", kind="file", env_var=ENV_ALIGNMENT)


def lora_dir(explicit: str | os.PathLike[str] | None = None) -> Path | None:
    """Directory of LoRA adapters, one subdirectory per voice. ``None`` when unconfigured."""
    value = str(explicit) if explicit is not None else _env(ENV_LORA_DIR)
    if value is None:
        return None
    return _require(value, what="LoRA directory", kind="dir", env_var=ENV_LORA_DIR)


def available_loras(explicit: str | os.PathLike[str] | None = None) -> list[str]:
    """Names of the LoRA adapters found in the LoRA directory, sorted."""
    root = lora_dir(explicit)
    if root is None:
        return []
    return sorted(p.name for p in root.iterdir() if (p / "adapter_config.json").is_file())


def lora_path(name: str, *, root: str | os.PathLike[str] | None = None) -> Path:
    """Directory of a single named LoRA adapter."""
    base = lora_dir(root)
    if base is None:
        raise MissingArtifact(
            f"No LoRA directory configured, so voice {name!r} cannot be resolved. "
            f"Set {ENV_LORA_DIR} in your .env."
        )
    path = base / name
    if not (path / "adapter_config.json").is_file():
        found = ", ".join(available_loras(base)) or "none"
        raise MissingArtifact(f"No LoRA adapter named {name!r} in {base}. Available: {found}.")
    return path
