"""The run configuration: one YAML file, validated into a dataclass.

Every field defaults to a value that trains a good adapter, so a usable config is three lines::

    dataset: data/my_voice/train.jsonl
    voice: my_voice
    output_dir: runs

Everything else -- LoRA rank, schedule, the ending-weight recipe -- only appears in a file when
someone is deliberately changing it, which makes a config a diff against the defaults rather
than a wall of values to check. YAML, so each of those changes can carry the comment saying why.

**Relative paths resolve against the directory containing the YAML file**, so a config can sit
next to its corpus and the pair moves between machines intact. Absolute paths are honoured but
belong in a local file, never in a config committed to a repository.

Unknown keys are an error, not a warning: a typo in ``max_lenght`` that silently trains at the
default would cost a GPU-hour to notice.
"""

from __future__ import annotations

import dataclasses
import difflib
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from kova_tts.finetune.loss import EndingWeight


class ConfigError(ValueError):
    """A config file is missing, malformed, or holds a value that cannot be trained with."""


@dataclass(frozen=True, slots=True)
class LoraSettings:
    """peft ``LoraConfig`` arguments.

    The default is r=64 with alpha=64 (scaling 1.0) on the attention projections only. Leaving
    the MLP and embedding layers frozen is what keeps an adapter at ~55 MB and keeps a voice from
    dragging the base model's pronunciation with it.
    """

    r: int = 64
    alpha: int = 64
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

    #: Extra modules trained in full and saved alongside the adapter. Almost always ``None``.
    modules_to_save: tuple[str, ...] | None = None

    def validate(self) -> None:
        if self.r < 1:
            raise ConfigError(f"lora.r must be >= 1, got {self.r}.")
        if self.alpha <= 0:
            raise ConfigError(f"lora.alpha must be > 0, got {self.alpha}.")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError(f"lora.dropout must be in [0.0, 1.0), got {self.dropout}.")
        if not self.target_modules:
            raise ConfigError(
                "lora.target_modules is empty, so the adapter would have no trainable weights. "
                "The default set is [q_proj, k_proj, v_proj, o_proj]."
            )


@dataclass(frozen=True, slots=True)
class WandbSettings:
    """Weights & Biases logging. Off unless a config turns it on.

    No project name is baked in: an open-source run must not report into someone else's
    workspace by default.
    """

    enabled: bool = False
    project: str | None = None
    entity: str | None = None
    tags: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.enabled and not self.project:
            raise ConfigError(
                "wandb.enabled is true but wandb.project is unset. Set wandb.project to the "
                "project the run should report into, or set wandb.enabled: false."
            )


@dataclass(frozen=True, slots=True)
class FinetuneConfig:
    """A complete, resolved LoRA finetuning run.

    Constructed from YAML by :func:`load_config`; every path is absolute by then and every
    numeric field has passed :meth:`validate`.
    """

    # ------------------------------------------------------------------------------- data
    #: JSONL corpus, one ``{"text": ...}`` object per line. The only required field; it is
    #: optional here only so that an incomplete config can be built and *then* rejected with a
    #: message, rather than dying in the constructor.
    dataset: Path | None = None

    #: Held-out corpus in the same format. When unset, ``val_split`` carves one out.
    val_dataset: Path | None = None

    #: Fraction of the training corpus held out when ``val_dataset`` is unset. 0 disables eval.
    val_split: float = 0.05

    #: Examples longer than this are skipped, not truncated -- truncation would cut off
    #: ``<|speech_end|>`` and teach the model that utterances never end. 4096 tokens is ~51 s.
    max_length: int = 4096

    # ------------------------------------------------------------------------------ model
    #: Base checkpoint: a local directory or a Hub id. Defaults to the configured checkpoint.
    model: str | None = None

    #: Continue training an existing adapter instead of initialising a fresh one. The adapter's
    #: own ``adapter_config.json`` then defines r/alpha/target_modules and ``lora:`` is ignored.
    init_from_adapter: Path | None = None

    #: Resume an interrupted run from a Trainer checkpoint directory.
    resume_from: Path | None = None

    lora: LoraSettings = field(default_factory=LoraSettings)

    # --------------------------------------------------------------------------- schedule
    epochs: float = 4.0
    batch_size: int = 2
    grad_accum: int = 4
    lr: float = 1.5e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.01
    weight_decay: float = 0.01
    seed: int = 42

    #: bf16 throughout. The base checkpoint is bf16; fp16 overflows on this vocabulary.
    bf16: bool = True

    #: Trades ~30% step time for the activation memory that makes 4096-token rows fit.
    gradient_checkpointing: bool = True

    ending_weight: EndingWeight = field(default_factory=EndingWeight)

    # ---------------------------------------------------------------------------- outputs
    #: Parent directory; each run gets a fresh timestamped subdirectory inside it.
    output_dir: Path = Path("runs")

    #: Name of the run subdirectory. Auto-generated (``ft_007_my_voice_2026-08-03_11-04-22``)
    #: when unset.
    run_name: str | None = None

    #: Voice being trained. Used in the run name and written into the adapter directory;
    #: defaults to the dataset's filename stem.
    voice: str | None = None

    logging_steps: int = 10
    eval_steps: int = 500
    save_strategy: str = "epoch"
    save_steps: int = 500
    save_total_limit: int = 4

    wandb: WandbSettings = field(default_factory=WandbSettings)

    # -------------------------------------------------------------------------- behaviour

    def validate(self) -> None:
        """Reject anything that cannot train, naming the key and what to do about it."""
        if self.dataset is None:
            raise ConfigError("'dataset' is required: the path to your JSONL training corpus.")
        if not self.dataset.is_file():
            raise ConfigError(
                f"dataset not found at {self.dataset}. Paths in a config are resolved relative "
                f"to the config file's own directory."
            )
        if self.val_dataset is not None and not self.val_dataset.is_file():
            raise ConfigError(f"val_dataset not found at {self.val_dataset}.")
        if not 0.0 <= self.val_split < 1.0:
            raise ConfigError(
                f"val_split must be in [0.0, 1.0), got {self.val_split}. It is a fraction of the "
                f"corpus, not a percentage; use 0 to train without evaluation."
            )
        if self.max_length < 16:
            raise ConfigError(
                f"max_length must be at least 16 tokens, got {self.max_length}; a real example "
                f"is thousands of tokens (80 audio tokens per second of speech)."
            )
        if self.epochs <= 0:
            raise ConfigError(f"epochs must be > 0, got {self.epochs}.")
        if self.batch_size < 1:
            raise ConfigError(f"batch_size must be >= 1, got {self.batch_size}.")
        if self.grad_accum < 1:
            raise ConfigError(f"grad_accum must be >= 1, got {self.grad_accum}.")
        if not 0.0 < self.lr < 1.0:
            raise ConfigError(
                f"lr must be in (0.0, 1.0), got {self.lr}. The default is 1.5e-4; note that YAML "
                f"parses 1.5e-4 as a string unless the exponent is signed, so prefer 0.00015."
            )
        if not 0.0 <= self.warmup_ratio <= 1.0:
            raise ConfigError(f"warmup_ratio must be in [0.0, 1.0], got {self.warmup_ratio}.")
        if self.weight_decay < 0.0:
            raise ConfigError(f"weight_decay must be >= 0, got {self.weight_decay}.")
        if self.save_strategy not in ("no", "epoch", "steps"):
            raise ConfigError(
                f"save_strategy must be one of no/epoch/steps, got {self.save_strategy!r}."
            )
        if self.logging_steps < 1:
            raise ConfigError(f"logging_steps must be >= 1, got {self.logging_steps}.")
        if self.eval_steps < 1:
            raise ConfigError(f"eval_steps must be >= 1, got {self.eval_steps}.")
        if self.save_steps < 1:
            raise ConfigError(f"save_steps must be >= 1, got {self.save_steps}.")
        if (
            self.init_from_adapter is not None
            and not (self.init_from_adapter / "adapter_config.json").is_file()
        ):
            raise ConfigError(
                f"init_from_adapter has no adapter_config.json: {self.init_from_adapter}. It "
                f"must point at a saved peft adapter directory."
            )
        if self.resume_from is not None and not self.resume_from.is_dir():
            raise ConfigError(f"resume_from is not a directory: {self.resume_from}.")

        self.lora.validate()
        self.wandb.validate()
        try:
            self.ending_weight.validate()
        except ValueError as exc:  # raised as plain ValueError by the loss module
            raise ConfigError(str(exc)) from exc

    @property
    def voice_name(self) -> str:
        """The voice being trained, falling back to the corpus filename."""
        return self.voice or (self.dataset.stem if self.dataset else "voice")

    def to_dict(self) -> dict[str, Any]:
        """YAML-safe plain-data view of the resolved config, for the run snapshot."""
        return _to_plain(self)


# ----------------------------------------------------------------------------------- loading


def load_config(path: str | os.PathLike[str]) -> FinetuneConfig:
    """Read and validate a YAML config. Relative paths resolve against its directory."""
    file = Path(path).expanduser()
    if not file.is_file():
        raise ConfigError(f"Config file not found: {file}")

    import yaml

    try:
        data = yaml.safe_load(file.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{file} is not valid YAML: {exc}") from exc

    if data is None:
        raise ConfigError(f"{file} is empty. A minimal config needs at least a 'dataset:' key.")
    if not isinstance(data, dict):
        raise ConfigError(
            f"{file} must contain a mapping of settings, got {type(data).__name__}. "
            f"A minimal config is 'dataset: path/to/train.jsonl'."
        )

    config = from_dict(data, base_dir=file.parent)
    config.validate()
    return config


def from_dict(data: dict[str, Any], *, base_dir: Path | None = None) -> FinetuneConfig:
    """Build a config from plain data, resolving relative paths against `base_dir`.

    Does not validate -- :func:`load_config` does that. Split out so tests and callers that
    build a config programmatically can check validation independently of YAML parsing.
    """
    root = (base_dir or Path.cwd()).expanduser().resolve()
    config = FinetuneConfig(**_coerce(FinetuneConfig, data, root, prefix=""))

    # Defaults are relative too ("runs"), and a default that silently means "relative to
    # whatever directory the trainer was launched from" is a config that behaves differently
    # depending on the shell's cwd. Anchor every path field to the config file instead.
    anchored = {
        name: _resolve(getattr(config, name), root, name)
        for name in _PATH_FIELDS
        if getattr(config, name) is not None
    }
    return dataclasses.replace(config, **anchored)


def snapshot(config: FinetuneConfig, directory: str | os.PathLike[str]) -> Path:
    """Write the resolved config into a run directory as ``config.yaml``.

    This is the record of what actually ran: defaults filled in, paths absolute. A run whose
    hyperparameters are only knowable from a config file that has since been edited is a run
    that cannot be reproduced.
    """
    import yaml

    target = Path(directory).expanduser() / "config.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return target


# ------------------------------------------------------------------------------------ helpers

#: Fields holding filesystem paths, resolved against the config file's directory.
_PATH_FIELDS = frozenset(
    {"dataset", "val_dataset", "output_dir", "init_from_adapter", "resume_from"}
)

#: Nested config sections, by field name.
_SECTIONS: dict[str, type] = {
    "lora": LoraSettings,
    "wandb": WandbSettings,
    "ending_weight": EndingWeight,
}


def _coerce(cls: type, data: dict[str, Any], root: Path, *, prefix: str) -> dict[str, Any]:
    """Validate keys and convert values for one dataclass level."""
    known = {f.name for f in fields(cls)}
    values: dict[str, Any] = {}
    for key, value in data.items():
        if key not in known:
            raise ConfigError(_unknown_key_message(prefix + str(key), known, prefix))
        if key in _SECTIONS and prefix == "":
            if not isinstance(value, dict):
                raise ConfigError(
                    f"'{key}' must be a mapping of settings, got {type(value).__name__}."
                )
            section = _SECTIONS[key]
            values[key] = section(**_coerce(section, value, root, prefix=f"{key}."))
            continue
        values[key] = _convert(cls, key, value, root, prefix=prefix)
    return values


def _convert(cls: type, key: str, value: Any, root: Path, *, prefix: str) -> Any:
    """Convert one scalar to the field's declared shape."""
    if value is None:
        return None
    if key in _PATH_FIELDS and prefix == "":
        return _resolve(value, root, key)
    if key == "model":
        # A Hub id ("kova-ai/kova-tts-1") is left alone; an explicitly relative path is not.
        text = str(value)
        return str(_resolve(text, root, key)) if text.startswith((".", "~")) else text
    if key in ("target_modules", "modules_to_save", "tags"):
        if isinstance(value, str):
            raise ConfigError(
                f"'{prefix}{key}' must be a list, got the string {value!r}. Write it as a YAML "
                f"list, e.g. [q_proj, k_proj, v_proj, o_proj]."
            )
        return tuple(str(item) for item in value)
    declared = {f.name: f.type for f in fields(cls)}[key]
    return _cast_scalar(value, declared, f"{prefix}{key}")


def _cast_scalar(value: Any, declared: Any, label: str) -> Any:
    """Cast YAML scalars to the declared type, so ``lr: 2e-5`` survives a naive parser."""
    # With `from __future__ import annotations` a field's type is its source text, so this is a
    # substring test against strings like "int", "float | None", "str".
    text = str(declared)
    if "bool" in text:
        if isinstance(value, str):
            raise ConfigError(
                f"'{label}' must be true or false, got the string {value!r}. YAML quotes some "
                f"words automatically; write it bare as `{label}: true`."
            )
        return bool(value)
    if "float" not in text and "int" not in text:
        return value
    try:
        return float(value) if "float" in text else int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"'{label}' must be a number, got {value!r}.") from exc


def _resolve(value: Any, root: Path, key: str) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = root / path
    return Path(os.path.normpath(path))


def _unknown_key_message(key: str, known: set[str], prefix: str) -> str:
    close = difflib.get_close_matches(key.rsplit(".", 1)[-1], sorted(known), n=1)
    hint = f" Did you mean '{prefix}{close[0]}'?" if close else ""
    return f"Unknown config key '{key}'.{hint} Valid keys here: {', '.join(sorted(known))}."


def _to_plain(value: Any) -> Any:
    """Recursively convert a config to YAML-safe primitives."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _to_plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple | list):
        return [_to_plain(item) for item in value]
    return value
