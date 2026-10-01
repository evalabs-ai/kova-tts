"""Folding a LoRA adapter back into the base weights.

An adapter is the right thing to ship when one process serves many voices: the base model is
loaded once and adapters are swapped per request. Merging is the right thing when a deployment
serves exactly one voice, or when a runtime cannot apply adapters at all -- the merged model is
an ordinary causal LM directory that any tool can load, at the cost of a full copy of the
weights per voice.

The merge itself is exact for a plain LoRA: ``W + (alpha/r) * B @ A`` folded into ``W``, with no
approximation. Pass the dtype the model will be served in -- merging in float32 and then casting
is not the same arithmetic as merging in bfloat16.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kova_tts import paths

logger = logging.getLogger(__name__)

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def base_model_of(adapter: str | os.PathLike[str]) -> str | None:
    """The base checkpoint an adapter was trained against, from its ``adapter_config.json``."""
    config_file = Path(adapter).expanduser() / "adapter_config.json"
    if not config_file.is_file():
        raise FileNotFoundError(
            f"{config_file} not found -- {adapter} is not a saved peft adapter directory."
        )
    return json.loads(config_file.read_text(encoding="utf-8")).get("base_model_name_or_path")


def merge_adapter(
    adapter: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    base_model: str | os.PathLike[str] | None = None,
    dtype: str = "bfloat16",
) -> Path:
    """Merge `adapter` into its base model and save a standalone checkpoint to `output`.

    `base_model` defaults to the base recorded in the adapter, then to the configured
    checkpoint -- an adapter trained on another machine records a path that does not exist here,
    which is why the fallback exists rather than a hard error. A recorded Hub repository id (the
    published voices record ``kova-ai/kova-tts-1``) is used as is.
    """
    from peft import PeftModel

    adapter_dir = Path(adapter).expanduser()
    target = Path(output).expanduser()
    if dtype not in _DTYPES:
        raise ValueError(f"dtype must be one of {', '.join(_DTYPES)}, got {dtype!r}.")

    source = base_model or base_model_of(adapter_dir)
    if (
        source is not None
        and paths._looks_local(str(source))
        and not Path(str(source)).expanduser().exists()
    ):
        logger.warning(
            "Adapter records base model %s, which does not exist here; falling back to the "
            "configured checkpoint.",
            source,
        )
        source = None
    resolved = paths.model_path(source)

    logger.info("Merging %s into %s", adapter_dir, resolved)
    model = AutoModelForCausalLM.from_pretrained(resolved, dtype=_DTYPES[dtype])
    merged = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()

    target.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(target))
    # The tokenizer travels with the adapter when one was saved there, so a merged voice keeps
    # whatever vocabulary it was actually trained with.
    tokenizer_source = adapter_dir if (adapter_dir / "tokenizer.json").is_file() else resolved
    AutoTokenizer.from_pretrained(str(tokenizer_source)).save_pretrained(str(target))
    logger.info("Merged model saved to %s", target)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kova-tts merge",
        description="Merge a LoRA adapter into its base model and save a standalone checkpoint.",
    )
    parser.add_argument("adapter", help="directory holding adapter_config.json")
    parser.add_argument("output", help="directory to write the merged model to")
    parser.add_argument("--base-model", help="override the base checkpoint (path or Hub id)")
    parser.add_argument("--dtype", default="bfloat16", choices=sorted(_DTYPES))
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    try:
        target = merge_adapter(
            args.adapter, args.output, base_model=args.base_model, dtype=args.dtype
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(target)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
