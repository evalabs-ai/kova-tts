"""Training a per-voice LoRA adapter.

The whole run is :func:`run`: load the base checkpoint, wrap it in a peft adapter, build the
masked-causal datasets, and hand the lot to ``transformers.Trainer`` with the ending-weighted
loss. It returns the directory holding the finished adapter, which is what every downstream
tool (inference, merging, evaluation) actually wants.

Single GPU by design. A 1B backbone with rank-64 attention adapters and gradient checkpointing
trains a voice in minutes on one consumer card; distributed training would add a launcher, a
sharding story and a class of bugs that this workload does not need. Pin the card with
``CUDA_VISIBLE_DEVICES`` if the machine has more than one.

Training metrics are loss only. :func:`run` accepts extra ``transformers.TrainerCallback``
objects, so anything more -- periodic sample audio, a logger of your own -- can be attached from
outside without changing this module.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import re
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from kova_tts import paths
from kova_tts.finetune.config import ConfigError, FinetuneConfig, load_config, snapshot
from kova_tts.finetune.dataset import build_datasets
from kova_tts.finetune.loss import EndingWeightCollator, EndingWeightTrainer

logger = logging.getLogger(__name__)

#: Auto-generated run directories look like ``ft_007_my_voice_2026-08-03_11-04-22``.
_RUN_DIR_RE = re.compile(r"^ft_(\d+)_")

#: Subdirectory of the run holding the adapter that should be used.
FINAL_DIR = "final"

#: transformers 5 folded ``warmup_ratio`` into ``warmup_steps`` (which now accepts a fraction)
#: and left the old name behind as a deprecated alias defaulting to ``None``; transformers 4
#: only understands ``warmup_ratio``. Detected from the field's default rather than a version
#: string, which lies in prereleases.
_WARMUP_KEY = (
    "warmup_steps"
    if any(
        f.name == "warmup_ratio" and f.default is None
        for f in dataclasses.fields(TrainingArguments)
    )
    else "warmup_ratio"
)


def run(
    config: FinetuneConfig,
    *,
    callbacks: Sequence[TrainerCallback] | None = None,
) -> Path:
    """Train one adapter. Returns the directory it was saved to.

    `callbacks` are extra ``transformers.TrainerCallback`` objects appended to the built-in
    ones, which is how anything beyond loss metrics is attached to a run.
    """
    peft = _import_peft()

    model_source = paths.model_path(config.model)
    run_dir = _prepare_run_dir(config)
    logger.info("Run directory: %s", run_dir)
    snapshot(config, run_dir)
    _configure_wandb(config)

    tokenizer = AutoTokenizer.from_pretrained(model_source)
    if tokenizer.pad_token is None:
        # Padding is masked out of the loss either way; reusing EOS avoids resizing embeddings.
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset, val_dataset = build_datasets(
        train_path=config.dataset,
        tokenizer=tokenizer,
        max_length=config.max_length,
        ending_weight=config.ending_weight,
        val_path=config.val_dataset,
        val_split=config.val_split,
        seed=config.seed,
    )
    logger.info(
        "Training on %d examples%s (longest %d of max_length %d)",
        len(train_dataset),
        f", validating on {len(val_dataset)}" if val_dataset else "",
        train_dataset.longest,
        config.max_length,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        dtype=torch.bfloat16 if config.bf16 else torch.float32,
        device_map={"": 0} if torch.cuda.is_available() else None,
    )
    model = _attach_adapter(model, config, peft)
    if config.gradient_checkpointing:
        # Without this the checkpointed base layers produce activations with no grad_fn, and
        # the adapter receives no gradient at all -- a run that trains and changes nothing.
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    ending = config.ending_weight
    collator_cls = EndingWeightCollator if ending.ramps else DataCollatorForSeq2Seq
    collator = collator_cls(tokenizer=tokenizer, padding=True, pad_to_multiple_of=8)

    args = TrainingArguments(
        output_dir=str(run_dir),
        run_name=run_dir.name,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.lr,
        lr_scheduler_type=config.lr_scheduler_type,
        weight_decay=config.weight_decay,
        bf16=config.bf16,
        seed=config.seed,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        eval_strategy="steps" if val_dataset is not None else "no",
        eval_steps=config.eval_steps,
        load_best_model_at_end=False,
        report_to=["wandb"] if config.wandb.enabled else [],
        # The ending-weight channel is not a model argument; Trainer would drop it as unused.
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        **{_WARMUP_KEY: config.warmup_ratio},
    )

    _pin_to_one_gpu(args)

    extra = list(callbacks or [])
    if val_dataset is not None:
        extra.append(EvalAtEpochEnd())

    common: dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": collator,
        "processing_class": tokenizer,
        "callbacks": extra,
    }
    if ending.active:
        logger.info("Ending-weight loss on: %s", ending)
        trainer = EndingWeightTrainer(
            speech_end_id=train_dataset.speech_end_id, ending=ending, **common
        )
    else:
        trainer = Trainer(**common)

    trainer.train(resume_from_checkpoint=str(config.resume_from) if config.resume_from else None)

    final = run_dir / FINAL_DIR
    model.save_pretrained(str(final))
    tokenizer.save_pretrained(str(final))
    logger.info("LoRA adapter saved to %s", final)
    return final


# --------------------------------------------------------------------------------- callbacks


class EvalAtEpochEnd(TrainerCallback):
    """Forces an evaluation at each epoch boundary when step-based eval just missed one.

    With ``eval_strategy="steps"`` and a small corpus an epoch can pass without ever evaluating,
    which leaves the per-epoch checkpoints unlabelled by any metric.
    """

    def on_epoch_end(self, args, state, control, **kwargs):
        recent = state.log_history[-5:]
        already = any(
            entry.get("step") == state.global_step and "eval_loss" in entry for entry in recent
        )
        if not already:
            control.should_evaluate = True
        return control


# ----------------------------------------------------------------------------------- helpers


def _import_peft():
    try:
        import peft
    except ImportError as exc:  # pragma: no cover - depends on how the package was installed
        raise ConfigError(
            "Finetuning needs peft, which is not installed. Install the extra: "
            "pip install 'kova-tts[finetune]'."
        ) from exc
    return peft


def _attach_adapter(model: Any, config: FinetuneConfig, peft: Any) -> Any:
    """Wrap the base model in a fresh LoRA adapter, or reopen an existing one for training."""
    if config.init_from_adapter is not None:
        logger.info("Continuing from existing adapter: %s", config.init_from_adapter)
        # The adapter's own adapter_config.json defines r/alpha/target_modules, so `lora:` in
        # the config is deliberately ignored here rather than silently half-applied.
        return peft.PeftModel.from_pretrained(
            model, str(config.init_from_adapter), is_trainable=True
        )
    lora = config.lora
    return peft.get_peft_model(
        model,
        peft.LoraConfig(
            task_type=peft.TaskType.CAUSAL_LM,
            r=lora.r,
            lora_alpha=lora.alpha,
            lora_dropout=lora.dropout,
            target_modules=list(lora.target_modules),
            modules_to_save=list(lora.modules_to_save) if lora.modules_to_save else None,
        ),
    )


def _pin_to_one_gpu(args: TrainingArguments) -> None:
    """Keep a multi-GPU box from silently becoming a ``nn.DataParallel`` run.

    The model is loaded entirely onto device 0, but ``TrainingArguments`` counts every visible
    CUDA device and wraps the model in the deprecated ``nn.DataParallel`` when there is more
    than one. That quietly multiplies the effective batch size by the device count -- so the
    configured schedule no longer means what the config says -- and for a 1B adapter it is slower
    than one card. Reading ``args.device`` first is required because the device count is
    computed lazily, and would otherwise overwrite this.
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        return
    _ = args.device
    args._n_gpu = 1
    logger.info(
        "%d CUDA devices visible; training on device 0 only. Set CUDA_VISIBLE_DEVICES to pick "
        "a different one.",
        torch.cuda.device_count(),
    )


def _prepare_run_dir(config: FinetuneConfig) -> Path:
    """Create the run directory, numbering it after the runs already in ``output_dir``."""
    base = Path(config.output_dir).expanduser()
    base.mkdir(parents=True, exist_ok=True)

    name = config.run_name
    if name is None:
        used = [
            int(match.group(1))
            for entry in base.iterdir()
            if entry.is_dir() and (match := _RUN_DIR_RE.match(entry.name))
        ]
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name = f"ft_{max(used, default=0) + 1:03d}_{config.voice_name}_{stamp}"

    run_dir = base / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _configure_wandb(config: FinetuneConfig) -> None:
    """Point wandb at the configured project. Nothing is hardcoded and nothing runs when off."""
    if not config.wandb.enabled:
        # Belt and braces: wandb auto-initialises from a stale environment otherwise.
        os.environ.setdefault("WANDB_DISABLED", "true")
        return
    os.environ["WANDB_PROJECT"] = config.wandb.project or ""
    if config.wandb.entity:
        os.environ["WANDB_ENTITY"] = config.wandb.entity
    if config.wandb.tags:
        os.environ["WANDB_TAGS"] = ",".join(config.wandb.tags)


# -------------------------------------------------------------------------------------- cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kova-tts finetune",
        description="Train a per-voice LoRA adapter from a JSONL corpus.",
    )
    parser.add_argument("--config", required=True, help="YAML run config")
    parser.add_argument("--dataset", help="override the training corpus")
    parser.add_argument("--output-dir", help="override the parent directory for run outputs")
    parser.add_argument("--model", help="override the base checkpoint (path or Hub id)")
    parser.add_argument("--voice", help="override the voice name used in the run directory")
    parser.add_argument("--epochs", type=float, help="override the epoch count")
    parser.add_argument("--lr", type=float, help="override the learning rate")
    parser.add_argument("--resume-from", help="resume from a Trainer checkpoint directory")
    parser.add_argument(
        "--no-wandb", action="store_true", help="disable wandb even if the config enables it"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    try:
        config = _apply_overrides(load_config(args.config), args)
        config.validate()
        adapter = run(config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(adapter)
    return 0


def _apply_overrides(config: FinetuneConfig, args: argparse.Namespace) -> FinetuneConfig:
    """Apply command-line overrides. Paths resolve against the cwd, not the config file."""
    changes: dict[str, Any] = {}
    for name, field_name in (
        ("dataset", "dataset"),
        ("output_dir", "output_dir"),
        ("resume_from", "resume_from"),
    ):
        value = getattr(args, name)
        if value is not None:
            changes[field_name] = Path(value).expanduser().resolve()
    for name in ("model", "voice", "epochs", "lr"):
        value = getattr(args, name)
        if value is not None:
            changes[name] = value
    if args.no_wandb:
        changes["wandb"] = dataclasses.replace(config.wandb, enabled=False)
    return dataclasses.replace(config, **changes) if changes else config


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
