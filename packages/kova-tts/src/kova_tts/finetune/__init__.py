"""LoRA finetuning: teach the base model a single voice from a few hundred clips.

Install the extra first (``pip install 'kova-tts[finetune]'``), write a short YAML config
(see ``configs/example_voice.yaml``), then::

    from kova_tts.finetune import load_config, run

    adapter = run(load_config("voice.yaml"))

The three pieces worth reading before changing anything:

* :mod:`~kova_tts.finetune.dataset` -- where loss masking is decided (everything up to and
  including ``<|speech_start|>`` is unsupervised).
* :mod:`~kova_tts.finetune.loss` -- the ending-weight recipe, which is what makes an adapter
  stop cleanly rather than trail off.
* :mod:`~kova_tts.finetune.config` -- every knob and its default.
"""

from __future__ import annotations

from kova_tts.finetune.config import (
    ConfigError,
    FinetuneConfig,
    LoraSettings,
    WandbSettings,
    load_config,
    snapshot,
)
from kova_tts.finetune.dataset import (
    DatasetError,
    LoadReport,
    MaskedCausalDataset,
    build_datasets,
)
from kova_tts.finetune.loss import (
    EndingWeight,
    EndingWeightCollator,
    EndingWeightTrainer,
    ramp_weights,
    weighted_lm_loss,
)
from kova_tts.finetune.merge import merge_adapter
from kova_tts.finetune.train import run

__all__ = [
    "ConfigError",
    "DatasetError",
    "EndingWeight",
    "EndingWeightCollator",
    "EndingWeightTrainer",
    "FinetuneConfig",
    "LoadReport",
    "LoraSettings",
    "MaskedCausalDataset",
    "WandbSettings",
    "build_datasets",
    "load_config",
    "merge_adapter",
    "ramp_weights",
    "run",
    "snapshot",
    "weighted_lm_loss",
]
