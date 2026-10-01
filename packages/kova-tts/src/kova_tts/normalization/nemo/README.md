# Vendored NeMo text normalization (English)

The English written-to-spoken grammars from
[NeMo-text-processing](https://github.com/NVIDIA/NeMo-text-processing) (Apache 2.0), with Kova's
patches to money, large numbers, phone numbers and acronyms. Only the deterministic English path
is here; `normalize.py` has been trimmed to it.

The three compiled grammars in `cache/` are loaded at startup. Building them from `en/`
takes minutes, so they are tracked. After changing a grammar, rebuild them and commit the
source and the caches together:

```bash
uv run python - <<'PY'
from kova_tts.normalization.normalizer import CACHE_DIR
from kova_tts.normalization.nemo import Normalizer
Normalizer(input_case="cased", lang="en", cache_dir=str(CACHE_DIR),
           overwrite_cache=True, post_process=True)
PY
uv run pytest packages/kova-tts/tests/test_normalization.py
```
