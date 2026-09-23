"""Copy the canonical code licenses into the separate model repositories.

Run from any directory. Python packages include these files automatically at build time.
The voice package's dataset NOTICE and LICENSE-SUPPLEMENT are maintained separately
and are never overwritten here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report differences without writing")
    parser.add_argument(
        "--model-repos-root",
        type=Path,
        required=True,
        help="sync existing kova-tts-1 and kova-tts-1-voices folders under this directory",
    )
    args = parser.parse_args()
    targets = [
        (args.model_repos_root / "kova-tts-1", "NOTICE"),
        (args.model_repos_root / "kova-tts-1-voices", "THIRD-PARTY-NOTICE"),
    ]
    for target, _ in targets:
        if not target.is_dir():
            parser.error(f"destination does not exist: {target}")

    sources = [ROOT / "LICENSE", ROOT / "NOTICE"] + sorted(
        path for path in (ROOT / "licenses").rglob("*") if path.is_file()
    )
    differences = 0
    for target, notice_name in targets:
        for source in sources:
            relative = Path(notice_name) if source.name == "NOTICE" else source.relative_to(ROOT)
            destination = target / relative
            payload = source.read_bytes()
            if destination.is_file() and destination.read_bytes() == payload:
                continue
            differences += 1
            if args.check:
                print(f"Out of sync: {destination}")
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(payload)
                print(f"Updated: {destination}")
    if args.check and differences:
        return 1
    print(
        f"License copies {'match' if args.check else 'synchronized'} across {len(targets)} folders."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
