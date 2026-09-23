"""Check legal-file metadata and contents in built archives or installed Kova packages."""

from __future__ import annotations

import argparse
import email
import tarfile
import zipfile
from importlib.metadata import distribution
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {"kova-tts", "kova-codec"}


def canonical_files() -> dict[str, bytes]:
    sources = [ROOT / "LICENSE", ROOT / "NOTICE"] + sorted(
        path for path in (ROOT / "licenses").rglob("*") if path.is_file()
    )
    return {
        "build/legal/" + source.relative_to(ROOT).as_posix(): source.read_bytes()
        for source in sources
    }


def check_metadata(raw: bytes, read_license, expected: dict[str, bytes]) -> str:
    metadata = email.message_from_bytes(raw)
    name = metadata["Name"]
    if name not in PACKAGES:
        raise ValueError(f"Unexpected package: {name}")
    if metadata["License-Expression"] != "LicenseRef-Kova-Research-NonCommercial":
        raise ValueError(f"{name}: missing or incorrect license expression")
    declared = metadata.get_all("License-File", [])
    if set(declared) != set(expected) or len(declared) != len(expected):
        raise ValueError(f"{name}: legal-file metadata does not match the canonical files")
    for path, content in expected.items():
        if read_license(path) != content:
            raise ValueError(f"{name}: legal file differs from the canonical copy: {path}")
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path("dist"))
    parser.add_argument(
        "--installed", action="store_true", help="check the active Python environment"
    )
    args = parser.parse_args()
    expected = canonical_files()
    if args.installed:
        for name in sorted(PACKAGES):
            package = distribution(name)
            metadata_path = next(
                path for path in package.files or [] if str(path).endswith(".dist-info/METADATA")
            )
            metadata_dir = package.locate_file(metadata_path).parent
            check_metadata(
                (metadata_dir / "METADATA").read_bytes(),
                lambda path, directory=metadata_dir: (directory / "licenses" / path).read_bytes(),
                expected,
            )
            print(f"{name}: installed legal files match the repository root")
        return

    checked = set()
    for path in sorted(args.directory.glob("*.whl")):
        with zipfile.ZipFile(path) as archive:
            metadata_path = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
            metadata_dir = metadata_path.rsplit("/", 1)[0]
            name = check_metadata(
                archive.read(metadata_path),
                lambda name, directory=metadata_dir, z=archive: z.read(
                    f"{directory}/licenses/{name}"
                ),
                expected,
            )
        checked.add((name, "wheel"))
        print(f"{path.name}: all {len(expected)} legal files match")
    for path in sorted(args.directory.glob("*.tar.gz")):
        with tarfile.open(path) as archive:
            prefix = archive.getnames()[0].split("/", 1)[0]

            def read(name: str, source=archive, base=prefix, label=path.name) -> bytes:
                member = source.extractfile(f"{base}/{name}")
                if member is None:
                    raise ValueError(f"Not a file in {label}: {name}")
                return member.read()

            name = check_metadata(read("PKG-INFO"), read, expected)
            # This is needed to rebuild a wheel without the original checkout.
            read("build/_license_metadata.py")
            read("hatch_build.py")
        checked.add((name, "sdist"))
        print(f"{path.name}: all {len(expected)} legal files and the build helper are present")
    required = {(name, kind) for name in PACKAGES for kind in ("wheel", "sdist")}
    if checked != required:
        raise ValueError(f"Missing build artifacts: {sorted(required - checked)}")


if __name__ == "__main__":
    main()
