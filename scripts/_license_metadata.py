"""Stage canonical legal files for Hatch without maintaining package-level copies."""

from __future__ import annotations

import shutil
from pathlib import Path

from hatchling.metadata.plugin.interface import MetadataHookInterface


class LicenseMetadataHook(MetadataHookInterface):
    def update(self, metadata: dict) -> None:
        package = Path(self.root)
        stage = package / "build" / "legal"
        helper = Path(__file__).resolve()
        repository = helper.parent.parent

        # A checkout reads the canonical root. An extracted sdist already contains
        # the staged legal files and this helper, so it needs no parent checkout.
        if helper.parent.name == "scripts":
            sources = [repository / "LICENSE", repository / "NOTICE"]
            sources += sorted(p for p in (repository / "licenses").rglob("*") if p.is_file())
            if len(sources) <= 2 or not all(p.is_file() for p in sources):
                raise ValueError("The repository's canonical legal files are missing")
            if stage.exists():
                shutil.rmtree(stage)
            for source in sources:
                destination = stage / source.relative_to(repository)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
            shutil.copyfile(helper, package / "build" / "_license_metadata.py")

        if not all((stage / name).is_file() for name in ("LICENSE", "NOTICE")):
            raise ValueError("The source distribution is missing its legal files")
        if not (stage / "licenses" / "third-party").is_dir():
            raise ValueError("The source distribution is missing third-party license files")

        metadata["license-files"] = [
            p.relative_to(package).as_posix() for p in sorted(stage.rglob("*")) if p.is_file()
        ]
