"""Load the shared license metadata hook from a checkout or a source archive."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def get_metadata_hook():
    package = Path(__file__).resolve().parent
    helper = package.parents[1] / "scripts" / "_license_metadata.py"
    if not helper.is_file():
        helper = package / "build" / "_license_metadata.py"
    spec = spec_from_file_location("kova_license_metadata", helper)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load license metadata hook: {helper}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LicenseMetadataHook
