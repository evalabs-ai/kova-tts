"""Sync the public README, media and licenses to the Hugging Face model repo.

GitHub owns this content. Hugging Face owns its model-card YAML and model files.
Only explicitly selected, git-tracked documentation files are uploaded; nothing is deleted.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import quote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
GITHUB_REPO = "evalabs-ai/kova-tts"
MODEL_REPO = "kova-ai/kova-tts-1"
GITHUB = f"https://github.com/{GITHUB_REPO}"
HUB = f"https://huggingface.co/{MODEL_REPO}"
LEGAL_FILES = {"LICENSE", "LICENSE-WEIGHTS", "NOTICE"}
MEDIA_SUFFIXES = {".svg", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp3", ".wav", ".mp4"}


def frontmatter(card: str) -> str:
    """Retain the live Hub metadata verbatim, including any future fields."""
    match = re.match(r"\A---[ \t]*\r?\n.*?\r?\n---[ \t]*(?:\r?\n|$)", card, re.DOTALL)
    if not match:
        raise ValueError("The Hugging Face README has no YAML header; refusing to replace it.")
    return match.group(0).rstrip("\r\n") + "\n"


def owned_path(path: str) -> bool:
    return (
        path in LEGAL_FILES
        or path.startswith("licenses/")
        or (path.startswith("assets/") and Path(path).suffix.lower() in MEDIA_SUFFIXES)
    )


def local_files(root: Path) -> dict[str, bytes]:
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    files = {}
    for name in tracked:
        if not owned_path(name):
            continue
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Refusing a symlink or a file outside the checkout: {name}")
        files[name] = path.read_bytes()
    for name in LEGAL_FILES:
        if name not in files:
            raise ValueError(f"Missing required tracked file: {name}")
    return files


def target_url(url: str, files: dict[str, bytes], root: Path, *, image: bool = False) -> str:
    if url.startswith("#"):
        return "#" + url[1:].lstrip("-")
    parsed = urlsplit(url)
    if parsed.scheme or parsed.netloc or url.startswith("/"):
        return url
    path = parsed.path.removeprefix("./")
    suffix = ("?" + parsed.query if parsed.query else "") + (
        "#" + parsed.fragment if parsed.fragment else ""
    )
    if path in files:
        kind = "resolve" if image else "blob"
        return f"{HUB}/{kind}/main/{quote(path, safe='/')}{suffix}"
    local = root / path
    if not local.exists():
        raise ValueError(f"README references a missing local file: {path}")
    kind = "tree" if local.is_dir() else "blob"
    return f"{GITHUB}/{kind}/main/{quote(path, safe='/')}{suffix}"


def render_readme(readme: str, metadata: str, files: dict[str, bytes], root: Path) -> str:
    """Use GitHub's prose, adapting links and embedded audio for the Hub."""
    readme = readme.replace(
        "This repository is its runtime:",
        f"The [GitHub repository]({GITHUB}) contains its runtime:",
    ).replace("The local runtime in this repository", "The local runtime in the GitHub repository")
    # GitHub's attachment player is not rendered by the Hub. Use the same public
    # sample recordings that are already tracked in the GitHub repository.
    if re.search(r"^https://github\.com/user-attachments/assets/[^\s]+$", readme, re.MULTILINE):
        rows = []
        for names in [("Mira", "Ash"), ("Owen", "Cal")]:
            cells = []
            for name in names:
                path = f"assets/samples/{name.lower()}.mp3"
                if path not in files:
                    raise ValueError(f"Missing public audio sample: {path}")
                url = f"{HUB}/resolve/main/{path}"
                cells.append(
                    f'    <td align="center"><b>{name}</b><br>'
                    f'<audio controls src="{url}"></audio></td>'
                )
            rows.append("  <tr>\n" + "\n".join(cells) + "\n  </tr>")
        audio = "<table>\n" + "\n".join(rows) + "\n</table>"
        readme = re.sub(
            r"^https://github\.com/user-attachments/assets/[^\s]+$",
            lambda _: audio,
            readme,
            flags=re.MULTILINE,
        )
    github_badge = (
        f'  <a href="{GITHUB}"><img src="https://img.shields.io/badge/'
        'GitHub-kova--tts-181717?logo=github&logoColor=white" alt="GitHub"></a>\n'
    )
    readme = readme.replace("  <br>\n", "  <br>\n" + github_badge, 1)

    def markdown_link(match: re.Match) -> str:
        url = target_url(match[3], files, root, image=bool(match[1]))
        return f"{match[1]}[{match[2]}]({url})"

    def html_link(match: re.Match) -> str:
        url = target_url(match[3], files, root, image=match[1] == "src")
        return f"{match[1]}={match[2]}{url}{match[2]}"

    result = []
    fence = None
    for line in readme.splitlines(keepends=True):
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            if fence is None:
                fence = marker[1]
            elif marker[1][0] == fence[0] and len(marker[1]) >= len(fence):
                fence = None
            result.append(line)
            continue
        if fence is not None:
            result.append(line)
            continue
        heading = re.match(r"^#{1,6}\s+(.+)", line)
        if heading:
            slug = re.sub(r"[^a-z0-9\s-]", "", heading[1].lower())
            slug = re.sub(r"\s+", "-", slug.strip())
            result.append(f'<a id="{slug}"></a>\n')
        line = re.sub(r"(!?)\[([^\]]+)\]\(([^\s)]+)\)", markdown_link, line)
        line = re.sub(r"""\b(src|href)=(["'])(.*?)\2""", html_link, line)
        result.append(line)
    notice = f"<!-- Synced from {GITHUB_REPO}/README.md. Edit the content on GitHub. -->\n"
    return metadata + "\n" + notice + "\n" + "".join(result)


def matches(remote, payload: bytes) -> bool:
    if remote is None:
        return False
    if remote.lfs is not None:
        return remote.lfs.sha256 == hashlib.sha256(payload).hexdigest()
    header = f"blob {len(payload)}\0".encode()
    return remote.blob_id == hashlib.sha1(header + payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="preview without publishing")
    parser.add_argument("--output-dir", type=Path, help="write the dry-run files here")
    parser.add_argument(
        "--model-card", type=Path, help="use a local Hub README for an offline preview"
    )
    args = parser.parse_args()
    if not args.dry_run and (args.output_dir or args.model_card):
        parser.error("--output-dir and --model-card require --dry-run")
    token = os.environ.get("HF_TOKEN")
    if not args.dry_run and not token:
        parser.error("Set the GitHub Actions secret HF_TOKEN to a token that can write this model.")

    files = local_files(ROOT)
    info = None
    if args.model_card:
        card = args.model_card.read_text(encoding="utf-8")
    else:
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi(token=token or False)
        info = api.model_info(MODEL_REPO, revision="main", files_metadata=True)
        card_path = hf_hub_download(
            MODEL_REPO, "README.md", revision=info.sha, token=token or False
        )
        card = Path(card_path).read_text(encoding="utf-8")
    files["README.md"] = render_readme(
        (ROOT / "README.md").read_text(encoding="utf-8"), frontmatter(card), files, ROOT
    ).encode()
    remote_files = {file.rfilename: file for file in info.siblings} if info else {}
    changes = {
        name: data for name, data in files.items() if not matches(remote_files.get(name), data)
    }
    print(f"Target: {MODEL_REPO}; {len(changes)} changed documentation/media files.")
    for name in sorted(changes):
        print(f"  {name}")
    if args.dry_run:
        if args.output_dir:
            for name, payload in files.items():
                target = args.output_dir / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
        print("Dry run: no Hub changes made.")
        return 0
    if not changes:
        print("Already synchronized.")
        return 0

    from huggingface_hub import CommitOperationAdd

    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    commit = api.create_commit(
        repo_id=MODEL_REPO,
        repo_type="model",
        revision="main",
        parent_commit=info.sha,
        operations=[
            CommitOperationAdd(path_in_repo=name, path_or_fileobj=data)
            for name, data in changes.items()
        ],
        commit_message=f"Sync public documentation from {GITHUB_REPO}@{source_sha[:12]}",
    )
    verified = api.model_info(MODEL_REPO, revision=commit.oid, files_metadata=True)
    after = {file.rfilename: file for file in verified.siblings}
    for name, payload in files.items():
        if not matches(after.get(name), payload):
            raise RuntimeError(f"Uploaded content failed verification: {name}")
    for name, before in remote_files.items():
        if name not in files and (name not in after or before.blob_id != after[name].blob_id):
            raise RuntimeError(f"An unowned Hub file changed: {name}")
    print(f"Verified sync: {commit.commit_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
