# Hugging Face documentation sync

The GitHub README is the source for the public model card at
[kova-ai/kova-tts-1](https://huggingface.co/kova-ai/kova-tts-1).
The **Sync Hugging Face model card** GitHub Actions workflow updates it after relevant
changes reach `main`. It can also be run manually from GitHub Actions.

## Initial setup

Add an Actions repository secret named `HF_TOKEN` in
[the GitHub repository settings](https://github.com/evalabs-ai/kova-tts/settings/secrets/actions).
Use a Hugging Face fine-grained access token with write access to `kova-ai/kova-tts-1`.
The token's account must also have permission to write to the `kova-ai` organization.
Never commit the token to either repository.

Run the workflow once to publish the current README. A missing token fails with a setup
message, rather than reporting a successful sync.

## What is synchronized

- The GitHub README body, including the team section and social links.
- Git-tracked media in `assets/`, including the public audio samples.
- `LICENSE`, `LICENSE-WEIGHTS`, `NOTICE`, and the files in `licenses/`.

The script adapts relative documentation and package links to GitHub, adds explicit
section anchors, and uses Hugging Face audio players for the public samples. The model
card's YAML header is copied verbatim from the current Hugging Face version, preserving
its language, task, tags, license metadata and any additional fields.

Weights, tokenizers, configuration, repository attributes and other Hugging Face files
are outside the upload list. The sync never deletes files or rewrites git history.
The separate `kova-tts-1-voices` repository is maintained independently.

Uploads use the observed Hugging Face commit as their parent, so a concurrent edit fails
instead of overwriting a newer version. Every upload is verified by file hash, including
a check that files outside the upload list still match the previous commit. Unchanged
content does not create a Hugging Face commit.

## Preview or run locally

With `huggingface_hub` installed, a read-only preview does not require authentication:

```bash
python scripts/sync_huggingface.py --dry-run --output-dir /tmp/kova-model-card
```

For an offline preview, add `--model-card /path/to/current-hub-README.md`.
To publish, set `HF_TOKEN` in your environment and run `python scripts/sync_huggingface.py`.
Edit shared content on GitHub; changes to the Hugging Face README body will be replaced
by the next sync. Edit Hugging Face metadata in its YAML header on the Hub.
