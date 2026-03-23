# HF Export

This folder contains the public checkpoint export and upload tooling.

It does two jobs:

1. Export a relayered model variant as a normal Hugging Face `safetensors` checkpoint.
2. Upload that exported folder to Hugging Face.

The exporter physically duplicates decoder layers into a new checkpoint. The output model does not depend on the runtime relayer wrapper in this repo.

## Supported Config Syntax

Provide exactly one relayer spec:

- `--blocks "30,34"`
- `--blocks "31,34;43,45"`
- `--layer-list "0,1,2,3,3,4,5,..."`
- `--spec "blocks:30,34"`
- `--spec "layers:0,1,2,3,3,4,5"`

The exporter uses the repo's canonical layer-config logic from `src/core/layer_config.py`.

## Colab Notebook

A minimal Colab notebook is available at:

- `hf_export/colab/export_upload_minimal.ipynb`

It installs only `huggingface_hub` and `safetensors`, clones or unpacks this repo, downloads a base model from Hugging Face, exports one relayered variant, verifies the exported layer count, and optionally uploads the result.

For a private repo, the notebook supports two paths:

- set `GITHUB_TOKEN` and clone over HTTPS from GitHub
- upload a `.zip` or `.tar.gz` archive to Colab and set `REPO_ARCHIVE_PATH`

## Export Examples

Single block:

```bash
uv run python -m hf_export.export_model \
  --source /path/to/base-model \
  --source-repo-id some/model \
  --output exports/model-block-30-34 \
  --blocks "30,34"
```

Multi-block:

```bash
uv run python -m hf_export.export_model \
  --source /path/to/base-model \
  --source-repo-id some/model \
  --output exports/model-31_34__43_45 \
  --blocks "31,34;43,45"
```

Dry-run validation without writing shards:

```bash
uv run python -m hf_export.export_model \
  --source /path/to/base-model \
  --source-repo-id some/model \
  --output tmp/export-check \
  --blocks "30,34" \
  --dry-run \
  --overwrite
```

## Output Files

Each export writes:

- `config.json` with updated layer count and layer-type order
- `model.safetensors.index.json`
- new `model.safetensors-xxxxx-of-xxxxx.safetensors` shards
- copied tokenizer / generation / template files from the source model
- `rys_export_manifest.json` with provenance and the exact layer list

## Upload Example

```bash
export HF_TOKEN=...

uv run python -m hf_export.upload_to_hf \
  --folder exports/model-block-30-34 \
  --repo-id your-name/model-block-30-34
```

Private repo:

```bash
uv run python -m hf_export.upload_to_hf \
  --folder exports/model-block-30-34 \
  --repo-id your-name/model-block-30-34 \
  --private
```

## Notes

- The exporter keeps the original shard count and rewrites the shard index for the new layer order.
- Mixed layer-type metadata is rewritten when present under `num_hidden_layers` and `text_config.layer_types`.
- The upload script uses `huggingface_hub.HfApi.upload_folder`.
- Make sure you have enough disk space before exporting. Output size grows roughly with the number of duplicated layers.
