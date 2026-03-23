from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload an exported model folder to Hugging Face.")
    parser.add_argument("--folder", required=True, type=Path, help="Local model folder to upload.")
    parser.add_argument("--repo-id", required=True, help="Target Hugging Face repo id, e.g. user/model-name.")
    parser.add_argument("--private", action="store_true", help="Create/upload to a private repo.")
    parser.add_argument("--revision", default="main", help="Revision to upload to.")
    parser.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="Environment variable containing the HF token. Default: HF_TOKEN.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload relayered model export",
        help="Commit message used for upload_folder.",
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Upload dotfiles too. By default hidden files are skipped.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folder = args.folder.resolve()
    if not folder.exists():
        raise SystemExit(f"Folder does not exist: {folder}")

    token = os.environ.get(args.token_env)
    if not token:
        raise SystemExit(f"Missing token env var {args.token_env}.")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    ignore_patterns = None if args.include_hidden else [".*", "__pycache__/*"]
    print(f"[upload] repo_id={args.repo_id}")
    print(f"[upload] folder={folder}")
    api.upload_folder(
        folder_path=str(folder),
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        commit_message=args.commit_message,
        ignore_patterns=ignore_patterns,
    )
    print("[upload] done")


if __name__ == "__main__":
    main()

