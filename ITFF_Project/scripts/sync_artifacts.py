import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Optional

import requests
from requests import Response

DEFAULT_TIMEOUT = 30


def sha256sum(path: Path) -> str:
    hash_obj = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def download_file(url: str, dest: Path, overwrite: bool, timeout: int = DEFAULT_TIMEOUT) -> Response:
    if dest.exists() and not overwrite:
        return None  # type: ignore[return-value]

    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()

    tmp_path = dest.with_suffix(dest.suffix + ".tmp")
    with tmp_path.open("wb") as fh:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            fh.write(chunk)
    tmp_path.replace(dest)
    return response


def iter_manifest(manifest_path: Path) -> Iterable[dict]:
    data = json.loads(manifest_path.read_text())
    if isinstance(data, dict):
        data = data.get("artifacts", [])
    if not isinstance(data, list):
        raise ValueError("Manifest must be a list or contain an 'artifacts' list")
    for item in data:
        if "source" not in item or "destination" not in item:
            raise ValueError("Manifest entries must include 'source' and 'destination'")
        yield item


def sync_artifacts(
    manifest: Path,
    base_url: Optional[str],
    root: Path,
    overwrite: bool,
    timeout: int,
) -> None:
    for entry in iter_manifest(manifest):
        source = entry["source"]
        if base_url and not source.startswith("http"):
            source = f"{base_url.rstrip('/')}/{source.lstrip('/')}"

        destination = root / entry["destination"]
        response = download_file(source, destination, overwrite=overwrite, timeout=timeout)

        if response is None:
            print(f"[skip] {destination} already exists (use --overwrite to replace)")
            continue

        expected_sha = entry.get("sha256")
        if expected_sha:
            actual_sha = sha256sum(destination)
            if actual_sha.lower() != expected_sha.lower():
                destination.unlink(missing_ok=True)
                raise ValueError(f"SHA256 mismatch for {destination}: expected {expected_sha}, got {actual_sha}")

        print(f"[ok] {destination} ({len(response.content)} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download model/scaler artifacts from remote storage.")
    parser.add_argument("--manifest", default="deploy/artifacts.manifest.json", help="Path to manifest JSON file")
    parser.add_argument("--base-url", help="Base URL prepended to relative sources in the manifest")
    parser.add_argument("--root", default=".", help="Root directory for downloaded artifacts")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Request timeout in seconds")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    root = Path(args.root)
    sync_artifacts(
        manifest=manifest_path,
        base_url=args.base_url or os.getenv("ITFF_ARTIFACT_BASE_URL"),
        root=root,
        overwrite=args.overwrite,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()

