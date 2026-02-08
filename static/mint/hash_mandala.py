from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import lunamint


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_fft_data(values: Iterable[int]) -> str:
    data = bytes([max(0, min(255, int(v))) for v in values])
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a mandala hash using lunamint metadata.")
    parser.add_argument("file", help="Path to the uploaded mandala file")
    parser.add_argument("--name", default="mandala", help="Seed name for lunamint basename")
    parser.add_argument("--denom", default="1", help="Denomination string for lunamint basename")
    parser.add_argument("--fft", help="Path to JSON array of FFT bytes", default=None)
    args = parser.parse_args()

    if args.fft:
        fft_path = Path(args.fft).expanduser().resolve()
        if not fft_path.exists():
            raise FileNotFoundError(f"FFT file not found: {fft_path}")
        fft_values = json.loads(fft_path.read_text(encoding="utf-8"))
        timestamp_ms = lunamint.generate_timestamp_ms_precise()
        basename = lunamint.create_basename(args.name, str(args.denom), timestamp_ms, "FFT")
        digest = sha256_fft_data(fft_values)
        payload = {
            "fft_file": str(fft_path),
            "timestamp_ms": timestamp_ms,
            "basename": basename,
            "sha256": digest,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    file_path = Path(args.file).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    timestamp_ms = lunamint.generate_timestamp_ms_precise()
    basename = lunamint.create_basename(args.name, str(args.denom), timestamp_ms, args.name)
    digest = sha256_file(file_path)

    payload = {
        "file": str(file_path),
        "size_bytes": file_path.stat().st_size,
        "timestamp_ms": timestamp_ms,
        "basename": basename,
        "sha256": digest,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
