from __future__ import annotations

import hashlib
from typing import List

from fastapi import Body, FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import lunamint

from hash_mandala import sha256_fft_data

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def sha256_bytes(data: bytes) -> str:
    hasher = hashlib.sha256()
    hasher.update(data)
    return hasher.hexdigest()


@app.post("/hash")
async def hash_mandala(
    file: UploadFile = File(...),
    name: str = Form("mandala"),
    denom: str = Form("1"),
) -> dict:
    payload = await file.read()
    timestamp_ms = lunamint.generate_timestamp_ms_precise()
    basename = lunamint.create_basename(name, str(denom), timestamp_ms, "MANDALA")
    digest = sha256_bytes(payload)
    return {
        "filename": file.filename,
        "size_bytes": len(payload),
        "timestamp_ms": timestamp_ms,
        "basename": basename,
        "sha256": digest,
    }


@app.post("/hash-fft")
async def hash_fft(
    fft: List[int] = Body(..., embed=True),
    name: str = Body("mandala", embed=True),
    denom: str = Body("1", embed=True),
) -> dict:
    timestamp_ms = lunamint.generate_timestamp_ms_precise()
    basename = lunamint.create_basename(name, str(denom), timestamp_ms, "FFT")
    digest = sha256_fft_data(fft)
    return {
        "bins": len(fft),
        "timestamp_ms": timestamp_ms,
        "basename": basename,
        "sha256": digest,
    }
