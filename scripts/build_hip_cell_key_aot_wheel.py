#!/usr/bin/env python3
"""Build a gfx936-specific ops wheel containing the native cell-key artifact.

The regular Hatchling wheel remains pure Python.  This script compiles the
optional HIP extension with the active Hygon PyTorch, injects that shared
object into a copy of the wheel, and rewrites WHEEL/RECORD with a CPython and
platform-specific tag.  It intentionally does not register a runtime backend.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import sysconfig
import zipfile
from pathlib import Path


def _sha256_record(data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return f"sha256={digest.decode('ascii')}"


def _wheel_tag() -> str:
    implementation = f"cp{sys.version_info.major}{sys.version_info.minor}"
    platform = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    return f"{implementation}-{implementation}-{platform}"


def _find_base_wheel(directory: Path) -> Path:
    wheels = sorted(directory.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected exactly one base wheel in {directory}, found {wheels}")
    return wheels[0]


def _build_base_wheel(source_dir: Path, base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["uv", "build", "--wheel", "--directory", str(source_dir), "--out-dir", str(base_dir)],
        check=True,
    )
    return _find_base_wheel(base_dir)


def _rewrite_wheel_metadata(data: bytes, tag: str) -> bytes:
    lines = data.decode("utf-8").splitlines()
    retained = [
        line
        for line in lines
        if not line.startswith("Root-Is-Purelib:") and not line.startswith("Tag:")
    ]
    retained.extend(["Root-Is-Purelib: false", f"Tag: {tag}", ""])
    return "\n".join(retained).encode("utf-8")


def _record(rows: list[tuple[str, bytes]], record_name: str) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    for name, data in rows:
        writer.writerow((name, _sha256_record(data), len(data)))
    writer.writerow((record_name, "", ""))
    return output.getvalue().encode("utf-8")


def _write_aot_wheel(base_wheel: Path, artifact: Path, output_dir: Path) -> Path:
    tag = _wheel_tag()
    prefix = base_wheel.name.removesuffix(".whl").rsplit("-", 3)[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_wheel = output_dir / f"{prefix}-{tag}.whl"
    if output_wheel.exists():
        raise FileExistsError(f"refusing to overwrite existing wheel: {output_wheel}")

    with zipfile.ZipFile(base_wheel) as source:
        wheel_name = next(
            name for name in source.namelist() if name.endswith(".dist-info/WHEEL")
        )
        record_name = next(
            name for name in source.namelist() if name.endswith(".dist-info/RECORD")
        )
        rows = []
        for info in source.infolist():
            if info.filename == record_name:
                continue
            data = source.read(info.filename)
            if info.filename == wheel_name:
                data = _rewrite_wheel_metadata(data, tag)
            rows.append((info.filename, data))

    artifact_name = "nvalchemiops/_native/nvalchemi_cell_key_hip.so"
    rows.append((artifact_name, artifact.read_bytes()))
    provenance_name = wheel_name.removesuffix("WHEEL") + "hip_cell_key_aot.json"
    provenance = json.dumps(
        {
            "artifact": artifact_name,
            "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "hip": __import__("torch").version.hip,
            "platform_tag": tag,
            "pytorch_rocm_arch": os.environ.get("PYTORCH_ROCM_ARCH"),
        },
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    rows.append((provenance_name, provenance))

    with zipfile.ZipFile(output_wheel, "w", compression=zipfile.ZIP_DEFLATED) as target:
        for name, data in rows:
            target.writestr(name, data)
        target.writestr(record_name, _record(rows, record_name))
    return output_wheel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path("packages/ops"))
    parser.add_argument("--base-wheel", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--arch", default="gfx936")
    args = parser.parse_args()

    import torch

    if not torch.version.hip:
        raise RuntimeError("the active PyTorch does not expose a HIP build")
    source_dir = args.source_dir.resolve()
    output_dir = args.out_dir.resolve()
    build_dir = args.build_dir.resolve()
    os.environ["PYTORCH_ROCM_ARCH"] = args.arch
    os.environ["NVALCHEMI_HIP_CELL_KEY_BUILD_DIR"] = str(build_dir)
    base_wheel = (
        args.base_wheel.resolve()
        if args.base_wheel is not None
        else _build_base_wheel(source_dir, output_dir / "base")
    )
    if not base_wheel.is_file():
        raise FileNotFoundError(base_wheel)

    from nvalchemiops._hip_cell_key import _load_jit_extension

    artifact = Path(_load_jit_extension().__file__).resolve()
    if not artifact.is_file():
        raise RuntimeError(f"native HIP build did not produce an artifact: {artifact}")
    wheel = _write_aot_wheel(base_wheel, artifact, output_dir)
    print(
        json.dumps(
            {
                "arch": args.arch,
                "artifact": str(artifact),
                "base_wheel": str(base_wheel),
                "wheel": str(wheel),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
