import argparse
import json
import struct
import subprocess
import sys
from pathlib import Path
from typing import Callable, Sequence

import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = PROJECT_ROOT / "assets" / "models"


def _is_up_to_date(output_path: Path, source_paths: Sequence[Path]) -> bool:
    if not output_path.is_file():
        return False
    output_timestamp = output_path.stat().st_mtime_ns
    return all(output_timestamp >= path.stat().st_mtime_ns for path in source_paths)


def _has_draco_extension(model_path: Path) -> bool:
    data = model_path.read_bytes()
    if len(data) < 20 or data[:4] != b"glTF":
        raise ValueError(f"Invalid binary glTF model: {model_path}")
    json_length, chunk_type = struct.unpack_from("<I4s", data, 12)
    if chunk_type != b"JSON" or 20 + json_length > len(data):
        raise ValueError(f"Invalid glTF JSON chunk: {model_path}")
    document = json.loads(
        data[20 : 20 + json_length].decode("utf-8").rstrip(" \t\r\n\0")
    )
    extensions = set(document.get("extensionsUsed", ()))
    extensions.update(document.get("extensionsRequired", ()))
    return "KHR_draco_mesh_compression" in extensions


def convert_stl_directory(
    stl_directory: Path, output_path: Path, force: bool = False
) -> bool:
    stl_files = sorted(
        (
            path
            for path in stl_directory.iterdir()
            if path.is_file() and path.suffix.lower() == ".stl"
        ),
        key=lambda path: path.name.casefold(),
    )
    if not stl_files:
        raise ValueError(f"No STL files found in {stl_directory}")
    if not force and _is_up_to_date(output_path, stl_files):
        return False

    scene = trimesh.Scene()
    for stl_file in stl_files:
        mesh = trimesh.load(stl_file, file_type="stl", force="mesh")
        scene.add_geometry(mesh, geom_name=stl_file.stem, node_name=stl_file.stem)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(scene.export(file_type="glb"))
    return True


def decompress_cassini(
    source_path: Path, output_path: Path, force: bool = False
) -> bool:
    if not source_path.is_file():
        raise FileNotFoundError(f"Cassini source model not found: {source_path}")
    if not force and _is_up_to_date(output_path, [source_path]):
        try:
            if not _has_draco_extension(output_path):
                return False
        except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
            pass

    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "npx",
            "--yes",
            "--package=@gltf-transform/cli@4.5.1",
            "--package=draco3dgltf@1.5.7",
            "gltf-transform",
            "copy",
            str(source_path),
            str(output_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return True


def _convert_model(
    label: str, conversion: Callable[..., bool], *args: object, force: bool
) -> bool:
    try:
        converted = conversion(*args, force=force)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        details = " ".join(getattr(error, "stdout", "").split())
        message = f"{label}: échec — {error}"
        if details:
            message += f" — {details}"
        print(message, file=sys.stderr)
        return False
    status = "converti" if converted else "déjà à jour"
    print(f"{label}: {status}")
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert supported vehicle model assets for the Panda3D viewer."
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help=f"Model asset directory (default: {DEFAULT_MODELS_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild outputs even when they are already up to date.",
    )
    args = parser.parse_args(argv)
    models_dir = args.models_dir

    results = (
        _convert_model(
            "SOFIA",
            convert_stl_directory,
            models_dir / "sofia",
            models_dir / "sofia.glb",
            force=args.force,
        ),
        _convert_model(
            "SATURN V",
            convert_stl_directory,
            models_dir / "saturn_v",
            models_dir / "saturn_v.glb",
            force=args.force,
        ),
        _convert_model(
            "CASSINI-HUYGENS",
            decompress_cassini,
            models_dir / "Cassini-Huygens (B).glb",
            models_dir / "cassini_huygens.glb",
            force=args.force,
        ),
    )
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
