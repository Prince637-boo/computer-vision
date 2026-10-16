import json
import struct
from pathlib import Path
from typing import Dict, Optional


class ModelRegistryMixin:
    MODEL_CATALOG = (
        ("vehicle", "MERCEDES 300 SEL", ("mercedes",)),
        ("probe", "CASSINI-HUYGENS", ("cassini", "huygens")),
        ("rocket", "SATURN V", ("saturnv", "saturn5")),
        ("aircraft", "SOFIA", ("sofia",)),
    )
    MODEL_EXTENSIONS = {".bam", ".glb", ".gltf", ".obj"}

    def _exclude_draco_models(self, model_paths: Dict[str, Path]) -> Dict[str, str]:
        unavailable_reasons = {}
        for model_id, model_file in tuple(model_paths.items()):
            if self._requires_draco(model_file):
                model_paths.pop(model_id)
                unavailable_reasons[model_id] = "DRACO NON SUPPORTE"
        return unavailable_reasons

    @classmethod
    def _discover_model_assets(
        cls, model_dir: Optional[Path] = None
    ) -> Dict[str, Path]:
        if model_dir is None:
            model_dir = Path(__file__).resolve().parents[2] / "assets" / "models"
        if not model_dir.is_dir():
            return {}
        priority = {".glb": 0, ".gltf": 1, ".bam": 2, ".obj": 3}
        files = sorted(
            (
                path
                for path in model_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in cls.MODEL_EXTENSIONS
            ),
            key=lambda path: (
                priority[path.suffix.lower()],
                len(path.relative_to(model_dir).parts),
                str(path),
            ),
        )
        discovered: Dict[str, Path] = {}
        for model_id, _, aliases in cls.MODEL_CATALOG:
            matching_files = []
            for model_file in files:
                relative_path = "/".join(model_file.relative_to(model_dir).parts)
                normalized_path = "".join(
                    character.lower()
                    for character in relative_path
                    if character.isalnum()
                )
                if any(alias in normalized_path for alias in aliases):
                    matching_files.append(model_file)
            if matching_files:
                if model_id == "probe":
                    matching_files = [
                        model_file
                        for model_file in matching_files
                        if not cls._requires_draco(model_file)
                    ] or matching_files
                discovered[model_id] = matching_files[0]
        return discovered

    @staticmethod
    def _requires_draco(model_file: Path) -> bool:
        if model_file.suffix.lower() == ".glb":
            data = model_file.read_bytes()
            if len(data) < 20 or data[:4] != b"glTF":
                raise ValueError(f"Invalid binary glTF model: {model_file}")
            json_length, chunk_type = struct.unpack_from("<I4s", data, 12)
            if chunk_type != b"JSON" or 20 + json_length > len(data):
                raise ValueError(f"Invalid glTF JSON chunk: {model_file}")
            document = json.loads(
                data[20 : 20 + json_length]
                .decode("utf-8")
                .rstrip(" \t\r\n\0")
            )
        elif model_file.suffix.lower() == ".gltf":
            document = json.loads(model_file.read_text(encoding="utf-8"))
        else:
            return False
        extensions = set(document.get("extensionsUsed", ()))
        extensions.update(document.get("extensionsRequired", ()))
        return "KHR_draco_mesh_compression" in extensions
