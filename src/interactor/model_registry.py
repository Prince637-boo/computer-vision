import json
import logging
import math
import struct
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, Optional, Tuple


LOGGER = logging.getLogger(__name__)
DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[2] / "assets" / "models" / "models.json"
)


@dataclass(frozen=True)
class ModelPart:
    match: str
    label: str
    info: str
    explode_axis: Tuple[float, float, float]


@dataclass(frozen=True)
class ModelDefinition:
    id: str
    label: str
    file: str
    description: str
    parts: Tuple[ModelPart, ...]
    training_steps: Tuple[str, ...]


class ModelRegistry:
    """Validated model metadata loaded from the models.json manifest."""

    def __init__(
        self,
        models: Tuple[ModelDefinition, ...],
        manifest_path: Path,
        errors: Tuple[str, ...] = (),
    ):
        self.models = models
        self.manifest_path = manifest_path
        self.assets_dir = manifest_path.parent
        self.errors = errors
        self.by_id = {model.id: model for model in models}

    @classmethod
    def load(cls, manifest_path: Path = DEFAULT_MANIFEST) -> "ModelRegistry":
        manifest_path = Path(manifest_path)
        try:
            document = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            message = f"Cannot load model manifest {manifest_path}: {error}"
            LOGGER.warning("%s; no models will be available.", message)
            return cls((), manifest_path, (message,))

        if not isinstance(document, dict) or not isinstance(
            document.get("models"), list
        ):
            message = (
                f"Invalid model manifest {manifest_path}: expected an object "
                "with a 'models' list; no models will be available."
            )
            LOGGER.warning(message)
            return cls((), manifest_path, (message,))

        valid_models = []
        errors = []
        seen_ids = set()
        for index, raw_model in enumerate(document["models"]):
            try:
                model = cls._parse_model(raw_model)
                if model.id in seen_ids:
                    raise ValueError(f"duplicate model id {model.id!r}")
                seen_ids.add(model.id)
                valid_models.append(model)
            except (OverflowError, TypeError, ValueError) as error:
                model_id = (
                    raw_model.get("id", "<unknown>")
                    if isinstance(raw_model, dict)
                    else "<unknown>"
                )
                message = (
                    f"Skipping invalid model entry #{index} ({model_id}): {error}"
                )
                LOGGER.warning(message)
                errors.append(message)

        return cls(tuple(valid_models), manifest_path, tuple(errors))

    @classmethod
    def _parse_model(cls, raw_model: object) -> ModelDefinition:
        if not isinstance(raw_model, dict):
            raise ValueError("expected a JSON object")

        model_id = cls._required_string(raw_model, "id")
        label = cls._required_string(raw_model, "label")
        model_file = cls._required_string(raw_model, "file")
        description = cls._required_string(raw_model, "description")
        relative_path = PurePosixPath(model_file)
        if (
            relative_path.is_absolute()
            or "\\" in model_file
            or any(part in ("", ".", "..") for part in relative_path.parts)
        ):
            raise ValueError("'file' must be a safe path relative to the manifest")

        raw_parts = raw_model.get("parts")
        if not isinstance(raw_parts, list):
            raise ValueError("'parts' must be a list")
        parts = tuple(cls._parse_part(part) for part in raw_parts)
        part_matches = [part.match for part in parts]
        if len(part_matches) != len(set(part_matches)):
            raise ValueError("part 'match' values must be unique")

        raw_steps = raw_model.get("training_steps", [])
        if not isinstance(raw_steps, list) or any(
            not isinstance(step, str) or not step for step in raw_steps
        ):
            raise ValueError("'training_steps' must be a list of non-empty strings")
        unknown_steps = set(raw_steps) - set(part_matches)
        if unknown_steps:
            raise ValueError(
                f"'training_steps' refers to unknown parts: "
                f"{', '.join(sorted(unknown_steps))}"
            )

        return ModelDefinition(
            id=model_id,
            label=label,
            file=model_file,
            description=description,
            parts=parts,
            training_steps=tuple(raw_steps),
        )

    @classmethod
    def _parse_part(cls, raw_part: object) -> ModelPart:
        if not isinstance(raw_part, dict):
            raise ValueError("each 'parts' entry must be a JSON object")
        match = cls._required_string(raw_part, "match")
        label = cls._required_string(raw_part, "label")
        info = raw_part.get("info")
        if not isinstance(info, str):
            raise ValueError(f"part {match!r} must have a string 'info' field")

        axis = raw_part.get("explode_axis")
        if not isinstance(axis, list) or len(axis) != 3 or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in axis
        ):
            raise ValueError(
                f"part {match!r} 'explode_axis' must be a non-zero vector "
                "of three finite numbers"
            )
        try:
            axis_values = tuple(float(value) for value in axis)
        except OverflowError as error:
            raise ValueError(
                f"part {match!r} 'explode_axis' must be finite"
            ) from error
        if not all(math.isfinite(value) for value in axis_values):
            raise ValueError(f"part {match!r} 'explode_axis' must be finite")
        magnitude = max(abs(value) for value in axis_values)
        if magnitude == 0:
            raise ValueError(
                f"part {match!r} 'explode_axis' must be a non-zero vector"
            )
        axis_values = tuple(value / magnitude for value in axis_values)

        return ModelPart(
            match=match,
            label=label,
            info=info,
            explode_axis=axis_values,
        )

    @staticmethod
    def _required_string(data: dict, key: str) -> str:
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"'{key}' must be a non-empty string")
        return value.strip()


class ModelRegistryMixin:
    MODEL_EXTENSIONS = {".bam", ".glb", ".gltf", ".obj"}

    def _exclude_draco_models(self, model_paths: Dict[str, Path]) -> Dict[str, str]:
        unavailable_reasons = {}
        for model_id, model_file in tuple(model_paths.items()):
            try:
                requires_draco = self._requires_draco(model_file)
            except (OSError, ValueError) as error:
                LOGGER.warning(
                    "Skipping invalid model asset %s: %s", model_file, error
                )
                model_paths.pop(model_id)
                unavailable_reasons[model_id] = "ASSET INVALIDE"
                continue
            if requires_draco:
                model_paths.pop(model_id)
                unavailable_reasons[model_id] = "DRACO NON SUPPORTE"
        return unavailable_reasons

    @classmethod
    def _discover_model_assets(
        cls,
        model_dir: Optional[Path] = None,
        registry: Optional[ModelRegistry] = None,
    ) -> Dict[str, Path]:
        if model_dir is None:
            model_dir = DEFAULT_MANIFEST.parent
        if registry is None:
            registry = ModelRegistry.load(Path(model_dir) / "models.json")

        discovered = {}
        for model in registry.models:
            model_file = registry.assets_dir / model.file
            if (
                model_file.is_file()
                and model_file.suffix.lower() in cls.MODEL_EXTENSIONS
            ):
                discovered[model.id] = model_file.resolve()
        return discovered

    def _model_definition_for_path(
        self, model_file: Path
    ) -> Optional[ModelDefinition]:
        resolved_path = model_file.resolve()
        return next(
            (
                model
                for model in self._model_registry.models
                if (self._model_registry.assets_dir / model.file).resolve()
                == resolved_path
            ),
            None,
        )

    def _part_definition_for_node(self, node_name: str) -> Optional[ModelPart]:
        if self._active_model_definition is None:
            return None
        matching_parts = [
            part
            for part in self._active_model_definition.parts
            if part.match.casefold() in node_name.casefold()
        ]
        return max(matching_parts, key=lambda part: len(part.match), default=None)

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
        if not isinstance(document, dict):
            raise ValueError(f"Invalid glTF document: {model_file}")
        extensions_used = document.get("extensionsUsed", [])
        extensions_required = document.get("extensionsRequired", [])
        if not isinstance(extensions_used, list) or not isinstance(
            extensions_required, list
        ) or not all(
            isinstance(extension, str)
            for extension in extensions_used + extensions_required
        ):
            raise ValueError(f"Invalid glTF extension lists: {model_file}")
        extensions = set(extensions_used)
        extensions.update(extensions_required)
        return "KHR_draco_mesh_compression" in extensions
