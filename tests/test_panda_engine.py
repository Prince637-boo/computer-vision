import json
import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from panda3d.core import Point3, Vec3

from src.interactor.model_registry import ModelRegistry, ModelRegistryMixin
from src.interactor.model_view import ModelViewMixin
from src.interactor.panda_engine import Panda3DEngine


class ExplosionDirectionTests(unittest.TestCase):
    def test_named_vehicle_parts_explode_along_technical_axes(self):
        engine = Panda3DEngine.__new__(Panda3DEngine)
        engine._active_model_definition = ModelRegistry.load().by_id["vehicle"]
        center = Vec3(0, 0, 0)

        self.assertEqual(
            engine._explosion_direction(
                "MERC_300SEL_ENGINE_LADX", Point3(0, 0, 0), center
            ),
            Vec3(0, 0, 1),
        )
        self.assertEqual(
            engine._explosion_direction(
                "MERC_300SEL_WHEEL_LF_LADX", Point3(-1, 0, 0), center
            ),
            Vec3(-1, 0, 0),
        )
        self.assertEqual(
            engine._explosion_direction(
                "MERC_300SEL_WHEEL_LF_LADX", Point3(1, 0, 0), center
            ),
            Vec3(1, 0, 0),
        )
        self.assertEqual(
            engine._explosion_direction(
                "MERC_300SEL_BONNET_LADX", Point3(0, 0, 0), center
            ),
            Vec3(0, 1, 0),
        )
        self.assertEqual(
            engine._explosion_direction(
                "MERC_300SEL_BOOT_LADX", Point3(0, 0, 0), center
            ),
            Vec3(0, -1, 0),
        )

    def test_unclassified_parts_use_their_radial_position_without_randomness(self):
        engine = Panda3DEngine.__new__(Panda3DEngine)
        engine._active_model_definition = ModelRegistry.load().by_id["vehicle"]
        direction = engine._explosion_direction(
            "SOLAR_PANEL", Point3(3, 4, 0), Vec3(0, 0, 0)
        )

        self.assertAlmostEqual(direction.length(), 1)
        self.assertEqual(direction, Vec3(0.6, 0.8, 0))
        self.assertEqual(
            engine._explosion_direction(
                "ANTENNA", Point3(0, 0, 0), Vec3(0, 0, 0)
            ),
            Vec3(0, 0, 1),
        )

    def test_current_cassini_asset_is_detected_as_draco_compressed(self):
        model_file = Path("assets/models/Cassini-Huygens (B).glb")

        self.assertTrue(ModelRegistryMixin._requires_draco(model_file))


class ModelAssetTests(unittest.TestCase):
    def test_discovery_uses_manifest_paths_in_nested_directories(self):
        with TemporaryDirectory() as temporary_directory:
            model_dir = Path(temporary_directory)
            sofia_asset = model_dir / "sofia" / "download" / "scene.gltf"
            saturn_asset = model_dir / "saturn_v" / "scene.glb"
            sofia_asset.parent.mkdir(parents=True)
            saturn_asset.parent.mkdir(parents=True)
            sofia_asset.touch()
            saturn_asset.touch()
            (model_dir / "models.json").write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "id": "aircraft",
                                "label": "SOFIA",
                                "file": "sofia/download/scene.gltf",
                                "description": "Aircraft model.",
                                "parts": [],
                            },
                            {
                                "id": "rocket",
                                "label": "Saturn V",
                                "file": "saturn_v/scene.glb",
                                "description": "Rocket model.",
                                "parts": [],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            discovered = ModelRegistryMixin._discover_model_assets(model_dir)

        self.assertEqual(discovered["aircraft"], sofia_asset)
        self.assertEqual(discovered["rocket"], saturn_asset)

    def test_discovery_uses_the_manifest_selected_decompressed_asset(self):
        with TemporaryDirectory() as temporary_directory:
            model_dir = Path(temporary_directory)

            def write_glb(path, requires_draco=False):
                document = {"asset": {"version": "2.0"}}
                if requires_draco:
                    document["extensionsUsed"] = ["KHR_draco_mesh_compression"]
                json_chunk = json.dumps(document).encode("utf-8")
                json_chunk += b" " * (-len(json_chunk) % 4)
                total_length = 12 + 8 + len(json_chunk)
                path.write_bytes(
                    b"glTF"
                    + struct.pack("<II", 2, total_length)
                    + struct.pack("<I4s", len(json_chunk), b"JSON")
                    + json_chunk
                )

            model_names = (
                "mercedes_benz.glb",
                "saturn_v.glb",
                "sofia.glb",
                "cassini_huygens.glb",
            )
            for model_name in model_names[:-1]:
                write_glb(model_dir / model_name)
            write_glb(model_dir / "Cassini-Huygens (B).glb", requires_draco=True)
            decoded_cassini = model_dir / "cassini_huygens.glb"
            write_glb(decoded_cassini)
            (model_dir / "models.json").write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "id": model_id,
                                "label": model_id,
                                "file": model_file,
                                "description": model_id,
                                "parts": [],
                            }
                            for model_id, model_file in zip(
                                ("vehicle", "rocket", "aircraft", "probe"),
                                model_names,
                            )
                        ]
                    }
                ),
                encoding="utf-8",
            )

            discovered = ModelRegistryMixin._discover_model_assets(model_dir)
            self.assertFalse(ModelRegistryMixin._requires_draco(decoded_cassini))

        self.assertEqual(
            set(discovered), {"vehicle", "probe", "rocket", "aircraft"}
        )
        self.assertEqual(discovered["probe"], decoded_cassini)

    def test_fit_to_target_centers_and_normalizes_model_bounds(self):
        model = Mock()
        model.get_tight_bounds.return_value = (
            Point3(-1, -2, -3),
            Point3(3, 2, 1),
        )
        engine = Panda3DEngine.__new__(Panda3DEngine)

        ModelViewMixin._fit_to_target(engine, model)

        scale = ModelViewMixin.TARGET_SIZE / 4
        model.set_scale.assert_called_once_with(scale)
        model.set_pos.assert_called_once_with(Vec3(-scale, 0, scale))


if __name__ == "__main__":
    unittest.main()
