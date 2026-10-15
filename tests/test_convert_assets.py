import tempfile
import unittest
from pathlib import Path

import trimesh

from tools.convert_assets import convert_stl_directory, decompress_cassini


class ConvertAssetsTests(unittest.TestCase):
    def test_stl_directory_exports_one_glb_node_per_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            stl_directory = root / "parts"
            stl_directory.mkdir()
            for name in ("wing", "body"):
                mesh = trimesh.creation.box()
                (stl_directory / f"{name}.stl").write_bytes(
                    mesh.export(file_type="stl")
                )

            output_path = root / "vehicle.glb"
            converted = convert_stl_directory(stl_directory, output_path)
            scene = trimesh.load(output_path, file_type="glb", force="scene")
            converted_again = convert_stl_directory(stl_directory, output_path)

        self.assertTrue(converted)
        self.assertEqual(set(scene.graph.nodes_geometry), {"body", "wing"})
        self.assertFalse(converted_again, "A current output should not be converted again.")

    def test_current_decompressed_cassini_output_is_not_rebuilt(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "cassini_source.glb"
            output_path = root / "cassini_huygens.glb"
            source_path.write_bytes(b"source")
            output_path.write_bytes(
                trimesh.creation.box().export(file_type="glb")
            )

            converted = decompress_cassini(source_path, output_path)

        self.assertFalse(converted)


if __name__ == "__main__":
    unittest.main()
