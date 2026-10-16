import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import trimesh

from src.interactor.model_registry import ModelRegistry
from src.interactor.training import TrainingPhase, TrainingSession, TrainingStep


class ModelRegistryTests(unittest.TestCase):
    def test_project_manifest_has_four_valid_models_and_safe_asset_paths(self):
        manifest_path = (
            Path(__file__).resolve().parents[1]
            / "assets"
            / "models"
            / "models.json"
        )

        registry = ModelRegistry.load(manifest_path)

        self.assertEqual(
            set(registry.by_id), {"vehicle", "probe", "rocket", "aircraft"}
        )
        self.assertFalse(registry.errors)
        for model in registry.models:
            asset_path = registry.assets_dir / model.file
            self.assertTrue(asset_path.is_file(), model.file)
            self.assertEqual(asset_path.resolve().parent, registry.assets_dir.resolve())
            geometry_names = trimesh.load(
                asset_path, force="scene"
            ).geometry.keys()
            for part in model.parts:
                self.assertTrue(
                    any(
                        part.match.casefold() in name.casefold()
                        for name in geometry_names
                    ),
                    f"{model.id} part {part.match!r} has no matching mesh",
                )
            for step in model.training_steps:
                self.assertTrue(
                    any(
                        step.casefold() in name.casefold()
                        for name in geometry_names
                    ),
                    f"{model.id} training step {step!r} has no matching mesh",
                )

    def test_invalid_entry_is_skipped_while_valid_entries_remain_available(self):
        with TemporaryDirectory() as directory:
            manifest = Path(directory) / "models.json"
            manifest.write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "id": "good",
                                "label": "Good",
                                "file": "good.glb",
                                "description": "A valid model.",
                                "parts": [],
                            },
                            {
                                "id": "bad",
                                "label": "Bad",
                                "file": "../outside.glb",
                                "description": "Unsafe path.",
                                "parts": [],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            registry = ModelRegistry.load(manifest)

        self.assertEqual(set(registry.by_id), {"good"})
        self.assertEqual(len(registry.errors), 1)
        self.assertIn("bad", registry.errors[0])
        self.assertIn("safe path", registry.errors[0])

    def test_invalid_part_axis_invalidates_only_its_model(self):
        with TemporaryDirectory() as directory:
            manifest = Path(directory) / "models.json"
            manifest.write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "id": "bad",
                                "label": "Bad",
                                "file": "bad.glb",
                                "description": "Invalid part.",
                                "parts": [
                                    {
                                        "match": "engine",
                                        "label": "Engine",
                                        "info": "",
                                        "explode_axis": [0, 0, 0],
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            registry = ModelRegistry.load(manifest)

        self.assertEqual(registry.models, ())
        self.assertIn("non-zero vector", registry.errors[0])


class TrainingSessionTests(unittest.TestCase):
    def setUp(self):
        self.session = TrainingSession(
            [
                TrainingStep("engine", "Moteur"),
                TrainingStep("wheel", "Roue"),
            ],
            ["Moteur", "Roue", "Aile"],
        )

    def test_only_the_current_step_part_advances_the_disassembly(self):
        self.assertFalse(self.session.select_part("front wheel"))
        self.assertEqual(self.session.step_index, 0)
        self.assertTrue(self.session.select_part("vehicle ENGINE block"))
        self.assertEqual(self.session.current_step.label, "Roue")
        self.assertTrue(self.session.select_part("left wheel"))
        self.assertEqual(self.session.phase, TrainingPhase.QUIZ)
        self.assertEqual(self.session.current_question, "Moteur")

    def test_final_quiz_scores_answers_and_completes(self):
        self.session.select_part("engine")
        self.session.select_part("wheel")

        choices = self.session.quiz_choices
        self.assertIn("Moteur", choices)
        self.assertFalse(self.session.answer_quiz("Aile"))
        self.assertEqual(self.session.current_question, "Roue")
        self.assertTrue(self.session.answer_quiz("Roue"))
        self.assertEqual(self.session.phase, TrainingPhase.COMPLETE)
        self.assertEqual(self.session.score, 1)
        self.assertEqual(self.session.quiz_choices, ())

    def test_training_requires_steps_and_known_quiz_labels(self):
        with self.assertRaisesRegex(ValueError, "at least one step"):
            TrainingSession([], ["Moteur"])
        with self.assertRaisesRegex(ValueError, "quiz answer option"):
            TrainingSession([TrainingStep("engine", "Moteur")], ["Roue"])


if __name__ == "__main__":
    unittest.main()
