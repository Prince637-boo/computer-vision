from pathlib import Path

from panda3d.core import NodePath, Shader, TransparencyAttrib


class ModelViewMixin:
    BASE_SCALE = 1.2
    TARGET_SIZE = 3.5
    RENDER_MODES = ("HOLOGRAM", "WIREFRAME", "ORIGINAL")

    def load_car_model(self, model_path: str) -> None:
        """Load a model, apply the selected render mode, and index its parts."""
        model_file = Path(model_path)
        if not model_file.is_file():
            raise FileNotFoundError(f"3D model not found: {model_file}")

        new_model = self.loader.load_model(str(model_file))
        if new_model.is_empty():
            raise RuntimeError(f"Panda3D could not load model: {model_file}")

        geometry_nodes = list(new_model.find_all_matches("**/+GeomNode"))
        if not geometry_nodes:
            new_model.remove_node()
            raise RuntimeError(f"The model contains no geometry: {model_file}")

        self._ensure_hologram_shader()
        previous_model = self.car_model
        if self.grabbed_node is not None:
            self._release_grabbed_part()

        pivot = self.render.attach_new_node("model_pivot")
        new_model.reparent_to(pivot)
        new_model.set_two_sided(True)
        self._fit_to_target(new_model)
        pivot.set_pos(0, 5, 0)
        pivot.set_hpr(0, 0, 0)
        pivot.set_scale(self.BASE_SCALE)
        self.car_model = pivot
        self._model_name = self._name_for_model_path(model_file)
        self._anchor_position = pivot.get_pos()
        self._target_scale = self.BASE_SCALE
        self._current_scale = self.BASE_SCALE
        self._spin_h = 0.0
        self._spin_p = 0.0
        self._previous_rotation_pos = None
        self._explosion_target = 0.0
        self._explosion_amount = 0.0
        self._index_parts(geometry_nodes)
        self._prepare_explosion(geometry_nodes)
        self._apply_render_mode()
        if previous_model is not None:
            previous_model.remove_node()
        self._refresh_model_buttons()
        self._show_controls()

    def _ensure_hologram_shader(self) -> None:
        if self._hologram_shader is not None:
            return
        shader_dir = Path(__file__).resolve().parents[2] / "assets" / "shaders"
        vertex_source = (shader_dir / "hologram.vert").read_text(encoding="utf-8")
        fragment_source = (shader_dir / "hologram.frag").read_text(encoding="utf-8")
        shader = Shader.make(Shader.SL_GLSL, vertex_source, fragment_source)
        if shader is None:
            raise RuntimeError("Panda3D could not create the hologram shader.")
        self._hologram_shader = shader

    def _fit_to_target(self, model: NodePath) -> None:
        bounds = model.get_tight_bounds()
        if bounds is None:
            return
        lower, upper = bounds
        size = max(
            upper.x - lower.x,
            upper.y - lower.y,
            upper.z - lower.z,
        )
        if size <= 0:
            return

        scale = self.TARGET_SIZE / size
        center = (lower + upper) / 2
        model.set_scale(scale)
        model.set_pos(-center * scale)

    def _name_for_model_path(self, model_file: Path) -> str:
        resolved_path = model_file.resolve()
        for model_id, label, _ in self.MODEL_CATALOG:
            if self._model_paths.get(model_id) == resolved_path:
                return label
        return model_file.stem.replace("_", " ").upper()

    def select_model(self, model_id: str) -> None:
        model_file = self._model_paths.get(model_id)
        if model_file is None:
            reason = self._model_unavailable_reasons.get(model_id, "ASSET ABSENT")
            raise FileNotFoundError(
                f"Model {model_id} is unavailable: {reason}."
            )
        self.load_car_model(str(model_file))

    def set_render_mode(self, mode: str) -> None:
        if mode not in self.RENDER_MODES:
            raise ValueError(f"Unsupported render mode: {mode}")
        if mode == "HOLOGRAM":
            self._ensure_hologram_shader()
        if self.grabbed_node is not None:
            self._release_grabbed_part()
        self._render_mode = mode
        self._apply_render_mode()
        self._refresh_mode_buttons()
        self._show_controls()

    def _apply_render_mode(self) -> None:
        if self.car_model is None:
            return
        model = self.car_model
        model.set_render_mode_filled()
        model.clear_color()
        model.clear_color_scale()
        model.clear_transparency()
        model.clear_depth_write()
        model.clear_shader()
        if self._render_mode == "HOLOGRAM":
            self._ensure_hologram_shader()
            model.set_shader(self._hologram_shader)
            model.set_transparency(TransparencyAttrib.M_alpha)
            model.set_depth_write(False)
        elif self._render_mode == "WIREFRAME":
            model.set_render_mode_wireframe(2)
            model.set_color(0.18, 0.82, 1.0, 1)
