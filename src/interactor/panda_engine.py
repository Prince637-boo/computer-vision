import json
import math
import struct
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from direct.gui.DirectGui import DGG, DirectButton, DirectFrame, DirectLabel
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    AmbientLight,
    CardMaker,
    ClockObject,
    DirectionalLight,
    NodePath,
    Point2,
    Point3,
    Shader,
    TextNode,
    Texture,
    TransparencyAttrib,
    Vec3,
    WindowProperties,
)

import gltf  # Registers Panda3D's glTF loader.
from src.config import RENDER_FPS
from src.vision.gestures import GestureType


class Panda3DEngine(ShowBase):
    """Render selectable 3D assets and translate hand gestures into direct controls."""

    BASE_SCALE = 1.2
    TARGET_SIZE = 3.5
    EXPLOSION_DISTANCE = 2.4
    TOUCH_PLANE_Y = 5.0  # Profondeur fixe de l'écran tactile virtuel
    FIST_HOLD_SECONDS = 0.5
    DEAD_ZONE = 0.002
    RENDER_MODES = ("HOLOGRAM", "WIREFRAME", "ORIGINAL")
    MODEL_CATALOG = (
        ("vehicle", "MERCEDES 300 SEL", ("mercedes",)),
        ("probe", "CASSINI-HUYGENS", ("cassini", "huygens")),
        ("rocket", "SATURN V", ("saturnv", "saturn5")),
        ("aircraft", "SOFIA", ("sofia",)),
    )
    MODEL_EXTENSIONS = {".bam", ".glb", ".gltf", ".obj"}

    _PART_CATEGORIES = (
        ("CALIPER", "Brake caliper"),
        ("ENGINE", "Engine"),
        ("DISC", "Brake disc"),
        ("WHEEL", "Wheel"),
        ("SUSP", "Suspension"),
        ("BONNET", "Bonnet"),
        ("BOOT", "Boot"),
        ("BODY", "Body panel"),
        ("CHASSIS", "Chassis"),
    )

    _PART_DETAILS = {
        "CALIPER": (
            "Hydraulic brake component that clamps the disc.",
            "Exact dimensions and compound are not stored in the model.",
        ),
        "DISC": (
            "Rotating friction surface used by the braking system.",
            "Disc material and dimensions are not stored in the model.",
        ),
        "ENGINE": (
            "Mercedes-Benz 300 SEL AMG model-asset power unit.",
            "Exact displacement and output are not embedded in the mesh.",
        ),
        "WHEEL": (
            "Wheel and tire assembly transferring torque to the road.",
            "Tire pressure and fitment data are not stored in the model.",
        ),
        "SUSP": (
            "Suspension assembly supporting the wheel and controlling motion.",
            "Spring and damper settings are not stored in the model.",
        ),
        "BONNET": ("Front body panel covering the engine bay.", ""),
        "BOOT": ("Rear luggage-compartment body panel.", ""),
        "BODY": ("Vehicle bodywork and exterior panels.", ""),
        "CHASSIS": ("Structural vehicle frame and supporting components.", ""),
    }

    _EXPLOSION_AXES = {
        "CALIPER": (1, 0, 0),
        "DISC": (1, 0, 0),
        "ENGINE": (0, 0, 1),
        "WHEEL": (1, 0, 0),
        "SUSP": (1, 0, 0),
        "BONNET": (0, 1, 0),
        "BOOT": (0, -1, 0),
        "CHASSIS": (0, 0, -1),
    }

    def __init__(self):
        super().__init__()

        props = WindowProperties()
        props.setTitle("AR Holographic Lab | Interactive 3D Viewer")
        props.setSize(960, 720)
        if hasattr(self.win, "requestProperties"):
            self.win.requestProperties(props)
        self.set_background_color(0, 0, 0, 1)

        clock = ClockObject.get_global_clock()
        clock.set_mode(ClockObject.MLimited)
        clock.set_frame_rate(float(RENDER_FPS))

        self.cam2d.node().get_display_region(0).set_sort(-10)
        self.bg_texture = Texture("camera_feed")
        self._bg_ready = False
        card = CardMaker("camera_background")
        card.set_frame(-1, 1, -1, 1)
        self.bg_card = self.render2d.attach_new_node(card.generate())
        self.bg_card.set_texture(self.bg_texture)

        ambient_light = AmbientLight("studio_ambient")
        ambient_light.set_color((0.42, 0.46, 0.52, 1))
        self.render.set_light(self.render.attach_new_node(ambient_light))
        key_light = DirectionalLight("studio_key")
        key_light.set_color((0.82, 0.88, 1.0, 1))
        key_light_path = self.render.attach_new_node(key_light)
        key_light_path.set_hpr(-35, -35, 0)
        self.render.set_light(key_light_path)

        self.disable_mouse()
        self.camera.set_pos(0, -12, 1)
        self.camera.look_at(0, 4, 0)

        self.car_model: Optional[NodePath] = None
        self._hologram_shader: Optional[Shader] = None
        self._render_mode = "HOLOGRAM"
        self._model_name = "MODEL"
        self._model_paths = self._discover_model_assets()
        self._model_unavailable_reasons: Dict[str, str] = {}
        for model_id, model_file in tuple(self._model_paths.items()):
            if self._requires_draco(model_file):
                self._model_paths.pop(model_id)
                self._model_unavailable_reasons[model_id] = "DRACO NON SUPPORTE"
        self._anchor_enabled = False
        self._anchor_position = Vec3(0, 5, 0)
        self.parts_dict: Dict[str, NodePath] = {}
        self.grabbed_node: Optional[NodePath] = None
        self.grab_offset = Vec3(0, 0, 0)
        self._previous_rotation_pos: Optional[Tuple[float, float]] = None
        self._spin_h = 0.0
        self._spin_p = 0.0
        self._hand_sample_time: Optional[float] = None
        self._hand_sample_interval = 1.0 / RENDER_FPS
        self._previous_pinch_pos: Optional[Tuple[float, float]] = None
        self._current_pinch_pos: Optional[Tuple[float, float]] = None
        self._previous_palm_pos: Optional[Tuple[float, float]] = None
        self._current_palm_pos: Optional[Tuple[float, float]] = None
        self._previous_primary_wrist_pos: Optional[Tuple[float, float]] = None
        self._current_primary_wrist_pos: Optional[Tuple[float, float]] = None
        self._previous_second_wrist_pos: Optional[Tuple[float, float]] = None
        self._current_second_wrist_pos: Optional[Tuple[float, float]] = None
        self._fist_time = 0.0
        self._fist_fired = False
        self._explosion_target = 0.0
        self._explosion_amount = 0.0
        self._explosion_center = Vec3(0, 0, 0)
        self._explosion_parts: List[
            Tuple[NodePath, Vec3, Vec3, Vec3]
        ] = []
        self._base_scale = self.BASE_SCALE
        self._target_scale = self.BASE_SCALE
        self._current_scale = self.BASE_SCALE
        self._zoom_start_distance: Optional[float] = None
        self._zoom_start_scale = self.BASE_SCALE
        self._main_status = "MAIN: NO HAND -> WAIT"
        self._hud_detail = ""

        self.hand_cursor = self.loader.load_model("models/misc/sphere")
        if self.hand_cursor:
            self.hand_cursor.reparent_to(self.render)
            self.hand_cursor.set_scale(0.12)
            self.hand_cursor.set_color_scale(0.0, 0.95, 1.0, 0.85)
            self.hand_cursor.set_transparency(TransparencyAttrib.M_alpha)
            self.hand_cursor.hide()

        self._hud_text = ""
        self.hud = TextNode("interaction_hud")
        self.hud.set_text_color(0.42, 0.95, 1.0, 1.0)
        self.hud.set_shadow(0.02, 0.02)
        self.hud.set_shadow_color(0.0, 0.8, 1.0, 0.65)
        self.hud.set_card_color(0.01, 0.04, 0.07, 0.82)
        self.hud.set_card_as_margin(0.12, 0.12, 0.1, 0.1)
        self.hud.set_card_decal(True)
        self.hud.set_wordwrap(42)
        self.hud_path = self.aspect2d.attach_new_node(self.hud)
        self.hud_path.set_pos(-1.72, 0, 0.45)
        self.hud_path.set_scale(0.04)
        self._build_control_panel()
        self._show_controls()

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

    def _name_for_model_path(self, model_file: Path) -> str:
        resolved_path = model_file.resolve()
        for model_id, label, _ in self.MODEL_CATALOG:
            if self._model_paths.get(model_id) == resolved_path:
                return label
        return model_file.stem.replace("_", " ").upper()

    def _build_control_panel(self) -> None:
        self.control_panel = DirectFrame(
            parent=self.aspect2d,
            frameColor=(0.015, 0.055, 0.085, 0.94),
            frameSize=(-1.72, 1.72, -0.22, 0.22),
            pos=(0, 0, 0.77),
            relief=DGG.FLAT,
        )
        DirectLabel(
            parent=self.control_panel,
            text="LABORATOIRE AR  /  CONTROLE DES MODELES",
            text_fg=(0.5, 0.9, 1, 1),
            text_scale=0.038,
            frameColor=(0, 0, 0, 0),
            pos=(0, 0, 0.16),
        )

        model_positions = (-1.25, -0.42, 0.42, 1.25)
        self._model_buttons: Dict[str, DirectButton] = {}
        for (model_id, label, _), x_pos in zip(
            self.MODEL_CATALOG, model_positions
        ):
            available = model_id in self._model_paths
            button_label = label if available else f"{label} / ABSENT"
            self._model_buttons[model_id] = DirectButton(
                parent=self.control_panel,
                text=button_label,
                text_fg=(0.76, 0.91, 0.96, 1),
                text_scale=0.027,
                textMayChange=True,
                frameColor=(0.04, 0.16, 0.21, 1),
                frameSize=(-0.37, 0.37, -0.04, 0.04),
                pos=(x_pos, 0, 0.055),
                relief=DGG.FLAT,
                state=DGG.NORMAL if available else DGG.DISABLED,
                command=self.select_model,
                extraArgs=[model_id],
            )

        mode_positions = (-1.12, -0.38, 0.38, 1.12)
        mode_labels = ("HOLOGRAMME", "FILAIRE", "ORIGINAL", "ANCRAGE OFF")
        self._mode_buttons: Dict[str, DirectButton] = {}
        for mode, label, x_pos in zip(
            (*self.RENDER_MODES, "ANCHOR"),
            mode_labels,
            mode_positions,
        ):
            command = (
                self.toggle_anchor
                if mode == "ANCHOR"
                else self.set_render_mode
            )
            self._mode_buttons[mode] = DirectButton(
                parent=self.control_panel,
                text=label,
                text_fg=(0.63, 0.82, 0.87, 1),
                text_scale=0.029,
                textMayChange=True,
                frameColor=(0.025, 0.105, 0.145, 1),
                frameSize=(-0.34, 0.34, -0.04, 0.04),
                pos=(x_pos, 0, -0.105),
                relief=DGG.FLAT,
                command=command,
                extraArgs=[] if mode == "ANCHOR" else [mode],
            )
        self._refresh_model_buttons()
        self._refresh_mode_buttons()

    def _refresh_model_buttons(self) -> None:
        for model_id, button in self._model_buttons.items():
            available = model_id in self._model_paths
            label = next(
                name for key, name, _ in self.MODEL_CATALOG if key == model_id
            )
            unavailable = self._model_unavailable_reasons.get(model_id, "ABSENT")
            button["text"] = (
                label if available else f"{label} / {unavailable}"
            )
            if available:
                button["frameColor"] = (
                    (0.025, 0.3, 0.36, 1)
                    if label == self._model_name
                    else (0.04, 0.16, 0.21, 1)
                )

    def _refresh_mode_buttons(self) -> None:
        for mode, button in self._mode_buttons.items():
            selected = (
                self._anchor_enabled if mode == "ANCHOR" else mode == self._render_mode
            )
            button["frameColor"] = (
                (0.025, 0.3, 0.36, 1)
                if selected
                else (0.025, 0.105, 0.145, 1)
            )
        self._mode_buttons["ANCHOR"]["text"] = (
            "ANCRAGE ON" if self._anchor_enabled else "ANCRAGE OFF"
        )

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

    def toggle_anchor(self) -> None:
        self._anchor_enabled = not self._anchor_enabled
        self._anchor_position = (
            self.car_model.get_pos() if self.car_model is not None else Vec3(0, 5, 0)
        )
        self._spin_h = 0.0
        self._spin_p = 0.0
        self._previous_rotation_pos = None
        if self._anchor_enabled:
            self._release_grabbed_part()
        self._refresh_mode_buttons()
        self._show_controls()
        self._set_main_status(
            "ANCRAGE ACTIF : balayez la paume pour orienter"
            if self._anchor_enabled
            else "ANCRAGE DESACTIVE"
        )

    def _index_parts(self, geometry_nodes: List[NodePath]) -> None:
        self.parts_dict.clear()
        for node in geometry_nodes:
            name = node.get_name()
            if self._part_category(name) is not None:
                self.parts_dict[f"{name}_{len(self.parts_dict)}"] = node

    def _prepare_explosion(self, geometry_nodes: List[NodePath]) -> None:
        if self.car_model is None:
            return

        centers = [
            self.car_model.get_relative_point(node, node.get_bounds().get_center())
            for node in geometry_nodes
        ]
        if centers:
            self._explosion_center = Vec3(
                (min(point.x for point in centers) + max(point.x for point in centers)) / 2,
                (min(point.y for point in centers) + max(point.y for point in centers)) / 2,
                (min(point.z for point in centers) + max(point.z for point in centers)) / 2,
            )

        self._explosion_parts.clear()
        for index, node in enumerate(geometry_nodes):
            base_pos = node.get_pos(self.car_model)
            direction = self._explosion_direction(
                node.get_name(), centers[index], self._explosion_center
            )
            self._explosion_parts.append(
                (node, base_pos, node.get_hpr(self.car_model), direction)
            )

    @classmethod
    def _explosion_direction(
        cls, node_name: str, part_center: Point3, model_center: Vec3
    ) -> Vec3:
        category = cls._part_category(node_name)
        if category in cls._EXPLOSION_AXES:
            direction = Vec3(*cls._EXPLOSION_AXES[category])
            if direction.x:
                if abs(part_center.x - model_center.x) > 0.08:
                    direction.x *= 1 if part_center.x > model_center.x else -1
                else:
                    normalized_name = node_name.upper()
                    direction.x *= (
                        -1
                        if any(
                            token in normalized_name
                            for token in ("LEFT", "_L", " LH", "LH_")
                        )
                        else 1
                    )
            return direction

        direction = Vec3(part_center - model_center)
        if direction.length() < 0.15:
            return Vec3(0, 0, 1)
        direction.normalize()
        return direction

    def update_background(self, frame: Tuple[int, int, bytes]) -> None:
        """Upload a camera frame already converted by the capture worker."""
        width, height, rgb_bytes = frame
        if not self._bg_ready:
            self.bg_texture.setup_2d_texture(
                width, height, Texture.T_unsigned_byte, Texture.F_rgb
            )
            self._bg_ready = True
        self.bg_texture.set_ram_image(rgb_bytes)

    def update_hand_interaction(
        self,
        hand_norm_pos: Tuple[float, float],
        is_pinching: bool,
        second_hand_norm_pos: Optional[Tuple[float, float]] = None,
        gesture: GestureType = GestureType.IDLE,
        palm_norm_pos: Optional[Tuple[float, float]] = None,
        primary_wrist_norm_pos: Optional[Tuple[float, float]] = None,
        hand_sample_time: Optional[float] = None,
    ) -> None:
        """Apply current hand controls using a direct touch-plane mapping."""
        if self.car_model is None:
            return

        dt = min(max(ClockObject.get_global_clock().get_dt(), 1 / 120), 0.1)
        if gesture == GestureType.NONE:
            self._hand_sample_time = None
            self._previous_pinch_pos = None
            self._current_pinch_pos = None
            self._previous_palm_pos = None
            self._current_palm_pos = None
            self._previous_primary_wrist_pos = None
            self._current_primary_wrist_pos = None
            self._previous_second_wrist_pos = None
            self._current_second_wrist_pos = None
            self._previous_rotation_pos = None
            self._zoom_start_distance = None
            self._fist_time = 0.0
            self._fist_fired = False
            self._set_main_status("MAIN: NO HAND -> WAIT")
            self._release_grabbed_part()
            if self.hand_cursor:
                self.hand_cursor.hide()
            self._update_rotation(None, dt)
            self._update_explosion(dt)
            self._update_scale(dt)
            if self._anchor_enabled:
                self.car_model.set_pos(self._anchor_position)
            return

        if hand_sample_time is not None and hand_sample_time != self._hand_sample_time:
            if self._hand_sample_time is not None:
                self._hand_sample_interval = max(
                    hand_sample_time - self._hand_sample_time,
                    1.0 / RENDER_FPS,
                )
                self._previous_pinch_pos = self._current_pinch_pos
                self._previous_palm_pos = self._current_palm_pos
                self._previous_primary_wrist_pos = self._current_primary_wrist_pos
                self._previous_second_wrist_pos = self._current_second_wrist_pos
            else:
                self._previous_pinch_pos = hand_norm_pos
                self._previous_palm_pos = palm_norm_pos or hand_norm_pos
                self._previous_primary_wrist_pos = primary_wrist_norm_pos
                self._previous_second_wrist_pos = second_hand_norm_pos
            self._current_pinch_pos = hand_norm_pos
            self._current_palm_pos = palm_norm_pos or hand_norm_pos
            self._current_primary_wrist_pos = primary_wrist_norm_pos
            self._current_second_wrist_pos = second_hand_norm_pos
            self._hand_sample_time = hand_sample_time
        elif hand_sample_time is None:
            self._hand_sample_time = None
            self._previous_pinch_pos = None
            self._current_pinch_pos = None
            self._previous_palm_pos = None
            self._current_palm_pos = None
            self._previous_primary_wrist_pos = None
            self._current_primary_wrist_pos = None
            self._previous_second_wrist_pos = None
            self._current_second_wrist_pos = None

        (
            interpolated_pinch_pos,
            interpolated_palm_pos,
            interpolated_primary_wrist,
            interpolated_second_wrist,
        ) = self._interpolated_hand_positions(
            hand_norm_pos,
            palm_norm_pos or hand_norm_pos,
            primary_wrist_norm_pos,
            second_hand_norm_pos,
        )

        two_open_palms = (
            gesture == GestureType.OPEN_PALM and second_hand_norm_pos is not None
        )
        if self._anchor_enabled and two_open_palms:
            self._set_main_status("ANCRAGE : DEUX PAUMES -> ZOOM")
        elif self._anchor_enabled and gesture == GestureType.OPEN_PALM:
            self._set_main_status("ANCRAGE : BALAYEZ POUR ORIENTER")
        elif self._anchor_enabled and gesture == GestureType.FIST:
            action = "ASSEMBLER" if self._explosion_target else "ECLATER"
            self._set_main_status(f"ANCRAGE : POING -> {action}")
        elif self._anchor_enabled:
            self._set_main_status("ANCRAGE : PAUME OUVERTE POUR ORIENTER")
        elif two_open_palms:
            self._set_main_status("MAIN: TWO OPEN PALMS -> ZOOM")
        elif gesture == GestureType.OPEN_PALM:
            self._set_main_status("MAIN: OPEN PALM -> ROTATE")
        elif gesture == GestureType.PINCH and is_pinching:
            self._set_main_status("MAIN: PINCH -> GRAB")
        elif gesture == GestureType.FIST:
            if self._fist_fired:
                action = (
                    "REASSEMBLE"
                    if self._explosion_target == 0
                    else "EXPLODE"
                )
                self._set_main_status(f"MAIN: FIST -> {action}")
            else:
                self._set_main_status(
                    f"MAIN: FIST -> HOLD {self._fist_time:.1f}/{self.FIST_HOLD_SECONDS:.1f}s"
                )
        else:
            self._set_main_status(f"MAIN: {gesture.value} -> WAIT")

        cursor_pos = (
            interpolated_palm_pos
            if gesture == GestureType.OPEN_PALM
            else interpolated_pinch_pos
        )
        target_pos = self._project_hand_to_touch_plane(cursor_pos)
        if self.hand_cursor:
            self.hand_cursor.show()
            self.hand_cursor.set_pos(target_pos)
            if (
                gesture == GestureType.PINCH
                and is_pinching
                and not self._anchor_enabled
            ):
                self.hand_cursor.set_color_scale(0.1, 1.0, 0.35, 0.9)
            elif gesture == GestureType.FIST and not self._fist_fired:
                self.hand_cursor.set_color_scale(1.0, 0.48, 0.08, 0.95)
            else:
                self.hand_cursor.set_color_scale(0.0, 0.95, 1.0, 0.85)

        if two_open_palms:
            self._update_zoom(
                interpolated_primary_wrist or interpolated_palm_pos,
                interpolated_second_wrist or second_hand_norm_pos,
            )
            self._previous_rotation_pos = None
            self._update_rotation(None, dt)
        else:
            self._zoom_start_distance = None
            if gesture == GestureType.OPEN_PALM:
                self._update_rotation(interpolated_palm_pos, dt)
            else:
                self._previous_rotation_pos = None
                self._update_rotation(None, dt)

        if gesture == GestureType.FIST:
            self._fist_time += dt
            if self._fist_time >= self.FIST_HOLD_SECONDS and not self._fist_fired:
                self._explosion_target = (
                    0.0 if self._explosion_target > 0 else 1.0
                )
                self._release_grabbed_part()
                self._fist_fired = True
                action = "REASSEMBLE" if self._explosion_target == 0 else "EXPLODE"
                self._set_main_status(f"MAIN: FIST -> {action}")
        else:
            self._fist_time = 0.0
            self._fist_fired = False

        if (
            is_pinching
            and gesture == GestureType.PINCH
            and self._explosion_target == 0
            and not self._anchor_enabled
        ):
            self._grab_or_move_part(target_pos)
        else:
            self._release_grabbed_part()

        self._update_explosion(dt)
        self._update_scale(dt)

    def _interpolated_hand_positions(
        self,
        pinch_pos: Tuple[float, float],
        palm_pos: Tuple[float, float],
        primary_wrist_pos: Optional[Tuple[float, float]],
        second_wrist_pos: Optional[Tuple[float, float]],
    ) -> Tuple[
        Tuple[float, float],
        Tuple[float, float],
        Optional[Tuple[float, float]],
        Optional[Tuple[float, float]],
    ]:
        if self._hand_sample_time is None:
            return pinch_pos, palm_pos, primary_wrist_pos, second_wrist_pos

        alpha = min(
            max(
                (time.monotonic() - self._hand_sample_time)
                / self._hand_sample_interval,
                0.0,
            ),
            1.0,
        )
        if self._previous_pinch_pos is None or self._current_pinch_pos is None:
            return pinch_pos, palm_pos, primary_wrist_pos, second_wrist_pos

        def interpolate(
            previous: Tuple[float, float], current: Tuple[float, float]
        ) -> Tuple[float, float]:
            return (
                previous[0] + (current[0] - previous[0]) * alpha,
                previous[1] + (current[1] - previous[1]) * alpha,
            )

        interpolated_primary = primary_wrist_pos
        if (
            self._previous_primary_wrist_pos is not None
            and self._current_primary_wrist_pos is not None
        ):
            interpolated_primary = interpolate(
                self._previous_primary_wrist_pos,
                self._current_primary_wrist_pos,
            )

        interpolated_second = second_wrist_pos
        if (
            self._previous_second_wrist_pos is not None
            and self._current_second_wrist_pos is not None
        ):
            interpolated_second = interpolate(
                self._previous_second_wrist_pos,
                self._current_second_wrist_pos,
            )

        return (
            interpolate(self._previous_pinch_pos, self._current_pinch_pos),
            interpolate(
                self._previous_palm_pos or palm_pos,
                self._current_palm_pos or palm_pos,
            ),
            interpolated_primary,
            interpolated_second,
        )

    def _update_zoom(
        self, first_hand: Tuple[float, float], second_hand: Tuple[float, float]
    ) -> None:
        distance = math.hypot(
            second_hand[0] - first_hand[0], second_hand[1] - first_hand[1]
        )
        if self._zoom_start_distance is None:
            self._zoom_start_distance = max(distance, 0.05)
            self._zoom_start_scale = self._target_scale
        zoom_ratio = distance / self._zoom_start_distance
        self._target_scale = self._zoom_start_scale * min(
            max(zoom_ratio, 0.65), 1.8
        )

    def _update_rotation(
        self, palm_pos: Optional[Tuple[float, float]], dt: float
    ) -> None:
        if self.car_model is None:
            return
        if self._anchor_enabled:
            if palm_pos is None:
                self._previous_rotation_pos = None
                self._spin_h = 0.0
                self._spin_p = 0.0
                return
            if self._previous_rotation_pos is not None:
                delta_x = palm_pos[0] - self._previous_rotation_pos[0]
                delta_y = palm_pos[1] - self._previous_rotation_pos[1]
                if abs(delta_x) >= self.DEAD_ZONE:
                    self.car_model.set_h(self.car_model.get_h() - delta_x * 220)
                if abs(delta_y) >= self.DEAD_ZONE:
                    self.car_model.set_p(
                        min(
                            max(self.car_model.get_p() + delta_y * 140, -45),
                            45,
                        )
                    )
            self._previous_rotation_pos = palm_pos
            self.car_model.set_pos(self._anchor_position)
            return

        if palm_pos is not None and palm_pos != self._previous_rotation_pos:
            if self._previous_rotation_pos is not None:
                delta_x = palm_pos[0] - self._previous_rotation_pos[0]
                delta_y = palm_pos[1] - self._previous_rotation_pos[1]
                self._spin_h = -delta_x * 220 if abs(delta_x) >= self.DEAD_ZONE else 0
                self._spin_p = delta_y * 140 if abs(delta_y) >= self.DEAD_ZONE else 0
            self._previous_rotation_pos = palm_pos

        self.car_model.set_h(self.car_model.get_h() + self._spin_h)
        self.car_model.set_p(
            min(max(self.car_model.get_p() + self._spin_p, -45), 45)
        )
        decay = 0.9 ** (dt * RENDER_FPS)
        self._spin_h *= decay
        self._spin_p *= decay

    def _project_hand_to_touch_plane(self, hand_pos: Tuple[float, float]) -> Point3:
        """Projette directement la main sur un plan plat virtuel face à la caméra (effet tactile)."""
        near = Point3()
        far = Point3()
        if not self.camLens.extrude(Point2(hand_pos[0], hand_pos[1]), near, far):
            return Point3(0, self.TOUCH_PLANE_Y, 0)

        ray_origin = self.render.get_relative_point(self.cam, near)
        ray_end = self.render.get_relative_point(self.cam, far)
        direction = ray_end - ray_origin
        direction.normalize()

        # Interception directe sur le plan fixe Y de la voiture
        if abs(direction.y) > 0.0001:
            t = (self.TOUCH_PLANE_Y - ray_origin.y) / direction.y
            return ray_origin + direction * t

        return ray_origin + direction * 17.0

    def _grab_or_move_part(self, target_pos: Vec3) -> None:
        if self.car_model is None:
            return
        if self.grabbed_node is None:
            best_part: Optional[NodePath] = None
            min_dist = 2.0
            for part_node in self.parts_dict.values():
                center = part_node.get_bounds().get_center()
                world_pos = self.render.get_relative_point(part_node, center)
                distance = (world_pos - target_pos).length()
                if distance < min_dist:
                    min_dist = distance
                    best_part = part_node
            if best_part is not None:
                self.grabbed_node = best_part
                self.grab_offset = best_part.get_pos(self.render) - target_pos
                self.grabbed_node.set_color_scale(1.2, 1.55, 1.7, 1.0)
                self._show_part_details(self.grabbed_node.get_name())

        if self.grabbed_node is not None:
            self.grabbed_node.set_pos(self.render, target_pos + self.grab_offset)

    def _release_grabbed_part(self) -> None:
        if self.grabbed_node is None:
            return
        released_node = self.grabbed_node
        released_node.set_color_scale(1, 1, 1, 1)
        self.grabbed_node = None
        self._refresh_explosion_part(released_node)
        self._show_controls()

    def _refresh_explosion_part(self, node: NodePath) -> None:
        if self.car_model is None:
            return
        for index, (part, _, _, _) in enumerate(self._explosion_parts):
            if part != node:
                continue
            base_pos = node.get_pos(self.car_model)
            part_center = self.car_model.get_relative_point(
                node, node.get_bounds().get_center()
            )
            direction = self._explosion_direction(
                node.get_name(), part_center, self._explosion_center
            )
            self._explosion_parts[index] = (
                node,
                base_pos,
                node.get_hpr(self.car_model),
                direction,
            )
            break

    def _update_explosion(self, dt: float) -> None:
        if self.car_model is None or not self._explosion_parts:
            return
        previous_amount = self._explosion_amount
        alpha = 1.0 - math.exp(-dt * 5.0)
        self._explosion_amount += (
            self._explosion_target - self._explosion_amount
        ) * alpha
        if abs(self._explosion_target - self._explosion_amount) < 0.003:
            self._explosion_amount = self._explosion_target
        if abs(previous_amount - self._explosion_amount) < 0.0001:
            return

        amount = self._explosion_amount
        for node, base_pos, base_hpr, direction in self._explosion_parts:
            node.set_pos(
                self.car_model,
                base_pos + direction * self.EXPLOSION_DISTANCE * amount,
            )
            node.set_hpr(self.car_model, base_hpr)

        if amount == 1.0:
            self._set_hud(
                "EXPLODED VIEW\n"
                "Components separated along technical axes.\n"
                "Close your fist again to reassemble."
            )
        elif amount == 0.0:
            self._show_controls()

    def _update_scale(self, dt: float) -> None:
        if self.car_model is None:
            return
        alpha = 1.0 - math.exp(-dt / 0.12)
        self._current_scale = self._approach(
            self._current_scale, self._target_scale, alpha
        )
        self.car_model.set_scale(self._current_scale)

    @staticmethod
    def _approach(current: float, target: float, alpha: float) -> float:
        return current + (target - current) * alpha

    @classmethod
    def _part_category(cls, node_name: str) -> Optional[str]:
        normalized_name = node_name.upper()
        for token, _ in cls._PART_CATEGORIES:
            if token in normalized_name:
                return token
        return None

    def _show_part_details(self, node_name: str) -> None:
        category = self._part_category(node_name)
        if category is None:
            self._show_controls()
            return
        title = dict(self._PART_CATEGORIES)[category]
        detail, note = self._PART_DETAILS[category]
        lines = [title.upper(), node_name, detail]
        if note:
            lines.append(note)
        lines.append("Release the pinch to continue.")
        self._set_hud("\n".join(lines))

    def _show_controls(self) -> None:
        if self._anchor_enabled:
            lines = [
                f"{self._model_name}  /  {self._render_mode}  /  ANCRAGE",
                "Balayez paume ouverte : axes horizontal et vertical",
                "Saisie par pincement desactivee | Deux paumes : zoom",
                "Poing ferme : eclater / assembler",
            ]
        else:
            lines = [
                f"{self._model_name}  /  {self._render_mode}",
                "Pincez pour saisir une piece | Deux paumes ouvertes : zoom",
                "Paume ouverte : rotation | Poing ferme : eclater / assembler",
            ]
        self._set_hud("\n".join(lines))

    def _set_main_status(self, status: str) -> None:
        self._main_status = status
        self._refresh_hud()

    def _set_hud(self, text: str) -> None:
        self._hud_detail = text
        self._refresh_hud()

    def _refresh_hud(self) -> None:
        text = (
            f"{self._main_status}\n{self._hud_detail}"
            if self._hud_detail
            else self._main_status
        )
        if text != self._hud_text:
            self.hud.set_text(text)
            self._hud_text = text