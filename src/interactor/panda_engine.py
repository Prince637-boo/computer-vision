from typing import Dict, List, Optional, Tuple

from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    AmbientLight,
    CardMaker,
    ClockObject,
    ColorBlendAttrib,
    DirectionalLight,
    NodePath,
    PNMImage,
    Shader,
    Texture,
    TransparencyAttrib,
    Vec3,
    WindowProperties,
)

import gltf  # Registers Panda3D's glTF loader.
from src.config import RENDER_FPS
from src.interactor.controller import ControllerMixin
from src.interactor.explosion import ExplosionMixin
from src.interactor.model_registry import (
    ModelDefinition,
    ModelRegistry,
    ModelRegistryMixin,
)
from src.interactor.model_view import ModelViewMixin
from src.interactor.ui import UIMixin


class Panda3DEngine(
    ShowBase,
    ModelRegistryMixin,
    ModelViewMixin,
    ControllerMixin,
    ExplosionMixin,
    UIMixin,
):
    """Render selectable 3D assets and translate hand gestures into direct controls."""

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
        self._model_registry = ModelRegistry.load()
        self._model_paths = self._discover_model_assets(registry=self._model_registry)
        self._model_unavailable_reasons = self._exclude_draco_models(
            self._model_paths
        )
        self._active_model_definition: Optional[ModelDefinition] = None
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
        self._effect_time = 0.0
        self._fps_update_elapsed = 0.0
        self.hand_cursor = self.loader.load_model("models/misc/sphere")
        if self.hand_cursor:
            self.hand_cursor.reparent_to(self.render)
            self.hand_cursor.set_scale(0.12)
            self.hand_cursor.set_color_scale(0.0, 0.95, 1.0, 0.85)
            self.hand_cursor.set_transparency(TransparencyAttrib.M_alpha)
            self.hand_cursor.hide()

        self._create_pedestal()
        self._initialize_ui()
        self.taskMgr.add(self._update_visual_effects, "VisualEffectsTask")

    def _create_pedestal(self) -> None:
        image = PNMImage(96, 96, 4)
        for y in range(96):
            for x in range(96):
                distance = (((x + 0.5 - 48) ** 2 + (y + 0.5 - 48) ** 2) ** 0.5) / 48
                alpha = max(0.0, 1.0 - distance) ** 2 * 0.72
                image.set_xel(x, y, 0.02, 0.62, 1.0)
                image.set_alpha(x, y, alpha)

        texture = Texture("hologram_pedestal_glow")
        texture.load(image)
        texture.set_minfilter(Texture.FT_linear)
        texture.set_magfilter(Texture.FT_linear)

        card = CardMaker("hologram_pedestal")
        card.set_frame(-2.5, 2.5, -2.5, 2.5)
        self.pedestal = self.render.attach_new_node(card.generate())
        self.pedestal.set_p(-90)
        self.pedestal.set_texture(texture)
        self.pedestal.set_attrib(
            ColorBlendAttrib.make(
                ColorBlendAttrib.M_add,
                ColorBlendAttrib.O_incoming_alpha,
                ColorBlendAttrib.O_one,
            )
        )
        self.pedestal.set_depth_write(False)
        self.pedestal.set_light_off()
        self.pedestal.hide()

    def _update_visual_effects(self, task):
        clock = ClockObject.get_global_clock()
        dt = max(clock.get_dt(), 0.0)
        self._effect_time += dt
        self._fps_update_elapsed += dt

        if self.car_model is not None and self._render_mode == "HOLOGRAM":
            self.car_model.set_shader_input("time", self._effect_time)
        self._update_training_visuals(self._effect_time)

        if self._fps_update_elapsed >= 0.25:
            frame_rate = clock.get_average_frame_rate()
            self._fps_label["text"] = (
                f"FPS: {frame_rate:.0f}" if frame_rate > 0 else "FPS: --"
            )
            self._fps_update_elapsed = 0.0
        return task.cont

    def update_background(self, frame: Tuple[int, int, bytes]) -> None:
        """Upload a camera frame already converted by the capture worker."""
        width, height, rgb_bytes = frame
        if not self._bg_ready:
            self.bg_texture.setup_2d_texture(
                width, height, Texture.T_unsigned_byte, Texture.F_rgb
            )
            self._bg_ready = True
        self.bg_texture.set_ram_image(rgb_bytes)
