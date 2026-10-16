from typing import Dict, List, Optional, Tuple

from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    AmbientLight,
    CardMaker,
    ClockObject,
    DirectionalLight,
    NodePath,
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
from src.interactor.model_registry import ModelRegistryMixin
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
        self._model_paths = self._discover_model_assets()
        self._model_unavailable_reasons = self._exclude_draco_models(
            self._model_paths
        )
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
        self.hand_cursor = self.loader.load_model("models/misc/sphere")
        if self.hand_cursor:
            self.hand_cursor.reparent_to(self.render)
            self.hand_cursor.set_scale(0.12)
            self.hand_cursor.set_color_scale(0.0, 0.95, 1.0, 0.85)
            self.hand_cursor.set_transparency(TransparencyAttrib.M_alpha)
            self.hand_cursor.hide()

        self._initialize_ui()

    def update_background(self, frame: Tuple[int, int, bytes]) -> None:
        """Upload a camera frame already converted by the capture worker."""
        width, height, rgb_bytes = frame
        if not self._bg_ready:
            self.bg_texture.setup_2d_texture(
                width, height, Texture.T_unsigned_byte, Texture.F_rgb
            )
            self._bg_ready = True
        self.bg_texture.set_ram_image(rgb_bytes)
