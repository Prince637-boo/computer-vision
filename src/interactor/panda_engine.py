import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    CardMaker,
    ClockObject,
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
    """Render a low-cost cyan hologram and translate hand gestures into direct touch controls."""

    BASE_SCALE = 1.2
    EXPLOSION_DISTANCE = 2.4
    TOUCH_PLANE_Y = 5.0  # Profondeur fixe de l'écran tactile virtuel
    FIST_HOLD_SECONDS = 0.5
    DEAD_ZONE = 0.002

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

    def __init__(self):
        super().__init__()

        props = WindowProperties()
        props.setTitle("Holographic Vehicle Lab | Mercedes-Benz 300 SEL AMG")
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

        self.disable_mouse()
        self.camera.set_pos(0, -12, 1)
        self.camera.look_at(0, 4, 0)

        self.car_model: Optional[NodePath] = None
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
            Tuple[NodePath, Vec3, Vec3, Vec3, Vec3]
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
        self.hud_path.set_pos(-1.72, 0, 0.93)
        self.hud_path.set_scale(0.045)
        self._show_controls()

    def load_car_model(self, model_path: str) -> None:
        """Load the vehicle, apply one inherited hologram shader, and index its parts."""
        model_file = Path(model_path)
        if not model_file.is_file():
            raise FileNotFoundError(f"Vehicle model not found: {model_file}")

        self.car_model = self.loader.load_model(str(model_file))
        if self.car_model.is_empty():
            raise RuntimeError(f"Panda3D could not load vehicle model: {model_file}")

        self.car_model.reparent_to(self.render)
        self.car_model.set_pos(0, 5, 0)
        self.car_model.set_hpr(0, 0, 0)
        self.car_model.set_scale(self.BASE_SCALE)

        shader_dir = Path(__file__).resolve().parents[2] / "assets" / "shaders"
        vertex_source = (shader_dir / "hologram.vert").read_text(encoding="utf-8")
        fragment_source = (shader_dir / "hologram.frag").read_text(encoding="utf-8")
        holo_shader = Shader.make(Shader.SL_GLSL, vertex_source, fragment_source)
        if holo_shader is None:
            raise RuntimeError("Panda3D could not create the hologram shader.")

        self.car_model.set_shader(holo_shader)
        self.car_model.set_transparency(TransparencyAttrib.M_alpha)
        self.car_model.set_depth_write(False)
        self.car_model.set_two_sided(True)

        geometry_nodes = list(self.car_model.find_all_matches("**/+GeomNode"))
        if not geometry_nodes:
            raise RuntimeError(f"The vehicle model contains no geometry: {model_file}")

        self._index_parts(geometry_nodes)
        self._prepare_explosion(geometry_nodes)

    def _index_parts(self, geometry_nodes: List[NodePath]) -> None:
        self.parts_dict.clear()
        for node in geometry_nodes:
            name = node.get_name()
            if self._part_category(name) is not None:
                self.parts_dict[f"{name}_{len(self.parts_dict)}"] = node

    def _prepare_explosion(self, geometry_nodes: List[NodePath]) -> None:
        if self.car_model is None:
            return

        positions = [node.get_pos(self.car_model) for node in geometry_nodes]
        if positions:
            self._explosion_center = Vec3(
                (min(point.x for point in positions) + max(point.x for point in positions)) / 2,
                (min(point.y for point in positions) + max(point.y for point in positions)) / 2,
                (min(point.z for point in positions) + max(point.z for point in positions)) / 2,
            )

        self._explosion_parts.clear()
        for index, node in enumerate(geometry_nodes):
            base_pos = node.get_pos(self.car_model)
            direction = base_pos - self._explosion_center
            if direction.length() < 0.15:
                direction = Vec3(
                    math.cos(index * 2.4),
                    math.sin(index * 2.4),
                    0.6 if index % 2 == 0 else -0.6,
                )
            direction.normalize()
            rotation = Vec3(
                45 + (index % 4) * 25,
                ((index % 3) - 1) * 22,
                30 + (index % 5) * 18,
            )
            self._explosion_parts.append(
                (node, base_pos, node.get_hpr(self.car_model), direction, rotation)
            )

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
        if two_open_palms:
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
            if gesture == GestureType.PINCH and is_pinching:
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
        for index, (part, _, _, _, rotation) in enumerate(self._explosion_parts):
            if part != node:
                continue
            base_pos = node.get_pos(self.car_model)
            direction = base_pos - self._explosion_center
            if direction.length() < 0.15:
                direction = Vec3(1, 0.6 if index % 2 else -0.6, 0.6)
            direction.normalize()
            self._explosion_parts[index] = (
                node,
                base_pos,
                node.get_hpr(self.car_model),
                direction,
                rotation,
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
        for node, base_pos, base_hpr, direction, rotation in self._explosion_parts:
            node.set_pos(
                self.car_model,
                base_pos + direction * self.EXPLOSION_DISTANCE * amount,
            )
            node.set_hpr(self.car_model, base_hpr + rotation * amount)

        if amount == 1.0:
            self._set_hud(
                "EXPLODED VIEW\n"
                "Vehicle components separated and rotated.\n"
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
        self._set_hud(
            "MERCEDES-BENZ 300 SEL AMG | CYAN HOLOGRAM\n"
            "Pinch: grab a component    Two open palms: zoom\n"
            "Open palm: rotate    Close fist: explode / reassemble"
        )

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