import math
import time
from typing import Optional, Tuple

from panda3d.core import ClockObject, NodePath, Point2, Point3, Vec3

from src.config import RENDER_FPS
from src.vision.gestures import GestureType


class ControllerMixin:
    TOUCH_PLANE_Y = 5.0
    FIST_HOLD_SECONDS = 0.5
    DEAD_ZONE = 0.002

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
