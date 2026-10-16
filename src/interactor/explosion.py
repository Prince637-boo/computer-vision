import math
from typing import List

from panda3d.core import NodePath, Point3, Vec3


class ExplosionMixin:
    EXPLOSION_DISTANCE = 2.4

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
