from typing import Dict, Optional

from direct.gui.DirectGui import DGG, DirectButton, DirectFrame, DirectLabel
from panda3d.core import TextNode


class UIMixin:
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

    def _initialize_ui(self) -> None:
        self._main_status = "MAIN: NO HAND -> WAIT"
        self._hud_detail = ""
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
