from typing import Dict, List, Optional

from direct.gui.DirectGui import DGG, DirectButton, DirectFrame, DirectLabel
from panda3d.core import NodePath, TextNode

from src.interactor.training import TrainingPhase, TrainingSession, TrainingStep


class UIMixin:
    def _initialize_ui(self) -> None:
        self._main_status = "MAIN: NO HAND -> WAIT"
        self._hud_detail = ""
        self._hud_text = ""
        self._training_session: Optional[TrainingSession] = None
        self._training_removed_nodes: List[NodePath] = []
        self._training_highlight_nodes: List[NodePath] = []
        self._training_blink_elapsed = 0.0
        self._training_blink_on = False
        self._quiz_buttons: List[DirectButton] = []
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
        self._fps_label = DirectLabel(
            parent=self.aspect2d,
            text="FPS: --",
            text_fg=(0.5, 0.9, 1, 1),
            text_scale=0.035,
            frameColor=(0.01, 0.04, 0.07, 0.82),
            pos=(1.1, 0, -0.94),
        )
        self._build_control_panel()
        self._show_controls()

    def _build_control_panel(self) -> None:
        self.control_panel = DirectFrame(
            parent=self.aspect2d,
            frameColor=(0.015, 0.055, 0.085, 0.94),
            frameSize=(-1.72, 1.72, -0.32, 0.22),
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

        models = tuple(self._model_registry.by_id.items())
        spacing = 2.5 / max(len(models) - 1, 1)
        model_positions = (
            (0.0,)
            if len(models) == 1
            else tuple(-1.25 + spacing * index for index in range(len(models)))
        )
        model_button_half_width = min(0.37, spacing * 0.45)
        self._model_buttons: Dict[str, DirectButton] = {}
        for (model_id, model), x_pos in zip(models, model_positions):
            label = model.label
            available = model_id in self._model_paths
            button_label = label if available else f"{label} / ABSENT"
            self._model_buttons[model_id] = DirectButton(
                parent=self.control_panel,
                text=button_label,
                text_fg=(0.76, 0.91, 0.96, 1),
                text_scale=0.027,
                textMayChange=True,
                frameColor=(0.04, 0.16, 0.21, 1),
                frameSize=(
                    -model_button_half_width,
                    model_button_half_width,
                    -0.04,
                    0.04,
                ),
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
        self._training_button = DirectButton(
            parent=self.control_panel,
            text="FORMATION",
            text_fg=(0.76, 0.91, 0.96, 1),
            text_scale=0.027,
            textMayChange=True,
            frameColor=(0.04, 0.16, 0.21, 1),
            frameSize=(-0.4, 0.4, -0.04, 0.04),
            pos=(0, 0, -0.255),
            relief=DGG.FLAT,
            command=self.toggle_training,
        )
        self._update_training_button()
        self._refresh_model_buttons()
        self._refresh_mode_buttons()

    def _refresh_model_buttons(self) -> None:
        for model_id, button in self._model_buttons.items():
            available = model_id in self._model_paths
            label = self._model_registry.by_id[model_id].label
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

    def _show_part_details(self, node_name: str) -> None:
        part = self._part_definition_for_node(node_name)
        if part is None:
            self._show_controls()
            return
        lines = [part.label.upper(), node_name]
        if part.info:
            lines.append(part.info)
        lines.append("Release the pinch to continue.")
        self._set_hud("\n".join(lines))

    def _show_controls(self) -> None:
        if self._training_session is not None:
            self._refresh_training_status()
            return
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

    def toggle_training(self) -> None:
        if self._training_session is not None:
            if self._training_session.phase != TrainingPhase.COMPLETE:
                self._stop_training()
                return
            self._stop_training()
        model = self._active_model_definition
        if model is None:
            self._set_hud("Chargez un modèle avant de lancer la formation.")
            return

        parts_by_match = {part.match: part for part in model.parts}
        steps = [
            (part_match, parts_by_match[part_match])
            for part_match in model.training_steps
            if part_match in parts_by_match
        ]
        if not steps or any(
            not any(
                part_match.casefold() in node.get_name().casefold()
                for node in self.parts_dict.values()
            )
            for part_match, _ in steps
        ):
            self._set_hud(
                "Formation indisponible : aucune séquence de pièces "
                "identifiables pour ce modèle."
            )
            return

        self._training_session = TrainingSession(
            [TrainingStep(match, part.label) for match, part in steps],
            [part.label for part in model.parts],
        )
        self._training_removed_nodes.clear()
        self._training_blink_elapsed = 0.0
        self._training_blink_on = False
        self._update_training_button()
        self._refresh_training_status()

    def _stop_training(self) -> None:
        self._hide_quiz_choices()
        for node in self._training_removed_nodes:
            if not node.is_empty():
                node.show()
        self._training_removed_nodes.clear()
        for node in self._training_highlight_nodes:
            if not node.is_empty():
                node.set_color_scale(1, 1, 1, 1)
        self._training_highlight_nodes.clear()
        self._training_session = None
        self._training_blink_on = False
        if hasattr(self, "_training_button"):
            self._update_training_button()
        if hasattr(self, "_hud_text"):
            self._show_controls()

    def _on_training_part_pinch(self, node_name: str) -> None:
        session = self._training_session
        if session is None:
            return
        step = session.current_step
        if step is None:
            return
        if not session.select_part(node_name):
            self._set_hud(f"Ce n'est pas la bonne pièce. Cherchez : {step.label}.")
            return

        removed_nodes = [
            node
            for node in self.parts_dict.values()
            if step.match.casefold() in node.get_name().casefold()
        ]
        for node in removed_nodes:
            node.hide()
        self._training_removed_nodes.extend(removed_nodes)
        self._refresh_training_status()

    def _refresh_training_status(self) -> None:
        session = self._training_session
        if session is None:
            return
        if session.phase == TrainingPhase.REMOVING_PARTS:
            step = session.current_step
            if step is None:
                return
            self._hide_quiz_choices()
            self._set_hud(
                f"FORMATION — étape {session.step_index + 1}/{len(session.steps)}\n"
                f"Pincez la pièce clignotante : {step.label}."
            )
        elif session.phase == TrainingPhase.QUIZ:
            self._set_hud(
                f"MINI-QUIZ — question {session.quiz_index + 1}/"
                f"{session.question_count}\nQuelle est cette pièce ?"
            )
            self._show_quiz_choices()
        else:
            self._hide_quiz_choices()
            self._set_hud(
                f"FORMATION TERMINÉE — score : "
                f"{session.score}/{session.question_count}"
            )
        self._update_training_button()

    def _show_quiz_choices(self) -> None:
        self._hide_quiz_choices()
        if self._training_session is None:
            return
        choices = self._training_session.quiz_choices
        if not choices:
            return
        spacing = 2.8 / len(choices)
        for index, choice in enumerate(choices):
            button = DirectButton(
                parent=self.aspect2d,
                text=choice,
                text_fg=(0.76, 0.91, 0.96, 1),
                text_scale=0.032,
                textMayChange=True,
                frameColor=(0.04, 0.16, 0.21, 1),
                frameSize=(-0.32, 0.32, -0.055, 0.055),
                pos=(-1.4 + spacing * (index + 0.5), 0, -0.72),
                relief=DGG.FLAT,
                command=self._answer_training_quiz,
                extraArgs=[choice],
            )
            self._quiz_buttons.append(button)

    def _hide_quiz_choices(self) -> None:
        for button in self._quiz_buttons:
            button.destroy()
        self._quiz_buttons.clear()

    def _answer_training_quiz(self, answer: str) -> None:
        if self._training_session is None:
            return
        self._training_session.answer_quiz(answer)
        self._refresh_training_status()

    def _update_training_visuals(self, elapsed: float) -> None:
        session = self._training_session
        if session is None or session.current_step is None:
            if self._training_highlight_nodes:
                for node in self._training_highlight_nodes:
                    if not node.is_empty():
                        node.set_color_scale(1, 1, 1, 1)
                self._training_highlight_nodes.clear()
            return

        if elapsed - self._training_blink_elapsed < 0.3:
            return
        self._training_blink_elapsed = elapsed
        self._training_blink_on = not self._training_blink_on
        for node in self._training_highlight_nodes:
            if not node.is_empty():
                node.set_color_scale(1, 1, 1, 1)
        self._training_highlight_nodes.clear()

        if self._training_blink_on:
            match = session.current_step.match.casefold()
            self._training_highlight_nodes = [
                node
                for node in self.parts_dict.values()
                if not node.is_hidden() and match in node.get_name().casefold()
            ]
            for node in self._training_highlight_nodes:
                node.set_color_scale(1.2, 2.0, 2.5, 1)

    def _update_training_button(self) -> None:
        if not hasattr(self, "_training_button"):
            return
        if self._training_session is not None:
            self._training_button["text"] = (
                "REJOUER FORMATION"
                if self._training_session.phase.value == "complete"
                else "QUITTER FORMATION"
            )
            self._training_button["state"] = DGG.NORMAL
        else:
            self._training_button["text"] = "FORMATION"
            enabled = bool(
                self._active_model_definition
                and self._active_model_definition.training_steps
            )
            self._training_button["state"] = (
                DGG.NORMAL if enabled else DGG.DISABLED
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
