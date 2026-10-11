import numpy as np
import cv2
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    Texture, CardMaker, NodePath, Vec3,
    Shader, TransparencyAttrib, WindowProperties, GeomNode
)
import gltf
from src.config import WORKSPACE_SCALE

class Panda3DEngine(ShowBase):
    """Panda3D interaction engine with true Holographic shader rendering and exploded view mechanics."""

    def __init__(self):
        super().__init__()

        props = WindowProperties()
        props.setTitle("Prince Vision - Mercedes-Benz Hologram AR")
        props.setSize(1280, 720)
        self.win.requestProperties(props)
        self.set_background_color(0, 0, 0, 1)

        # Arrière-plan Réalité Augmentée
        self.cam2d.node().get_display_region(0).set_sort(-10)
        self.bg_texture = Texture("camera_feed")
        self._bg_ready = False
        
        cm = CardMaker("bg_card")
        cm.set_frame(-1, 1, -1, 1)
        self.bg_card = self.render2d.attach_new_node(cm.generate())
        self.bg_card.set_texture(self.bg_texture)

        self.disable_mouse()
        # Caméra reculée pour englober toute la voiture
        self.camera.set_pos(0, -12, 1)
        self.camera.look_at(0, 4, 0)

        self.car_model: NodePath = None
        self.parts_dict = {}
        self.grabbed_node: NodePath = None
        self.grab_offset = Vec3(0, 0, 0)

        # Curseur de main plus visible
        self.hand_cursor = self.loader.load_model("models/misc/sphere")
        if self.hand_cursor:
            self.hand_cursor.reparent_to(self.render)
            self.hand_cursor.set_scale(0.2)
            self.hand_cursor.set_color(1, 0, 0, 1)

    def load_car_model(self, model_path: str):
        """Charge la Mercedes, applique le shader holographique partout et indexe les pièces."""
        print(f"🏎️ Chargement de la Mercedes-Benz : {model_path}...")
        self.car_model = self.loader.load_model(model_path)
        self.car_model.reparent_to(self.render)
        self.car_model.set_pos(0, 5, 0)
        self.car_model.set_hpr(0, 0, 0)
        # Augmentation de l'échelle pour qu'elle soit bien visible
        self.car_model.set_scale(1.2)

        # Chargement du shader GLSL
        try:
            holo_shader = Shader.load(Shader.SL_GLSL, "assets/shaders/hologram.sha")
        except Exception as e:
            print(f"⚠️ Erreur de chargement du shader : {e}")
            holo_shader = None

        print("⚙️ Application du style Hologramme et indexation...")
        
        # Parcourir tous les sous-nœuds pour appliquer le shader de manière agressive sur chaque géométrie
        for child in self.car_model.findAllMatches("**/+PandaNode"):
            name = child.getName()
            
            # Forcer le shader holographique sur les nœuds contenant de la géométrie
            if holo_shader and not child.is_empty():
                child.set_shader(holo_shader)
                child.set_transparency(TransparencyAttrib.M_alpha)
                child.set_depth_write(False)
                child.set_two_sided(True)

            # Filtrer et enregistrer les pièces mobiles/démontables
            if any(k in name for k in ["WHEEL", "BONNET", "BOOT", "SUSP", "BODY", "DOOR"]):
                self.parts_dict[name] = child
                print(f"   🔹 Pièce interactive prête : {name}")

    def update_background(self, cv_frame):
        """Injecte l'image de la caméra en arrière-plan."""
        h, w, _ = cv_frame.shape
        if not self._bg_ready:
            self.bg_texture.setup_2d_texture(w, h, Texture.T_unsigned_byte, Texture.F_rgb)
            self._bg_ready = True
        self.bg_texture.set_ram_image(cv2.flip(cv_frame, 0).tobytes())

    def update_hand_interaction(self, hand_norm_pos: tuple, is_pinching: bool):
        """Mappe le geste de la main aux coordonnées 3D et applique le déplacement absolu des pièces."""
        hx, hy = hand_norm_pos[:2]
        
        target_pos = Vec3(hx * WORKSPACE_SCALE * 1.5, 5, hy * WORKSPACE_SCALE * 1.5)
        
        if self.hand_cursor:
            self.hand_cursor.set_pos(target_pos)

        if is_pinching:
            if not self.grabbed_node:
                best_part = None
                min_dist = 2.0

                for name, part_node in self.parts_dict.items():
                    try:
                        world_pos = part_node.get_pos(self.render)
                        dist = (world_pos - target_pos).length()
                        if dist < min_dist:
                            min_dist = dist
                            best_part = part_node
                    except Exception:
                        continue

                if best_part:
                    self.grabbed_node = best_part
                    # Enregistrer la position initiale dans le repère de la scène (render)
                    self.grab_offset = self.grabbed_node.get_pos(self.render) - target_pos
                    print(f"🖐️ Pièce sélectionnée (Grab) : {self.grabbed_node.getName()}")

            if self.grabbed_node:
                # Forcer la position dans le repère global render pour éviter les conflits de parents
                self.grabbed_node.set_pos(self.render, target_pos + self.grab_offset)
        else:
            if self.grabbed_node:
                print(f"👌 Pièce relâchée : {self.grabbed_node.getName()}")
                self.grabbed_node = None