import pyvista as pv
import numpy as np
from typing import Tuple
from src.config import (
    OBJECT_DEFAULT_SIZE, INTERACTION_POINT_RADIUS, WORKSPACE_SCALE,
    COLOR_WORKSPACE_GRID
)

# Convert strings or float triples for VTK compatibility
COLOR_RGB_CYAN = (0.0, 1.0, 1.0)
COLOR_RGB_RED = (1.0, 0.0, 0.0)

class Scene3DEngine:
    """Manages the 3D scene rendering, object transformation, and grab mechanics."""

    def __init__(self, obj_mesh: pv.PolyData = pv.Cube()):
        # Initialize Plotter
        self.plotter = pv.Plotter(title="Prince Vision 3D Workspace")
        
        # 1. Setup Environment
        self.plotter.add_axes()
        self.plotter.show_grid(color=COLOR_WORKSPACE_GRID)
        self.plotter.set_background('black')
        self.plotter.set_position((0, 0, 10))
        self.plotter.camera.focal_point = (0.0, 0.0, 0.0)

        # 2. Add Main Manipulable Object (Cube)
        self.main_object_mesh = obj_mesh.scale(OBJECT_DEFAULT_SIZE)
        self.object_actor = self.plotter.add_mesh(
            self.main_object_mesh,
            color=COLOR_RGB_CYAN,
            smooth_shading=True,
            specular=1.0,
            show_edges=True
        )

        # 3. Add Dynamic Interaction Point (Virtual Finger / Cursor)
        self.interaction_sphere = pv.Sphere(radius=INTERACTION_POINT_RADIUS)
        self.interaction_actor = self.plotter.add_mesh(
            self.interaction_sphere,
            color=COLOR_RGB_RED,
            smooth_shading=True,
            opacity=1.0
        )

        # 4. Grab Logic state
        self.is_object_grabbed = False
        self.prev_object_pos = np.array([0.0, 0.0, 0.0])

        # Force interactive non-blocking update
        self.plotter.show(interactive_update=True, auto_close=False)

    def is_interaction_visible(self) -> bool:
        """Returns True if the PyVista render window is active."""
        return not getattr(self.plotter, '_closed', False)

    def render_update(self) -> None:
        """Required to update the PyVista rendering loop manually."""
        if not getattr(self.plotter, '_closed', False):
            self.plotter.update()

    def update_interaction_point(self, position_3d: Tuple[float, float, float], is_active: bool) -> None:
        """Updates the position and opacity of the virtual finger marker."""
        if not is_active:
            self.interaction_actor.SetVisibility(0)
            return
        
        # Map 2D normalized hand pos [-1, 1] to 3D scene space
        final_pos = np.array([position_3d[0] * WORKSPACE_SCALE, position_3d[1] * WORKSPACE_SCALE, 0.0])
        self.interaction_actor.SetPosition(final_pos)
        self.interaction_actor.SetVisibility(1)

    def process_grab_logic(self, is_pinching: bool, hand_norm_pos: Tuple[float, float, float]) -> None:
        """Handles object grabbing based on pinching gesture."""
        scaled_hand_pos = np.array([hand_norm_pos[0] * WORKSPACE_SCALE, hand_norm_pos[1] * WORKSPACE_SCALE, 0.0])
        
        # Current position of the 3D cube
        current_obj_pos = np.array(self.object_actor.GetPosition())
        
        # Distance between hand cursor and object
        distance = np.linalg.norm(scaled_hand_pos - current_obj_pos)

        # A. Trigger Grab when pinching close to the object
        if is_pinching and not self.is_object_grabbed:
            if distance < 2.0:  # Augmented grabbing zone threshold
                self.is_object_grabbed = True
                self.object_actor.GetProperty().SetColor(COLOR_RGB_RED)
                self.prev_object_pos = current_obj_pos

        # B. Move object while maintaining grab
        if self.is_object_grabbed:
            if is_pinching:
                self.object_actor.SetPosition(scaled_hand_pos)
                self.prev_object_pos = scaled_hand_pos
            else:
                # C. Release Grab
                self.is_object_grabbed = False
                self.object_actor.GetProperty().SetColor(COLOR_RGB_CYAN)
                self.object_actor.SetPosition(self.prev_object_pos)

    def close(self) -> None:
        """Safely shuts down the PyVista interactor."""
        try:
            self.plotter.close()
        except Exception:
            pass