# Holographic Vehicle Lab

A real-time hand-controlled Mercedes-Benz 300 SEL AMG model viewer built with
MediaPipe and Panda3D. The vehicle uses a lightweight, semi-transparent cyan
Fresnel shader for a self-illuminated AR-style outline; it does not use dynamic
shadows, post-processing, or external shader textures.

## Hand controls

- **Pinch** to grab and move a named vehicle component.
- **Two open palms** to zoom in and out by changing the distance between the wrists.
- **Open palm** to rotate the vehicle. Horizontal and vertical movement control
  heading and pitch; palm tilt controls roll.
- **Close fist** to toggle the animated exploded view. Close your fist again to
  reassemble the model.
- Grabbing a labeled engine, brake, wheel, or suspension component displays
  contextual information in the Panda3D HUD. Values not included in the model
  are identified as unavailable rather than guessed.

## Performance

Camera capture remains at 640 × 480 and targets 30 FPS. The Panda3D window is
limited to 60 FPS, camera buffering is minimized, and an adaptive One Euro
filter smooths only the palm center used for rotation. Pinch positions are not
double-smoothed. Camera capture, background-frame conversion, and MediaPipe
inference run off Panda3D's render thread; inference skips queued frames when needed.
All hologram shading is performed in a
single inherited shader without lighting passes or shadow maps.

## Project layout

```text
assets/
  models/mercedes_benz.glb
  models/hand_landmarker.task
  shaders/hologram.vert
  shaders/hologram.frag
src/
  config.py
  core/utils.py
  interactor/panda_engine.py
  vision/camera.py
  vision/gestures.py
  vision/hand_detector.py
  vision/worker.py
main.py
requirements.txt
```
