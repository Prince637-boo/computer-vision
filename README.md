# Holographic Vehicle Lab

A real-time hand-controlled 3D model viewer built with MediaPipe and Panda3D.
The Mercedes-Benz vehicle uses a lightweight, semi-transparent cyan
Fresnel shader for a self-illuminated AR-style outline; it does not use dynamic
shadows, post-processing, or external shader textures.

## Model and render controls

The control bar offers the four entries in `assets/models/models.json`: the
Mercedes-Benz 300 SEL AMG, the Cassini-Huygens probe, Saturn V, and SOFIA.
That manifest defines each asset path, public description, known named parts,
and per-part exploded-view direction. Invalid model entries are skipped with
a warning; unknown part information is left empty rather than inferred. Each
loaded model is centered and fitted to a common size. Choices without a
supported model file are disabled and marked unavailable. Run
`python -m tools.convert_assets` to build
`assets/models/sofia.glb` and `assets/models/saturn_v.glb` from their STL parts
and to create the Draco-decompressed `assets/models/cassini_huygens.glb`.
The original Cassini-Huygens asset is preserved. The converter uses the
glTF Transform CLI and Draco decoder via `npx` for this step; these pinned CLI
packages are stored in the npm cache, not the project manifest. `gltf-pipeline
-d` compresses meshes rather than decompressing them.
Supported model formats are `.bam`, `.glb`, `.gltf`, and `.obj`; the manifest
selects the model file explicitly. The converter skips outputs that are newer
than their source assets; use `--force` to rebuild them.

Use the render buttons to switch between the cyan hologram, a CAD-style
wireframe, and the model's original materials and textures. The viewer retains
the imported appearance in Original mode rather than applying the hologram
shader.

## Hand controls

- **Pinch** to grab and move a named vehicle component.
- **Two open palms** to zoom in and out by changing the distance between the wrists.
- **Open palm** to rotate the vehicle. Horizontal and vertical movement control
  heading and pitch.
- **Close fist** to toggle the animated exploded view. Close your fist again to
  reassemble the model.
- **Anchor mode** keeps the selected model at its current position and uses
  open-palm sweeps for precise heading and pitch control. Pinch-to-grab is
  disabled while anchoring is active.
- **Formation** blinks each known piece in the manifest's ordered sequence.
  Pinching the highlighted piece removes it from view and advances the lesson;
  the final multiple-choice quiz displays the score in the HUD. Cassini's
  imported mesh names are numeric, so no unsupported component labels or
  training sequence are fabricated for that asset.
- Pinching a labeled component displays its manifest information in the HUD.
  Unknown values are left blank rather than guessed.

## Performance

Camera capture remains at 640 × 480 and targets 30 FPS. The Panda3D window is
limited to 60 FPS, and the HUD displays Panda3D's measured average render FPS.
Camera buffering is minimized, and an adaptive One Euro filter smooths only
the palm center used for rotation. Pinch positions are not double-smoothed.
Camera capture, background-frame conversion, and MediaPipe inference run off
Panda3D's render thread; inference skips queued frames when needed. Animated
scanlines, Fresnel rim lighting, subtle flicker, and the additive CardMaker
pedestal use the existing scene render without additional render passes or
shadow maps.

## Project layout

```text
assets/
  models/mercedes_benz.glb
  models/Cassini-Huygens (B).glb
  models/models.json
  models/sofia/*.stl
  models/saturn_v/*.stl
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
