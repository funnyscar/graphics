# face

A procedural human head — rounded braincase, gentle brow, projecting nose,
defined lips and chin, soft cheekbones, tapering jaw — built and rendered
entirely in Blender.

Everything is generated from code: no sculpting, no external meshes. A
subdivided sphere is pushed into shape by closed-form displacement
(`_sculpt_vertex`), voxel-remeshed for even topology, subsurfed, and dressed
with a procedural skin shader (with SSS), a brown eye shader, and two short
scalp-hair particle systems. Separate meshes handle the eyeballs, ears and neck.

## files

| file | |
|---|---|
| `face.py` | the generator + render (runs inside Blender or the `bpy` module) |
| `modal_render.py` | runs `face.py` on a Modal GPU, pulls back the images |
| `face.png` / `face.blend` | latest render + scene (git-ignored) |

## run it

On a Modal GPU (recommended — Cycles + hair wants a GPU):

```bash
cd face
modal run modal_render.py                       # A10G, 200 samples, 1400px, hair on
modal run modal_render.py --samples 400 --res 1600
modal run modal_render.py --no-hair --debug     # fast; also writes 4 angle checks
modal run modal_render.py --gpu L4              # pick another GPU
```

Locally, if you have Blender or `pip install bpy` (needs Python 3.11):

```bash
blender -b -P face.py           # or:  python face.py
```

## tuning

Env vars read by `face.py`: `FACE_SAMPLES`, `FACE_RES`, `FACE_HAIR` (`0`/`1`),
`FACE_GPU` (`0` forces CPU), `FACE_OUT_DIR`, `FACE_DEBUG` (`1` → also render
front / 3-4 / side / low angle checks).

The face is shaped by landmark falloffs against the base sphere's stable
coords (`xc`, `zc`), so each feature can be dialed independently — see the
numbered sections of `_sculpt_vertex`:

- **head mass / jaw taper** — section 1
- **brow** — section 2
- **eye sockets + lids** — section 3 (`add_eyes()` sets eyeball size/position)
- **nose** — section 4: bridge, ridge `_band`, bulb tip, nostril wings, septum
- **lips** — section 5: upper/lower roll, seam, philtrum, cupid's bow
- **chin** — section 6
- **cheekbones** — section 7
- **jaw angle** — section 8
- **lighting** — `setup_lights()`: key / fill / rim / kick / catchlight areas
- **camera** — `setup_camera()`: 85 mm 3/4 portrait, DOF on at f/4
