# ape

A procedural ape head — heavy brow, prognathic muzzle, deep-set amber eyes,
receding chin — built and rendered entirely in Blender. Aimed at the
*Planet of the Apes* "Caesar" read (chimp proportions pushed a little).

Everything is generated from code: no sculpting, no external meshes. A
subdivided sphere is pushed into shape by closed-form displacement
(`_sculpt_vertex`), voxel-remeshed for clean topology, subsurfed, and
dressed with a procedural skin shader, amber eye shader, and two fur
particle systems.

## files

| file | |
|---|---|
| `ape_face.py` | the generator + render (runs inside Blender or the `bpy` module) |
| `modal_render.py` | runs `ape_face.py` on a Modal GPU, pulls back the images |
| `ape_face.png` / `ape_face.blend` | latest render + scene (git-ignored) |

## run it

On a Modal GPU (recommended — Cycles + fur wants a GPU):

```bash
cd ape
modal run modal_render.py                       # A10G, 200 samples, 1400px, fur on
modal run modal_render.py --samples 400 --res 1600
modal run modal_render.py --no-hair --debug     # fast; also writes 4 angle checks
modal run modal_render.py --gpu L4               # pick another GPU
```

Locally, if you have Blender or `pip install bpy` (needs Python 3.11):

```bash
blender -b -P ape_face.py           # or:  python ape_face.py
```

## tuning

Env vars read by `ape_face.py`: `APE_SAMPLES`, `APE_RES`, `APE_HAIR` (`0`/`1`),
`APE_GPU` (`0` forces CPU), `APE_OUT_DIR`, `APE_DEBUG` (`1` → also render
front / 3-4 / side / low angle checks).

The face is shaped by landmark falloffs against the base sphere's stable
coords, so each feature can be dialed independently:

- **brow** — section 2 of `_sculpt_vertex`: `0.25` forward push, glabella dip
- **muzzle** — section 3: `_band(-0.44, 0.02, …)` z-extent, `0.40` forward, `-0.12` droop
- **eyes** — `add_eyes()` position/size; socket rim in section 4
- **lighting** — `setup_lights()`: key / fill / front / rim / kick area lights
- **camera** — `setup_camera()`: 3/4 hero; DOF on, f/7

## status

Structure and the front / 3-4 / profile reads are there. Rougher spots:
the cranium is still a touch large, the skin wants more colour variation
(darker muzzle mask, pinker tones), and the fur could use a proper
clump/length map rather than the single `pelt` vertex group.
