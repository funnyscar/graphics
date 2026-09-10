# nerfrooms — the Big Bang Theory apartment as a NeRF

Reconstruct Sheldon & Leonard's apartment 4A (the living room: the couch and
Sheldon's spot, the kitchen, the stairwell door, the windows, the whiteboards)
as a **3D Gaussian splat** you can fly through in the browser — built only from
video scraped off the internet, trained on Modal GPUs.

The method follows Google's
[*Reconstructing indoor spaces with NeRF*](https://research.google/blog/reconstructing-indoor-spaces-with-nerf/):
many posed photos of an indoor space → structure-from-motion for camera poses →
a neural/gaussian scene fit → novel-view synthesis. Google captured restaurants
with rigs of DSLRs; we don't have the set, so the only usable substitute is
footage where **a real camera physically moves through the standing set** —
i.e. official *set-tour walkthrough* videos. Random episode screenshots can't be
used: the lighting rig, lens and props change shot to shot, and COLMAP will not
register them into one model.

## Pipeline

| stage | where | what |
| --- | --- | --- |
| `scrape` | Modal CPU | `yt-dlp` a few set-tour videos → `ffmpeg` frames → drop blurry / near-duplicate frames |
| `sfm` | Modal A10G | COLMAP (via `ns-process-data`) → camera poses + sparse cloud → `transforms.json` |
| `train` | Modal A100 | nerfstudio `splatfacto` (3D Gaussian Splatting) → `ns-export` a `.ply` splat + spiral preview |
| `fetch` | local | pull `out/point_cloud.ply`, `out/preview.mp4`, SfM stats |

```bash
modal run nerfrooms/pipeline.py::scrape
modal run nerfrooms/pipeline.py::sfm   --frames frames_all
modal run nerfrooms/pipeline.py::train --proc proc_frames_all --iters 15000
modal run nerfrooms/pipeline.py::fetch
```

All intermediates live in the Modal Volume `nerfrooms`, so stages are
independent and re-runnable. Inspect with `modal volume ls nerfrooms`.

## View it

```bash
cd nerfrooms/viewer
python3 -m http.server 8000
```

Open <http://localhost:8000>, click to enter, fly with `WASD` + `QE`, drag to
look, scroll to change speed. Load any other splat with `?url=…`.

## Reality check

Scraped internet footage is the hardest possible input for this — expect COLMAP
to register only a fraction of frames, and expect floaters where coverage is
thin. `out/*.transforms.json` reports the registration rate; if it's poor, the
honest fixes are a longer/steadier walkthrough video or a real capture
(`sfm --frames <your_folder>` works on any image set you drop in the Volume).
