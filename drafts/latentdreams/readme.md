# latentdreams

Mix the latent features of many women's faces with **StyleGAN2**.

A pretrained StyleGAN2-ADA generator (NVIDIA, FFHQ, 1024²) is a compressed model
of *what a human face is*. `latentdreams` walks that latent space:

| stage | what |
| --- | --- |
| **anchors** | sample a big pool of faces, score each with CLIP for *"a portrait photo of a woman"*, then farthest-point-sample a diverse set of women in `W` space |
| **stylemix** | the classic StyleGAN style-mixing figure — coarse layers (pose / face shape) from one woman, fine layers (skin, colour, micro-detail) from another |
| **dream** | a seamless looping video: the 18 `W` layers are split into coarse / mid / fine groups, and each group is driven by its own slowly drifting blend of *all* the anchors, so structure, lighting and detail morph on different rhythms — every frame is a mixture of every woman at once |

Everything runs on a Modal GPU. The FFHQ pickle and outputs are cached in the
Modal Volume `latentdreams`.

## run it

```bash
modal run latentdreams/latentdreams.py                      # A10G, 8 anchors, 240-frame loop
modal run latentdreams/latentdreams.py --anchors 12 --frames 480 --fps 30
modal run latentdreams/latentdreams.py --seeds 12,45,99,301  # skip CLIP, pick faces by seed
modal run latentdreams/latentdreams.py --gpu L4 --res 512    # cheaper / faster
```

Writes to `out/`:

| file | |
| --- | --- |
| `dream.mp4` | the latent-feature loop |
| `anchors.png` | the women being mixed |
| `stylemix.png` | coarse × fine style-mixing grid |
| `dream_contact.png` | nine frames from the loop |
| `anchors.npz` | anchor `W` vectors, seeds, CLIP `P(woman)` |

## tuning

| flag | default | |
| --- | --- | --- |
| `--pool` | 64 | faces sampled before CLIP filtering |
| `--anchors` | 8 | women kept and mixed |
| `--truncation` | 0.7 | lower = closer to the average face, safer; higher = more varied |
| `--stylemix-cut` | 8 | `W` layer where the style-mix switches from row to column |
| `--frames` / `--fps` | 240 / 30 | loop length; frames must divide evenly for a seamless loop |
| `--res` | 768 | output resolution (G is native 1024) |
| `--seed` | 0 | controls the sample pool and the dream's drift |

## view it

Open `latentdreams/viewer/index.html` (it reads `../out/`). Serve the folder if
your browser blocks local video:

```bash
cd latentdreams && python3 -m http.server 8000
```

## notes

The generator only knows FFHQ (Flickr faces) — its "average" skews toward that
distribution. StyleGAN2-ADA's custom CUDA ops are built on first run inside the
container (a minute or two); if the build fails it falls back to a slower native
path and still works.
