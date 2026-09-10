"""
latentdreams — mix the latent features of many women's faces with StyleGAN2.

A pretrained StyleGAN2-ADA generator (FFHQ, 1024²) is a compressed model of
"what a human face is". This walks its latent space:

  1. anchors   — sample a large pool of faces, score each with CLIP for
                 "a portrait photo of a woman", then farthest-point-sample a
                 diverse set of women in W space. These are the anchors.
  2. stylemix  — the classic StyleGAN style-mixing figure: coarse layers
                 (pose / face shape) from one woman, fine layers (skin, colour,
                 micro-detail) from another.
  3. dream     — a seamless looping video. The 18 W layers are split into
                 coarse / mid / fine groups, and each group is driven by its
                 own slowly drifting blend of *all* the anchors. Structure,
                 lighting and detail therefore morph on different rhythms —
                 every frame is a different mixture of every woman at once.

Everything runs on a Modal GPU. Pickle + outputs are cached in a Modal Volume.

    modal run latentdreams/latentdreams.py                    # A10G, 8 anchors, 240f loop
    modal run latentdreams/latentdreams.py --anchors 12 --frames 480 --fps 30
    modal run latentdreams/latentdreams.py --seeds 12,45,99,301,777   # skip CLIP, pick faces by seed
    modal run latentdreams/latentdreams.py --gpu L4 --res 512

Writes out/anchors.png, out/stylemix.png, out/dream.mp4, out/anchors.npz.
View with  latentdreams/viewer/index.html.
"""

import io
import pathlib

import modal

HERE = pathlib.Path(__file__).parent

SG2_REPO = "https://github.com/NVlabs/stylegan2-ada-pytorch.git"
FFHQ_PKL = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel"
    )
    .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC"})
    .run_commands(
        "ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime",
        "apt-get update && apt-get install -y --no-install-recommends "
        "git ninja-build ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*",
    )
    .pip_install(
        "numpy<2",
        "scipy==1.11.4",
        "click==8.1.7",
        "requests==2.31.0",
        "pillow==10.2.0",
        "imageio==2.34.0",
        "imageio-ffmpeg==0.4.9",
        "open_clip_torch==2.24.0",
        "tqdm",
    )
    .run_commands(
        f"git clone --depth 1 {SG2_REPO} /root/stylegan2-ada-pytorch",
        # sg2-ada was written for torch 1.7; these two guards are all inference needs on 2.x
        "sed -i 's/if not enabled:/if True:/' "
        "/root/stylegan2-ada-pytorch/torch_utils/ops/conv2d_gradfix.py || true",
    )
    .env({"TORCH_EXTENSIONS_DIR": "/root/torch_ext", "PYTHONUNBUFFERED": "1"})
    .add_local_file(HERE / "latentdreams.py", "/root/latentdreams.py")
)

vol = modal.Volume.from_name("latentdreams", create_if_missing=True)
DATA = "/data"

app = modal.App("latentdreams", image=image)


# --------------------------------------------------------------------------- #
# helpers (run inside the GPU container)
# --------------------------------------------------------------------------- #
def _load_G(device):
    import sys

    sys.path.append("/root/stylegan2-ada-pytorch")
    import dnnlib
    import legacy

    cache = pathlib.Path(DATA) / "ffhq.pkl"
    if not cache.exists():
        print("[latentdreams] downloading FFHQ pickle …")
        with dnnlib.util.open_url(FFHQ_PKL) as f:
            cache.write_bytes(f.read())
        vol.commit()
    with open(cache, "rb") as f:
        G = legacy.load_network_pkl(f)["G_ema"]
    return G.to(device).eval().requires_grad_(False)


def _to_uint8(img):
    """[N,3,H,W] in [-1,1]  ->  numpy [N,H,W,3] uint8"""
    x = (img * 127.5 + 128).clamp(0, 255).to("cpu", dtype=__import__("torch").uint8)
    return x.permute(0, 2, 3, 1).numpy()


def _grid(tiles, cols):
    import numpy as np

    n = len(tiles)
    rows = (n + cols - 1) // cols
    h, w, _ = tiles[0].shape
    out = np.full((rows * h, cols * w, 3), 12, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        out[r * h:(r + 1) * h, c * w:(c + 1) * w] = t
    return out


def _resize(img, res):
    import torch.nn.functional as F

    if img.shape[-1] == res:
        return img
    return F.interpolate(img, size=(res, res), mode="area")


# --------------------------------------------------------------------------- #
# 1. pick the anchor faces
# --------------------------------------------------------------------------- #
def _anchor_ws(G, device, *, pool, n_anchors, truncation, seed, seeds):
    import numpy as np
    import torch

    if seeds:
        chosen = [int(s) for s in seeds.split(",") if s.strip()]
        z = torch.from_numpy(
            np.stack([np.random.RandomState(s).randn(G.z_dim) for s in chosen])
        ).float().to(device)
        ws = G.mapping(z, None, truncation_psi=truncation)
        print(f"[anchors] {len(chosen)} faces from seeds {chosen}")
        return ws, np.array(chosen), np.ones(len(chosen))

    import open_clip
    from PIL import Image

    clip, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    clip = clip.to(device).eval()
    tok = open_clip.get_tokenizer("ViT-B-32")
    prompts = ["a portrait photo of a woman's face", "a portrait photo of a man's face"]
    with torch.no_grad():
        tfeat = clip.encode_text(tok(prompts).to(device))
        tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)

    rng = np.random.RandomState(seed)
    zs = rng.randn(pool, G.z_dim).astype(np.float32)
    woman_p, all_w = [], []
    for i in range(0, pool, 8):
        z = torch.from_numpy(zs[i:i + 8]).to(device)
        with torch.no_grad():
            w = G.mapping(z, None, truncation_psi=truncation)
            img = G.synthesis(w, noise_mode="const")
            batch = torch.stack(
                [preprocess(Image.fromarray(u)) for u in _to_uint8(img)]
            ).to(device)
            ifeat = clip.encode_image(batch)
            ifeat = ifeat / ifeat.norm(dim=-1, keepdim=True)
            p = (100.0 * ifeat @ tfeat.T).softmax(dim=-1)[:, 0]
        woman_p.extend(p.tolist())
        all_w.append(w.cpu())
    woman_p = np.array(woman_p)
    all_w = torch.cat(all_w)  # [pool, L, D]

    # keep a generous shortlist of "clearly a woman", then farthest-point-sample
    # for diversity in W space so the anchors aren't near-duplicates
    shortlist = np.argsort(-woman_p)[: max(n_anchors * 4, n_anchors)]
    shortlist = shortlist[woman_p[shortlist] > 0.5]
    if len(shortlist) < n_anchors:
        shortlist = np.argsort(-woman_p)[:n_anchors]

    feats = all_w[shortlist][:, 0, :].numpy()  # coarse W as the identity proxy
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    picked = [int(np.argmax(woman_p[shortlist]))]
    while len(picked) < n_anchors:
        d = np.min(
            [1.0 - feats @ feats[p] for p in picked], axis=0
        )
        d[picked] = -1
        picked.append(int(np.argmax(d)))
    idx = shortlist[picked]

    print(f"[anchors] pool={pool}  women>0.5: {(woman_p > 0.5).sum()}  "
          f"picked {len(idx)}  P(woman)={np.round(woman_p[idx], 3).tolist()}")
    return all_w[idx].to(device), idx.astype(int), woman_p[idx]


# --------------------------------------------------------------------------- #
# 2. style-mixing figure
# --------------------------------------------------------------------------- #
def _style_mix(G, ws, *, cut, res):
    import numpy as np
    import torch

    k = min(4, ws.shape[0])
    rows = ws[:k]          # coarse: pose + face shape
    cols = ws[k:2 * k] if ws.shape[0] >= 2 * k else ws[:k]

    tiles = [np.full((res, res, 3), 12, np.uint8)]  # empty corner
    for c in cols:
        tiles.append(_to_uint8(_resize(G.synthesis(c[None], noise_mode="const"), res))[0])
    for r in rows:
        tiles.append(_to_uint8(_resize(G.synthesis(r[None], noise_mode="const"), res))[0])
        for c in cols:
            w = r.clone()
            w[cut:] = c[cut:]     # fine layers (skin / colour / detail) from the column
            tiles.append(_to_uint8(_resize(G.synthesis(w[None], noise_mode="const"), res))[0])
    return _grid(tiles, len(cols) + 1)


# --------------------------------------------------------------------------- #
# 3. the dream loop
# --------------------------------------------------------------------------- #
def _dream(G, ws, *, frames, res, seed):
    import numpy as np
    import torch

    A, L, D = ws.shape
    # split the W layers into coarse / mid / fine groups
    g1, g2 = L // 3, 2 * L // 3
    groups = [slice(0, g1), slice(g1, g2), slice(g2, L)]

    rng = np.random.RandomState(seed + 1)
    t = np.linspace(0, 2 * np.pi, frames, endpoint=False)  # periodic -> seamless loop
    # per (group, anchor): a few integer-frequency sinusoids => smooth looping weights
    logits = np.zeros((len(groups), frames, A))
    for gi in range(len(groups)):
        for a in range(A):
            for _ in range(3):
                f = rng.randint(1, 4)
                ph = rng.uniform(0, 2 * np.pi)
                amp = rng.uniform(0.6, 1.6)
                logits[gi, :, a] += amp * np.sin(f * t + ph)
        logits[gi] += rng.uniform(-0.4, 0.4, A)          # static bias per anchor
    weights = np.exp(logits * 1.4)
    weights /= weights.sum(axis=2, keepdims=True)          # [G, frames, A]

    ws_cpu = ws.cpu().numpy()
    out = []
    for f0 in range(0, frames, 6):
        chunk = []
        for fr in range(f0, min(f0 + 6, frames)):
            w = np.empty((L, D), np.float32)
            for gi, sl in enumerate(groups):
                w[sl] = np.tensordot(weights[gi, fr], ws_cpu[:, sl, :], axes=([0], [0]))
            chunk.append(w)
        wt = torch.from_numpy(np.stack(chunk)).to(ws.device)
        img = _resize(G.synthesis(wt, noise_mode="const"), res)
        out.extend(_to_uint8(img))
        print(f"[dream] {min(f0 + 6, frames)}/{frames}")
    return out


# --------------------------------------------------------------------------- #
# GPU entrypoint
# --------------------------------------------------------------------------- #
@app.function(gpu="A10G", timeout=60 * 60, volumes={DATA: vol})
def run(pool: int, anchors: int, frames: int, fps: int, truncation: float,
        stylemix_cut: int, res: int, seed: int, seeds: str) -> dict:
    import imageio
    import numpy as np
    import torch

    device = torch.device("cuda")
    torch.manual_seed(seed)
    G = _load_G(device)
    print(f"[latentdreams] G: z_dim={G.z_dim} w_dim={G.w_dim} num_ws={G.num_ws} "
          f"img={G.img_resolution}")

    with torch.no_grad():
        ws, idx, probs = _anchor_ws(
            G, device, pool=pool, n_anchors=anchors,
            truncation=truncation, seed=seed, seeds=seeds,
        )

        anchor_tiles = [
            _to_uint8(_resize(G.synthesis(w[None], noise_mode="const"), res))[0]
            for w in ws
        ]
        anchors_png = _grid(anchor_tiles, min(len(anchor_tiles), 4))

        mix_png = _style_mix(G, ws, cut=stylemix_cut, res=res)
        dream_frames = _dream(G, ws, frames=frames, res=res, seed=seed)

    out = {}

    def _png(name, arr):
        buf = io.BytesIO()
        imageio.imwrite(buf, arr, format="png")
        out[name] = buf.getvalue()

    _png("anchors.png", anchors_png)
    _png("stylemix.png", mix_png)
    _png("dream_contact.png", _grid(dream_frames[:: max(1, len(dream_frames) // 9)][:9], 3))

    # ffmpeg/mp4 needs a seekable target -> write to a real file, then read back
    mp4_path = "/tmp/dream.mp4"
    w = imageio.get_writer(
        mp4_path, format="ffmpeg", mode="I", fps=fps, codec="libx264",
        quality=8, macro_block_size=1, output_params=["-pix_fmt", "yuv420p"],
    )
    for fr in dream_frames:
        w.append_data(fr)
    w.close()
    out["dream.mp4"] = pathlib.Path(mp4_path).read_bytes()

    nbuf = io.BytesIO()
    np.savez_compressed(nbuf, ws=ws.cpu().numpy(), seeds=idx, woman_p=probs,
                        truncation=truncation)
    out["anchors.npz"] = nbuf.getvalue()

    for k, v in out.items():
        print(f"[modal] {k}: {len(v) / 1e6:.2f} MB")
    return out


@app.local_entrypoint()
def main(gpu: str = "A10G", pool: int = 64, anchors: int = 8, frames: int = 240,
         fps: int = 30, truncation: float = 0.7, stylemix_cut: int = 8,
         res: int = 768, seed: int = 0, seeds: str = ""):
    fn = run.with_options(gpu=gpu) if gpu != "A10G" else run
    got = fn.remote(pool=pool, anchors=anchors, frames=frames, fps=fps,
                    truncation=truncation, stylemix_cut=stylemix_cut, res=res,
                    seed=seed, seeds=seeds)
    dst = HERE / "out"
    dst.mkdir(exist_ok=True)
    for name, data in got.items():
        (dst / name).write_bytes(data)
        print(f"wrote out/{name}  ({len(data) / 1e6:.2f} MB)")
    if "dream.mp4" not in got:
        raise SystemExit("no dream produced")
    print("\nopen latentdreams/viewer/index.html")
