"""
nerfrooms — reconstruct the Big Bang Theory apartment (4A living room) as a NeRF /
3D Gaussian splat from scraped internet video, on Modal GPUs.

Pipeline, in the spirit of Google's "Reconstructing indoor spaces with NeRF"
(https://research.google/blog/reconstructing-indoor-spaces-with-nerf/):

  1. scrape  — yt-dlp a few *set-tour walkthrough* videos (a camera physically
               moving through the standing set is the only kind of internet
               footage that structure-from-motion can register), cut them to
               sharp, deduplicated frames.
  2. sfm     — COLMAP (via nerfstudio's ns-process-data) recovers camera poses
               and a sparse point cloud -> transforms.json.
  3. train   — nerfstudio `splatfacto` (3D Gaussian Splatting) fits the scene,
               then ns-export dumps a .ply splat.
  4. fetch   — pull the .ply + a preview render + SfM stats back to nerfrooms/out.

Everything lives in a Modal Volume ("nerfrooms") so stages are independent:

    modal run nerfrooms/pipeline.py::scrape
    modal run nerfrooms/pipeline.py::sfm     --frames frames_all
    modal run nerfrooms/pipeline.py::train   --iters 15000
    modal run nerfrooms/pipeline.py::fetch

`out/point_cloud.ply` then loads in nerfrooms/viewer/index.html.
"""

import json
import pathlib

import modal

HERE = pathlib.Path(__file__).parent

# Candidate walkthroughs. yt-dlp resolves these at run time; metadata is logged
# so we can see what actually came down. Order = rough preference.
CANDIDATES = {
    "walkthrough": "https://www.youtube.com/watch?v=cV5N69gUUc8",
    "settour_simon_kunal": "https://www.youtube.com/watch?v=N_8PpITO6PY",
    "geekchic": "https://www.youtube.com/watch?v=o_zDomxzg2I",
}

vol = modal.Volume.from_name("nerfrooms", create_if_missing=True)
DATA = "/data"

# --------------------------------------------------------------------------- #
# images
# --------------------------------------------------------------------------- #
scrape_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("yt-dlp==2025.8.11", "opencv-python-headless==4.10.0.84", "numpy<2")
)

# Official nerfstudio image: CUDA 11.8, python 3.10, COLMAP 3.9.1 + hloc + gsplat
# + tinycudann all prebuilt.
ns_image = (
    modal.Image.from_registry(
        "ghcr.io/nerfstudio-project/nerfstudio:1.1.5", add_python=None
    )
    .env({"PYTHONUNBUFFERED": "1"})
    .run_commands("ln -sf $(which python3) /usr/local/bin/python || true")
)

app = modal.App("nerfrooms")


# --------------------------------------------------------------------------- #
# 1. scrape
# --------------------------------------------------------------------------- #
@app.function(image=scrape_image, volumes={DATA: vol}, timeout=60 * 60)
def scrape(fps: float = 2.0, max_frames_per_video: int = 350, long_edge: int = 1600):
    import os
    import subprocess

    import cv2
    import numpy as np

    raw = pathlib.Path(DATA) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    frames_all = pathlib.Path(DATA) / "frames_all"
    frames_all.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for name, url in CANDIDATES.items():
        mp4 = raw / f"{name}.mp4"
        if not mp4.exists():
            # grab <=720p mp4; log metadata
            meta = subprocess.run(
                ["yt-dlp", "--dump-json", "--no-warnings", url],
                capture_output=True, text=True,
            )
            if meta.returncode == 0:
                j = json.loads(meta.stdout)
                print(f"[{name}] {j.get('title')!r}  {j.get('duration')}s  "
                      f"by {j.get('uploader')!r}  {j.get('width')}x{j.get('height')}")
                manifest[name] = {k: j.get(k) for k in
                                  ("title", "duration", "uploader", "webpage_url", "width", "height")}
            else:
                print(f"[{name}] metadata failed:\n{meta.stderr[-1500:]}")
            dl = subprocess.run(
                ["yt-dlp", "-f", "bv*[height<=720][ext=mp4]+ba/b[height<=720]",
                 "--merge-output-format", "mp4", "-o", str(mp4), "--no-warnings", url],
                capture_output=True, text=True,
            )
            if dl.returncode != 0 or not mp4.exists():
                print(f"[{name}] download FAILED:\n{dl.stderr[-2000:]}")
                manifest.setdefault(name, {})["download_failed"] = True
                continue
        print(f"[{name}] {mp4.stat().st_size/1e6:.1f} MB")

        # extract frames
        stage = raw / f"frames_{name}"
        stage.mkdir(exist_ok=True)
        for f in stage.glob("*.jpg"):
            f.unlink()
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(mp4), "-vf",
             f"fps={fps},scale='min({long_edge},iw)':-2", "-q:v", "2",
             str(stage / f"{name}_%05d.jpg")],
            check=True, capture_output=True,
        )
        raw_frames = sorted(stage.glob("*.jpg"))

        # sharpness (variance of Laplacian) + crude near-duplicate drop
        scored = []
        prev = None
        for fp in raw_frames:
            im = cv2.imread(str(fp))
            if im is None:
                continue
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            sharp = cv2.Laplacian(g, cv2.CV_64F).var()
            small = cv2.resize(g, (32, 32)).astype(np.float32)
            if prev is not None and np.abs(small - prev).mean() < 4.0:
                continue  # too similar to the last kept frame
            prev = small
            scored.append((sharp, fp))
        scored.sort(reverse=True)
        keep = [fp for _, fp in scored[:max_frames_per_video]]
        keep.sort()
        for fp in keep:
            (frames_all / fp.name).write_bytes(fp.read_bytes())
        print(f"[{name}] {len(raw_frames)} raw -> {len(keep)} kept")
        manifest.setdefault(name, {})["kept_frames"] = len(keep)

    (pathlib.Path(DATA) / "scrape_manifest.json").write_text(json.dumps(manifest, indent=2))
    total = len(list(frames_all.glob("*.jpg")))
    print(f"\nTOTAL frames in frames_all/: {total}")
    vol.commit()
    return manifest


# --------------------------------------------------------------------------- #
# 2. structure from motion  (COLMAP via nerfstudio)
# --------------------------------------------------------------------------- #
@app.function(image=ns_image, volumes={DATA: vol}, gpu="A10G", timeout=3 * 60 * 60)
def sfm(frames: str = "frames_all", matching: str = "vocab_tree"):
    import shutil
    import subprocess

    src = pathlib.Path(DATA) / frames
    assert src.exists(), f"{src} missing — run scrape first"
    out = pathlib.Path(DATA) / f"proc_{frames}"
    if out.exists():
        shutil.rmtree(out)

    cmd = [
        "ns-process-data", "images",
        "--data", str(src),
        "--output-dir", str(out),
        "--matching-method", matching,
        "--sfm-tool", "colmap",
        "--num-downscales", "2",
        "--verbose",
    ]
    print(" ".join(cmd))
    p = subprocess.run(cmd, text=True)
    vol.commit()

    tj = out / "transforms.json"
    if tj.exists():
        j = json.loads(tj.read_text())
        n_reg = len(j.get("frames", []))
        n_in = len(list(src.glob("*.jpg")))
        print(f"\nREGISTERED {n_reg}/{n_in} frames "
              f"({100*n_reg/max(n_in,1):.0f}%) into {tj}")
    else:
        print(f"\nNO transforms.json produced (rc={p.returncode}) — COLMAP failed "
              f"to register a consistent model.")
    return str(tj) if tj.exists() else ""


# --------------------------------------------------------------------------- #
# 3. train splatfacto + export .ply
# --------------------------------------------------------------------------- #
@app.function(image=ns_image, volumes={DATA: vol}, gpu="A100-40GB", timeout=6 * 60 * 60)
def train(proc: str = "proc_frames_all", iters: int = 15000):
    import glob
    import subprocess

    data = pathlib.Path(DATA) / proc
    assert (data / "transforms.json").exists(), f"{data}/transforms.json missing — run sfm"
    train_out = pathlib.Path(DATA) / "train"
    train_out.mkdir(exist_ok=True)

    subprocess.run(
        ["ns-train", "splatfacto",
         "--data", str(data),
         "--output-dir", str(train_out),
         "--max-num-iterations", str(iters),
         "--viewer.quit-on-train-completion", "True",
         "--pipeline.model.cull-alpha-thresh", "0.005",
         "--vis", "tensorboard"],
        check=True, text=True,
    )
    cfg = sorted(glob.glob(str(train_out / "**" / "config.yml"), recursive=True))[-1]
    print("config:", cfg)

    exports = pathlib.Path(DATA) / "exports"
    exports.mkdir(exist_ok=True)
    subprocess.run(
        ["ns-export", "gaussian-splat", "--load-config", cfg,
         "--output-dir", str(exports)],
        check=True, text=True,
    )
    # a quick spiral render for a sanity-check preview
    try:
        subprocess.run(
            ["ns-render", "spiral", "--load-config", cfg,
             "--output-path", str(exports / "preview.mp4"), "--frame-rate", "24",
             "--seconds", "6"],
            check=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        print("preview render failed (non-fatal):", e)

    vol.commit()
    for p in exports.iterdir():
        print(f"  {p.name}: {p.stat().st_size/1e6:.2f} MB")
    return str(cfg)


# --------------------------------------------------------------------------- #
# 4. fetch artifacts locally
# --------------------------------------------------------------------------- #
@app.function(image=scrape_image, volumes={DATA: vol}, timeout=30 * 60)
def _collect() -> dict:
    out = {}
    root = pathlib.Path(DATA)
    for rel in ["exports/point_cloud.ply", "exports/splat.ply", "exports/preview.mp4",
                "scrape_manifest.json"]:
        p = root / rel
        if p.exists():
            out[p.name] = p.read_bytes()
    for tj in root.glob("proc_*/transforms.json"):
        out[f"{tj.parent.name}.transforms.json"] = tj.read_bytes()
    return out


@app.local_entrypoint()
def fetch():
    dst = HERE / "out"
    dst.mkdir(exist_ok=True)
    got = _collect.remote()
    for name, data in got.items():
        (dst / name).write_bytes(data)
        print(f"wrote out/{name}  ({len(data)/1e6:.2f} MB)")
    if not got:
        print("nothing to fetch yet")
