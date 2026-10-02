"""
Render dngr.py (the Thorne / Interstellar black-hole ray tracer) on a Modal GPU
and pull the results back into this folder.

    modal run modal_render.py                                   # 720p, 96 frames, A10G
    modal run modal_render.py --frames 240 --fps 24 --ssaa 3    # long, smooth
    modal run modal_render.py --still 0 --width 1920 --height 1080   # one hero still
    modal run modal_render.py --gpu L4                          # cheaper GPU
    modal run modal_render.py --shifts none --flare 0           # paint-swatch look (Fig 13)

Writes gargantua.mp4 (and any stills) next to this file.
"""

import pathlib

import modal

HERE = pathlib.Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install(
        "jax[cuda12]==0.11.1",
        "numpy",
        "imageio",
        "imageio-ffmpeg",
        "pillow",
    )
    .add_local_file(HERE / "dngr.py", "/root/dngr.py")
)

app = modal.App("dngr-interstellar", image=image)


@app.function(gpu="A10G", timeout=60 * 60 * 4)
def render(args: list[str]):
    import os
    import subprocess
    import sys

    os.makedirs("/root/out", exist_ok=True)
    cmd = [sys.executable, "-u", "/root/dngr.py", "--out", "/root/out", *args]
    print("[modal] running:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, stdout=None, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"dngr.py exited {proc.returncode}")

    out = {}
    for p in sorted(pathlib.Path("/root/out").iterdir()):
        if p.suffix == ".mp4" or (p.suffix == ".png" and "--still" in args
                                  and args[args.index("--still") + 1] != "-1"):
            out[p.name] = p.read_bytes()
            print(f"[modal] {p.name}: {len(out[p.name]) / 1e6:.2f} MB")
    return out


@app.local_entrypoint()
def main(
    gpu: str = "A10G",
    width: int = 1280,
    height: int = 720,
    frames: int = 96,
    fps: int = 24,
    ssaa: int = 2,
    spin: float = 0.6,
    radius: float = 30.0,
    incl_deg: float = 2.0,
    fov_deg: float = 42.0,
    orbit_frac: float = 1.0,
    disk_in: float = 9.26,
    disk_out: float = 18.70,
    disk_temp: float = 4500.0,
    disk_gain: float = 1.5,
    star_gain: float = 1.0,
    shifts: str = "full",
    star_shifts: int = 1,
    exposure: float = 1.5,
    flare: float = 0.16,
    max_steps: int = 2500,
    chunk: int = 400_000,
    still: int = -1,
):
    args = [
        "--width", str(width), "--height", str(height),
        "--frames", str(frames), "--fps", str(fps), "--ssaa", str(ssaa),
        "--spin", str(spin), "--radius", str(radius),
        "--incl-deg", str(incl_deg), "--fov-deg", str(fov_deg),
        "--orbit-frac", str(orbit_frac),
        "--disk-in", str(disk_in), "--disk-out", str(disk_out),
        "--disk-temp", str(disk_temp), "--disk-gain", str(disk_gain),
        "--star-gain", str(star_gain), "--shifts", shifts,
        "--star-shifts", str(star_shifts), "--exposure", str(exposure),
        "--flare", str(flare), "--max-steps", str(max_steps),
        "--chunk", str(chunk), "--still", str(still),
    ]
    fn = render.with_options(gpu=gpu) if gpu != "A10G" else render
    out = fn.remote(args)
    for name, data in out.items():
        dst = HERE / name
        dst.write_bytes(data)
        print(f"wrote {dst}  ({len(data) / 1e6:.2f} MB)")
    if not out:
        raise SystemExit("no output produced")
