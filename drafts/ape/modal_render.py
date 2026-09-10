"""
Render ape_face.py on a Modal GPU and pull the results back into this folder.

    modal run modal_render.py                 # default: A10G, 200 samples
    modal run modal_render.py --gpu L4 --samples 400 --res 1600
    modal run modal_render.py --no-hair       # skip fur (much faster)

Writes ape_face.png and ape_face.blend next to this file.
"""

import pathlib

import modal

HERE = pathlib.Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libxxf86vm1", "libxfixes3", "libxi6", "libxkbcommon0", "libxrender1",
        "libgl1", "libglx0", "libegl1", "libsm6", "libice6", "xz-utils",
    )
    .pip_install("bpy==4.2.0")
    .add_local_file(HERE / "ape_face.py", "/root/ape_face.py")
)

app = modal.App("ape-face", image=image)


@app.function(gpu="A10G", timeout=60 * 60)
def render(samples: int, res: int, hair: bool, gpu_name: str, debug: bool = False):
    import os
    import subprocess
    import sys

    os.environ.update(
        APE_OUT_DIR="/root/out",
        APE_SAMPLES=str(samples),
        APE_RES=str(res),
        APE_HAIR="1" if hair else "0",
        APE_GPU="1",
    )
    if os.environ.get("APE_DEBUG_PASSTHROUGH") or debug:
        os.environ["APE_DEBUG"] = "1"
    os.makedirs("/root/out", exist_ok=True)

    proc = subprocess.run([sys.executable, "/root/ape_face.py"], capture_output=True, text=True)
    print(proc.stdout[-8000:])
    if proc.returncode != 0:
        print(proc.stderr[-8000:])
        raise RuntimeError(f"ape_face.py exited {proc.returncode}")
    print(proc.stderr[-2000:])

    out = {}
    for p in sorted(pathlib.Path("/root/out").iterdir()):
        if p.suffix in (".png", ".blend"):
            out[p.name] = p.read_bytes()
            print(f"[modal] {p.name}: {len(out[p.name]) / 1e6:.2f} MB")
    return out


@app.local_entrypoint()
def main(gpu: str = "A10G", samples: int = 200, res: int = 1400, hair: bool = True, debug: bool = False):
    fn = render.with_options(gpu=gpu) if gpu != "A10G" else render
    out = fn.remote(samples=samples, res=res, hair=hair, gpu_name=gpu, debug=debug)
    for name, data in out.items():
        dst = HERE / name
        dst.write_bytes(data)
        print(f"wrote {dst}  ({len(data) / 1e6:.2f} MB)")
    if "ape_face.png" not in out:
        raise SystemExit("no render produced")
