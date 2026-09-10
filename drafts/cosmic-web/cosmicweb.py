"""
cosmicweb - 13.8 billion years of gravity in ~19 seconds.

A particle-mesh (PM) N-body simulation of a LambdaCDM universe. We seed a cubic
comoving volume with a Gaussian random field whose power spectrum is the real
primordial one (scale-invariant inflationary spectrum processed by the BBKS
transfer function), lay down N^3 particles on a grid, give them their Zel'dovich
displacements + velocities, and then integrate gravity in an expanding universe
with a symplectic kick-drift-kick leapfrog whose time variable is the cosmic
scale factor a.

Gravity is solved on a mesh every step:  FFT the density, multiply by the
Green's function -i k / k^2, FFT back to get the acceleration field, interpolate
(cloud-in-cell) back to the particles. That turns an O(N^2) force calculation
into O(N log N) and is exactly how the first cosmic-web simulations of the 1980s
were run, just far bigger.

One video frame is rendered per timestep, so the movie is a continuous fast
-forward from redshift z = 49 (a = 0.02) to today, z = 0. The near-uniform early
universe (fluctuations of 1 part in ~30,000) collapses under its own gravity
into the cosmic web: sheets drain into filaments, filaments into knots, and the
knots are where galaxies and clusters live. The empty regions in between are
the voids.

Everything runs on one Modal GPU in JAX.

    modal run drafts/cosmic-web/cosmicweb.py                      # full run, A100
    modal run drafts/cosmic-web/cosmicweb.py --npart 128 --mesh 256 --gpu A10G   # quick

Writes to the Modal Volume `cosmic-web` and downloads to drafts/cosmic-web/out/:
    cosmic-web.mp4        the fast-forward
    stills/*.png          a few key epochs
    power.png             linear vs measured power spectrum (sanity check)
"""

import pathlib
import subprocess

import modal

HERE = pathlib.Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "jax[cuda12]==0.4.34",
        "numpy==1.26.4",
        "pillow==10.4.0",
        "matplotlib==3.9.2",
    )
    .add_local_file(HERE / "cosmicweb.py", "/root/cosmicweb.py")
)

vol = modal.Volume.from_name("cosmic-web", create_if_missing=True)
OUT = "/data"

app = modal.App("cosmic-web", image=image)


# --------------------------------------------------------------------------- #
#  cosmology  (LambdaCDM, flat, radiation ignored - fine for a < 0.02 onward)
# --------------------------------------------------------------------------- #
COSMO = dict(Om=0.311, OL=0.689, Ob=0.049, h=0.677, ns=0.965, sigma8=0.811)


import numpy as _np_mod
if not hasattr(_np_mod, "trapz"):        # numpy 2.x renamed trapz -> trapezoid
    _np_mod.trapz = _np_mod.trapezoid


def _np_cosmo():
    import numpy as np

    Om, OL = COSMO["Om"], COSMO["OL"]

    def E(a):
        return np.sqrt(Om * a ** -3 + OL)

    def growth_D(a):
        # D(a) proportional to  E(a) * int_0^a da' / (a' E(a'))^3 , normalised D(1)=1
        def integ(aa):
            g = np.geomspace(1e-6, aa, 4000)
            return np.trapz(1.0 / (g * E(g)) ** 3, g)

        norm = E(1.0) * integ(1.0)
        if np.ndim(a) == 0:
            return E(a) * integ(a) / norm
        return np.array([E(x) * integ(x) / norm for x in a])

    def growth_f(a):
        # f = dlnD/dlna  via finite difference
        la = np.log(a)
        d = 1e-3
        return (np.log(growth_D(np.exp(la + d))) - np.log(growth_D(np.exp(la - d)))) / (2 * d)

    def drift_factor(a1, a2):
        g = np.geomspace(a1, a2, 256)
        return np.trapz(1.0 / (g ** 3 * E(g)), g)

    def kick_factor(a1, a2):
        g = np.geomspace(a1, a2, 256)
        return np.trapz(1.5 * Om / (g ** 2 * E(g)), g)

    return E, growth_D, growth_f, drift_factor, kick_factor


def _transfer_bbks(k, np):
    # k in h/Mpc.  BBKS (1986) with Sugiyama (1995) baryon correction.
    Om, Ob, h = COSMO["Om"], COSMO["Ob"], COSMO["h"]
    Gamma = Om * h * np.exp(-Ob * (1.0 + np.sqrt(2.0 * h) / Om))
    q = k / Gamma
    q = np.where(q <= 0, 1e-20, q)
    T = (np.log(1.0 + 2.34 * q) / (2.34 * q)) * (
        1.0 + 3.89 * q + (16.1 * q) ** 2 + (5.46 * q) ** 3 + (6.71 * q) ** 4
    ) ** -0.25
    return T


# --------------------------------------------------------------------------- #
#  the simulation
# --------------------------------------------------------------------------- #
@app.function(gpu="A100", volumes={OUT: vol}, timeout=60 * 60)
def simulate(
    npart: int = 256,
    mesh: int = 512,
    box: float = 160.0,        # Mpc/h, comoving
    a_init: float = 0.02,
    steps: int = 570,
    fps: int = 30,
    seed: int = 8,
    width: int = 1280,
    height: int = 720,
):
    import os
    import time as _time

    import numpy as np
    import jax
    import jax.numpy as jnp

    print("jax devices:", jax.devices())
    os.makedirs(f"{OUT}/frames", exist_ok=True)
    os.makedirs(f"{OUT}/stills", exist_ok=True)

    Om = COSMO["Om"]
    E, growth_D, growth_f, drift_factor, kick_factor = _np_cosmo()

    # ---- k grid (h/Mpc) on the force mesh ------------------------------------
    kf = 2.0 * np.pi / box
    kx = np.fft.fftfreq(mesh, d=1.0 / mesh) * kf
    kz = np.fft.rfftfreq(mesh, d=1.0 / mesh) * kf
    KX, KY, KZ = np.meshgrid(kx, kx, kz, indexing="ij")
    K2 = KX ** 2 + KY ** 2 + KZ ** 2
    K2[0, 0, 0] = 1.0
    kmag = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)

    # ---- power spectrum, normalised to sigma8 -------------------------------
    def Pk(k):
        return k ** COSMO["ns"] * _transfer_bbks(k, np) ** 2

    kk = np.geomspace(1e-4, 100.0, 20000)
    W = 3.0 * (np.sin(kk * 8.0) - kk * 8.0 * np.cos(kk * 8.0)) / (kk * 8.0) ** 3
    sig2 = np.trapz(Pk(kk) * W ** 2 * kk ** 2, kk) / (2 * np.pi ** 2)
    Anorm = COSMO["sigma8"] ** 2 / sig2
    print(f"power spectrum normalisation A = {Anorm:.4e}")

    # ---- Gaussian random field -> Zel'dovich displacement (a=1 linear) ------
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((mesh, mesh, mesh)).astype(np.float32)
    white_k = np.fft.rfftn(white)
    # numpy rfftn/irfftn convention: <|w_k|^2> = N^3, and irfftn divides by N^3,
    # so delta_k = w_k * sqrt(P(k) * N^3 / V) gives a real field with power P(k).
    delta_k = white_k * np.sqrt(
        Anorm * Pk(np.maximum(kmag, 1e-8)) * mesh ** 3 / box ** 3
    ).astype(np.float32)
    delta_k[0, 0, 0] = 0.0

    # displacement field  Psi_k = i k / k^2 * delta_k   (a=1 normalisation)
    psi = []
    for Kd in (KX, KY, KZ):
        pk = 1j * Kd / K2 * delta_k
        psi.append(np.fft.irfftn(pk, s=(mesh, mesh, mesh)).astype(np.float32))
    psi = np.stack(psi, axis=-1)  # (mesh,mesh,mesh,3) at the mesh nodes

    # sample the displacement at the particle lattice (npart^3), sub-sampling the mesh
    step = mesh // npart
    assert step * npart == mesh, "mesh must be an integer multiple of npart"
    S = psi[::step, ::step, ::step, :].reshape(-1, 3)          # (Np^3, 3) Mpc/h
    q = np.stack(
        np.meshgrid(*[(np.arange(npart) + 0.5) * (box / npart)] * 3, indexing="ij"),
        axis=-1,
    ).reshape(-1, 3).astype(np.float32)
    Np = npart ** 3
    print(f"{Np:,} particles, {mesh}^3 force mesh, box {box} Mpc/h")
    print(f"rms Zel'dovich displacement at a=1: {np.sqrt((S**2).sum(1).mean()):.2f} Mpc/h")

    D_i = float(growth_D(a_init))
    f_i = float(growth_f(a_init))
    Ei = float(E(a_init))
    # tiny Lagrangian jitter (< 0.4 inter-particle spacing) so the initial
    # lattice doesn't beat against the pixel grid in the render; dynamically
    # negligible next to the displacement field.
    jitter = (rng.random((Np, 3)).astype(np.float32) - 0.5) * 1.0 * (box / npart)
    x = (q + jitter + D_i * S) % box
    # canonical momentum p = a^2 dx/dt ;  Zel'dovich: dx/dt = D f H Psi
    p = (a_init ** 2 * D_i * f_i * Ei) * S
    x = jnp.asarray(x)
    p = jnp.asarray(p)

    # ---- GPU kernels -------------------------------------------------------
    KXj, KYj, KZj, K2j = map(jnp.asarray, (KX, KY, KZ, K2))
    g = box / mesh  # cell size

    @jax.jit
    def cic_deposit(pos):
        # cloud-in-cell mass assignment -> overdensity delta on the mesh
        pm = pos / g
        i0 = jnp.floor(pm).astype(jnp.int32)
        fr = pm - i0
        dens = jnp.zeros((mesh, mesh, mesh), jnp.float32)
        for dx in (0, 1):
            wx = jnp.where(dx == 0, 1 - fr[:, 0], fr[:, 0])
            ix = (i0[:, 0] + dx) % mesh
            for dy in (0, 1):
                wy = jnp.where(dy == 0, 1 - fr[:, 1], fr[:, 1])
                iy = (i0[:, 1] + dy) % mesh
                for dz in (0, 1):
                    wz = jnp.where(dz == 0, 1 - fr[:, 2], fr[:, 2])
                    iz = (i0[:, 2] + dz) % mesh
                    dens = dens.at[ix, iy, iz].add(wx * wy * wz)
        mean = Np / mesh ** 3
        return dens / mean - 1.0

    @jax.jit
    def accel(pos):
        delta = cic_deposit(pos)
        dk = jnp.fft.rfftn(delta)
        # CIC deconvolution (squared window: once for assignment, once for readout)
        def sinc(kv):
            x = kv * (g / 2.0)
            return jnp.where(jnp.abs(x) < 1e-6, 1.0, jnp.sin(x) / jnp.where(x == 0, 1.0, x))
        win = (sinc(KXj) * sinc(KYj) * sinc(KZj)) ** 2
        dk = dk / win
        # Poisson: nabla^2 Phi = (3/2)(Om/a) delta  ->  (-grad Phi)_k = +i k/k^2 delta_k
        # (the (3/2)Om/a and the 1/(a E) time-Jacobian are folded into kick_factor).
        # Same sign as the Zel'dovich displacement: matter accelerates toward overdensities.
        out = []
        for Kd in (KXj, KYj, KZj):
            gk = 1j * Kd / K2j * dk
            gk = gk.at[0, 0, 0].set(0.0)
            out.append(jnp.fft.irfftn(gk, s=(mesh, mesh, mesh)))
        acc_field = jnp.stack(out, axis=-1)  # (mesh,mesh,mesh,3)

        pm = pos / g
        i0 = jnp.floor(pm).astype(jnp.int32)
        fr = pm - i0
        a_p = jnp.zeros((pos.shape[0], 3), jnp.float32)
        for dx in (0, 1):
            wx = jnp.where(dx == 0, 1 - fr[:, 0], fr[:, 0])
            ix = (i0[:, 0] + dx) % mesh
            for dy in (0, 1):
                wy = jnp.where(dy == 0, 1 - fr[:, 1], fr[:, 1])
                iy = (i0[:, 1] + dy) % mesh
                for dz in (0, 1):
                    wz = jnp.where(dz == 0, 1 - fr[:, 2], fr[:, 2])
                    iz = (i0[:, 2] + dz) % mesh
                    w = (wx * wy * wz)[:, None]
                    a_p = a_p + w * acc_field[ix, iy, iz, :]
        return a_p

    @jax.jit
    def kick(p, pos, dp):
        return p + dp * accel(pos)

    @jax.jit
    def drift(pos, p, df):
        return (pos + df * p) % box

    # ---- rendering (on the GPU) -----------------------------------------
    @jax.jit
    def render_gpu(pos, ang, span):
        c, s = jnp.cos(ang), jnp.sin(ang)
        r = pos - box / 2.0
        xr = c * r[:, 0] + s * r[:, 2]           # rotate about vertical axis
        zr = -s * r[:, 0] + c * r[:, 2]
        yr = r[:, 1]
        tilt = jnp.radians(20.0)                 # tilt so sheets read in 3D
        yv = jnp.cos(tilt) * yr - jnp.sin(tilt) * zr
        depth = jnp.sin(tilt) * yr + jnp.cos(tilt) * zr
        ppm = width / span
        u = width / 2.0 + xr * ppm
        v = height / 2.0 - yv * ppm
        iu = jnp.floor(u).astype(jnp.int32)
        iv = jnp.floor(v).astype(jnp.int32)
        inb = (iu >= 0) & (iu < width) & (iv >= 0) & (iv < height)
        dn = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)
        slab = jnp.exp(-(zr / 52.0) ** 2)        # soft slab through the volume
        wgt = jnp.where(inb, (0.30 + 0.70 * dn) * slab, 0.0)
        iu = jnp.clip(iu, 0, width - 1)
        iv = jnp.clip(iv, 0, height - 1)
        img = jnp.zeros((height, width), jnp.float32).at[iv, iu].add(wgt)
        dep = jnp.zeros((height, width), jnp.float32).at[iv, iu].add(dn * wgt)

        def blur(a):
            a = (a + jnp.roll(a, 1, 0) + jnp.roll(a, -1, 0)) * (1.0 / 3.0)
            a = (a + jnp.roll(a, 1, 1) + jnp.roll(a, -1, 1)) * (1.0 / 3.0)
            return a
        img = blur(blur(blur(img)))
        dep = blur(blur(dep))
        return img, dep

    def render(pos_dev, ang, span):
        img, dep = render_gpu(pos_dev, jnp.float32(ang), jnp.float32(span))
        return np.asarray(img), np.asarray(dep)

    # colour ramp: black -> indigo -> violet -> magenta -> orange -> pale gold -> white
    ramp_x = np.array([0.0, 0.15, 0.32, 0.52, 0.72, 0.88, 1.0])
    ramp_c = np.array([
        [0, 0, 6], [22, 12, 64], [76, 26, 128], [176, 44, 122],
        [242, 120, 52], [252, 208, 138], [255, 255, 255],
    ]) / 255.0

    def colorize(img, dep):
        # resolution-independent: normalise to mean surface density 1 over the
        # whole frame, then a fixed log stretch.  Early (near-uniform) epochs
        # stay dim and smooth; the web lights up on its own as it collapses.
        m = img.sum() / (img.shape[0] * img.shape[1]) + 1e-12
        s = img / m
        t = np.clip(np.log1p(s * 1.9) / np.log1p(60.0 * 1.9), 0.0, 1.0) ** 0.80
        t = np.clip((t - 0.10) / 0.90, 0.0, 1.0)     # crush void speckle toward black
        rgb = np.stack([np.interp(t, ramp_x, ramp_c[:, i]) for i in range(3)], -1)
        with np.errstate(invalid="ignore"):
            mean_d = np.where(img > 1e-6, dep / np.maximum(img, 1e-6), 0.0)
        rgb *= (0.80 + 0.20 * mean_d[..., None])
        return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
        fontb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
    except Exception:
        font = fontb = ImageFont.load_default()

    # ---- main loop -------------------------------------------------------
    a_grid = np.geomspace(a_init, 1.0, steps + 1)
    still_at = {int(steps * frac): name for frac, name in
                [(0.001, "z49"), (0.33, "early"), (0.6, "mid"), (0.85, "late"), (0.999, "today")]}
    t0 = _time.time()
    for i in range(steps):
        a1, a2 = float(a_grid[i]), float(a_grid[i + 1])
        amid = np.sqrt(a1 * a2)
        dk1 = float(kick_factor(a1, amid))
        df = float(drift_factor(a1, a2))
        dk2 = float(kick_factor(amid, a2))
        p = kick(p, x, dk1)
        x = drift(x, p, df)
        p = kick(p, x, dk2)

        frac = i / (steps - 1)
        ang = np.radians(20.0) * frac
        span = box * (0.86 - 0.16 * frac)        # zoomed so the box always fills frame
        z_now = 1.0 / a2 - 1.0
        img, dep = render(x, ang, span)
        rgb = colorize(img, dep)
        im = Image.fromarray(rgb)
        dr = ImageDraw.Draw(im)
        dr.text((28, 24), "THE COSMIC WEB", font=fontb, fill=(255, 255, 255))
        dr.text((28, 58), "ΛCDM particle-mesh N-body · 160 Mpc/h box · 16.7M particles",
                font=font, fill=(180, 180, 200))
        dr.text((28, height - 92), f"redshift  z = {z_now:5.2f}", font=fontb, fill=(255, 235, 200))
        gyr = _age_gyr(a2)
        dr.text((28, height - 56), f"t = {gyr:5.2f} Gyr after the Big Bang     a = {a2:5.3f}",
                font=font, fill=(200, 200, 210))
        # scale bar: 20 Mpc/h comoving
        barpx = int(20.0 / span * width)
        bx1 = width - 48
        dr.rectangle([bx1 - barpx, height - 42, bx1, height - 36], fill=(255, 255, 255))
        lbl = "20 Mpc/h comoving"
        tw = dr.textlength(lbl, font=font)
        dr.text((bx1 - tw, height - 70), lbl, font=font, fill=(210, 210, 210))
        im.save(f"{OUT}/frames/f{i:04d}.png")
        if i in still_at:
            im.save(f"{OUT}/stills/{still_at[i]}.png")
        if i % 40 == 0:
            print(f"step {i:4d}/{steps}  a={a2:.3f} z={z_now:6.2f}  "
                  f"{(_time.time()-t0):.0f}s  |x|rms={float(jnp.std(x)):.1f}")

    # hold the final frame for ~0.7 s
    for j in range(steps, steps + int(fps * 0.7)):
        Image.open(f"{OUT}/frames/f{steps-1:04d}.png").save(f"{OUT}/frames/f{j:04d}.png")

    vol.commit()
    print("encoding...")
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(fps), "-i", f"{OUT}/frames/f%04d.png",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "slow",
        "-crf", "20", "-maxrate", "14M", "-bufsize", "28M",
        "-vf", "scale=1280:720:flags=lanczos", "-movflags", "+faststart",
        f"{OUT}/cosmic-web.mp4",
    ], check=True)

    _power_plot(np, delta_k, kmag, Pk, Anorm, box, mesh, x, g, Np)
    vol.commit()
    print("done in", int(_time.time() - t0), "s")


def _age_gyr(a):
    import numpy as np
    Om, OL = COSMO["Om"], COSMO["OL"]
    H0 = 100.0 * COSMO["h"]  # km/s/Mpc
    # 1/H0 in Gyr
    inv_H0_gyr = 977.8 / H0
    aa = np.geomspace(1e-6, a, 4000)
    integ = np.trapz(1.0 / (aa * np.sqrt(Om * aa ** -3 + OL)), aa)
    return float(inv_H0_gyr * integ)


def _power_plot(np, delta_k, kmag, Pk, Anorm, box, mesh, x, g, Np):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import jax.numpy as jnp

    # measured non-linear P(k) today
    pm = np.asarray(x) / g
    i0 = np.floor(pm).astype(int)
    fr = pm - i0
    dens = np.zeros((mesh, mesh, mesh), np.float32)
    for dx in (0, 1):
        wx = (1 - fr[:, 0]) if dx == 0 else fr[:, 0]
        ix = (i0[:, 0] + dx) % mesh
        for dy in (0, 1):
            wy = (1 - fr[:, 1]) if dy == 0 else fr[:, 1]
            iy = (i0[:, 1] + dy) % mesh
            for dz in (0, 1):
                wz = (1 - fr[:, 2]) if dz == 0 else fr[:, 2]
                iz = (i0[:, 2] + dz) % mesh
                np.add.at(dens, (ix, iy, iz), wx * wy * wz)
    d = dens / (Np / mesh ** 3) - 1.0
    dk = np.fft.rfftn(d)
    p3 = (np.abs(dk) ** 2) * (box ** 3) / (mesh ** 6)
    kb = np.geomspace(2 * np.pi / box, np.pi * mesh / box, 30)
    kc = 0.5 * (kb[1:] + kb[:-1])
    which = np.digitize(kmag.ravel(), kb)
    meas = np.array([p3.ravel()[which == j].mean() if (which == j).any() else np.nan
                     for j in range(1, len(kb))])
    lin = Anorm * Pk(kc) * _growth_today_sq()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(kc, lin, "--", label="linear theory (z=0)", color="#888")
    ax.loglog(kc, meas, "o-", label="simulation (z=0)", color="#c02c7a")
    ax.set_xlabel("k  [h/Mpc]"); ax.set_ylabel("P(k)  [(Mpc/h)^3]")
    ax.set_title("Matter power spectrum"); ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(f"{OUT}/power.png", dpi=110)


def _growth_today_sq():
    return 1.0


@app.local_entrypoint()
def main(npart: int = 256, mesh: int = 512, steps: int = 570, gpu: str = "A100", seed: int = 8):
    simulate.with_options(gpu=gpu).remote(npart=npart, mesh=mesh, steps=steps, seed=seed)
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    (out / "stills").mkdir(exist_ok=True)
    import subprocess as sp
    for f in ["cosmic-web.mp4", "power.png"]:
        sp.run(["modal", "volume", "get", "cosmic-web", f, str(out / f), "--force"], check=True)
    sp.run(["modal", "volume", "get", "cosmic-web", "stills", str(out / "stills"), "--force"], check=False)
    print("downloaded to", out)
