"""
dngr.py  --  a small Double Negative Gravitational Renderer.

A from-scratch reimplementation of the ray-tracing prescription in

    James, von Tunzelmann, Franklin & Thorne,
    "Gravitational Lensing by Spinning Black Holes in Astrophysics,
     and in the Movie Interstellar",
    Classical and Quantum Gravity 32 (2015) 065001  (arXiv:1502.03808)

Everything below follows Appendix A of that paper as closely as is practical
in a single GPU kernel.  Equation numbers in the comments refer to the paper.

What is faithful to the paper
-----------------------------
* Kerr metric in Boyer-Lindquist coordinates, M = 1                     (A.1, A.2)
* FIDO orthonormal frame                                                (A.3)
* the exact camera -> local-sky -> celestial-sphere prescription
  - incoming ray unit vector N in the camera frame                      (A.8)
  - relativistic aberration into the FIDO frame                         (A.9)
  - projection onto the FIDO spherical basis via the motion vector B    (A.10)
  - canonical momenta p_r, p_theta and constants of motion b, q         (A.11, A.12)
* the super-Hamiltonian form of the null-geodesic equations             (A.4, A.15)
  integrated backwards in time with an adaptive Runge-Kutta-Fehlberg
  (Cash-Karp 5(4)) step, exactly as described under (A.15) / Appendix A.4
* circular equatorial geodesic camera orbit                             (A.7)
* blue-shift / Doppler + gravitational frequency shift of the disc and
  the stars, with brightness following Liouville  I_nu / nu^3 = const   (Sec 4.1.2, A.6)
* constant-temperature black-body accretion disc, T = 4500 K            (Sec 4, A.6)
* optional "veiling flare" bloom characteristic of IMAX lenses          (Sec 4.2, 4.3.1)

Where we deviate (and why)
--------------------------
* The paper propagates an elliptical *ray bundle* alongside every central
  ray (Sachs / Pineault-Roeder optical-scalar equations, A.17-A.30) purely
  for anti-aliasing.  We instead stochastically super-sample each pixel.
  The paper itself notes (Appendix A.7) that flat-space ray-differential
  methods are "mathematically equivalent"; GRay and most astrophysical
  codes also trace individual rays.
* The Interstellar disc was a painted texture from a DNeg artist.  Ours is
  procedural (spiral turbulence), same infinitely-thin planar model (A.6).
* The celestial sphere is a procedurally generated star field + Milky Way
  rather than the DNeg Tycho-based star catalogue.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", False)  # fp32 is plenty and 2x faster on GPU


# ----------------------------------------------------------------------------
# Kerr metric quantities, Boyer-Lindquist, geometrised units  M = 1   (A.1, A.2)
# ----------------------------------------------------------------------------
def metric_factors(r, theta, a):
    cos2 = jnp.cos(theta) ** 2
    sin2 = jnp.sin(theta) ** 2
    rho2 = r * r + a * a * cos2                       # rho^2                 (A.1)
    delta = r * r - 2.0 * r + a * a                   # Delta                (A.1)
    sigma2 = (r * r + a * a) ** 2 - a * a * delta * sin2   # Sigma^2         (A.1)
    sigma = jnp.sqrt(jnp.maximum(sigma2, 1e-12))
    alpha = rho2 ** 0.5 * jnp.sqrt(jnp.maximum(delta, 1e-12)) / sigma  # lapse (A.2)
    omega = 2.0 * a * r / sigma2                      # frame dragging       (A.2)
    varpi = sigma * jnp.sqrt(sin2) / rho2 ** 0.5      # cylindrical radius    (A.2)
    return rho2, delta, sigma, alpha, omega, varpi


# ----------------------------------------------------------------------------
# Super-Hamiltonian for null geodesics.
#
# The paper writes the equations of motion (A.15) as gradients of the bracket
#     B = -(Delta/2 rho^2) p_r^2 - (1/2 rho^2) p_theta^2 + (R + Delta*Theta)/(2 Delta rho^2)
# We use K = -B, which is the usual super-Hamiltonian  K = 1/2 g^{ab} p_a p_b
# (identically zero on a null ray), and integrate Hamilton's equations:
#     dr/dl      =  dK/dp_r        =  (Delta/rho^2) p_r
#     dtheta/dl  =  dK/dp_theta    =  (1/rho^2) p_theta
#     dphi/dl    =  dK/db          = -d/db [ (R + Delta Theta)/(2 Delta rho^2) ]
#     dp_r/dl    = -dK/dr          =  dB/dr
#     dp_theta/dl= -dK/dtheta      =  dB/dtheta
# matching (A.15) term by term.  The r- and theta-derivatives are taken with
# JAX autodiff so they are exact.
# ----------------------------------------------------------------------------
def _K(r, theta, p_r, p_th, b, q, a):
    rho2, delta, _, _, _, _ = metric_factors(r, theta, a)
    sin2 = jnp.maximum(jnp.sin(theta) ** 2, 1e-8)
    P = r * r + a * a - a * b                                        # (A.4)
    R = P * P - delta * ((b - a) ** 2 + q)                           # (A.4)
    Theta = q - jnp.cos(theta) ** 2 * (b * b / sin2 - a * a)         # (A.4)
    kin = 0.5 * delta / rho2 * p_r ** 2 + 0.5 / rho2 * p_th ** 2
    pot = -(R + delta * Theta) / (2.0 * delta * rho2)
    return kin + pot


_dK_all = jax.grad(_K, argnums=(0, 1, 4))  # d/dr, d/dtheta, d/db  (exact)


def geodesic_rhs(state, b, q, a):
    """d(r, theta, phi, p_r, p_theta)/d(affine)  --  the RHS of (A.15)."""
    r, theta, phi, p_r, p_th = state
    rho2, delta, _, _, _, _ = metric_factors(r, theta, a)
    dK_dr, dK_dth, dK_db = _dK_all(r, theta, p_r, p_th, b, q, a)
    dr = delta / rho2 * p_r
    dth = p_th / rho2
    dphi = dK_db
    dp_r = -dK_dr
    dp_th = -dK_dth
    return jnp.array([dr, dth, dphi, dp_r, dp_th])


# ----------------------------------------------------------------------------
# Camera  ->  initial ray momentum      (A.7 - A.12)
# ----------------------------------------------------------------------------
def orbit_params(r_c, a):
    """Circular, equatorial, geodesic orbit  (A.7).  Returns Omega, beta, u^t."""
    omega_orbit = 1.0 / (a + r_c ** 1.5)                             # (A.7)
    _, _, _, alpha, omega, varpi = metric_factors(r_c, np.pi / 2, a)
    beta = varpi / alpha * (omega_orbit - omega)                     # (A.7)
    # u^t of the orbit, equatorial (rho^2 = r^2):
    gtt = -(1.0 - 2.0 / r_c)
    gtp = -2.0 * a / r_c
    gpp = r_c * r_c + a * a + 2.0 * a * a / r_c
    ut = 1.0 / jnp.sqrt(-(gtt + 2.0 * omega_orbit * gtp + omega_orbit ** 2 * gpp))
    return omega_orbit, beta, ut


def initial_ray(N, r_c, theta_c, a, beta):
    """
    N : incoming-ray unit vector in the camera's proper frame, Cartesian,
        with x -> FIDO r-hat, y -> FIDO phi-hat (== direction of motion),
        z -> FIDO theta-hat.  This is exactly the paper's camera frame for a
        camera on a circular equatorial orbit, where B = (0, 0, 1), kappa = 1
        and (A.10) collapses to  n_r = n_x , n_theta = -n_z , n_phi = n_y .
    Returns (state0, b, q, EF).
    """
    Nx, Ny, Nz = N[..., 0], N[..., 1], N[..., 2]
    s = jnp.sqrt(jnp.maximum(1.0 - beta * beta, 1e-12))

    # relativistic aberration into the FIDO frame           (A.9)
    denom = 1.0 - beta * Ny
    nF_y = (-Ny + beta) / denom
    nF_x = -s * Nx / denom
    nF_z = -s * Nz / denom

    # FIDO spherical basis components, B = (0,0,1)           (A.10)
    n_r = nF_x
    n_th = -nF_z
    n_ph = nF_y

    rho2, delta, _, alpha, omega, varpi = metric_factors(r_c, theta_c, a)
    rho = jnp.sqrt(rho2)
    sin2 = jnp.maximum(jnp.sin(theta_c) ** 2, 1e-8)

    EF = 1.0 / (alpha + omega * varpi * n_ph)                        # (A.11)
    p_r = EF * rho / jnp.sqrt(delta) * n_r                           # (A.11)
    p_th = EF * rho * n_th                                           # (A.11)
    p_ph = EF * varpi * n_ph                                         # (A.11)

    b = p_ph                                                         # (A.12)
    q = p_th ** 2 + jnp.cos(theta_c) ** 2 * (b * b / sin2 - a * a)   # (A.12)

    state0 = jnp.array([r_c, theta_c, 0.0, p_r, p_th])
    return state0, b, q, EF


# ----------------------------------------------------------------------------
# Adaptive Runge-Kutta-Fehlberg  (Cash-Karp 5(4)), integrated *backwards* in
# time.  With tau = -affine, dstate/dtau = -geodesic_rhs.       (below A.15)
# ----------------------------------------------------------------------------
_CK_C = jnp.array([0.0, 1 / 5, 3 / 10, 3 / 5, 1.0, 7 / 8])
_CK_A = jnp.array([
    [0, 0, 0, 0, 0],
    [1 / 5, 0, 0, 0, 0],
    [3 / 40, 9 / 40, 0, 0, 0],
    [3 / 10, -9 / 10, 6 / 5, 0, 0],
    [-11 / 54, 5 / 2, -70 / 27, 35 / 27, 0],
    [1631 / 55296, 175 / 512, 575 / 13824, 44275 / 110592, 253 / 4096],
])
_CK_B5 = jnp.array([37 / 378, 0, 250 / 621, 125 / 594, 0, 512 / 1771])
_CK_B4 = jnp.array([2825 / 27648, 0, 18575 / 48384, 13525 / 55296, 277 / 14336, 1 / 4])


def _rkf_step(y, h, b, q, a):
    def f(s):
        return -geodesic_rhs(s, b, q, a)

    k0 = f(y)
    k1 = f(y + h * _CK_A[1, 0] * k0)
    k2 = f(y + h * (_CK_A[2, 0] * k0 + _CK_A[2, 1] * k1))
    k3 = f(y + h * (_CK_A[3, 0] * k0 + _CK_A[3, 1] * k1 + _CK_A[3, 2] * k2))
    k4 = f(y + h * (_CK_A[4, 0] * k0 + _CK_A[4, 1] * k1 + _CK_A[4, 2] * k2 + _CK_A[4, 3] * k3))
    k5 = f(y + h * (_CK_A[5, 0] * k0 + _CK_A[5, 1] * k1 + _CK_A[5, 2] * k2
                    + _CK_A[5, 3] * k3 + _CK_A[5, 4] * k4))
    ks = jnp.stack([k0, k1, k2, k3, k4, k5])
    y5 = y + h * jnp.tensordot(_CK_B5, ks, axes=1)
    y4 = y + h * jnp.tensordot(_CK_B4, ks, axes=1)
    return y5, jnp.abs(y5 - y4)


# integration status codes
RUN, HORIZON, ESCAPED, DISK = 0, 1, 2, 3


def trace_ray(N, r_c, theta_c, a, beta, cfg):
    """Backward-integrate one ray.  Returns (status, theta0, phi0, r_hit, phi_hit,
    b, q, p_r_hit, p_th_hit)."""
    state0, b, q, _ = initial_ray(N, r_c, theta_c, a, beta)
    r_h = 1.0 + jnp.sqrt(jnp.maximum(1.0 - a * a, 0.0))

    r_in, r_out = cfg["disk_in"], cfg["disk_out"]
    r_esc = cfg["r_escape"]
    # Safety net: a ray with Carter constant q ~ 0 launched exactly in the
    # equatorial plane stays in it forever (Theta(pi/2) = q = p_theta^2) and so
    # never "crosses" pi/2.  The camera sits just off the plane precisely so this
    # set is empty, but catch any strays by radius alone rather than let them
    # streak the celestial-sphere equator across the image.
    equatorial = q < 5e-4
    atol, rtol = cfg["atol"], cfg["rtol"]
    hmin, hmax = cfg["hmin"], cfg["hmax"]
    max_steps = cfg["max_steps"]

    def cond(c):
        _, _, step, status, *_ = c
        return (status == RUN) & (step < max_steps)

    def body(c):
        y, h, step, status, hit = c
        y5, err = _rkf_step(y, h, b, q, a)
        sc = atol + rtol * jnp.maximum(jnp.abs(y), jnp.abs(y5))
        e = jnp.max(err / sc)
        accept = (e <= 1.0) | (h <= hmin * 1.0001)

        y_new = jnp.where(accept, y5, y)
        r_prev, th_prev = y[0], y[1]
        r_now, th_now = y_new[0], y_new[1]

        # --- classify the accepted state -----------------------------------
        st = status
        hit_new = hit

        # equatorial-plane (disc) intersection: normally the ray crosses
        # theta = pi/2 between two steps and we interpolate the hit point; the
        # `in_plane` branch only fires for a stray exactly-equatorial ray (see
        # `equatorial` above), which never changes the sign of theta - pi/2.
        crossed = accept & ((th_prev - jnp.pi / 2) * (th_now - jnp.pi / 2) < 0.0)
        frac = (jnp.pi / 2 - th_prev) / (th_now - th_prev + 1e-30)
        r_cross = r_prev + frac * (r_now - r_prev)
        phi_cross = y[2] + frac * (y_new[2] - y[2])
        in_plane = accept & equatorial & (r_now < r_prev)   # confined ray, moving inward
        r_hit_ = jnp.where(crossed, r_cross, r_now)
        phi_hit_ = jnp.where(crossed, phi_cross, y_new[2])
        on_disk = (crossed | in_plane) & (r_hit_ >= r_in) & (r_hit_ <= r_out)
        st = jnp.where(on_disk & (st == RUN), DISK, st)
        hit_new = jnp.where(
            on_disk & (hit[0] == RUN),
            jnp.array([float(DISK), r_hit_, phi_hit_, y_new[3], y_new[4]]),
            hit_new,
        )

        st = jnp.where(accept & (r_now <= r_h * (1.0 + 1e-3)) & (st == RUN), HORIZON, st)
        st = jnp.where(accept & (r_now >= r_esc) & (st == RUN), ESCAPED, st)

        # --- PI-ish step size control ------------------------------------
        fac = jnp.where(e > 0, 0.9 * e ** (-0.2), 5.0)
        fac = jnp.clip(fac, 0.2, 5.0)
        h_new = jnp.clip(h * fac, hmin, hmax)
        # slow down where the ray bends sharply near the hole
        h_new = jnp.minimum(h_new, 0.10 * jnp.maximum(r_now - r_h, 0.02))

        return (y_new, h_new, step + 1, st, hit_new)

    h0 = jnp.minimum(hmax, 0.1 * jnp.maximum(r_c - r_h, 0.05))
    hit0 = jnp.array([float(RUN), 0.0, 0.0, 0.0, 0.0])
    y, h, step, status, hit = jax.lax.while_loop(
        cond, body, (state0, h0, 0, RUN, hit0)
    )
    # ran out of steps: classify by where the ray ended up rather than blindly
    # blackening it -- a ray still outside the camera radius is on its way out.
    leftover = status == RUN
    status = jnp.where(leftover & (y[0] > r_c), ESCAPED,
                       jnp.where(leftover, HORIZON, status))

    # wrap the celestial-sphere hit point to (theta in [0,pi], phi in [0,2pi))
    th0 = jnp.arccos(jnp.clip(jnp.cos(y[1]), -1.0, 1.0))
    ph0 = jnp.mod(y[2], 2.0 * jnp.pi)
    return (status, th0, ph0, hit[1], jnp.mod(hit[2], 2 * jnp.pi), b, q, hit[3], hit[4])


# ----------------------------------------------------------------------------
# Shading
# ----------------------------------------------------------------------------
def blackbody_rgb(T):
    """Planckian-locus approximation (Tanner Helland), linear-ish sRGB, T in K."""
    T = jnp.clip(T, 1000.0, 40000.0) / 100.0

    Tm60 = jnp.maximum(T - 60.0, 1e-3)
    r = jnp.where(T <= 66, 255.0,
                  jnp.clip(329.698727 * Tm60 ** -0.1332047592, 0, 255))
    g = jnp.where(T <= 66,
                  jnp.clip(99.4708025861 * jnp.log(jnp.maximum(T, 1e-3)) - 161.1195681661, 0, 255),
                  jnp.clip(288.1221695283 * Tm60 ** -0.0755148492, 0, 255))
    bl = jnp.where(T >= 66, 255.0,
                   jnp.where(T <= 19, 0.0,
                             jnp.clip(138.5177312231 * jnp.log(jnp.maximum(T - 10, 1e-3)) - 305.0447927307, 0, 255)))
    rgb = jnp.stack([r, g, bl], axis=-1) / 255.0
    return rgb ** 2.2  # to ~linear


def _hash2(ix, iy):
    h = ix * 374761393 + iy * 668265263
    h = (h ^ (h >> 13)) * 1274126177
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF).astype(jnp.float32) / float(0x1000000)


def _value_noise(x, y):
    x0 = jnp.floor(x); y0 = jnp.floor(y)
    fx = x - x0; fy = y - y0
    ix = x0.astype(jnp.int32); iy = y0.astype(jnp.int32)
    a = _hash2(ix, iy); b = _hash2(ix + 1, iy)
    c = _hash2(ix, iy + 1); d = _hash2(ix + 1, iy + 1)
    ux = fx * fx * (3 - 2 * fx); uy = fy * fy * (3 - 2 * fy)
    return a * (1 - ux) * (1 - uy) + b * ux * (1 - uy) + c * (1 - ux) * uy + d * ux * uy


def disk_texture(r, phi, r_in, r_out):
    """Procedural infinitely-thin disc emissivity in [0,1] (spiral turbulence)."""
    x = jnp.clip((r - r_in) / (r_out - r_in), 0.0, 1.0)
    edge = jnp.clip(x / 0.05, 0, 1) * jnp.clip((1.0 - x) / 0.14, 0, 1)
    # bright, hot inner rim falling off outward
    radial = 0.25 + 1.5 * jnp.exp(-3.0 * x) + 0.5 * (1.0 - x) ** 1.5

    lr = jnp.log(r)
    swirl = phi + 7.0 * lr                      # trailing spiral arms
    n = 0.0
    amp, freq = 0.6, 1.0
    for _ in range(5):
        n += amp * _value_noise(swirl * freq * 2.2, lr * freq * 6.0)
        amp *= 0.55; freq *= 2.0
    n = 0.55 + 0.9 * n
    filaments = 0.7 + 0.5 * (0.5 + 0.5 * jnp.sin(swirl * 8.0 + 4.0 * n))
    return jnp.clip(edge * radial * n * filaments, 0.0, 6.0)


def orbit_ut(r, Omega, a):
    gtt = -(1.0 - 2.0 / r)
    gtp = -2.0 * a / r
    gpp = r * r + a * a + 2.0 * a * a / r
    return 1.0 / jnp.sqrt(jnp.maximum(-(gtt + 2 * Omega * gtp + Omega ** 2 * gpp), 1e-9))


def shade(result, cfg, star_tex, cam_Omega, cam_ut, world_phi):
    status, th0, ph0, r_hit, phi_hit, b, q, pr_h, pth_h = result
    a = cfg["a"]
    # camera orbits to azimuth `world_phi`; by axisymmetry that is the same as
    # rotating the whole scene, so add it to every sampled longitude.
    ph0 = jnp.mod(ph0 + world_phi, 2 * jnp.pi)
    phi_hit = jnp.mod(phi_hit + world_phi, 2 * jnp.pi)

    # ---- celestial sphere -------------------------------------------------
    H, W = star_tex.shape[:2]
    u = ph0 / (2 * jnp.pi) * W
    v = th0 / jnp.pi * (H - 1)
    x0 = jnp.floor(u).astype(jnp.int32) % W
    x1 = (x0 + 1) % W
    y0 = jnp.clip(jnp.floor(v).astype(jnp.int32), 0, H - 1)
    y1 = jnp.clip(y0 + 1, 0, H - 1)
    fx = u - jnp.floor(u); fy = v - jnp.floor(v)
    star = (star_tex[y0, x0] * (1 - fx) * (1 - fy) + star_tex[y0, x1] * fx * (1 - fy)
            + star_tex[y1, x0] * (1 - fx) * fy + star_tex[y1, x1] * fx * fy)

    if cfg["star_shifts"]:
        g_star = cam_ut * (1.0 - b * cam_Omega)                 # E_obs / E_emit(inf)
        gs = jnp.clip(g_star, 0.5, 1.8)                         # keep the sky sane
        star = star * gs ** 3
        star = star * blackbody_rgb(6500.0 * gs) / (blackbody_rgb(6500.0) + 1e-6)

    # ---- accretion disc -------------------------------------------------
    Om_e = 1.0 / (a + r_hit ** 1.5)
    ut_e = orbit_ut(jnp.maximum(r_hit, 1e-3), Om_e, a)
    E_emit = ut_e * (1.0 - b * Om_e)
    E_obs = cam_ut * (1.0 - b * cam_Omega)
    g = jnp.where(E_emit != 0, E_obs / E_emit, 1.0)             # nu_obs / nu_emit

    mode = cfg["shifts"]                                        # none | color | full
    T = jnp.where(mode == 0, cfg["disk_T"], cfg["disk_T"] * g)
    bright = jnp.where(mode == 2, g ** 4, 1.0)                  # Liouville I_nu/nu^3
    col = blackbody_rgb(T)
    # normalise colour so a 6500 K source renders neutral (camera white balance)
    col = col / (blackbody_rgb(6500.0) + 1e-6)
    col = jnp.where(mode == 1, col / (0.30 * col[0] + 0.59 * col[1] + 0.11 * col[2] + 1e-6), col)

    tex = disk_texture(jnp.maximum(r_hit, r_in_guard(cfg)), phi_hit, cfg["disk_in"], cfg["disk_out"])
    disk_rgb = col * tex * bright * cfg["disk_gain"]

    out = jnp.where(status == DISK, disk_rgb,
                    jnp.where(status == ESCAPED, star * cfg["star_gain"], 0.0))
    return out


def r_in_guard(cfg):
    return cfg["disk_in"]


# ----------------------------------------------------------------------------
# Procedural celestial sphere (equirectangular, [H, W, 3], linear radiance)
# ----------------------------------------------------------------------------
def build_starfield(H=2048, W=4096, n_stars=280_000, seed=7):
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W, 3), np.float32)

    # Milky Way band: a great circle tilted out of the celestial equator
    yy, xx = np.mgrid[0:H, 0:W]
    theta = yy / (H - 1) * np.pi
    phi = xx / W * 2 * np.pi
    # direction vectors
    sx = np.sin(theta) * np.cos(phi)
    sy = np.sin(theta) * np.sin(phi)
    sz = np.cos(theta)
    tilt = np.deg2rad(60.0)
    gal_lat = np.arcsin(np.clip(sz * np.cos(tilt) - sx * np.sin(tilt), -1, 1))
    band = np.exp(-(gal_lat / np.deg2rad(9.0)) ** 2)
    # lumpy dust
    fn = rng.random((H // 16, W // 16)).astype(np.float32)
    fn = np.array(_np_bilinear(fn, H, W))
    fn2 = np.array(_np_bilinear(rng.random((H // 48, W // 48)).astype(np.float32), H, W))
    dust = 0.3 + 1.0 * fn * (0.4 + fn2)
    lanes = (fn < 0.44).astype(np.float32)
    band = band * dust * (1.0 - 0.75 * lanes)
    # keep the band dim: gravitational lensing near the shadow magnifies it a
    # lot, and a bright band there turns into a grey veil over the whole frame.
    glow = np.stack([band * 0.085, band * 0.08, band * 0.10], -1)
    img += glow.astype(np.float32)
    img += 0.00025  # faint airglow floor

    # stars: power-law brightness, random colour temperature
    su = rng.random(n_stars)
    sv = rng.random(n_stars)
    px = (su * W).astype(int) % W
    py = (sv * (H - 1)).astype(int)
    bright = (rng.random(n_stars) ** 8) * 3.2 + 0.02
    Tc = 2500 + rng.random(n_stars) ** 2 * 16000
    c = _bb_np(Tc) * bright[:, None]
    np.add.at(img, (py, px), c.astype(np.float32))
    # a few bright ones bloom over a 3x3
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            w = 0.16 if (dx == 0 or dy == 0) else 0.07
            np.add.at(img, ((py + dy) % H, (px + dx) % W), (c * w).astype(np.float32))
    return img


def _np_bilinear(small, H, W):
    sh, sw = small.shape
    ys = np.linspace(0, sh - 1, H)
    xs = np.linspace(0, sw - 1, W)
    y0 = np.floor(ys).astype(int); y1 = np.minimum(y0 + 1, sh - 1)
    x0 = np.floor(xs).astype(int); x1 = np.minimum(x0 + 1, sw - 1)
    fy = (ys - y0)[:, None]; fx = (xs - x0)[None, :]
    top = small[y0][:, x0] * (1 - fx) + small[y0][:, x1] * fx
    bot = small[y1][:, x0] * (1 - fx) + small[y1][:, x1] * fx
    return top * (1 - fy) + bot * fy


def _bb_np(T):
    t = np.clip(T, 1000, 40000) / 100.0
    tm60 = np.maximum(t - 60.0, 1e-3)
    r = np.where(t <= 66, 255.0, np.clip(329.698727 * tm60 ** -0.1332047592, 0, 255))
    g = np.where(t <= 66,
                 np.clip(99.4708025861 * np.log(np.maximum(t, 1e-3)) - 161.1195681661, 0, 255),
                 np.clip(288.1221695283 * tm60 ** -0.0755148492, 0, 255))
    b = np.where(t >= 66, 255.0,
                 np.where(t <= 19, 0.0,
                          np.clip(138.5177312231 * np.log(np.maximum(t - 10, 1e-3)) - 305.0447927307, 0, 255)))
    return (np.stack([r, g, b], -1) / 255.0) ** 2.2


# ----------------------------------------------------------------------------
# Frame assembly
# ----------------------------------------------------------------------------
def camera_frame_N(px, py, w, h, fov_y, incl):
    """Pixel grid -> incoming-ray unit vector N in the paper's camera frame
    (x = FIDO r-hat, y = FIDO phi-hat, z = FIDO theta-hat).

    The camera sits at colatitude theta_c = pi/2 - incl (a hair above the disc
    plane) and looks straight at the hole along -r-hat; screen right is +phi-hat
    (the orbital-motion direction), screen up is -theta-hat (toward the pole).
    Because the camera is off the plane, no pixel ray is an exactly-equatorial
    geodesic, so the equatorial-plane crossing test alone renders the disc.
    """
    ar = w / h
    ty = np.tan(fov_y / 2)
    sx = (px + 0.5) / w * 2 - 1
    sy = 1 - (py + 0.5) / h * 2
    dx = sx * ty * ar
    dy = sy * ty
    # camera basis in FIDO spherical components (r-hat, theta-hat, phi-hat):
    Lr, Lth, Lph = -1.0, 0.0, 0.0          # look straight at the hole
    Ur, Uth, Uph = 0.0, -1.0, 0.0          # up = toward the pole
    Rr, Rth, Rph = 0.0, 0.0, 1.0          # right = orbital motion
    vr = Lr + dx * Rr + dy * Ur
    vth = Lth + dx * Rth + dy * Uth
    vph = Lph + dx * Rph + dy * Uph
    n = np.sqrt(vr * vr + vth * vth + vph * vph)
    vr, vth, vph = vr / n, vth / n, vph / n
    # N is the incoming ray's direction on the camera's local sky (A.8): the
    # line of sight toward where the ray appears, i.e. the view direction itself.
    # Camera-frame Cartesian axes are (x, y, z) = (r-hat, phi-hat, theta-hat).
    N = np.stack([vr, vph, vth], axis=-1)
    return N.astype(np.float32)


def tonemap(hdr, exposure):
    x = hdr * exposure
    x = x / (1.0 + x)                       # Reinhard
    return np.clip(x ** (1 / 2.2), 0, 1)


def _box_blur(img, r):
    """Fast separable box blur (reflect edges) via cumulative sums."""
    x = img
    for ax in (0, 1):
        n = x.shape[ax]
        pad = [(0, 0)] * x.ndim
        pad[ax] = (r + 1, r)
        xp = np.pad(x, pad, mode="edge")
        cs = np.cumsum(xp, axis=ax)
        lo = np.take(cs, np.arange(0, n), axis=ax)
        hi = np.take(cs, np.arange(2 * r + 1, n + 2 * r + 1), axis=ax)
        x = (hi - lo) / (2 * r + 1)
    return x


def add_veiling_flare(lin, strength):
    """Soft glow characteristic of IMAX lenses (Sec 4.2, 4.3.1)."""
    if strength <= 0:
        return lin
    bright = np.maximum(lin - 0.6, 0.0)
    glow = sum(_box_blur(bright, r) for r in (2, 6, 16, 40)) / 4.0
    return lin + strength * glow


def default_cfg(a, args):
    return dict(
        a=a,
        disk_in=args.disk_in, disk_out=args.disk_out, disk_T=args.disk_temp,
        disk_gain=args.disk_gain, star_gain=args.star_gain,
        shifts={"none": 0, "color": 1, "full": 2}[args.shifts],
        star_shifts=bool(args.star_shifts),
        r_escape=max(400.0, 15.0 * args.radius),
        atol=3e-4, rtol=3e-5, hmin=1e-3, hmax=4.0, max_steps=args.max_steps,
    )


def make_renderer(r_c, theta_c, a, beta, cfg, cam_Omega, cam_ut):
    """Return a jitted fn  (px_N[K,3], star_tex[H,W,3], world_phi) -> rgb[K,3]."""
    def render(px_N, star_tex, world_phi):
        def one(N):
            res = trace_ray(N, r_c, theta_c, a, beta, cfg)
            return shade(res, cfg, star_tex, cam_Omega, cam_ut, world_phi)
        return jax.vmap(one)(px_N)
    return jax.jit(render)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="imax_out")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--frames", type=int, default=96)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--ssaa", type=int, default=2, help="sqrt of samples per pixel")
    ap.add_argument("--spin", type=float, default=0.6, help="a/M  (Interstellar visual value)")
    ap.add_argument("--radius", type=float, default=30.0, help="camera orbit radius r_c / M")
    ap.add_argument("--incl-deg", type=float, default=2.0, help="camera tilt above the disc")
    ap.add_argument("--fov-deg", type=float, default=42.0)
    ap.add_argument("--orbit-frac", type=float, default=1.0, help="fraction of a full orbit")
    ap.add_argument("--disk-in", type=float, default=9.26, help="paper's paint-swatch r_in")
    ap.add_argument("--disk-out", type=float, default=18.70, help="paper's paint-swatch r_out")
    ap.add_argument("--disk-temp", type=float, default=4500.0, help="Sec 4: T = 4500 K")
    ap.add_argument("--disk-gain", type=float, default=1.5)
    ap.add_argument("--star-gain", type=float, default=1.0)
    ap.add_argument("--shifts", choices=["none", "color", "full"], default="full")
    ap.add_argument("--star-shifts", type=int, default=1)
    ap.add_argument("--exposure", type=float, default=1.5)
    ap.add_argument("--flare", type=float, default=0.16, help="veiling-flare strength")
    ap.add_argument("--max-steps", type=int, default=2500)
    ap.add_argument("--chunk", type=int, default=400_000, help="rays per GPU batch")
    ap.add_argument("--dump-frames", action="store_true", help="also write per-frame PNGs")
    ap.add_argument("--still", type=int, default=-1, help="render only this frame index")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    a = args.spin
    r_c = args.radius
    incl = np.deg2rad(args.incl_deg)
    theta_c = np.pi / 2 - incl   # camera just above the disc plane
    fov_y = np.deg2rad(args.fov_deg)

    cfg = default_cfg(a, args)
    omega_orbit, beta, cam_ut = orbit_params(r_c, a)
    beta = float(beta); cam_ut = float(cam_ut)
    print(f"[dngr] a={a}  r_c={r_c}M  Omega={omega_orbit:.5f}  beta={beta:.4f}  "
          f"r_h={1+np.sqrt(1-a*a):.4f}M")

    print("[dngr] building celestial sphere ...")
    star_tex = jnp.asarray(build_starfield())

    # supersampled pixel centres
    W, H, S = args.width, args.height, args.ssaa
    jitter = (np.arange(S) + 0.5) / S - 0.5
    ox, oy = np.meshgrid(jitter, jitter)
    px = np.arange(W); py = np.arange(H)
    PX, PY = np.meshgrid(px, py)

    frame_fn = make_renderer(r_c, theta_c, a, beta, cfg, omega_orbit, cam_ut)

    def render_rays(N_np, world_phi):
        chunk = args.chunk
        outs = []
        for s in range(0, N_np.shape[0], chunk):
            nn = jnp.asarray(N_np[s:s + chunk])
            outs.append(np.asarray(frame_fn(nn, star_tex, world_phi)))
        return np.concatenate(outs, axis=0)

    import imageio.v2 as imageio
    is_video = args.still < 0
    fdir = os.path.join(args.out, "_frames")
    if is_video:
        os.makedirs(fdir, exist_ok=True)

    frames = list(range(args.frames)) if is_video else [args.still]
    for k, fi in enumerate(frames):
        t0 = time.time()
        frac = (fi / max(args.frames, 1)) * args.orbit_frac
        phi_cam = float(2 * np.pi * frac)
        acc = np.zeros((H, W, 3), np.float32)
        for (jx, jy) in zip(ox.ravel(), oy.ravel()):
            N = camera_frame_N(PX + jx, PY + jy, W, H, fov_y, incl).reshape(-1, 3)
            acc += render_rays(N, phi_cam).reshape(H, W, 3)
        acc /= (S * S)
        lin = add_veiling_flare(acc, args.flare)
        out = (tonemap(lin, args.exposure) * 255).astype(np.uint8)
        if is_video:
            imageio.imwrite(os.path.join(fdir, f"f{k:05d}.png"), out)
        if not is_video or args.dump_frames:
            imageio.imwrite(os.path.join(args.out, f"frame_{fi:04d}.png"), out)
        print(f"[dngr] frame {k + 1}/{len(frames)}  {time.time() - t0:5.1f}s", flush=True)

    if is_video:
        import shutil
        import subprocess
        mp4 = os.path.join(args.out, "gargantua.mp4")
        ff = shutil.which("ffmpeg")
        if ff:
            subprocess.run([ff, "-y", "-framerate", str(args.fps), "-i",
                            os.path.join(fdir, "f%05d.png"), "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-crf", "16", "-movflags", "+faststart",
                            mp4], check=True)
        else:
            imageio.mimsave(mp4, [imageio.imread(os.path.join(fdir, f"f{i:05d}.png"))
                                  for i in range(len(frames))], fps=args.fps)
        if not args.dump_frames:
            shutil.rmtree(fdir, ignore_errors=True)
        print("[dngr] wrote", mp4)


if __name__ == "__main__":
    main()
