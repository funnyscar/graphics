# cosmic-web

**13.8 billion years of gravity in ~19 seconds.**

A particle-mesh (PM) N-body simulation of a ΛCDM universe, integrated on one
Modal GPU in JAX. We start just after the Big Bang with a near-uniform sea of
matter (density contrast ~1 part in 30,000) and let gravity run. The movie is
one rendered frame per timestep, so it is a continuous fast-forward from
redshift *z* = 49 to today, *z* = 0. The featureless early universe collapses
into the **cosmic web**: pancakes drain into filaments, filaments into knots,
and the knots become the clusters of galaxies. The dark gaps are the voids.

The physics is validated by the matter power spectrum `power.png`: on large
scales the simulation tracks linear theory exactly; on small scales it shows the
non-linear boost from gravitational collapse.

## what it computes

| stage | what |
| --- | --- |
| **initial conditions** | a Gaussian random field with the real primordial power spectrum — scale-invariant inflationary spectrum `P(k) ∝ k^nₛ` processed by the BBKS transfer function, normalised to `σ₈ = 0.811`. Particles are placed on a 256³ grid and given their **Zel'dovich** displacements and velocities. |
| **gravity solve (every step)** | deposit mass on a 512³ mesh (cloud-in-cell), FFT, multiply by the Green's function `i k / k²`, FFT back to get the acceleration field, interpolate back to particles. O(N log N) instead of O(N²). |
| **time integration** | symplectic kick–drift–kick leapfrog whose time variable is the scale factor `a`; 570 log-spaced steps from `a = 0.02` to `a = 1`. Drift/kick factors are the exact ΛCDM integrals `∫ da / (a³ E)` and `∫ 3Ωₘ da / (2 a² E)`. |
| **render** | project a soft slab of the box to 720p, additive splat, depth shading, a fixed log-density colour ramp; a slow 20° rotation + zoom. Redshift, cosmic time and a comoving scale bar are burned in. |

Box: 160 Mpc/h comoving. Cosmology: Ωₘ = 0.311, Ω_Λ = 0.689, h = 0.677,
nₛ = 0.965, σ₈ = 0.811.

## recreate it

Needs a [Modal](https://modal.com) account (`pip install modal && modal setup`).

```bash
# full run — A100-80GB, ~15-20 min, ~16.7M particles on a 512³ force mesh
modal run drafts/cosmic-web/cosmicweb.py --npart 256 --mesh 512 --steps 570 --gpu "A100-80GB"

# quick look — A10G, ~1-2 min
modal run drafts/cosmic-web/cosmicweb.py --npart 96 --mesh 192 --steps 90 --gpu A10G
```

`npart` must divide `mesh` evenly. Outputs land in the Modal Volume `cosmic-web`
and are downloaded to `drafts/cosmic-web/out/`:

| file | |
| --- | --- |
| `cosmic-web.mp4` | the fast-forward, 1280×720, 30 fps |
| `stills/*.png` | z = 49 / 3 / 1.7 / 0.7 / 0 key epochs |
| `power.png` | measured vs linear matter power spectrum (sanity check) |

## notes

- Radiation is neglected — fine from `a = 0.02` (z = 49) onward for a visual.
- The force is PM-only, so structure is resolved down to ~1 mesh cell
  (0.31 Mpc/h); individual dark-matter halos are blobs, not resolved sub-structure.
- A small (<1 cell) Lagrangian jitter is added to the initial grid so it does
  not beat against the pixel raster in the first frames; it is dynamically
  negligible.
- `viewer/index.html` is a minimal local player for `out/`.

## build the writeup

`index.html` (the Observable-Framework synopsis) is built from `page/`:

```bash
cd drafts/cosmic-web/page && npm install && npm run build
# → dist/ ; deployed copy lives in the repo-root dist/<slug>/
```
