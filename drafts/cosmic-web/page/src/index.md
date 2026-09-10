---
title: The Cosmic Web
---

# The Cosmic Web

## 13.8 billion years of gravity, condensed into nineteen seconds

<figure class="hero">
  <video src="./cosmic-web.mp4" controls autoplay muted loop playsinline></video>
  <figcaption>
    A ΛCDM particle-mesh N-body simulation: 16.7 million particles in a
    160&nbsp;Mpc/<i>h</i> comoving box, integrated on one GPU from redshift
    <i>z</i>&nbsp;=&nbsp;49 to today. One frame per timestep. The clock in the
    corner is cosmic time; the bar is a fixed comoving ruler.
  </figcaption>
</figure>

The early universe was almost perfectly smooth. The matter it contained was
spread out evenly to about **one part in thirty thousand** — the same faint
ripples we still see imprinted on the cosmic microwave background. Gravity spent
the next 13.8 billion years amplifying them. Anywhere slightly denser than
average pulled in a little extra matter, which made it denser still, which made
it pull harder. Left running long enough, that instability turns a bland grey
fog into the **cosmic web**: a filigree of sheets and filaments strung between
dense knots, with vast near-empty **voids** in between. The knots are where
galaxies and galaxy clusters live. This simulation runs that process forward and
plays it back as a film.

Nothing here is artistic licence. The starting field has the measured
statistical properties of the real primordial fluctuations; the only force is
Newtonian gravity in an expanding space; the cosmological parameters are the
current best-fit values. What you are watching is the standard model of
cosmology doing the one thing it does really well.

---

## What the machine is actually doing

### 1 · An expanding stage

Distances are tracked in **comoving coordinates** $\mathbf{x}$, which factor out
the overall expansion. A physical separation is $\mathbf{r} = a(t)\,\mathbf{x}$,
where the **scale factor** $a$ grows from $a \approx 0$ at the Big Bang to
$a = 1$ today. Its growth obeys the Friedmann equation, which for a flat
universe of matter plus a cosmological constant is

$$
\left(\frac{\dot a}{a}\right)^{2} = H^2(a) = H_0^2\left[\,\Omega_m\,a^{-3} + \Omega_\Lambda\,\right].
$$

This run uses $\Omega_m = 0.311$, $\Omega_\Lambda = 0.689$, $h = 0.677$. Early
on the $a^{-3}$ term dominates and gravity wins; once $\Omega_\Lambda$ takes
over (around $a \approx 0.6$, redshift $z \approx 0.7$) the expansion
accelerates and structure growth stalls. In the film you can see the web stop
sharpening in the last few seconds.

### 2 · The seed field

Inflation predicts a nearly **scale-invariant** Gaussian random field of density
contrast $\delta(\mathbf{x}) \equiv \rho/\bar\rho - 1$. Its power spectrum — the
variance carried by each spatial frequency $k$ — is

$$
P(k) \propto k^{n_s}\,T^2(k), \qquad n_s = 0.965,
$$

where $T(k)$ is the **transfer function**, encoding how radiation pressure in
the first ~50,000 years suppressed growth on small scales (here the
Bardeen–Bond–Kaiser–Szalay fitting form). The field is built by drawing white
noise on the mesh, multiplying its Fourier transform by $\sqrt{P(k)}$, and
normalising so that the variance smoothed on an $8\,\mathrm{Mpc}/h$ sphere is
$\sigma_8 = 0.811$. The check below compares the simulation's power spectrum at
$z = 0$ against theory:

<figure class="framed">
  <img src="./power.png" alt="Matter power spectrum: simulation vs linear theory" />
  <figcaption>
    Large scales (small <i>k</i>) match linear theory exactly. Small scales sit
    <i>above</i> it — the signature of non-linear gravitational collapse, the
    web building itself.
  </figcaption>
</figure>

### 3 · Laying down particles — the Zel'dovich approximation

Matter is represented by $256^3$ particles. Rather than freezing them on a
static grid, each is given the first-order Lagrangian displacement from its grid
point $\mathbf{q}$,

$$
\mathbf{x}(\mathbf{q}, a) = \mathbf{q} + D(a)\,\boldsymbol{\Psi}(\mathbf{q}),
\qquad
\boldsymbol{\Psi}_{\mathbf{k}} = \frac{i\,\mathbf{k}}{k^2}\,\delta_{\mathbf{k}},
$$

with the matching velocity $\dot{\mathbf{x}} = D f H\,\boldsymbol{\Psi}$. Here
$D(a)$ is the **linear growth factor**, the solution of

$$
\ddot D + 2H\dot D - \tfrac{3}{2}\,\Omega_m H_0^2\,a^{-3} D = 0,
$$

normalised to $D(1) = 1$, and $f = \mathrm{d}\ln D / \mathrm{d}\ln a$ is the
growth rate. This "Zel'dovich" push is exact while fluctuations are small, and
it already contains the key insight: matter collapses first into **sheets**
(Zel'dovich pancakes), which is why the mature web looks like crushed paper.

### 4 · Gravity on a mesh, every step

The particles feel their mutual gravity through the comoving Poisson equation

$$
\nabla^2 \Phi = \frac{3}{2}\,\Omega_m\,H_0^2\,\frac{\delta}{a}.
$$

Solving it directly is $O(N^2)$ — hopeless for $N = 1.7\times10^7$. The
**particle-mesh** method does it in Fourier space in $O(N \log N)$: deposit each
particle's mass onto the $512^3$ grid with a cloud-in-cell kernel to get
$\delta(\mathbf{x})$; one forward FFT to $\delta_{\mathbf{k}}$; divide by $-k^2$
(the Laplacian is algebraic in Fourier space) so the acceleration field is

$$
\mathbf{g}_{\mathbf{k}} = -\,i\,\mathbf{k}\,\Phi_{\mathbf{k}}
= \frac{i\,\mathbf{k}}{k^2}\,\Big(\tfrac{3}{2}\,\Omega_m H_0^2\,\delta/a\Big)_{\mathbf{k}};
$$

three inverse FFTs give $\mathbf{g}(\mathbf{x})$; then interpolate it back to
each particle with the same kernel. This is exactly how the first cosmic-web
simulations of the 1980s were run — the modern part is only the scale and the
hardware. The whole force solve is a handful of 512³ FFTs per step on a GPU.

### 5 · Marching through time

The integrator is a **symplectic kick–drift–kick leapfrog** whose time variable
is the scale factor $a$ itself — 570 log-spaced steps from $a = 0.02$ to
$a = 1$. With canonical momentum $\mathbf{p} = a^2 \dot{\mathbf{x}}$, one step
from $a_1$ to $a_2$ (mid-point $a_m$) is

$$
\mathbf{p} \mathrel{+}= \mathbf{g}(\mathbf{x})\!\int_{a_1}^{a_m}\!\frac{\tfrac{3}{2}\Omega_m}{a^2 E(a)}\,\mathrm{d}a,
\quad
\mathbf{x} \mathrel{+}= \mathbf{p}\!\int_{a_1}^{a_2}\!\frac{\mathrm{d}a}{a^3 E(a)},
\quad
\mathbf{p} \mathrel{+}= \mathbf{g}(\mathbf{x})\!\int_{a_m}^{a_2}\!\frac{\tfrac{3}{2}\Omega_m}{a^2 E(a)}\,\mathrm{d}a,
$$

with $E(a) = H(a)/H_0$. Symplectic integrators do not conserve energy exactly,
but they conserve a nearby "shadow" energy, so there is no long-term drift: the
web that forms is stable, not a numerical artefact.

---

## Reading the film

| you see | it is |
| --- | --- |
| the near-uniform violet haze at the start | the smooth early universe; the faint grain is the real primordial fluctuation field |
| a spidery pattern fading in | sheets and filaments, where matter has fallen out of the expansion and begun to collapse |
| bright yellow-white points | collapsed halos — the sites of galaxy groups and clusters |
| growing black patches | voids, expanding faster than average and draining themselves empty |
| the pattern freezing near the end | dark energy taking over; gravity can no longer keep pace with the expansion |

The slow rotation and zoom are the only camera moves; the structure itself is
never touched.

---

## Method, briefly

- **Cosmology.** Flat ΛCDM: $\Omega_m = 0.311$, $\Omega_\Lambda = 0.689$,
  $h = 0.677$, $n_s = 0.965$, $\sigma_8 = 0.811$. Radiation neglected (fine for
  $a \gtrsim 0.02$).
- **Simulation.** $256^3$ particles, $512^3$ PM force mesh, $160\,\mathrm{Mpc}/h$
  box, cloud-in-cell assignment with window deconvolution, 570 KDK leapfrog
  steps in $\log a$. JAX on one A100-80GB GPU via Modal, ~15–20 minutes.
- **Initial conditions.** Zel'dovich approximation from a BBKS ΛCDM power
  spectrum, $\sigma_8$-normalised.
- **Render.** Additive projection of a soft $\sim 50\,\mathrm{Mpc}/h$ slab,
  depth-shaded, fixed log-density colour ramp, 20° rotation + zoom, 1280×720 at
  30 fps.

Full source and a one-command reproduction are in
[`drafts/cosmic-web/`](https://github.com/funnyscar/graphics/tree/main/drafts/cosmic-web)
of the graphics repo.

<style>
:root { --serif: "Source Serif 4", Georgia, serif; }
figure.hero { max-width: 960px; margin: 1.5rem 0; }
figure.hero video { width: 100%; border-radius: 8px; background: #000; display: block; }
figure.framed { max-width: 720px; margin: 1.5rem 0; }
figure.framed img { width: 100%; border-radius: 6px; }
figcaption { font-size: 14px; color: var(--theme-foreground-muted); line-height: 1.45; margin-top: 0.5rem; }
table { font-size: 15px; }
.katex { font-size: 1.02em; }
h1 { margin-bottom: 0.2rem; }
h1 + h2 { color: var(--theme-foreground-muted); font-weight: 400; font-style: italic; }
</style>
