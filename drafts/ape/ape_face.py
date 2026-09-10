"""
Procedural "Planet of the Apes"-style ape head, built and rendered in Blender.

Runs two ways:
  - inside Blender:      blender -b -P ape_face.py
  - via the bpy module:  python ape_face.py       (pip install bpy)
  - on a GPU in Modal:   modal run modal_render.py

Output:
  ape_face.blend  - the assembled scene
  ape_face.png    - a Cycles render (3/4 portrait)

Env knobs (all optional):
  APE_OUT_DIR   directory for the .blend / .png            (default: script dir)
  APE_SAMPLES   Cycles samples                             (default: 200)
  APE_RES       square-ish render, long edge in px         (default: 1400)
  APE_GPU       "1" to try to enable GPU compute in Cycles (default: auto)
  APE_HAIR      "1"/"0" toggle fur particle systems        (default: 1)
"""

import math
import os
import sys

import bpy
import mathutils

# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
OUT_DIR = os.environ.get("APE_OUT_DIR") or os.path.dirname(os.path.abspath(__file__))
SAMPLES = int(os.environ.get("APE_SAMPLES", "200"))
RES = int(os.environ.get("APE_RES", "1400"))
WANT_GPU = os.environ.get("APE_GPU", "auto")
WANT_HAIR = os.environ.get("APE_HAIR", "1") != "0"

BLEND_PATH = os.path.join(OUT_DIR, "ape_face.blend")
PNG_PATH = os.path.join(OUT_DIR, "ape_face.png")


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def add_meta_ellipsoid(mball_obj, co, radii, stiffness=2.0, negative=False):
    """Add one ellipsoid element to a metaball object. `radii` is (rx, ry, rz)."""
    el = mball_obj.data.elements.new(type="ELLIPSOID")
    el.co = mathutils.Vector(co)
    el.size_x, el.size_y, el.size_z = radii
    el.radius = stiffness
    el.use_negative = negative
    return el


def add_meta_ball(mball_obj, co, radius, stiffness=2.0, negative=False):
    el = mball_obj.data.elements.new(type="BALL")
    el.co = mathutils.Vector(co)
    el.radius = radius
    el.stiffness = stiffness
    el.use_negative = negative
    return el


def new_object(name, data):
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    return obj


# --------------------------------------------------------------------------- #
# head volume  (a subdivided sphere, sculpted by closed-form displacement)
# --------------------------------------------------------------------------- #
# Coordinate convention:
#   +Y  = forward (the face looks down +Y)
#   +Z  = up
#   +X  = the ape's left
#
# Proportions are pushed toward a male chimpanzee / "Caesar" read:
#   low flat cranium, huge supraorbital brow, projecting (prognathic) muzzle,
#   flared zygomatic cheeks, heavy but receding jaw, small low-set ears.

def _g(d, s):
    """Unit gaussian falloff."""
    return math.exp(-(d * d) / (s * s))


def _smoothstep(a, b, t):
    t = max(0.0, min(1.0, (t - a) / (b - a)))
    return t * t * (3.0 - 2.0 * t)


# vertical (z) offsets that define the facial profile, before x-falloff.
# +value pushes that band of the face forward (+Y).
def _band(lo, hi, edge, t):
    """A smooth plateau: ~1 for lo<t<hi, falling to 0 over `edge` on each side."""
    return _smoothstep(lo - edge, lo, t) * (1.0 - _smoothstep(hi, hi + edge, t))


def _sculpt_vertex(x, y, z):
    """Map a point on the unit sphere to a point on the ape head.

    All facial features are placed against stable reference coords (zc, xc)
    taken from the base sphere, so later offsets don't drift the landmarks.
    """
    # ---- 1. base cranial mass: small rounded braincase, a touch wide ----
    px = x * 0.96
    py = y * 0.98
    pz = z * 0.95

    # flatten the crown a little and slope the skull back off the brow
    if pz > 0.34:
        over = pz - 0.34
        pz = 0.34 + over * 0.74
        py -= over * 0.42
    if py < -0.1:                               # tuck the back of the head in
        py += _g(py + 0.1, 0.55) * 0.10 * max(0.0, pz)

    zc, xc = z * 0.95, x * 0.96                 # stable landmark references
    front = _smoothstep(-0.15, 0.35, y)         # 0 at back, 1 over the face
    midline = _g(xc, 0.46)
    face = front * _g(xc, 0.66)                  # facial-plane mask
    cheek_x = _g(xc, 0.5)

    # ---- 2. brow ridge: a rounded bar that juts forward over the eyes ----
    brow = _g(zc - 0.28, 0.12) * _g(xc, 0.56) * front
    brow *= 1.0 - 0.35 * _g(xc, 0.07)           # slight glabella dip on the midline
    py += 0.22 * brow
    pz -= 0.05 * brow                            # roll its lower edge down over the eyes

    # ---- 3. muzzle: a blunt snout projecting forward and drooping down ----
    muz = _band(-0.44, 0.02, 0.15, zc) * face
    py += 0.40 * muz
    pz -= 0.12 * muz                             # the snout angles downward
    px *= 1.0 - 0.15 * muz * (1.0 - cheek_x)     # keep its sides from ballooning

    # ---- 4. eye recess + sockets under the brow ----
    # the brow + muzzle already leave a natural hollow here; only a light touch.
    py -= 0.05 * _g(zc - 0.03, 0.12) * face
    for ex in (-0.29, 0.29):
        # a wide almond opening: elliptical, shallower than before
        d = math.sqrt(((xc - ex) / 1.5) ** 2 + ((zc - 0.01) / 0.85) ** 2)
        ring = _g(d, 0.15) - 0.8 * _g(d, 0.10)         # rim with a raised centre
        py -= 0.07 * max(0.0, ring) * front
    py += 0.05 * _g(xc, 0.09) * _g(zc - 0.08, 0.18) * front   # inter-orbital bridge

    # ---- 5. nose: bridge -> flat pad low on the snout -> nostrils underneath ----
    py += 0.03 * _g(xc, 0.075) * _g(zc - 0.02, 0.13) * front          # bony bridge
    py += 0.05 * _g(xc, 0.17) * _g(zc + 0.08, 0.08) * front           # nose pad
    for nx in (-0.07, 0.07):                                          # nostrils, opening down
        n = _g(xc - nx, 0.038) * _g(zc + 0.17, 0.045) * front
        py -= 0.035 * n
        pz -= 0.045 * n

    # ---- 6. mouth: a wide closed line with a faint upper/lower lip roll ----
    py -= 0.03 * _g(zc + 0.30, 0.02) * _g(xc, 0.32) * front           # lip seam
    py += 0.025 * _g(zc + 0.24, 0.035) * _g(xc, 0.30) * front         # upper lip
    py += 0.02 * _g(zc + 0.37, 0.035) * _g(xc, 0.28) * front          # lower lip

    # ---- 7. receding chin ----
    py -= 0.16 * _band(-0.72, -0.46, 0.12, zc) * _smoothstep(0.0, 0.5, y) * cheek_x

    # ---- 8. zygomatic (cheek) flare ----
    for cx in (-0.58, 0.58):
        d = math.sqrt((xc - cx) ** 2 + (y - 0.30) ** 2 * 0.7 + (zc + 0.05) ** 2)
        px += (0.09 if cx > 0 else -0.09) * _g(d, 0.32)

    # ---- 9. jaw: a little extra width at the angle ----
    px *= 1.0 + 0.11 * _g(zc + 0.46, 0.26) * _g(y - 0.05, 0.7)
    if zc < -0.5:                                # soft throat / underjaw
        py += _g(zc + 0.8, 0.26) * 0.13 * midline

    return mathutils.Vector((px, py, pz))


def build_head():
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=160, ring_count=120)
    head = bpy.context.active_object
    head.name = "ApeHead"

    me = head.data
    for v in me.vertices:
        d = v.co.normalized()
        v.co = _sculpt_vertex(d.x, d.y, d.z)

    bpy.context.view_layer.objects.active = head
    head.select_set(True)

    # a light voxel remesh -> clean even topology for fur + subsurf
    m = head.modifiers.new("remesh", "REMESH")
    m.mode = "VOXEL"
    m.voxel_size = 0.011
    m.use_smooth_shade = True
    bpy.ops.object.modifier_apply(modifier=m.name)

    # gentle relax only - keep the sculpted features
    m = head.modifiers.new("smooth", "SMOOTH")
    m.iterations = 2
    m.factor = 0.3
    bpy.ops.object.modifier_apply(modifier=m.name)

    sub = head.modifiers.new("subsurf", "SUBSURF")
    sub.levels = 1
    sub.render_levels = 3

    # fine skin break-up (subtle, high frequency)
    tex = bpy.data.textures.new("skinBump", type="CLOUDS")
    tex.noise_scale = 0.06
    tex.noise_depth = 3
    disp = head.modifiers.new("skin_disp", "DISPLACE")
    disp.texture = tex
    disp.texture_coords = "LOCAL"
    disp.strength = 0.01
    disp.mid_level = 0.5

    bpy.ops.object.shade_smooth()
    return head


# --------------------------------------------------------------------------- #
# eyes
# --------------------------------------------------------------------------- #
def add_eyes():
    eyes = []
    for side, x in (("L", -0.30), ("R", 0.30)):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.205, segments=48, ring_count=32)
        eye = bpy.context.active_object
        eye.name = f"Eye_{side}"
        eye.location = (x - (0.02 if x > 0 else -0.02), 0.83, 0.0)
        eye.rotation_euler = (math.radians(14), 0.0, math.radians(3 if x > 0 else -3))
        bpy.ops.object.shade_smooth()
        eyes.append(eye)
    return eyes


def add_ears():
    ears = []
    for side, sgn in (("L", -1.0), ("R", 1.0)):
        bpy.ops.mesh.primitive_ico_sphere_add(radius=1.0, subdivisions=4)
        ear = bpy.context.active_object
        ear.name = f"Ear_{side}"
        me = ear.data
        for v in me.vertices:
            v.co.x *= 0.06                        # thin flap
            v.co.z *= 0.20
            v.co.y *= 0.17
            outer = v.co.x * sgn
            rim = _g(math.hypot(v.co.z, v.co.y) - 0.16, 0.05)   # ring near the edge
            if outer > 0.0:                       # front face: deep concha bowl, keep the rim
                v.co.x -= sgn * 0.055 * _g(math.hypot(v.co.z, v.co.y), 0.11) * (1.0 - rim)
            v.co.x += sgn * 0.02 * rim            # roll the rim forward
        ear.location = (sgn * 0.82, -0.02, 0.10)
        ear.rotation_euler = (math.radians(6), 0.0, math.radians(-16 * sgn))
        m = ear.modifiers.new("sub", "SUBSURF")
        m.levels = 1
        m.render_levels = 2
        bpy.ops.object.shade_smooth()
        ears.append(ear)
    return ears


# --------------------------------------------------------------------------- #
# fur
# --------------------------------------------------------------------------- #
def add_fur(head):
    if not WANT_HAIR:
        return

    # vertex group: where fur is long / present.  scalp + surround, not the face.
    vg = head.vertex_groups.new(name="pelt")
    mesh = head.data
    for v in mesh.vertices:
        co = v.co
        # long on crown & back, fade out across the muzzle and around the eyes
        crown = max(0.0, (co.z - 0.1)) * 1.1
        back = max(0.0, (-co.y - 0.1)) * 0.8
        muzzle = max(0.0, (co.y - 0.35)) * 1.4
        eyes = 0.0
        for ex in (-0.34, 0.34):
            d = math.dist((co.x, co.y, co.z), (ex, 0.66, 0.14))
            if d < 0.28:
                eyes = max(eyes, (0.28 - d) * 3.0)
        w = max(0.0, min(1.0, 0.35 + crown + back - muzzle - eyes))
        vg.add([v.index], w, "REPLACE")

    def make_psys(name, count, length, root, tip, seed, kids):
        head.modifiers.new(name, "PARTICLE_SYSTEM")
        ps = head.particle_systems[-1]
        s = ps.settings

        def st(obj, attr, val):
            try:
                setattr(obj, attr, val)
            except (AttributeError, TypeError) as e:
                print(f"[ape] fur: skipped {attr} ({e})")

        st(ps, "seed", seed)
        st(s, "type", "HAIR")
        st(s, "count", count)
        st(s, "hair_length", length)
        st(s, "hair_step", 5)
        st(s, "use_advanced_hair", True)
        st(s, "root_radius", root)
        st(s, "tip_radius", tip)
        st(s, "radius_scale", 0.004)
        st(s, "brownian_factor", 0.15)
        st(s, "child_type", "INTERPOLATED")
        st(s, "child_percent", kids)          # bpy 4.x name (was child_nbr)
        st(s, "child_nbr", kids)
        st(s, "rendered_child_count", kids)
        st(s, "child_length", 0.9)
        st(s, "clump_factor", 0.72)
        st(s, "roughness_1", 0.18)
        st(s, "roughness_2", 0.12)
        st(s, "roughness_endpoint", 0.10)
        st(s, "kink", "CURL")
        st(s, "kink_amplitude", 0.006)
        st(s, "kink_frequency", 3.0)
        st(ps, "vertex_group_density", "pelt")
        st(ps, "vertex_group_length", "pelt")
        st(s.effector_weights, "gravity", 0.4)
        return ps

    make_psys("fur_under", 2600, 0.12, 0.9, 0.12, 1, kids=60)
    make_psys("fur_over", 1500, 0.16, 1.0, 0.05, 7, kids=45)


# --------------------------------------------------------------------------- #
# materials
# --------------------------------------------------------------------------- #
def _principled(mat):
    return mat.node_tree.nodes["Principled BSDF"]


def skin_material():
    mat = bpy.data.materials.new("ApeSkin")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = _principled(mat)

    bsdf.inputs["Base Color"].default_value = (0.055, 0.042, 0.038, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.62
    for key, val in (("Subsurface Weight", 0.12), ("Subsurface", 0.12)):
        if key in bsdf.inputs:
            bsdf.inputs[key].default_value = val
    if "Subsurface Radius" in bsdf.inputs:
        bsdf.inputs["Subsurface Radius"].default_value = (0.12, 0.045, 0.028)
    if "Subsurface Color" in bsdf.inputs:
        bsdf.inputs["Subsurface Color"].default_value = (0.28, 0.06, 0.045, 1.0)

    # procedural pore / wrinkle bump
    tc = nt.nodes.new("ShaderNodeTexCoord")
    n1 = nt.nodes.new("ShaderNodeTexNoise")
    n1.inputs["Scale"].default_value = 34.0
    n1.inputs["Detail"].default_value = 8.0
    n2 = nt.nodes.new("ShaderNodeTexNoise")
    n2.inputs["Scale"].default_value = 6.0
    n2.inputs["Detail"].default_value = 4.0
    mix = nt.nodes.new("ShaderNodeMixRGB")
    mix.inputs["Fac"].default_value = 0.4
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.5
    bump.inputs["Distance"].default_value = 0.02

    nt.links.new(tc.outputs["Object"], n1.inputs["Vector"])
    nt.links.new(tc.outputs["Object"], n2.inputs["Vector"])
    nt.links.new(n1.outputs["Fac"], mix.inputs["Color1"])
    nt.links.new(n2.outputs["Fac"], mix.inputs["Color2"])
    nt.links.new(mix.outputs["Color"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    # base colour: warm mottled grey-brown, darker in the creases
    cr = nt.nodes.new("ShaderNodeValToRGB")
    cr.color_ramp.elements[0].color = (0.020, 0.014, 0.012, 1.0)
    cr.color_ramp.elements[1].color = (0.095, 0.064, 0.05, 1.0)
    mid = cr.color_ramp.elements.new(0.5)
    mid.color = (0.05, 0.033, 0.027, 1.0)
    nt.links.new(n2.outputs["Fac"], cr.inputs["Fac"])

    # darken into the eye sockets and along the muzzle ridge (cavity-ish)
    dark = nt.nodes.new("ShaderNodeMixRGB")
    dark.blend_type = "MULTIPLY"
    dark.inputs["Fac"].default_value = 0.55
    ao = nt.nodes.new("ShaderNodeAmbientOcclusion")
    ao.inputs["Distance"].default_value = 0.25
    nt.links.new(cr.outputs["Color"], dark.inputs["Color1"])
    nt.links.new(ao.outputs["Color"], dark.inputs["Color2"])
    nt.links.new(dark.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def eye_material():
    mat = bpy.data.materials.new("ApeEye")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = _principled(mat)
    bsdf.inputs["Roughness"].default_value = 0.34
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = 0.15
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.4

    # measure distance from the *front pole* of the eyeball (local +Y extreme),
    # so the iris sits on the face-forward cap, not around the equator.
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mapp = nt.nodes.new("ShaderNodeMapping")
    mapp.inputs["Location"].default_value = (-0.5, -1.0, -0.5)
    mapp.inputs["Scale"].default_value = (2.4, 2.4, 2.4)
    grad = nt.nodes.new("ShaderNodeTexGradient")
    grad.gradient_type = "SPHERICAL"
    nt.links.new(tc.outputs["Generated"], mapp.inputs["Vector"])
    nt.links.new(mapp.outputs["Vector"], grad.inputs["Vector"])

    ramp = nt.nodes.new("ShaderNodeValToRGB")
    cr = ramp.color_ramp
    stops = [
        (0.00, (0.010, 0.006, 0.004)),   # pupil
        (0.13, (0.010, 0.006, 0.004)),   # pupil edge
        (0.17, (0.34, 0.19, 0.06)),      # inner iris
        (0.42, (0.66, 0.42, 0.15)),      # amber iris
        (0.58, (0.40, 0.24, 0.09)),      # outer iris
        (0.63, (0.16, 0.09, 0.04)),      # limbal ring
        (0.69, (0.40, 0.32, 0.24)),      # sclera (warm, dim - not white)
        (1.00, (0.46, 0.37, 0.28)),
    ]
    cr.elements[0].position, cr.elements[0].color = stops[0][0], (*stops[0][1], 1.0)
    cr.elements[1].position, cr.elements[1].color = stops[1][0], (*stops[1][1], 1.0)
    for pos, col in stops[2:]:
        cr.elements.new(pos).color = (*col, 1.0)
    nt.links.new(grad.outputs["Color"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    # roughness: wet and sharp over the cornea, drier over the sclera
    rr = nt.nodes.new("ShaderNodeValToRGB")
    rr.color_ramp.elements[0].position = 0.55
    rr.color_ramp.elements[0].color = (0.04, 0.04, 0.04, 1.0)
    rr.color_ramp.elements[1].position = 0.72
    rr.color_ramp.elements[1].color = (0.42, 0.42, 0.42, 1.0)
    nt.links.new(grad.outputs["Color"], rr.inputs["Fac"])
    nt.links.new(rr.outputs["Color"], bsdf.inputs["Roughness"])
    return mat


def hair_material():
    mat = bpy.data.materials.new("ApeFur")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        if n.type != "OUTPUT_MATERIAL":
            nt.nodes.remove(n)
    out = nt.nodes["Material Output"]
    try:
        h = nt.nodes.new("ShaderNodeBsdfHairPrincipled")
        if "Color" in h.inputs:
            h.inputs["Color"].default_value = (0.035, 0.022, 0.016, 1.0)
        if "Roughness" in h.inputs:
            h.inputs["Roughness"].default_value = 0.42
        if "Radial Roughness" in h.inputs:
            h.inputs["Radial Roughness"].default_value = 0.5
    except RuntimeError:
        h = nt.nodes.new("ShaderNodeBsdfHair")
    nt.links.new(h.outputs[0], out.inputs["Surface"])
    return mat


def assign(obj, mat):
    obj.data.materials.clear()
    obj.data.materials.append(mat)


# --------------------------------------------------------------------------- #
# scene: camera, lights, world
# --------------------------------------------------------------------------- #
def setup_world():
    world = bpy.data.worlds.new("ApeWorld")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.018, 0.02, 0.026, 1.0)
    bg.inputs["Strength"].default_value = 0.35


def setup_camera():
    cam_data = bpy.data.cameras.new("Cam")
    cam_data.lens = 58
    cam_data.dof.use_dof = True
    cam = new_object("Camera", cam_data)
    cam.location = mathutils.Vector((2.7, 6.6, 0.7))

    target_loc = mathutils.Vector((0.0, 0.5, 0.02))
    target = new_object("CamTarget", None)
    target.location = target_loc

    direction = target_loc - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    cam_data.dof.focus_object = target
    cam_data.dof.aperture_fstop = 7.0
    bpy.context.scene.camera = cam
    return cam


def setup_lights():
    def area(name, loc, energy, size, color=(1, 1, 1)):
        d = bpy.data.lights.new(name, "AREA")
        d.energy = energy
        d.size = size
        d.color = color
        o = new_object(name, d)
        o.location = loc
        con = o.constraints.new("TRACK_TO")
        tgt = new_object(name + "_t", None)
        tgt.location = (0, 0.6, 0.0)
        con.target = tgt
        con.track_axis = "TRACK_NEGATIVE_Z"
        return o

    area("Key", (3.2, 3.6, 2.3), 520, 1.5, color=(1.0, 0.95, 0.88))
    area("Fill", (-3.6, 3.2, 0.8), 240, 3.4, color=(0.74, 0.82, 1.0))
    area("Front", (0.5, 5.6, 1.1), 260, 0.6, color=(0.98, 0.96, 0.95))
    area("Rim", (-1.6, -2.8, 2.6), 460, 1.3, color=(0.8, 0.88, 1.0))
    area("Kick", (2.6, -1.2, -1.6), 120, 1.5, color=(1.0, 0.78, 0.55))
    area("Glint", (1.2, 5.6, 1.5), 260, 0.28, color=(1.0, 0.98, 0.95))   # eye catchlight


# --------------------------------------------------------------------------- #
# render settings
# --------------------------------------------------------------------------- #
def enable_gpu():
    prefs = bpy.context.preferences.addons["cycles"].preferences
    chosen = None
    for backend in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
        try:
            prefs.compute_device_type = backend
        except TypeError:
            continue
        prefs.get_devices() if hasattr(prefs, "get_devices") else None
        try:
            devs = prefs.devices
        except Exception:
            devs = []
        gpus = [d for d in devs if d.type == backend]
        if gpus:
            chosen = backend
            for d in devs:
                d.use = (d.type == backend) or (d.type == "CPU" and False)
            break
    if chosen:
        bpy.context.scene.cycles.device = "GPU"
        print(f"[ape] Cycles GPU backend: {chosen} ({len(gpus)} device(s))")
    else:
        print("[ape] no GPU found - rendering on CPU")
    return chosen


def setup_render():
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = SAMPLES
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 12
    scene.render.resolution_x = int(RES * 0.82)
    scene.render.resolution_y = RES
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "AgX" if _has_agx() else "Filmic"
    scene.view_settings.look = "AgX - Medium High Contrast" if _has_agx() else "Medium High Contrast"
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = PNG_PATH

    if WANT_GPU != "0":
        enable_gpu()
    else:
        print("[ape] GPU disabled by APE_GPU=0")


def _has_agx():
    try:
        return "AgX" in bpy.context.scene.view_settings.bl_rna.properties["view_transform"].enum_items
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    print(f"[ape] out dir: {OUT_DIR}  samples={SAMPLES}  res={RES}  hair={WANT_HAIR}")
    reset_scene()

    head = build_head()
    bpy.context.view_layer.update()

    eyes = add_eyes()
    ears = add_ears()

    skin = skin_material()
    eye_mat = eye_material()
    fur_mat = hair_material()

    assign(head, skin)
    for e in ears:
        assign(e, skin)
    for e in eyes:
        assign(e, eye_mat)

    add_fur(head)
    if head.data.materials and WANT_HAIR:
        head.data.materials.append(fur_mat)
        for ps in head.particle_systems:
            ps.settings.material = len(head.data.materials)

    setup_world()
    setup_camera()
    setup_lights()
    setup_render()

    bpy.ops.wm.save_as_mainfile(filepath=BLEND_PATH)
    print(f"[ape] saved {BLEND_PATH}")

    bpy.ops.render.render(write_still=True)
    print(f"[ape] wrote {PNG_PATH}")

    if os.environ.get("APE_DEBUG"):
        scene = bpy.context.scene
        cam = scene.camera
        tgt = mathutils.Vector((0.0, 0.4, -0.1))
        views = {
            "front": mathutils.Vector((0.0, 6.2, 0.15)),
            "q34": mathutils.Vector((3.2, 5.2, 0.6)),
            "side": mathutils.Vector((6.2, 0.4, 0.1)),
            "low": mathutils.Vector((0.6, 5.6, -1.6)),
        }
        for name, loc in views.items():
            cam.location = loc
            cam.rotation_euler = (tgt - loc).to_track_quat("-Z", "Y").to_euler()
            scene.render.filepath = os.path.join(OUT_DIR, f"ape_debug_{name}.png")
            bpy.ops.render.render(write_still=True)
            print(f"[ape] wrote ape_debug_{name}.png")


if __name__ == "__main__":
    main()
