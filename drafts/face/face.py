"""
Procedural human face / head, built and rendered entirely in Blender.

Runs three ways:
  - inside Blender:      blender -b -P face.py
  - via the bpy module:  python face.py            (pip install bpy, needs 3.11)
  - on a GPU in Modal:   modal run modal_render.py

Everything is generated from code: an icosphere is pushed into the shape of a
head by closed-form displacement (`_sculpt_vertex`), voxel-remeshed for even
topology, subsurfed, then dressed with a procedural skin shader, a separate
sclera + brown-iris eye, and (optionally) short scalp hair.  Separate meshes
handle the eyeballs, irises, ears and neck.

Output:
  face.blend  - the assembled scene
  face.png    - a Cycles 3/4 portrait render

Env knobs (all optional):
  FACE_OUT_DIR   directory for the .blend / .png           (default: script dir)
  FACE_SAMPLES   Cycles samples                             (default: 200)
  FACE_RES       render, long edge in px                    (default: 1400)
  FACE_GPU       "1"/"0" force GPU compute in Cycles        (default: auto)
  FACE_HAIR      "1"/"0" toggle scalp hair                  (default: 1)
  FACE_DEBUG     "1" -> also render front / 3-4 / side / low angle checks
"""

import math
import os

import bpy
import mathutils

# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
OUT_DIR = os.environ.get("FACE_OUT_DIR") or os.path.dirname(os.path.abspath(__file__))
SAMPLES = int(os.environ.get("FACE_SAMPLES", "200"))
RES = int(os.environ.get("FACE_RES", "1400"))
WANT_GPU = os.environ.get("FACE_GPU", "auto")
WANT_HAIR = os.environ.get("FACE_HAIR", "1") != "0"

BLEND_PATH = os.path.join(OUT_DIR, "face.blend")
PNG_PATH = os.path.join(OUT_DIR, "face.png")


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def new_object(name, data):
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    return obj


def _g(d, s):
    """Unit gaussian falloff."""
    return math.exp(-(d * d) / (s * s))


def _smoothstep(a, b, t):
    t = max(0.0, min(1.0, (t - a) / (b - a)))
    return t * t * (3.0 - 2.0 * t)


def _band(lo, hi, edge, t):
    """A smooth plateau: ~1 for lo<t<hi, falling to 0 over `edge` on each side."""
    return _smoothstep(lo - edge, lo, t) * (1.0 - _smoothstep(hi, hi + edge, t))


# --------------------------------------------------------------------------- #
# head volume  (an icosphere, sculpted by closed-form displacement)
# --------------------------------------------------------------------------- #
# Coordinate convention:
#   +Y  = forward (the face looks down +Y)
#   +Z  = up
#   +X  = the subject's left
#
# Proportions aim at a fairly neutral adult human head: rounded braincase,
# a gentle brow, projecting nose, defined lips and a broad chin, soft
# cheekbones, a moderately tapering jaw.

def _sculpt_vertex(x, y, z):
    """Map a point on the unit sphere to a point on the head."""
    # ---- 1. base head mass: narrower than a sphere, a little taller ----
    px = x * 0.80
    py = y * 0.95
    pz = z * 1.05

    zc, xc = z, x                               # stable landmark references
    front = _smoothstep(-0.10, 0.42, y)         # 0 at back, 1 over the face
    midline = _g(xc, 0.45)

    lower = 1.0 - _smoothstep(-0.85, -0.02, zc)
    px *= 1.0 - 0.13 * lower                    # jaw narrower than the cheeks
    crown = _smoothstep(0.15, 0.70, zc)
    px *= 1.0 - 0.10 * crown                    # taper the braincase
    py *= 1.0 - 0.05 * crown
    if pz > 0.30:                               # flatten the top of the skull
        pz = 0.30 + (pz - 0.30) * 0.80
    if abs(xc) > 0.5:                           # flatten the temples / sides
        px *= 1.0 - 0.06 * _smoothstep(0.5, 0.95, abs(xc)) * _g(y, 0.6)
    py += 0.05 * _g(y + 0.55, 0.42) * _smoothstep(-0.6, 0.2, zc)   # round the occiput out

    # forehead: set well back from the brow, flattened
    fore = _smoothstep(0.06, 0.52, zc) * front * _g(xc, 0.72)
    py -= 0.10 * fore

    # ---- 2. brow ridge ----
    brow = _g(zc - 0.16, 0.10) * _g(xc, 0.50) * front
    brow *= 1.0 - 0.20 * _g(xc, 0.12)
    py += 0.045 * brow

    # ---- 3. eyes: a broad orbit recess, one soft lid mound above each ----
    for ex in (-0.245, 0.245):
        d = math.sqrt(((xc - ex) / 1.45) ** 2 + ((zc + 0.01) / 0.80) ** 2)
        py -= 0.050 * _g(d, 0.19) * front                                  # orbit recess (broad)
        ud = math.hypot((xc - ex) / 1.30, (zc - 0.085) / 0.60)
        py += 0.032 * _g(ud, 0.085) * front                                # upper lid mound
        py += 0.014 * _g(math.hypot((xc - ex) / 1.1, (zc + 0.13) / 0.6), 0.06) * front  # lower lid
    py += 0.014 * _g(xc, 0.11) * _g(zc - 0.02, 0.13) * front               # nasion (broad, soft)

    # ---- 4. nose: a broad dorsum + a tip ball + nostril divots ----
    dorsum = _g(xc, 0.13) * _band(-0.20, 0.12, 0.16, zc) * front
    py += 0.105 * dorsum
    tip = _g(math.hypot(xc / 1.25, (zc + 0.15) / 1.0), 0.11) * front
    py += 0.052 * tip
    px *= 1.0 + 0.05 * tip                                                 # a little tip width
    for nx in (-0.072, 0.072):                                            # nostril divots + wings
        n = _g(xc - nx, 0.052) * _g(zc + 0.165, 0.06) * front
        pz -= 0.018 * n
        px += (0.020 if nx > 0 else -0.020) * n

    # ---- 5. lips: closed mouth - vermilion swell over a pressed seam ----
    lipw = _g(xc, 0.185) * front
    lip_mound = _g(zc + 0.385, 0.078) * lipw
    py += 0.030 * lip_mound
    py += 0.014 * _g(zc + 0.360, 0.022) * lipw                             # upper vermilion
    py += 0.016 * _g(zc + 0.415, 0.024) * lipw                             # lower vermilion
    py -= 0.032 * _g(zc + 0.388, 0.014) * _g(xc, 0.16) * front             # the seam
    px *= 1.0 + 0.04 * lip_mound                                          # widen at the mouth
    py -= 0.013 * _g(zc + 0.50, 0.045) * _g(xc, 0.22) * front              # mentolabial crease

    # ---- 6. chin + a forward mandible plane so the profile isn't round ----
    py += 0.045 * _band(-0.78, -0.28, 0.18, zc) * front * _g(xc, 0.46)     # bring the lower face forward
    chin = _band(-0.86, -0.48, 0.17, zc) * front * _g(xc, 0.52)
    py += 0.068 * chin

    # ---- 7. cheekbones (zygomatic) + faint sub-cheek hollow ----
    for cx in (-0.50, 0.50):
        d = math.sqrt((xc - cx) ** 2 + (y - 0.22) ** 2 * 0.8 + (zc - 0.03) ** 2)
        px += (0.040 if cx > 0 else -0.040) * _g(d, 0.33)
        py += 0.020 * _g(d, 0.30)
    py -= 0.010 * _g(xc, 0.5) * _g(zc + 0.09, 0.10) * front

    # ---- 8. jaw angle + underjaw ----
    jaw = _g(zc + 0.42, 0.17) * _g(y - 0.05, 0.72)
    px *= 1.0 + 0.06 * jaw
    if zc < -0.5:
        py += _smoothstep(-0.5, -0.9, zc) * 0.06 * midline

    return mathutils.Vector((px, py, pz))


def build_head():
    # dense, uniform icosphere: analytic displacement stays smooth on it, so we
    # skip the voxel remesh (which terraces shallow features)
    bpy.ops.mesh.primitive_ico_sphere_add(radius=1.0, subdivisions=7)
    head = bpy.context.active_object
    head.name = "Head"

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=1e-4)
    bpy.ops.object.mode_set(mode="OBJECT")

    me = head.data
    for v in me.vertices:
        d = v.co.normalized()
        v.co = _sculpt_vertex(d.x, d.y, d.z)

    bpy.context.view_layer.objects.active = head
    head.select_set(True)

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")

    m = head.modifiers.new("smooth", "SMOOTH")     # tiny relax, feature-preserving
    m.iterations = 1
    m.factor = 0.15
    bpy.ops.object.modifier_apply(modifier=m.name)

    bpy.ops.object.shade_smooth()
    return head


# --------------------------------------------------------------------------- #
# eyes, ears, neck
# --------------------------------------------------------------------------- #
EYE_X = 0.245
EYE_Y = 0.775
EYE_Z = 0.005
EYE_R = 0.100


def add_eyes():
    eyeballs, irises = [], []
    for side, sgn in (("L", -1.0), ("R", 1.0)):
        x = sgn * EYE_X
        bpy.ops.mesh.primitive_uv_sphere_add(radius=EYE_R, segments=48, ring_count=32)
        eye = bpy.context.active_object
        eye.name = f"Eye_{side}"
        eye.location = (x, EYE_Y, EYE_Z)
        bpy.ops.object.shade_smooth()
        eyeballs.append(eye)

        # iris: a shallow dome whose front pokes just past the sclera surface,
        # nudged toward the nose so the gaze converges on the viewer
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.050, segments=44, ring_count=28)
        iris = bpy.context.active_object
        iris.name = f"Iris_{side}"
        iris.location = (x * 0.93, EYE_Y + EYE_R - 0.013, EYE_Z + 0.006)
        iris.scale = (1.0, 0.40, 1.0)
        bpy.ops.object.shade_smooth()
        irises.append(iris)
    return eyeballs, irises


def add_ears():
    ears = []
    for side, sgn in (("L", -1.0), ("R", 1.0)):
        bpy.ops.mesh.primitive_ico_sphere_add(radius=1.0, subdivisions=4)
        ear = bpy.context.active_object
        ear.name = f"Ear_{side}"
        me = ear.data
        for v in me.vertices:
            v.co.x *= 0.045
            v.co.z *= 0.23
            v.co.y *= 0.185
            r = math.hypot(v.co.z, v.co.y)
            rim = _g(r - 0.17, 0.045)
            if v.co.x * sgn > 0.0:                       # front: scoop the concha, keep the rim
                v.co.x -= sgn * 0.06 * _g(r, 0.10) * (1.0 - rim)
            v.co.x += sgn * 0.02 * rim
            if v.co.z < -0.10:                           # earlobe
                v.co.z -= 0.03 * _smoothstep(-0.10, -0.20, v.co.z)
        ear.location = (sgn * 0.75, -0.16, -0.05)
        ear.rotation_euler = (math.radians(6), 0.0, math.radians(-10 * sgn))
        m = ear.modifiers.new("sub", "SUBSURF")
        m.levels = 1
        m.render_levels = 2
        bpy.ops.object.shade_smooth()
        ears.append(ear)
    return ears


def add_neck():
    # open-ended tube (no n-gon caps -> no subsurf fan artefacts)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.40, depth=1.3, vertices=64,
                                        end_fill_type="NOTHING")
    neck = bpy.context.active_object
    neck.name = "Neck"
    neck.location = (0.0, -0.06, -1.02)
    neck.rotation_euler = (math.radians(-5), 0.0, 0.0)
    for v in neck.data.vertices:                 # widen at the base, taper to the jaw
        t = _smoothstep(-0.65, 0.65, v.co.z)
        s = 1.0 - 0.16 * t
        v.co.x *= s
        v.co.y *= s * (1.0 + 0.10 * (1.0 - t))   # a little oval, deeper front-to-back low down
    m = neck.modifiers.new("sub", "SUBSURF")
    m.levels = 2
    m.render_levels = 3
    for p in neck.data.polygons:
        p.use_smooth = True
    return neck


# --------------------------------------------------------------------------- #
# scalp hair
# --------------------------------------------------------------------------- #
def _st(obj, attr, val):
    try:
        setattr(obj, attr, val)
    except (AttributeError, TypeError) as e:
        print(f"[face] hair: skipped {attr} ({e})")


def add_hair(head):
    """Eyebrows always; scalp hair when WANT_HAIR."""
    # ---- vertex groups ----
    scalp = head.vertex_groups.new(name="scalp")
    brow = head.vertex_groups.new(name="brow")
    for v in head.data.vertices:
        co = v.co
        cr = max(0.0, co.z - 0.10) * 1.5
        back = max(0.0, -co.y - 0.05) * 0.9
        face_front = max(0.0, co.y - 0.05) * max(0.0, co.z + 0.15) * 2.4
        w = max(0.0, min(1.0, cr + back - face_front))
        if co.y > 0.30 and co.z > 0.20:                     # forehead hairline
            w *= _smoothstep(0.58, 0.32, co.z)
        scalp.add([v.index], w, "REPLACE")

        bw = 0.0
        for ex in (-0.245, 0.245):
            d = math.dist((co.x, co.y, co.z), (ex, 0.80, 0.135))
            if d < 0.17 and co.y > 0.4:
                bw = max(bw, _smoothstep(0.17, 0.05, d))
        brow.add([v.index], bw, "REPLACE")

    def make_psys(name, count, length, seed, kids, vg, gravity, clump,
                  root=0.7, kink_amp=0.003, step=6):
        head.modifiers.new(name, "PARTICLE_SYSTEM")
        ps = head.particle_systems[-1]
        s = ps.settings
        _st(ps, "seed", seed)
        _st(s, "type", "HAIR")
        _st(s, "count", count)
        _st(s, "hair_length", length)
        _st(s, "hair_step", step)
        _st(s, "use_advanced_hair", True)
        _st(s, "root_radius", root)
        _st(s, "tip_radius", 0.0)
        _st(s, "radius_scale", 0.003)
        _st(s, "brownian_factor", 0.04)
        _st(s, "child_type", "INTERPOLATED")
        _st(s, "child_percent", kids)
        _st(s, "child_nbr", kids)
        _st(s, "rendered_child_count", kids)
        _st(s, "clump_factor", clump)
        _st(s, "roughness_1", 0.09)
        _st(s, "roughness_2", 0.07)
        _st(s, "roughness_endpoint", 0.05)
        _st(s, "kink", "CURL")
        _st(s, "kink_amplitude", kink_amp)
        _st(s, "kink_frequency", 2.0)
        _st(ps, "vertex_group_density", vg)
        _st(ps, "vertex_group_length", vg)
        _st(s.effector_weights, "gravity", gravity)
        return ps

    make_psys("brows", 700, 0.035, 3, kids=30, vg="brow", gravity=0.15,
              clump=0.4, root=0.5, kink_amp=0.001, step=4)

    if WANT_HAIR:
        make_psys("hair_under", 1500, 0.26, 1, kids=50, vg="scalp", gravity=0.8, clump=0.6)
        make_psys("hair_over", 950, 0.32, 7, kids=40, vg="scalp", gravity=0.8, clump=0.6)


# --------------------------------------------------------------------------- #
# materials
# --------------------------------------------------------------------------- #
def _principled(mat):
    return mat.node_tree.nodes["Principled BSDF"]


def skin_material():
    mat = bpy.data.materials.new("Skin")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = _principled(mat)

    bsdf.inputs["Base Color"].default_value = (0.52, 0.34, 0.26, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.52
    for key, val in (("Subsurface Weight", 0.16), ("Subsurface", 0.16)):
        if key in bsdf.inputs:
            bsdf.inputs[key].default_value = val
    if "Subsurface Radius" in bsdf.inputs:
        bsdf.inputs["Subsurface Radius"].default_value = (0.36, 0.18, 0.12)
    if "Subsurface Color" in bsdf.inputs:
        bsdf.inputs["Subsurface Color"].default_value = (0.62, 0.20, 0.15, 1.0)

    tc = nt.nodes.new("ShaderNodeTexCoord")

    n1 = nt.nodes.new("ShaderNodeTexNoise")
    n1.inputs["Scale"].default_value = 60.0
    n1.inputs["Detail"].default_value = 8.0
    n2 = nt.nodes.new("ShaderNodeTexNoise")
    n2.inputs["Scale"].default_value = 7.0
    n2.inputs["Detail"].default_value = 4.0
    mix = nt.nodes.new("ShaderNodeMixRGB")
    mix.inputs["Fac"].default_value = 0.35
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.30
    bump.inputs["Distance"].default_value = 0.012
    nt.links.new(tc.outputs["Object"], n1.inputs["Vector"])
    nt.links.new(tc.outputs["Object"], n2.inputs["Vector"])
    nt.links.new(n1.outputs["Fac"], mix.inputs["Color1"])
    nt.links.new(n2.outputs["Fac"], mix.inputs["Color2"])
    nt.links.new(mix.outputs["Color"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    cr = nt.nodes.new("ShaderNodeValToRGB")
    cr.color_ramp.elements[0].color = (0.47, 0.30, 0.23, 1.0)
    cr.color_ramp.elements[1].color = (0.57, 0.39, 0.30, 1.0)
    mid = cr.color_ramp.elements.new(0.5)
    mid.color = (0.52, 0.34, 0.26, 1.0)
    nt.links.new(n2.outputs["Fac"], cr.inputs["Fac"])

    # warm the mid-face band (nose / lips / cheeks)
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Generated"], sep.inputs["Vector"])
    wm = nt.nodes.new("ShaderNodeValToRGB")
    wm.color_ramp.elements[0].position = 0.30
    wm.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    wm.color_ramp.elements[1].position = 0.46
    wm.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    wm.color_ramp.elements.new(0.62).color = (0.0, 0.0, 0.0, 1.0)
    nt.links.new(sep.outputs["Z"], wm.inputs["Fac"])
    warm = nt.nodes.new("ShaderNodeMixRGB")
    warm.inputs["Color2"].default_value = (0.58, 0.26, 0.22, 1.0)
    wfac = nt.nodes.new("ShaderNodeMath")
    wfac.operation = "MULTIPLY"
    wfac.inputs[1].default_value = 0.34
    nt.links.new(wm.outputs["Color"], wfac.inputs[0])
    nt.links.new(wfac.outputs["Value"], warm.inputs["Fac"])
    nt.links.new(cr.outputs["Color"], warm.inputs["Color1"])

    dark = nt.nodes.new("ShaderNodeMixRGB")
    dark.blend_type = "MULTIPLY"
    dark.inputs["Fac"].default_value = 0.22
    ao = nt.nodes.new("ShaderNodeAmbientOcclusion")
    ao.inputs["Distance"].default_value = 0.14
    nt.links.new(warm.outputs["Color"], dark.inputs["Color1"])
    nt.links.new(ao.outputs["Color"], dark.inputs["Color2"])
    nt.links.new(dark.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def sclera_material():
    mat = bpy.data.materials.new("Sclera")
    mat.use_nodes = True
    bsdf = _principled(mat)
    bsdf.inputs["Base Color"].default_value = (0.85, 0.83, 0.80, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.22
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.5
    for key, val in (("Subsurface Weight", 0.10), ("Subsurface", 0.10)):
        if key in bsdf.inputs:
            bsdf.inputs[key].default_value = val
    return mat


def iris_material():
    mat = bpy.data.materials.new("Iris")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = _principled(mat)
    bsdf.inputs["Roughness"].default_value = 0.09
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = 0.6
    if "Coat Roughness" in bsdf.inputs:
        bsdf.inputs["Coat Roughness"].default_value = 0.03

    tc = nt.nodes.new("ShaderNodeTexCoord")
    mapp = nt.nodes.new("ShaderNodeMapping")
    mapp.inputs["Location"].default_value = (-0.5, -0.5, -0.5)
    mapp.inputs["Scale"].default_value = (2.15, 0.0, 2.15)   # measure radius in the disc plane only
    grad = nt.nodes.new("ShaderNodeTexGradient")
    grad.gradient_type = "SPHERICAL"
    nt.links.new(tc.outputs["Generated"], mapp.inputs["Vector"])
    nt.links.new(mapp.outputs["Vector"], grad.inputs["Vector"])

    # radial streaks in the iris
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 3.0
    noise.inputs["Detail"].default_value = 6.0
    disturb = nt.nodes.new("ShaderNodeMixRGB")
    disturb.inputs["Fac"].default_value = 0.12
    nt.links.new(grad.outputs["Color"], disturb.inputs["Color1"])
    nt.links.new(noise.outputs["Fac"], disturb.inputs["Color2"])

    ramp = nt.nodes.new("ShaderNodeValToRGB")
    c = ramp.color_ramp
    stops = [
        (0.00, (0.006, 0.005, 0.004)),   # pupil
        (0.22, (0.006, 0.005, 0.004)),   # pupil edge
        (0.30, (0.10, 0.05, 0.02)),      # inner iris
        (0.60, (0.38, 0.22, 0.10)),      # brown
        (0.82, (0.20, 0.12, 0.05)),      # outer iris
        (0.92, (0.04, 0.025, 0.012)),    # limbal ring
        (1.00, (0.04, 0.025, 0.012)),
    ]
    c.elements[0].position, c.elements[0].color = stops[0][0], (*stops[0][1], 1.0)
    c.elements[1].position, c.elements[1].color = stops[1][0], (*stops[1][1], 1.0)
    for pos, col in stops[2:]:
        c.elements.new(pos).color = (*col, 1.0)
    nt.links.new(disturb.outputs["Color"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def hair_material():
    mat = bpy.data.materials.new("Hair")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        if n.type != "OUTPUT_MATERIAL":
            nt.nodes.remove(n)
    out = nt.nodes["Material Output"]
    try:
        h = nt.nodes.new("ShaderNodeBsdfHairPrincipled")
        if "Color" in h.inputs:
            h.inputs["Color"].default_value = (0.020, 0.013, 0.009, 1.0)
        if "Roughness" in h.inputs:
            h.inputs["Roughness"].default_value = 0.35
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
# scene: world, camera, lights, render
# --------------------------------------------------------------------------- #
def setup_world():
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.021, 0.024, 0.030, 1.0)
    bg.inputs["Strength"].default_value = 0.4


def setup_camera():
    cam_data = bpy.data.cameras.new("Cam")
    cam_data.lens = 85
    cam_data.dof.use_dof = True
    cam = new_object("Camera", cam_data)
    cam.location = mathutils.Vector((1.7, 6.4, 0.30))

    target_loc = mathutils.Vector((0.0, 0.22, -0.03))
    target = new_object("CamTarget", None)
    target.location = target_loc
    cam.rotation_euler = (target_loc - cam.location).to_track_quat("-Z", "Y").to_euler()

    cam_data.dof.focus_object = target
    cam_data.dof.aperture_fstop = 5.6
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
        tgt.location = (0, 0.3, 0.0)
        con.target = tgt
        con.track_axis = "TRACK_NEGATIVE_Z"
        return o

    area("Key", (3.0, 4.0, 2.6), 520, 1.7, color=(1.0, 0.96, 0.90))
    area("Fill", (-3.8, 3.4, 0.6), 210, 3.6, color=(0.76, 0.83, 1.0))
    area("Rim", (-1.8, -3.0, 2.4), 420, 1.4, color=(0.82, 0.9, 1.0))
    area("Kick", (2.8, -1.4, -1.2), 80, 1.6, color=(1.0, 0.8, 0.58))
    area("Catch", (1.0, 5.8, 1.2), 200, 0.22, color=(1.0, 0.98, 0.96))


def enable_gpu():
    prefs = bpy.context.preferences.addons["cycles"].preferences
    chosen = None
    gpus = []
    for backend in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
        try:
            prefs.compute_device_type = backend
        except TypeError:
            continue
        if hasattr(prefs, "get_devices"):
            prefs.get_devices()
        try:
            devs = list(prefs.devices)
        except Exception:
            devs = []
        gpus = [d for d in devs if d.type == backend]
        if gpus:
            chosen = backend
            for d in devs:
                d.use = (d.type == backend)
            break
    if chosen:
        bpy.context.scene.cycles.device = "GPU"
        print(f"[face] Cycles GPU backend: {chosen} ({len(gpus)} device(s))")
    else:
        print("[face] no GPU found - rendering on CPU")
    return chosen


def _has_agx():
    try:
        return "AgX" in bpy.context.scene.view_settings.bl_rna.properties["view_transform"].enum_items
    except Exception:
        return False


def setup_render():
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = SAMPLES
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 12
    scene.render.resolution_x = int(RES * 0.80)
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
        print("[face] GPU disabled by FACE_GPU=0")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    print(f"[face] out dir: {OUT_DIR}  samples={SAMPLES}  res={RES}  hair={WANT_HAIR}")
    reset_scene()

    head = build_head()
    bpy.context.view_layer.update()

    eyeballs, irises = add_eyes()
    ears = add_ears()
    neck = add_neck()

    skin = skin_material()
    sclera = sclera_material()
    iris_mat = iris_material()
    hair_mat = hair_material()

    assign(head, skin)
    assign(neck, skin)
    for e in ears:
        assign(e, skin)
    for e in eyeballs:
        assign(e, sclera)
    for i in irises:
        assign(i, iris_mat)

    add_hair(head)
    if head.particle_systems:
        head.data.materials.append(hair_mat)
        for ps in head.particle_systems:
            ps.settings.material = len(head.data.materials)

    setup_world()
    setup_camera()
    setup_lights()
    setup_render()

    bpy.ops.wm.save_as_mainfile(filepath=BLEND_PATH)
    print(f"[face] saved {BLEND_PATH}")

    bpy.ops.render.render(write_still=True)
    print(f"[face] wrote {PNG_PATH}")

    if os.environ.get("FACE_DEBUG"):
        scene = bpy.context.scene
        cam = scene.camera
        tgt = mathutils.Vector((0.0, 0.2, -0.05))
        views = {
            "front": mathutils.Vector((0.0, 6.4, 0.12)),
            "q34": mathutils.Vector((3.2, 5.2, 0.6)),
            "side": mathutils.Vector((6.2, 0.2, 0.05)),
            "low": mathutils.Vector((0.6, 5.6, -1.8)),
        }
        for name, loc in views.items():
            cam.location = loc
            cam.rotation_euler = (tgt - loc).to_track_quat("-Z", "Y").to_euler()
            scene.render.filepath = os.path.join(OUT_DIR, f"face_debug_{name}.png")
            bpy.ops.render.render(write_still=True)
            print(f"[face] wrote face_debug_{name}.png")


if __name__ == "__main__":
    main()
