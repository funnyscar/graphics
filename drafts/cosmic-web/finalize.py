"""
Publish the cosmic-web graphic:

  * upload dist/<slug>/**  to  r2://experiments/graphics/<slug>/
  * ensure machinedata.graphics exists
  * embed the story with text-embedding-3-large (via OpenRouter)
  * upsert the row (slug, title, description, r2_link, created_at, video_link,
    image_link, vector)  —  cover.jpg is a representative still from the video

    modal run drafts/cosmic-web/finalize.py --dist-dir dist/cosmic-web
"""
import pathlib
import modal

app = modal.App("cosmic-web-finalize")
img = modal.Image.debian_slim().pip_install("psycopg2-binary", "boto3", "requests")

SLUG = "cosmic-web"
TITLE = "The Cosmic Web — 13.8 Billion Years of Gravity in 19 Seconds"
DESCRIPTION = (
    "A ΛCDM particle-mesh N-body simulation run on one GPU: 16.7 million particles "
    "in a 160 Mpc/h box, integrated with a symplectic leapfrog from redshift z=49 to "
    "today. A near-uniform early universe (fluctuations of 1 part in 30,000) collapses "
    "under gravity into the cosmic web — sheets draining into filaments, filaments into "
    "the dense knots where galaxy clusters form, with expanding voids in between. "
    "Initial conditions use the Zel'dovich approximation from a BBKS power spectrum "
    "normalised to σ8=0.811; gravity is solved each step with FFTs on a 512³ mesh. "
    "The measured matter power spectrum tracks linear theory on large scales and shows "
    "the non-linear collapse boost on small scales."
)
STORY = TITLE + "\n\n" + DESCRIPTION + """

The film is a continuous fast-forward: one rendered frame per simulation timestep,
570 log-spaced steps in the cosmic scale factor a. Early on the matter density term
drives gravitational growth; near a≈0.6 (z≈0.7) dark energy takes over, the
expansion accelerates, and the growth of structure visibly freezes. The physics is
Newtonian gravity in comoving coordinates governed by the Friedmann equation, the
particle-mesh Poisson solve g_k = i k/k^2 (3/2)(Ω_m/a) δ_k, and a kick-drift-kick
integrator whose exact ΛCDM drift and kick factors are ∫da/(a^3 E) and
∫(3Ω_m/2)da/(a^2 E). Cosmology: Ω_m=0.311, Ω_Λ=0.689, h=0.677, n_s=0.965.
Rendered as an additive projection of a soft 50 Mpc/h slab with a log-density colour
ramp, a slow 20-degree rotation and zoom, at 1280x720.
"""

R2_LINK = f"https://experiments.funnyscar.com/graphics/{SLUG}/index.html"
VIDEO_LINK = f"https://experiments.funnyscar.com/graphics/{SLUG}/cosmic-web.mp4"
IMAGE_LINK = f"https://experiments.funnyscar.com/graphics/{SLUG}/cover.jpg"

CT = {
    ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8", ".mp4": "video/mp4", ".png": "image/png",
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".json": "application/json", ".svg": "image/svg+xml", ".woff2": "font/woff2",
    ".woff": "font/woff", ".ttf": "font/ttf", ".map": "application/json",
}


@app.function(image=img, secrets=[
    modal.Secret.from_name("r2"),
    modal.Secret.from_name("do_db"),
    modal.Secret.from_name("openrouter-secret"),
])
def publish(files: dict):
    import os, io, json, datetime
    import boto3, requests, psycopg2

    # ---- 1. upload to R2 -------------------------------------------------
    s3 = boto3.client(
        "s3", endpoint_url=os.environ["ENDPOINT"],
        aws_access_key_id=os.environ["ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["SECRET_KEY_ID"], region_name="auto",
    )
    for rel, blob in files.items():
        ext = "." + rel.rsplit(".", 1)[-1] if "." in rel else ""
        s3.put_object(
            Bucket="experiments", Key=f"graphics/{SLUG}/{rel}", Body=blob,
            ContentType=CT.get(ext, "application/octet-stream"),
            CacheControl="public, max-age=3600",
        )
        print(f"  uploaded graphics/{SLUG}/{rel}  ({len(blob):,} B)")

    # ---- 2. embedding -------------------------------------------------
    r = requests.post(
        "https://openrouter.ai/api/v1/embeddings",
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
        json={"model": "openai/text-embedding-3-large", "input": STORY},
        timeout=90,
    )
    r.raise_for_status()
    vec = r.json()["data"][0]["embedding"]
    assert len(vec) == 3072, len(vec)
    print(f"  embedding ok: {len(vec)} dims")

    # ---- 3. database -------------------------------------------------
    c = psycopg2.connect(
        host=os.environ["HOST"], port=os.environ["PORT"], user=os.environ["USERNAME"],
        password=os.environ["PASSWORD"], dbname="machinedata", sslmode="require",
    )
    c.autocommit = True
    cur = c.cursor()
    cur.execute("create extension if not exists vector")
    cur.execute("""
        create table if not exists graphics (
            id          bigint generated always as identity primary key,
            slug        text unique not null,
            title       text not null,
            description text,
            r2_link     text,
            created_at  timestamptz not null default now(),
            video_link  text,
            image_link  text,
            vector      vector(3072)
        )
    """)
    cur.execute("alter table graphics add column if not exists image_link text")
    cur.execute("""
        insert into graphics (slug, title, description, r2_link, created_at, video_link, image_link, vector)
        values (%s,%s,%s,%s,%s,%s,%s,%s)
        on conflict (slug) do update set
            title=excluded.title, description=excluded.description,
            r2_link=excluded.r2_link, video_link=excluded.video_link,
            image_link=excluded.image_link,
            vector=excluded.vector, created_at=excluded.created_at
    """, (SLUG, TITLE, DESCRIPTION, R2_LINK,
          datetime.datetime.now(datetime.timezone.utc), VIDEO_LINK, IMAGE_LINK,
          "[" + ",".join(f"{x:.7f}" for x in vec) + "]"))
    cur.execute("select slug,title,r2_link,video_link,image_link,created_at,vector is not null from graphics where slug=%s", (SLUG,))
    print("  row:", cur.fetchone())
    c.close()
    return R2_LINK


@app.local_entrypoint()
def main(dist_dir: str = "dist/cosmic-web"):
    root = pathlib.Path(dist_dir)
    assert (root / "index.html").exists(), f"no index.html in {root}"
    assert (root / "cover.jpg").exists(), f"no cover.jpg in {root} (screenshot a representative video frame)"
    files = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            files[str(p.relative_to(root))] = p.read_bytes()
    print(f"{len(files)} files, {sum(len(v) for v in files.values()):,} B total")
    link = publish.remote(files)
    print("\nLIVE:", link)
