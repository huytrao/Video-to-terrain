# Moving files between two Kaggle notebooks

Everything needed to transfer a file from one running Kaggle notebook to another —
including the case both notebooks have internet, which has more options than the
no-internet case and a different best answer depending on size.

Every number in this document was measured on this project, not estimated. Where a
limit is stated as "silent", it means the failure produces no error and a plausible
result, which is the only reason it is worth a document.

---

## 0. The one idea everything rests on

> A notebook's **Jupyter Contents API is served by Kaggle's platform, not by the
> kernel process.** The "Internet" toggle only blocks network calls that the
> *kernel's own process* initiates. It has no say over the platform accepting an
> authenticated file read or write issued from somewhere else.

Consequences, and they are not obvious:

- A notebook with **Internet: off** can still **receive** a file pushed to it.
- A notebook with **Internet: off** can still have a file **read out of it**.
- Neither direction requires that notebook's kernel to fetch or send anything. Its
  kernel is only ever asked to concatenate parts already sitting on its disk.
- So the internet toggle is not a transfer barrier at all. It is a barrier only for
  code *inside* that kernel calling out.

This is why "internet → internet" and "internet → no internet" use the same
transport. Having internet on both ends does not unlock a faster path; it unlocks
*additional* paths (datasets, Hugging Face) that are slower but more durable.

---

## 1. Which route to use

| Situation | Route | Speed | Durable? | Use when |
|---|---|---|---|---|
| Both have internet, payload ≤ ~15 GB, you want it to survive the session | **A. Kaggle Dataset** | ~110 MB/s up, then minutes of processing | Yes — versioned, permanent | Results you will re-attach later, or share |
| Either or both, any size, you want it *now* | **B. Direct Contents API relay** | **44–51 MB/s** kernel→kernel | No — dies with the session | Moving work between live kernels mid-run |
| Both have internet, large model weights | **C. Hugging Face Hub** | ~80–150 MB/s | Yes | Checkpoints, anything you would `from_pretrained` |
| Receiver has no internet | **B only** (A and C are impossible from its side) | 44–51 MB/s | No | Feeding an offline GPU kernel |
| Via your laptop | **Don't** | **4.6 MB/s** | — | Never for bulk. 10× slower than B |

The laptop number is the one that decides architecture. Routing a payload through
your own machine is roughly **ten times slower** than kernel-to-kernel, and on this
project it also cost a 121 MB upload to proxy instability. Keep the bytes on
Kaggle's side of the wire.

---

## 2. Prerequisites (once per session)

### 2.1 Get each notebook's proxy URL

Both notebooks must be **running** (not just saved).

1. Open the notebook in a browser.
2. Kebab menu (**⋮ More settings**) → **Kaggle Jupyter Server**.
3. Copy the URL. It looks like:

```
https://kkb-production.jupyter-proxy.kaggle.net/k/<kernel-id>/<long-jwt>/proxy
```

That trailing token is a **JWT and it expires** — typically within hours, and always
when the session restarts. An expired URL is not a transient failure (§6).

### 2.2 Put them in `.env`, never in a script

```bash
# .env  (gitignored)
KAGGLE_HAS_INTERNET=https://kkb-production.jupyter-proxy.kaggle.net/k/<id>/<jwt>/proxy
KAGGLE_WITHOUT_INTERNET=https://kkb-production.jupyter-proxy.kaggle.net/k/<id>/<jwt>/proxy
KAGGLE_USERNAME=<your-kaggle-username>
KAGGLE_KEY=<your-kaggle-api-key>
HF_TOKEN=<your-hf-token>
```

For an internet→internet transfer, the names still work — read `KAGGLE_HAS_INTERNET`
as "the sender" and `KAGGLE_WITHOUT_INTERNET` as "the receiver". Nothing in the
transport inspects whether the receiver actually has internet.

**Do not `source .env`.** `. ./.env` executes every line. A key whose *name* contains
a dash is not a valid shell identifier, so the shell treats the line as a command and
echoes the **value** in its "command not found" error — which is how a live credential
lands in a terminal log. Parse it instead:

```bash
# scripts/kenv.sh — prints only the keys you ask for, executes nothing
eval "$(scripts/kenv.sh)"
```

### 2.3 Install

```bash
pip install requests websocket-client
```

---

## 3. Route A — via a Kaggle Dataset (both ends need internet)

The durable route. The sender publishes a dataset; the receiver attaches it.

### 3.1 Sender side, run *on kernel A*

```python
import json, os, subprocess, tarfile

# 1. Credentials, staged as a file rather than interpolated into a command string.
#    An interpolated key ends up in a command that gets echoed and scrolled back.
creds = json.load(open("/kaggle/working/_secret/kaggle_creds.json"))
os.environ["KAGGLE_USERNAME"] = creds["username"]
os.environ["KAGGLE_KEY"]      = creds["key"]
kdir = os.path.expanduser("~/.kaggle"); os.makedirs(kdir, exist_ok=True)
json.dump(creds, open(os.path.join(kdir, "kaggle.json"), "w"))
os.chmod(os.path.join(kdir, "kaggle.json"), 0o600)

# 2. Stage the payload. It MUST be under /kaggle/working.
STAGE = "/kaggle/working/kaggle_upload"
os.makedirs(STAGE, exist_ok=True)

# 3. Metadata. `id` must be "<username>/<slug>"; slug lowercase, hyphenated.
meta = {"title": "My transfer payload",          # 6-50 characters, enforced
        "id": f"{creds['username']}/my-payload",
        "licenses": [{"name": "CC0-1.0"}]}
json.dump(meta, open(os.path.join(STAGE, "dataset-metadata.json"), "w"), indent=2)

# 4. Create, or add a version if it already exists.
st = subprocess.run(["kaggle", "datasets", "status", meta["id"]],
                    capture_output=True, text=True)
exists = st.returncode == 0 and "404" not in (st.stdout + st.stderr)
cmd = (["kaggle", "datasets", "version", "-p", STAGE, "-m", "new run", "--dir-mode", "zip"]
       if exists else
       ["kaggle", "datasets", "create", "-p", STAGE, "--dir-mode", "zip"])
r = subprocess.run(cmd, capture_output=True, text=True)
print(r.stdout[-3000:], r.stderr[-2000:])
r.returncode == 0 or (_ for _ in ()).throw(SystemExit("kaggle CLI failed"))
```

### 3.2 Receiver side

In the receiving notebook's sidebar: **Add Input → Datasets → your dataset**. It
appears at `/kaggle/input/<slug>/`, read-only.

### 3.3 What to know

- `--dir-mode zip` matters. Without it, a directory of thousands of small files
  uploads one request per file and takes far longer than the bytes justify.
- **Processing delay.** After upload the dataset is unavailable until Kaggle finishes
  ingesting it — minutes for GBs. `kaggle datasets status <id>` polls it. This is why
  Route A is wrong for mid-run handoff.
- The receiver must **restart its session** to pick up a newly attached dataset.
- `/kaggle/input` is read-only. Copy out before modifying.
- Nested `sdist`/source trees on a read-only mount can sit one directory deeper than
  expected; copy to `/tmp` before building anything from them.

---

## 4. Route B — direct Contents API relay (the fast one)

Works in both directions and regardless of either notebook's internet toggle. This is
the route to use for live kernel-to-kernel movement.

### 4.1 Sender A → receiver B

```bash
eval "$(scripts/kenv.sh)"
export KAGGLE_INTERNET="$KAGGLE_HAS_INTERNET"     # A: runs the driver
# KAGGLE_WITHOUT_INTERNET  = B: receives

python3 scripts/relay_push.py /tmp/payload.tar.gz \
        --dest /tmp/payload.tar.gz \
        --workers 4 --subchunk 8
```

Add `--extract-to /tmp/unpacked` to untar on B in the same pass.

### 4.2 Receiver B → sender A (pulling results back out)

```bash
python3 scripts/relay_pull.py relay_out/deliver.tar.gz \
        --dest /tmp/deliver.tar.gz
```

The source path is **relative to B's `/kaggle/working`**, because that is the only
tree B's Contents API exposes.

### 4.3 Exactly what happens

```
   your laptop            notebook A                    notebook B
        |                      |                             |
        |-- upload ~4 KB ----->|  (driver script only)        |
        |                      |                             |
        |                      |-- PUT part_00_00000.bin ---->|  /kaggle/working/relay_parts/
        |                      |-- PUT part_01_00000.bin ---->|      (4 workers in parallel)
        |                      |-- ...  8 MB each  ---------->|
        |                      |                             |
        |                      |-- ONE websocket exec ------->|  concatenate parts -> dest,
        |                      |    (execute_request)         |  then delete the parts dir
        |<--- short progress --|                             |
```

Five design points, each of which is load-bearing:

1. **Parallel byte-range workers.** One stream does not saturate the link; four do.
   The payload is split into `workers` contiguous ranges, each streamed and PUT
   independently.
2. **Contents API for bulk, websocket for control.** Pushing bytes through
   `execute_request` messages is roughly **40× slower** than the Contents API. The
   websocket is used exactly once, to run the concatenation.
3. **8 MB per PUT.** See §5.1 — this is not a tuning preference.
4. **Retry with exponential backoff on every PUT** (6 attempts, capped at 20 s) and on
   every source read (8 attempts, capped at 30 s). The proxy drops requests without
   provocation; a single-shot transfer of thousands of parts will not finish.
5. **The laptop never sees the bytes.** It uploads a ~4 KB driver and reads progress
   lines. This is what makes 44–51 MB/s achievable instead of 4.6.

### 4.4 The wrapper that makes it survive a flaky proxy

`scripts/kx.sh` retries the executor with linear backoff and — importantly —
distinguishes failure kinds, because retrying the wrong one wastes minutes:

```bash
KX_KERNEL=has   scripts/kx.sh exec --timeout 3600 "$(cat job.py)"   # the internet kernel
KX_KERNEL=run   scripts/kx.sh upload local.bin remote/path.bin      # the offline kernel
```

- `HTTP 404` on `/api` → **the JWT expired**. Exits 3 immediately; no retry revives it.
- `ReadTimeout` / `ConnectionError` / `HTTP 5xx` / `RemoteDisconnected` → transport,
  retried with backoff.
- Anything else → a real error in your code. Fails immediately rather than six times.

---

## 5. The limits that fail silently

These are the ones worth memorising. Each produces a plausible-looking result.

### 5.1 A PUT larger than ~24 MB is silently truncated

Kaggle's Contents API accepts a PUT near or above ~24 MB, returns success, and stores
**less than you sent**. No error, no warning. The assembled file is then short — and a
short tarball fails much later with a confusing message, or worse, a short binary
loads and misbehaves.

**Use 8 MB.** `relay_push.py` defaults to `--subchunk 8` for this reason.

> ⚠️ **`kaggle_relay.py` in this repo defaults to `--subchunk 24`**, which sits exactly
> on the boundary. Pass `--subchunk 8` explicitly when using it for anything whose
> integrity you care about.

### 5.2 `/kaggle/working` is the only tree the Contents API can see

`/tmp` is **invisible** to the Contents API. You cannot PUT into it or GET out of it
directly. This is why the pattern is always: work in `/tmp`, then stage a copy under
`/kaggle/working` for transfer.

### 5.3 The two filesystems have opposite weaknesses

| Path | Size | Survives a session restart? | Visible to Contents API? |
|---|---|---|---|
| `/kaggle/working` | **20 GB** (loop mount) | Yes | **Yes** |
| `/tmp` | **1.1 TB** | **No — wiped** | No |
| `/kaggle/input/...` | per dataset | Yes (read-only) | Yes |

So: compute in `/tmp` (space), stage in `/kaggle/working` (visibility), and check free
space **before** staging. A 20 GB mount will not hold an 8 GB payload plus its tarball
plus whatever else is already there. Refuse rather than half-write:

```python
import shutil
total, used, free = shutil.disk_usage("/kaggle/working")
if payload_bytes > free * 0.8:
    raise SystemExit(f"{payload_bytes/1e9:.2f} GB payload, only {free/1e9:.1f} GB free")
```

### 5.4 A single large GET read-times-out at 30 s

Reading a 27 MB file in one Contents API response times out. Split on the producing
side into parts that share a filename prefix, then read them in order —
`relay_pull.py` does exactly this, listing the directory and sorting by name.

### 5.5 `tarfile` with `w:gz` becomes the bottleneck

Python's `tarfile` gzip is single-threaded. On ~8 GB dominated by float32 arrays it
spent **longer compressing than the transfer would have taken**, turning a 45 MB/s
link into a CPU-bound wait. Binary PLY and model checkpoints barely compress anyway.

```python
import shutil, subprocess
pigz = shutil.which("pigz")
cmd = (["tar", "-I", "pigz", "-cf", TAR, "-C", SRC, "-T", listing] if pigz
       else ["tar", "-cf", TAR, "-C", SRC, "-T", listing])   # uncompressed beats slow
```

### 5.6 `cmd | tail` reports tail's exit status

```bash
python3 transfer.py | tail -20     # ALWAYS "succeeds" — tail exited 0
```

A dead proxy read as "OK on attempt 1" this way. Use `set -o pipefail`, or capture to
a file and check the status separately.

### 5.7 A `.pth` in site-packages does not affect the already-running kernel

If you extend `sys.path` by dropping a `.pth` file, a kernel that is *already running*
will not see it. Append the path explicitly in that session too.

### 5.8 The kernel can be busy

A second `execute_request` while one is running queues or errors depending on timing.
Treat "busy" as a reason to wait, not to fail. `kx.sh` does.

---

## 6. Verify every transfer — SHA256, computed on the kernel

A transfer that reports success and delivers a truncated file is the failure mode this
transport actually has (§5.1). So the check is not optional, and it must be computed
**on the receiving kernel**, not inferred from byte counts the sender printed.

```python
# run on the RECEIVER
import hashlib
h = hashlib.sha256()
with open(DEST, "rb") as fh:
    while block := fh.read(1 << 20):
        h.update(block)
print(f"bytes {os.path.getsize(DEST):,}  sha256 {h.hexdigest()}")
```

Compare against the local digest. `scripts/kupload.py` automates this for
laptop→kernel: it splits, uploads, then computes the digest on the kernel and fails
loudly on mismatch rather than suggesting you retry with a smaller chunk.

---

## 7. Measured throughput

| Link | Rate | Note |
|---|---|---|
| Kernel A → Kernel B (Contents API, 4 workers, 8 MB parts) | **44–51 MB/s** | The fast path |
| Kernel → Kaggle Dataset (`kaggle datasets version`) | **~110 MB/s** | Then minutes of processing |
| Laptop → Kernel | **4.6 MB/s** | Avoid for bulk |
| Kernel bulk via `execute_request` instead of Contents API | **~40× slower** | Never do this |
| 3.44 GB relayed end-to-end | **~2 minutes** | 4 parallel byte-range workers |

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Cannot reach Jupyter API: HTTP 404` | JWT expired | Re-copy the proxy URL from the notebook; update `.env`. Retrying cannot help |
| Repeated `ReadTimeout` on `/api` | Normal proxy flakiness | Retry with backoff (`kx.sh`); not a real error |
| Transfer "succeeded", file is short | PUT > 24 MB truncated silently | `--subchunk 8`, then verify SHA256 |
| Assembled tar won't extract | Same as above | Same |
| `no parts found for X on B` | Wrong path, or path not under `/kaggle/working` | Paths are relative to B's `/kaggle/working` |
| Push fails part-way, disk full | `/kaggle/working` is 20 GB | Check free space first; stage a subset |
| Dataset attached but not visible | Session predates the attach | Restart the receiver's session |
| Slow despite a good link | Single-threaded gzip | `pigz`, or don't compress |
| A credential appeared in your terminal | You sourced `.env` | Parse it (`kenv.sh`); rotate the key |
| Everything ran twice | A decorator that executes at definition *and* is called | Check for `@step`-style decorators invoked explicitly |

---

## 9. Security

- **Never put credentials in a `.py` file.** A key in a tracked file is one
  `git add -A` from being published. `kaggle_upload_example.py` in the parent project
  holds a live key inline; it should move to `.env` and be **rotated**.
- Stage credentials to the kernel as a **file** (`/kaggle/working/_secret/...`), not
  interpolated into a command string — command strings get echoed and scrolled back.
- `chmod 600` the staged `kaggle.json`.
- A proxy URL **is** a credential: its JWT grants full read/write to that notebook's
  `/kaggle/working` and the ability to execute code in it. Treat a pasted proxy URL
  like a password, and prefer letting it expire over reusing it.
- `.gitignore` must cover `.env`, `server-connect.txt`, and any `_secret/` tree.

---

## 10. Copy-paste: internet → internet, verified

```bash
# 0. both notebooks running; URLs in .env
eval "$(scripts/kenv.sh)"
export KAGGLE_INTERNET="$KAGGLE_HAS_INTERNET"        # sender
                                                     # receiver = KAGGLE_WITHOUT_INTERNET

# 1. on the SENDER: stage under /kaggle/working, archive without slow gzip
KX_KERNEL=has scripts/kx.sh exec --timeout 3600 '
import os, shutil, subprocess
SRC, OUT = "/tmp/results", "/kaggle/working/relay_out"
os.makedirs(OUT, exist_ok=True)
files = [os.path.join(dp, f) for dp, dn, fn in os.walk(SRC) for f in fn]
total = sum(os.path.getsize(p) for p in files)
_, _, free = shutil.disk_usage("/kaggle/working")
assert total < free * 0.8, f"{total/1e9:.1f} GB payload vs {free/1e9:.1f} GB free"
listing = os.path.join(OUT, "files.txt")
open(listing, "w").write("\n".join(os.path.relpath(p, SRC) for p in sorted(files)) + "\n")
tar = os.path.join(OUT, "payload.tar")
pigz = shutil.which("pigz")
cmd = ["tar"] + (["-I", "pigz"] if pigz else []) + ["-cf", tar, "-C", SRC, "-T", listing]
subprocess.run(cmd, check=True)
print("STAGED", tar, os.path.getsize(tar))
'

# 2. relay it, 8 MB parts, 4 workers — bytes never touch this machine
python3 scripts/relay_pull.py relay_out/payload.tar --dest /tmp/payload.tar

# 3. verify on the receiver before trusting it
KX_KERNEL=run scripts/kx.sh exec '
import hashlib, os
h = hashlib.sha256()
with open("/tmp/payload.tar", "rb") as fh:
    while b := fh.read(1 << 20): h.update(b)
print(os.path.getsize("/tmp/payload.tar"), h.hexdigest())
'
```

Compare that digest against the sender's. If they differ, the chunk size is the first
suspect (§5.1) — not the network.

---

## 11. Why there is no direct socket between two notebooks

Neither notebook accepts inbound connections: no public address, no port forwarding,
and outbound-only egress even when Internet is on. Every route above therefore goes
through a third party that both can reach — Kaggle's own platform API (Routes A, B) or
a public host (Route C). There is no peer-to-peer option, and any guide claiming one
is describing something else.
