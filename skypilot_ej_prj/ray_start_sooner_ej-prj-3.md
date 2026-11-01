# Start the head's Ray sooner (`ej-prj-3`)

- **Branch:** `ej-prj-3`, based on `ej-prj-bonus` (`6db5eba63`). Not yet
  committed.
- **Context:** idea #1 in `launch_latency_analysis_4_nodes.md`. Ray is the
  critical path of a steady 4-node launch on kind: about 10 s are spent
  waiting for Ray after runtime setup ends.

## 1. Where the head's Ray start went (measured first)

**Timestamped pod trace.** I set `PS4` with the date during one launch (a
temporary change, since reverted) to time every command in the step-2
script:

| Step | Head | Worker |
|---|---|---|
| `uv pip install 'setuptools<70'` (always a download: the image ships 84.0.0) | **12.1 s** in that run; 0.6 s usually | 0.6 s |
| `apply_patches.py` | 1.0 s | 1.0 s |
| `ray stop` on a pod that never ran Ray | **2.9 s** (while the dry run competed for CPU) | **3.8 s** |
| `ray start` | 6.4 s (`--head`) | 4.8 s |

**Inside a running head pod:** 3 repetitions each.

| Measurement | Result |
|---|---|
| Needless `ray stop` (no Ray running) | 1.42 / 1.44 / 1.49 s |
| `ray start --head`: GCS port open | ~1.2 s, with or without the dashboard |
| `ray start --head` returns, with the dashboard | 4.38–4.73 s |
| `ray start --head` returns, `--include-dashboard=false` | 3.66–3.72 s |

## 2. Changes

1. **`sky/provision/instance_setup.py`: `RAY_STOP_IF_RUNNING`.**
   - `ray stop` now runs only when a Ray process is running: the GCS, a
     raylet, or the dashboard. Both the head start command and the
     (restarting) worker start command use it.
   - **The guard can't match itself.** The startup script's own command line
     contains the `pgrep` pattern. Bracketing the first letters
     (`[g]cs_server`) means the pattern no longer matches that text.
   - **Fallback:** without `pgrep` (images with no procps), it stops Ray as
     before.
   - **Restart paths are unchanged.** Where Ray is running (the provisioner
     restarting Ray, the hostNetwork relaunch), it is still stopped first.
2. **`Dockerfile_k8s`, `Dockerfile_k8s_gpu` and the docs' custom-image snippet:**
   pre-install `setuptools<70` and `click<8.3.0`. Those are the pins
   `RAY_INSTALLATION_COMMANDS` applies at pod start. When they're already met,
   the pod only checks them locally instead of downloading setuptools from
   PyPI before Ray can start.
3. **Not changed:**
   - **The Ray dashboard.** Turning it off saves ~0.9 s, but only on the
     head's own `ray start` return: the GCS port, which the workers wait for,
     opens just as early. It isn't on the critical path, and users may rely
     on port 8266.
   - **Ray patches baked into the image.** That needs a version stamp so pods
     can skip re-applying them; it would save ~1 s. Left for later.
   - **Dropping or reordering the setuptools pin.** Ray 2.56 doesn't use
     `pkg_resources` (and nothing in the runtime environment uses
     `pkg_resources.packaging`), but a cluster pinned to an older Ray might.

## 3. Verification

- **Unit tests:** `tests/unit_tests/test_sky/provision/test_ray_stop_if_running.py`
  (new, 6 tests):
  - the guard stops Ray when `pgrep` finds it, skips when it doesn't, and
    stops anyway without `pgrep`;
  - the real `pgrep` doesn't match the guard's own command line;
  - both start commands stop Ray only through the guard, and before
    `ray start`.

  The provision, Kubernetes, template and backend unit tests pass (1809), and
  pylint is 10.00/10.
- **In a live pod:** with Ray running, the guard stops it; after Ray is
  stopped, it skips; it never matched its own `bash -c`.
- **Image:** `docker build -f Dockerfile_k8s` succeeds (local tag
  `skypilot-k8s-ej-prj-3:test`). In it, setuptools is 69.5.1 and click 8.1.8,
  and both pin installs report `Checked 1 package in 1–2ms`, with no download.
  This could **not** be measured on kind: the cluster pulls the published
  image (`imagePullPolicy: Always`), and the new image has to be published
  first.

## 4. Results on kind

4-node `ej/job.yaml`. One warm-up launch, then 3 measured runs (p3r1–p3r3).
The baseline is the 5 runs of `ej-prj-bonus` from
`launch_latency_analysis_4_nodes.md`. Times are in seconds.

| Metric | Baseline median (range) | `ej-prj-3` median (range) | Δ |
|---|---|---|---|
| Head container start → GCS first log | 7.31 (6.90–7.66) | **5.89** (5.67–5.90) | **−1.4** |
| Head container start → head raylet | 11.19 (10.22–11.97) | **9.58** (9.15–9.74) | **−1.6** |
| Head container start → last raylet (whole cluster in Ray) | 12.95 (12.74–15.22; 26.6 outlier) | **11.37** (10.97–11.53) | **−1.6** |
| Runtime setup end → "Cluster launched" | 10.06 | 6.68 (8.37 / 3.23* / 6.68) | −2 to −3.7 |
| Client total, runs without a slow pod | 33.4–35.0 (median 34.2) | **32.0 / 32.4** | **~−2** |
| Client total, run with a slow pod | 51.0 | 40.2 (p3r2*) | (not this change) |

\* p3r2 had a slow pod: one pod's step-2 marker came 16.4 s after its
container started, against ~4 s for the others. Runtime setup waited for it
(14.2 s). By then Ray was already up, so the wait after runtime setup was
only the skylet start (3.2 s).

**Reading the results:**
- **Skipping `ray stop` brings the cluster up ~1.6 s sooner, every run,**
  with tight ranges. It shows up as ~2 s off the launch.
- **The setuptools image change targets the biggest remaining source of
  variance** (the PyPI download before `ray start`: 0.6 s usually, 12 s in the
  traced run). Its effect will only show once the image is published. The
  other remaining PyPI step is the uv dry run; dropping it is idea #5
  (install the wheel offline).
- **Skylet is next.** Once Ray is up ~2 s sooner, the wait after runtime
  setup (3.2–8.4 s) gets close to the skylet start (3.2–4.3 s). Shortening
  Ray further gains less unless skylet also starts earlier.
