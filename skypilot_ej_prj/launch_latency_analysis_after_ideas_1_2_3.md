# Launch latency after ideas #1, #2 and #3

- **Branch:** `ej-prj-bonus` (`6db5eba63`). It contains idea #1 (pods start
  Ray first), idea #2 (one blocking Ray wait), idea #3 (skylet started during
  the wait), and the fail-fast change for deleted pods.
- **Task:** `ej/job.yaml`, 6 nodes on `kind-skypilot` (one kind node, 12 host
  cores).
- **Runs:** `ej-analyze` (provision log only) and `ej-analyze2` (also traced
  with `SKYPILOT_TIMELINE_FILE_PATH`, plus timestamps of the bootstrap marker
  files and Ray's first log lines in every pod).

## 1. Where the 45.5 s go (`ej-analyze2`, as the client sees it)

| # | Phase | Time | Evidence |
|---|---|---|---|
| 1 | CLI start, `/validate`, request queued | 3.7 s | Client start 21:35:41.9; `execution.launch` starts ~21:35:45.6. The optimizer itself takes 0.06 s |
| 2 | `write_cluster_config` | **3.0 s** | Timeline span. Not instrumented inside |
| 3 | Create pods, wait until all are Running | 5.7 s | `_create_pods` 5.68 s. Each pod sits in `Pulling` (from `imagePullPolicy: Always`) for 1–6 s |
| 4 | `internal_file_mounts` | 1.3 s | |
| 5 | `setup_runtime_on_cluster` | **14.3 s** | Waits for the slowest pod's step-2 marker. worker4 wrote it 15.3 s after its container started; the other pods took 2.6–7.0 s (table below) |
| 6 | Skylet start, with the Ray wait alongside | 3.3 s | Ray was already up when runtime setup ended, so this run waited on skylet, not Ray |
| 7 | Job submission: `add_job` + `queue_job` | **3.3 s** | Two `run_on_head` calls of 1.64 s each (`kubectl exec` + Python start). gRPC is off by default (`SKYPILOT_ENABLE_GRPC=False`) |
| 8 | Driver `ray.init`, placement group, first rank starts | ~3.7 s | Job dir created 16.8, driver connected to Ray 19.5, tasks started 22.1 |
| 9 | Ranks starting one after another | 2.1 s | rank 0 at 22.07, last rank at 24.15 |
| 10 | Client log tail ends, status | ~3.2 s | `run.log` last written 24.19; client returned 27.39 |

The first run (`ej-analyze`) had no slow pod: runtime setup took 1.7 s. There,
**Ray** was the critical path instead: all 6 nodes joined 12.2 s after runtime
setup ended. Total time was the same, 45.6 s. Which step is the bottleneck
varies from run to run, so the fixes below target all of them.

### Per-pod bootstrap (`ej-analyze2`, UTC, from file times inside each pod)

| Pod | Container start | Step-2 marker (Ray installed + uv cache warmed) | Wheel installed | Ray: first log line |
|---|---|---|---|---|
| head | 35:54 | 35:58.18 (+4.2 s) | 35:58.67 | GCS 36:01.78, raylet 36:06.09 |
| worker1 | 35:52 | 35:55.17 (+3.2 s) | 35:57.79 | raylet 36:09.50 |
| worker2 | 35:53 | 36:00.81 (+7.0 s) | 36:01.09 | raylet 36:07.87 |
| worker3 | 35:51 | 35:54.62 (+2.6 s) | 35:57.68 | raylet 36:07.95 |
| worker4 | 35:55 | **36:10.53 (+15.3 s)** | 36:11.19 | raylet 36:09.49 |
| worker5 | 35:53 | 35:56.53 (+3.5 s) | 35:57.72 | raylet 36:07.55 |

Observations:
- **The straggler is the uv-cache dry run.** It resolves
  `skypilot[kubernetes,remote]` against PyPI on all six pods at once.
  worker4's Ray was running (raylet at 36:09.5) before its own marker
  (36:10.5), so the slow part ran after `ray start` began, and that is the dry
  run. Runtime setup waits on that marker, so one slow PyPI round trip delays
  the whole launch.
- **Ray on the head is slow to start.** About 3.6 s pass from the marker to
  the GCS's first log line. That includes `ray stop` on a pod that has never
  run Ray. Another 4.3 s pass before the raylet, while the dashboard and agents
  start. Workers can only start Ray once the GCS port is up, and their raylets
  then take another 6–8 s on a busy kind node.
- **Six pods start Ray, uv and Python at the same moment on one node,** so
  kind overstates contention. A real multi-node cluster will be less noisy,
  but the order of the steps is the same.

## 1b. Four traced runs (runs 2–5)

Runs 3–5 used the same task, branch and tracing as run 2. Each was torn down
before the next one started. Times are in seconds.

| Metric | run2 | run3 | run4 | run5 | median | range |
|---|---|---|---|---|---|---|
| **Client total** | 45.5 | 43.5 | 39.8 | 44.2 | **43.8** | 39.8–45.5 |
| CLI start → launch starts | 3.7 | 3.8 | 1.5 | 1.6 | 2.6 | 1.5–3.8 (slower right after a server restart) |
| `write_cluster_config` | 3.0 | 3.0 | 2.7 | 2.7 | **2.9** | 2.7–3.0 |
| Create pods, wait for Running | 5.7 | 5.7 | 5.7 | 5.9 | **5.7** | 5.7–5.9 |
| `internal_file_mounts` | 1.3 | 1.4 | 1.3 | 1.2 | 1.3 | 1.2–1.4 |
| `setup_runtime_on_cluster` | 14.3 | 11.2 | 2.7 | 13.6 | 12.4 | **2.7–14.3** |
| Runtime setup end → "Cluster launched" (Ray wait ∥ skylet) | 3.4 | 4.1 | 12.4 | 6.1 | 5.1 | **3.4–12.4** |
| ↳ skylet start (runs alongside) | 3.3 | 4.1 | 4.1 | 4.1 | 4.1 | 3.3–4.1 |
| **Runtime setup + Ray wait** | 17.7 | 15.3 | 15.1 | 19.7 | **16.5** | 15.1–19.7 |
| Job submission (2 × `run_on_head`) | 3.3 | 3.3 | 3.3 | 3.2 | **3.3** | 3.2–3.3 |
| "Cluster launched" → first rank starts | 7.4 | 7.4 | 7.5 | 7.1 | **7.4** | 7.1–7.5 |
| First → last rank | 2.1 | 2.5 | 2.1 | 2.2 | 2.2 | 2.1–2.5 |
| Last rank → client returns | 3.2 | 3.1 | 3.1 | 3.0 | 3.1 | 3.0–3.2 |

Pod side (per run: slowest pod and median pod):

| Metric | run2 | run3 | run4 | run5 |
|---|---|---|---|---|
| Container start → step-2 marker, **slowest pod** | 15.3 (w4) | 15.1 (w5) | 6.5 (w3) | 15.9 (w3) |
| Container start → step-2 marker, median pod | ~3.9 | 4.1 | 3.5 | 4.1 |
| Head: marker → GCS first log | 3.6 | 3.9 | 3.4 | 3.9 |
| Head: GCS → raylet | 4.3 | 4.3 | 4.6 | 3.1 |
| Head GCS up → last raylet | 7.7 | 6.3 | 6.8 | 14.2 |

What is consistent:
- **The setup stretch (runtime setup + Ray wait) is ~15–20 s and varies.**
  Most of it is on the pods, not in SkyPilot's server.
  - In 3 of 4 runs, one pod's step-2 marker came ~15 s after its container
    started, against ~4 s for the median pod. Runtime setup waits for every
    pod's marker, so that slowest pod sets the pace. The dry run that warms
    the uv cache (a PyPI resolve on every pod at once) is the only network
    step there.
  - When no pod straggles (run 4), the Ray wait takes over (12.4 s). Fixing
    only one of the two moves the bottleneck to the other.
- **After "Cluster launched", 12–13 s go to the job every time:** 7.4 s
  until the first rank starts, 2.2 s until the last, 3.1 s until the client
  returns. That's as much as provisioning (2.9 + 5.7 + 1.3 s) and almost
  no variance, which makes it the most predictable gain. Of the 7.4 s,
  3.3 s are the two `kubectl exec` calls that submit the job, since gRPC is
  off by default. The other ~4 s are the driver connecting to Ray, the
  STRICT_SPREAD placement group, and cold Ray worker processes.
- **The fixed provisioning costs are stable:** `write_cluster_config`
  2.7–3.0 s, pods 5.7–5.9 s, file mounts 1.2–1.4 s.
- **The head's Ray start is consistent:** 3.4–3.9 s to the GCS (that includes
  `ray stop` on a new pod), then 3.1–4.6 s to the raylet.

## 2. What else could be done (ranked by expected gain; see 1b for the data)

| # | Idea | Phase | Expected gain | Effort / risk |
|---|---|---|---|---|
| 1 | **Install the wheel offline first.** In `SKYPILOT_WHEEL_INSTALLATION_COMMANDS`, try `uv pip install --offline <wheel>[...]` and fall back to the online install. Then drop the step-2 dry run | 5 | Removes the straggler (~10 s in run 2), ~1–1.5 s of CPU and network per pod, and the per-launch dependency on PyPI | Low. The default image already has all 107 dependencies, and the online fallback keeps custom images working |
| 2 | **Submit the job over gRPC.** Turn on `SKYPILOT_ENABLE_GRPC` (find out why it's off by default), or merge `add_job` and `queue_job` into one exec | 7 | ~3 s (two `kubectl exec` + Python starts → two gRPC calls of a few ms), on every launch and every `sky exec` | Low–medium: the gRPC path exists but is behind a flag |
| 3 | **Start the head's Ray sooner.** Skip `ray stop` on a freshly booted pod (`pgrep -f raylet && ray stop`). Bake the `setuptools<70` / `click` pins and the Ray patches into the image (the rest of idea #1). Check whether SkyPilot needs the Ray dashboard; if not, pass `--include-dashboard=false` | Ray (run 1) | 2–4 s when Ray is the critical path. Workers can only start Ray after the GCS is up, so every second saved on the head is saved on every node | Medium: the dashboard question needs checking; the image change needs a rebuild |
| 4 | **Start skylet earlier or faster.** Start skylet in the same exec as the head's runtime setup, or from the pod bootstrap right after the wheel install. Don't wait for gRPC readiness in that exec | 6 | ~2–3 s when Ray is already up (run 2) | Medium: skylet has to survive the provisioner's restart logic and version checks |
| 5 | **Resolve the image tag to a digest once on the server,** then use `image@sha256:...` with `imagePullPolicy: IfNotPresent` | 3 | 1–4 s of pod wait on kind (each pod's `Pulling` step makes a registry round trip). It also guarantees every pod runs the same image | Low–medium: needs a registry lookup with a cache, and a fallback to `Always` |
| 6 | **Profile `write_cluster_config`** (3.0 s). The likely cost is Kubernetes API calls building the deploy variables (contexts, nodes, labels); cache them per context with a short TTL | 2 | 1–2 s | Low to measure. The fix depends on what the profile shows |
| 7 | **Start the job's ranks sooner.** The driver's `ray.init` plus a STRICT_SPREAD placement group plus cold Ray workers take ~3.7 s before rank 0, and ranks start 2.1 s apart. Options: prestart a Python worker per node during setup; reuse a warm driver | 8–9 | 2–4 s | Medium–high: Ray internals |
| 8 | **Client side:** CLI start plus `/validate` (~1.7 s before the first request), and the log tail ending ~3 s after the job finished | 1, 10 | 1–3 s of perceived latency | Low–medium |

Rough total if 1–6 land: ~10–15 s off the ~45 s launch on kind. Ideas 1 and 2
alone are the cheapest and remove the most variance.

## 3. Suggested next step

Ideas 1 and 2 are small, separate changes with clear before/after metrics.
Measure each with the A/B benchmark
(`tests/kubernetes/launching_benchmarks/bench_launch_latency.py`), using
`ej-prj-bonus` as the baseline. The benchmark's current metrics cover up to
"Cluster launched". To capture ideas 2, 7 and 8, add a metric for the time
until the first rank starts (the task already prints each rank's start
time).
