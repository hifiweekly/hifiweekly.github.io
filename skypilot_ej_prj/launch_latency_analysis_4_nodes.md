# Launch latency: 4-node `ej/job.yaml` on kind (5 runs)

- **Branch:** `ej-prj-bonus` (`6db5eba63`). It contains idea #1 (pods start
  Ray first), idea #2 (one blocking Ray wait), idea #3 (skylet started during
  the wait), and the fail-fast change for deleted pods.
- **Task:** `ej/job.yaml` with `num_nodes: 4` (1 CPU, 1 GB each), on
  `kind-skypilot` (one kind node, 12 host cores).
- **Method:** 5 launches in a row (`ej-n4r1`–`ej-n4r5`). Each was torn down
  before the next one started. The API server ran with
  `SKYPILOT_TIMELINE_FILE_PATH`. For each run I collected:
  - the server's timeline spans;
  - the provision log;
  - the client's start and end times;
  - the rank start times the task prints;
  - inside each pod: the container start, the time the bootstrap step-2
    marker and the wheel were written, and the first log lines of Ray's GCS
    and raylets.

  The scripts are `run_one.sh` and `analyze_runs.py` in the session scratchpad.

## 1. Results

Times are in seconds. "Steady" is the median of runs 2–5. Run 1 is an outlier
(section 3).

| Metric | r1 | r2 | r3 | r4 | r5 | **steady median** | steady range |
|---|---|---|---|---|---|---|---|
| **Client total** | 51.0 | 33.4 | 33.5 | 34.2 | 35.0 | **33.9** | 33.4–35.0 |
| CLI start → launch starts | 3.8 | 1.6 | 1.5 | 1.5 | 1.6 | 1.5 | 1.5–1.6 |
| `write_cluster_config` | 3.0 | 2.7 | 2.7 | 2.7 | 2.7 | 2.7 | 2.7–2.7 |
| Create pods, wait for Running | 4.6 | 4.4 | 4.5 | 4.4 | 3.4 | 4.4 | 3.4–4.5 |
| `internal_file_mounts` | 0.9 | 1.0 | 0.9 | 1.0 | 1.0 | 1.0 | 0.9–1.0 |
| `setup_runtime_on_cluster` | **20.6** | 1.6 | 1.1 | 1.6 | 2.1 | 1.6 | 1.1–2.1 |
| Runtime setup end → "Cluster launched" (Ray wait) | 5.0 | 9.8 | 10.7 | 10.1 | 10.8 | **10.4** | 9.8–10.8 |
| ↳ skylet start, running alongside | 3.4 | 3.8 | 3.7 | 3.7 | 4.6 | 3.8 | 3.7–4.6 |
| Job submission (2 × `run_on_head`) | 3.4 | 3.4 | 3.4 | 3.6 | 3.3 | 3.4 | 3.3–3.6 |
| "Cluster launched" → first rank starts | 7.2 | 7.1 | 7.0 | 7.3 | 7.2 | **7.1** | 7.0–7.3 |
| First → last rank | 1.8 | 1.8 | 1.7 | 1.7 | 1.8 | 1.8 | 1.7–1.8 |
| Last rank → client returns | 2.4 | 2.5 | 2.5 | 3.0 | 3.3 | 2.7 | 2.5–3.3 |

Pod side:

| Metric | r1 | r2 | r3 | r4 | r5 | steady median |
|---|---|---|---|---|---|---|
| Container start → step-2 marker, **slowest pod** | **25.3** (w1) | 4.2 | 3.5 | 4.0 | 4.3 | 4.1 |
| Container start → step-2 marker, median pod | 4.1 | 3.6 | 3.2 | 3.5 | 3.6 | 3.5 |
| Head: marker → GCS first log | 3.6 | 3.7 | 3.8 | 3.8 | 4.2 | 3.8 |
| Head: GCS → raylet | 3.3 | 4.0 | 3.7 | 3.9 | 4.3 | 4.0 |
| Head GCS up → last raylet | 19.8 | 5.5 | 5.5 | 5.6 | 7.6 | 5.5 |
| Runtime setup end → last raylet | 2.8 | 7.4 | 8.0 | 7.6 | 8.5 | 7.8 |

## 2. Where a steady launch spends ~34 s

```
CLI + validate        1.5  ▇
write_cluster_config  2.7  ▇▇
pods Running          4.4  ▇▇▇▇
file mounts           1.0  ▇
runtime setup         1.6  ▇
Ray wait (∥ skylet)  10.4  ▇▇▇▇▇▇▇▇▇▇      <- critical path: Ray
job submit (exec ×2)  3.4  ▇▇▇
driver + PG → rank 0  3.7  ▇▇▇▇
ranks spread          1.8  ▇▇
client log tail       2.7  ▇▇▇
                     ----
                     33.9
```

Three things stand out:

1. **Ray is the critical path in every steady run.** Runtime setup finishes
   7.4–8.5 s before the last worker's raylet starts, so the provisioner then
   waits ~10 s for Ray. Skylet (3.8 s) is fully hidden behind that wait. From
   a pod's container start:
   - ~3.5 s to the step-2 marker (Ray installed and patched, uv cache warmed);
   - ~3.8 s more until the head's GCS starts logging. That includes `ray stop`
     on a pod that has never run Ray, and the start of `ray start --head`;
   - ~4.0 s more until the head's raylet;
   - workers can only start Ray once the GCS port is up, and their raylets
     come ~5.5 s after the GCS.

   So the head takes ~11 s from container start to its raylet, and the whole
   cluster ~13 s until every node has joined.
2. **After "Cluster launched", ~12.3 s go to the job, with almost no
   variance:**
   - 3.4 s submitting it with two `kubectl exec` calls (gRPC is off by
     default);
   - ~3.7 s for the driver to connect to Ray, create the STRICT_SPREAD
     placement group, and start a cold Ray worker for rank 0;
   - 1.8 s until the last rank starts;
   - ~2.7 s until the client's log tail notices the job finished.

   That's more than pod creation, file mounts and runtime setup put together.
3. **The fixed costs are tight:** `write_cluster_config` 2.7 s and pods 4.4 s
   (each pod still checks the registry for its image) are the same in every
   run.

## 3. The outlier (run 1: 51.0 s, +17 s)

- `worker1` took **25.3 s** to write its step-2 marker; the other pods took
  ~4 s. Its raylet also came late (19.8 s after the head's GCS). Runtime setup
  waits for every pod's marker, so it took 20.6 s, and that delay accounts for
  the extra ~17 s.
- Only the dry run, which warms uv's cache with a PyPI resolve, depends on
  the network in that step. That makes it the likely cause, as in the 6-node
  runs.
- It was also the first launch after the API server restarted. That explains
  the 3.8 s CLI-to-launch time, but not the slow pod: pod bootstrap doesn't
  depend on the server.
- **4 nodes compared with 6 nodes:** a slow pod showed up in 1 of 5 runs here,
  against 3 of 4 runs with 6 nodes. Fewer pods starting at once means less
  contention for CPU and network, so it happens less often, but it still
  happens.

## 4. What else could be done, for this workload

| # | Idea | Phase it shortens | Expected gain (steady) |
|---|---|---|---|
| 1 | **Start the head's Ray sooner:** skip `ray stop` on a freshly booted pod; bake the `setuptools<70` / `click` pins and the Ray patches into the image; turn off the Ray dashboard if SkyPilot doesn't use it | Ray wait (critical path, 10.4 s) | 2–4 s. Every second on the head is a second for the workers too, since they wait for its GCS port |
| 2 | **Submit the job over gRPC** (`SKYPILOT_ENABLE_GRPC`, or make it the default), or merge `add_job` and `queue_job` into one exec | Job submit (3.4 s) | ~3 s, every launch and every `sky exec` |
| 3 | **Start the job's ranks sooner:** prestart a Python Ray worker on each node during setup; look at the driver's `ray.init` and the STRICT_SPREAD placement group | Driver + placement group (3.7 s) + rank spread (1.8 s) | 2–4 s |
| 4 | **Shorten the client's log-tail lag** once the job ends (poll more often, or end the stream on job completion) | Client tail (2.7 s) | ~2 s you'd notice |
| 5 | **Install the wheel offline first** and drop the dry run | Rare slow pod (+17 s in 1 of 5 runs) | Removes the tail risk and the per-launch PyPI dependency |
| 6 | Profile `write_cluster_config`; pin the image by digest with `IfNotPresent` | 2.7 s + 4.4 s | 1–3 s |

**Rough effect if 1–4 land:** ~9–13 s off a ~34 s launch, which puts a 4-node
launch on kind at around 21–25 s. Idea 5 keeps the occasional ~50 s launch
from happening.

**Interaction with skylet.** Once idea 1 shortens the Ray wait to about the
skylet start (3.8 s), skylet becomes the bottleneck. Starting skylet earlier
(from the pod's own bootstrap, or in the same exec as runtime setup) is the
follow-up that keeps the gain.

## 5. Caveats

- One kind node runs all 4 pods, so they compete for CPU and network. On a
  real multi-node cluster, Ray start and the dry run should be faster and less
  noisy, and pod creation includes scheduling and image pulls on separate
  nodes. Confirm the ranking there before investing in ideas 1 and 6.
- 5 runs (4 steady) is enough to rank the phases, because the steady
  spreads are small. It's not enough to estimate how often the slow-pod tail
  happens.

## 6. Experiment: job submission over gRPC (idea #2)

**Setup:** same branch, task and method as above. The only change is
`SKYPILOT_ENABLE_GRPC=1`, set on the API server (and the client). I ran one
untimed warm-up launch after the server restart, then 3 measured runs
(`ej-g1`–`ej-g3`).

**gRPC was really used.** The trace of every gRPC run has **no `run_on_head`
calls** (each baseline run has 2: `add_job` and `queue_job`). Neither the
request logs nor the server log mention `gRPC failed` or `falling back to SSH`.

| Metric (s) | Baseline: steady median (runs 2–5, §1) | gRPC: g1 | g2 | g3 | gRPC median | Δ median |
|---|---|---|---|---|---|---|
| `Backend.execute` (job submission) | 3.68 | 1.36 | 0.48 | 0.41 | **0.48** | **−3.2** |
| "Cluster launched" → first rank | 7.15 | 5.32 | 4.25 | 4.13 | **4.25** | **−2.9** |
| First → last rank | 1.80 | 1.90 | 1.77 | 1.82 | 1.82 | 0 |
| Last rank → client returns | 2.51 | 2.36 | 2.02 | 2.23 | 2.23 | −0.3 |
| **"Cluster launched" → client returns** | 11.7 | 9.6 | 8.0 | 8.2 | **8.2** | **−3.5** |
| Client total | 33.9 | 36.2 | 39.5 | 29.9 | 36.2 | (see below) |
| `setup_runtime_on_cluster` | 1.6 | 10.2 | 11.1 | 2.1 | 10.2 | (slow pod, unrelated) |

What it shows:

- **Job submission drops from ~3.7 s to ~0.5 s.** The two `kubectl exec` +
  Python round trips become two gRPC calls to skylet. g1 took 1.4 s,
  probably the cost of opening the skylet tunnel to a new cluster for the
  first time. Every launch is a new cluster, so expect that cost on some
  launches.
- **The time from "Cluster launched" until the client returns falls by
  ~3.5 s,** from 11.7 to 8.2 s. A little of that (~0.3 s) comes after the
  job, presumably because the client's job-status checks also go over gRPC.
- **Client totals are noisy for an unrelated reason.** g1 and g2 hit the
  slow-pod problem (one pod took 11–15 s to finish its bootstrap, so runtime
  setup took 10–11 s). That happens before any job is submitted, so gRPC
  can't affect it. g3 had no slow pod and finished in **29.9 s**, against
  33.4–35.0 s for the steady baseline runs: about −4 s, which matches the
  per-phase numbers.
- **The slow pod appeared in 2 of 3 runs here,** against 1 of 5 in the
  baseline. With so few runs that's noise, but it confirms that slow pods
  (idea #5: install offline, drop the dry run) are now the main source of
  variance.

**Conclusion:** turning on gRPC saves ~3–3.5 s on every launch (and on every
`sky exec`, which uses the same path), with no fallbacks seen. Before making
it the default, find out why `SKYPILOT_ENABLE_GRPC` is off by default (search
git history and issues), and test clusters whose skylet tunnel can't be
reached, to check that the fallback to SSH still works.
