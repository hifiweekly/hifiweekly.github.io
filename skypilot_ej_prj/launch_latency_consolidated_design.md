# Design: faster Kubernetes launches for `sky launch` and `sky jobs launch`

This doc consolidates five changes:

| Branch | Change | Layer | Detailed design |
|---|---|---|---|
| `ej-prj-0` | Start Ray in the pod before preparing SkyPilot | Pod template | `ray_first_pod_bootstrap_design.md` |
| `ej-prj-1` | Wait for Ray in one blocking call; start skylet during the wait | Provisioner | `ray_wait_and_skylet_overlap_design.md` |
| `ej-prj-3` | Skip the needless `ray stop`; pre-install the runtime pins in the image | Ray start commands, image | `ray_start_sooner_ej-prj-3.md` |
| `ej-prj-job0` | Fetch status and submitted-at in one call; skip the first status-check sleep when the job already finished; re-check status when the controller logs | Managed-jobs controller, client log follower | `launch_latency_managed_jobs_ej-prj-3.md` §3, §7, §10 |
| `ej-prj-job1` | Look up the enabled storage clouds once per request | Server request (`jobs.launch`) | §12 of this doc |

- **Combined branch:** `ej-prj-3-job1` (renamed from `ej-prj-3-job0`), which has all five. It is local only, not pushed yet.
- **Also in that branch:** `ej-prj-bonus`, the fail-fast change for pods lost during setup (`pod_loss_fail_fast_design.md`). It doesn't change launch latency, so it's only covered where it touches the same code.

## 1. Summary

A multi-node launch on Kubernetes spent most of its time waiting for Ray to come up on every pod. For managed jobs, another 15–35 s went to controller polling.

- The pod-side and provisioner changes (`ej-prj-0`, `ej-prj-1`, `ej-prj-3`) start Ray sooner and notice sooner that it's up.
- The controller change (`ej-prj-job0`) notices sooner that a managed job has started and finished.
- The request change (`ej-prj-job1`) stops the server's `jobs.launch` from running the storage check twice. It saves only about 0.2 s; the first check (4.8 s) remains (§13).

Headline numbers, from kind with 4 nodes running `ej/job.yaml` (details in §7):

| Measure | Before | After |
|---|---|---|
| `sky launch`: runtime setup end → "Cluster launched" | 12.5 s | 9.4 s (`ej-prj-0` + `ej-prj-1`) |
| `sky launch`: provisioning total | 20.9 s | 16.8 s (`ej-prj-0` + `ej-prj-1`) |
| `sky launch`: head container start → whole cluster in Ray | 12.95 s | 11.37 s (adding `ej-prj-3`) |
| `sky jobs launch`: provisioning start → job submitted | 24.9 s | 22.5 s (19.1–20.1 s in fast runs) |
| `sky jobs launch`: job ended → SUCCEEDED (F + G) | 14.3 s (random) | **4.9 s** (steady) |
| `sky jobs launch`: client start → job SUCCEEDED | 59.0 s | **48.4 s** |
| `sky jobs launch`: client total | 61.5 s | **50.1 s** (−11.4 s) |

The `sky jobs launch` rows are medians of an interleaved A/B: 8 runs each of master (`815e61a76`) and `ej-prj-3-job1` (`7aeec8ede`), §13.

## 2. Background: where a launch spent its time

### 2.1 `sky launch` on master

```
CLI ─▶ write config ─▶ create pods ─▶ pods Running ─▶ file mounts ─▶ runtime setup ─▶ wait for Ray ─▶ skylet ─▶ submit job
                                                                                     (61–67% of provisioning)
```

Inside each pod, step 2 of the template's startup script ran in sequence:
1. Install Ray, and the `setuptools<70` and `click<8.3.0` pins.
2. Install SkyPilot from PyPI.
3. Apply the Ray patches.
4. Run `ray stop`.
5. Run `ray start`.

Workers wait for the head's GCS port before they start, so a late head delays the whole cluster.

The provisioner polled for Ray once a second. Each probe was a `kubectl exec` that started three Python interpreters, about 2.3–4 s, so it could learn that Ray was up as much as a full probe late. Skylet started only after that.

### 2.2 `sky jobs launch` on top of that

The jobs controller (in consolidation mode, inside the API server) runs the same `sky.launch` path for each job's cluster. It adds its own steps:

| Phase | Median on master |
|---|---|
| Client start → job row created (the server's `jobs.launch`) | 7.4 s |
| Row created → controller picks the job up | 2.5 s |
| Controller → provisioning starts | 3.1 s |
| **Provisioning → job submitted on the cluster** | 26.8 s |
| Job runs (a one-line `echo` on 4 nodes) | 7.7 s |
| Job finished → controller marks it RUNNING | 3.0 s |
| **RUNNING → SUCCEEDED** | 7.7–15.8 s (random) |
| SUCCEEDED → client exits | 2.6–2.8 s |

## 3. Design

### 3.1 `ej-prj-0`: start Ray before preparing SkyPilot in the pod

**Where:** `sky/templates/kubernetes-ray.yml.j2` (step 2), plus docstrings in `sky/provision/instance_setup.py`.

**Problem.**
- The PyPI install of `skypilot[kubernetes,remote]` sat on Ray's critical path.
- Nothing before `ray start` imports `sky`: the patches and the hostNetwork probe are shipped inline.
- The installed copy is thrown away when `setup_runtime_on_cluster` installs the cluster's own wheel.

**New order of step 2:**

```
ray install (unchanged)
[HA only] uv pip install skypilot[kubernetes,remote]
apply Ray patches (unchanged)
( ray start --head | wait for head port + ray start ) &      # background
[non-HA] uv pip install --dry-run 'skypilot[kubernetes,remote]'   # warms uv's cache
touch /tmp/ray_skypilot_installation_complete               # marker for setup_commands
wait $SKY_RAY_START_PID                                      # step still fails if Ray fails
```

**Key points:**
- **The dry run is needed.** The old install also warmed uv's cache. Without it, the wheel install's dependency resolve took 4.5–6 s instead of about 20 ms.
  - `--dry-run` changes nothing in the environment, so it is safe to run while Ray starts.
  - It is best-effort (`|| echo WARNING`).
- **The marker keeps its meaning:** "setup_commands may install the wheel now". It is written after every step that changes the environment.
- **A Ray start failure still fails the step.** Under `set -e`, the final `wait` propagates Ray's exit code, so `/tmp/runtime-setup.failed` is written and the pod exits, as before.
- **HA controllers keep the PyPI install before `ray start`.** HA recovery imports `sky` before `setup_commands` run again.

### 3.2 `ej-prj-1`: one blocking Ray wait, with skylet started meanwhile

**Where:** `sky/provision/instance_setup.py` (`_WAIT_ALIVE_NODES_PY`, `ray_wait_for_alive_nodes_command`, `RAY_WAIT_DONE_MARKER`) and `sky/provision/provisioner.py` (`wait_for_ray_cluster_healthy`, `_run_alongside`, `parse_ray_probe`).

**Blocking wait.** The provisioner makes one `kubectl exec` that runs a small Python script on the head. The script uses only the standard library and `ray`, and is sent base64-encoded. Each round it:

1. Re-reads the Ray port file.
2. TCP-checks the port.
3. Keeps one `GcsClient` open and counts ALIVE nodes.
4. Sleeps 0.2 s.

It returns when `alive == num_nodes` or the deadline passes.

- **Output** has the same format as the old probe, plus a final `SKY_RAY_WAIT_DONE` line.
- **Missing done line.** Output without it means the exec dropped, not that Ray is down. The call is retried with the time left.
- **Hung exec.** A local timeout of `remaining + 30 s` bounds a hung exec.

**Skylet during the wait.** `_run_alongside(main, background)` runs `start_skylet_on_head_node` in a thread while the main thread waits for Ray.
- The thread gets the provision log path, which is thread-local, and the request's `contextvars`.
- It is joined before returning, and its errors are re-raised.
- Skylet's startup doesn't use Ray, and no job can be submitted until both finish, so starting skylet early is safe.

**Unchanged:**
- the exact-count health rule;
- the 90 s fallback that starts Ray from the provisioner;
- the `no_restart=True` path for workers;
- relaunches of an existing cluster and non-K8s clouds, which still use the single probe.

### 3.3 `ej-prj-3`: start the head's Ray sooner

**Where:** `sky/provision/instance_setup.py` (`RAY_STOP_IF_RUNNING`), `Dockerfile_k8s`, `Dockerfile_k8s_gpu`, and `docs/source/reference/kubernetes/kubernetes-getting-started.rst`.

**Guarded `ray stop`.**
- **The cost:** both start commands ran `ray stop` before `ray start`. On a fresh pod there is nothing to stop, yet it costs a Python start: 1.4–1.5 s idle, 2.9–3.8 s while the pod is busy.
- **The guard:** `RAY_STOP_IF_RUNNING` now runs it only if a GCS, raylet or dashboard process is running.

```bash
if ! command -v pgrep > /dev/null 2>&1 || pgrep -f \
  "ray/core/src/ray/[g]cs/gcs_server|ray/core/src/ray/[r]aylet/raylet|ray/[d]ashboard/" \
  > /dev/null 2>&1; then ray stop; fi;
```

- **It can't match itself.** A bracketed letter means the pattern doesn't match the command line of the shell running it.
- **Without `pgrep`, Ray is always stopped,** as before.
- **Restart paths still stop Ray first,** because Ray is running there.

**Pins in the image.**
- **The cost:** the template pins `setuptools<70` and `click<8.3.0` at pod start, but the image ships setuptools 84. Every pod downloaded setuptools from PyPI before it could start Ray: 0.6 s usually, 12 s in one traced launch.
- **The fix:** the Dockerfiles and the docs' custom-image snippet now install both pins, so a pod only checks them (about 2 ms).
- **It takes effect only after the image is rebuilt and published.**

**Not changed:**
- **The dashboard.** Disabling it saves about 0.9 s on the head's own `ray start`, but not on the GCS port the workers wait for. Users may also rely on it.
- **Baking the Ray patches into the image.** That saves about 1 s, but needs a version stamp.

### 3.4 `ej-prj-job0`: notice a short managed job's end sooner

**Where:**
- `sky/jobs/recovery_strategy.py`: `_wait_until_job_starts_on_cluster` and `pop_job_status_seen_at_launch`.
- `sky/jobs/controller.py`: the first sleep of the monitor loop.
- `sky/jobs/utils.py`: `get_job_status_and_submitted_at`, `_job_status_and_submitted_at_on_head` and `_fetch_job_status_on_head`.
- `sky/skylet/job_lib.py`: `JobLibCodeGen.get_job_status_and_submitted_timestamp`.

**One head command for status and submitted-at.**
- **The cost:** after `sky.launch` returns, the controller confirms that the job started by fetching its status, then its submitted-at time. Without gRPC, each fetch is a `kubectl exec` of about 1.6 s.
- **The fix:** a new codegen prints both payloads from one command. The parser takes them in order.
- **It works on older cluster runtimes.** It calls only `job_lib.get_statuses_payload` and `job_lib.get_job_submitted_or_ended_timestamp_payload`, which today's code already calls on clusters.
- **Fallback:** with gRPC, or if the time is missing, the old separate call is still made.
- **Error handling is unchanged.** `_fetch_job_status_on_head` is shared with `get_job_status`, so transient and fatal errors are told apart exactly as before.

**No first-check sleep when the launch already saw the job finish.**
- **The cost:** the monitor loop sleeps `uniform(0, 15)` s before its first status check, to spread out loops that start together.
- **Why it was wasted:** for short jobs, the launch had already seen the job finish, and threw that status away.
- **The fix:** the executor now remembers the status the launch (or a recovery) saw. If it is terminal, the loop checks immediately.
- **Running jobs keep the jittered first gap.** Checking a running job sooner wouldn't help: the next check is still a full gap later.
- **Executors built without `__init__` are covered.** The remembered status is a class-level default of `None`, so subclasses and tests that skip `__init__` work.
- **Stale statuses are cleared.** The value is reset at the start of every wait, and when the timestamp fetch fails, so it can't carry over from an earlier launch.

## 4. How the changes compose

They change different code and different phases of the launch, so their gains add up:

| Phase | Changed by |
|---|---|
| Server request: storage-cloud lookups in `jobs.launch` | `ej-prj-job1` |
| Pod startup: PyPI install before `ray start` | `ej-prj-0` |
| Pod startup: `ray stop` before `ray start`; setuptools download | `ej-prj-3` |
| Provisioner: noticing Ray is up; skylet start | `ej-prj-1` |
| Controller: confirming the job started | `ej-prj-job0` (status + submitted-at) |
| Controller: noticing the job finished | `ej-prj-job0` (first-check sleep) |

**Interactions:**
- **`ej-prj-0` needs `ej-prj-1` to show.** Alone, its earlier Ray was usually seen on the same 3–5 s probe, so its A/B was not significant (§7.1). With the 0.2 s blocking wait, the gain appears.
- **`ej-prj-3` moves the bottleneck.** Once Ray is up sooner, the skylet start (3–4 s, running alongside the wait) is close to being the critical path. Further Ray-side gains matter less unless skylet also starts earlier.
- **All of the provisioning gains reach managed jobs.** The controller launches job clusters through the same `sky.launch`. In consolidation mode it runs the server's code; otherwise the controller cluster gets a wheel built from the same code.
- **`ej-prj-job0` makes managed-job results comparable.** It removes the controller's random 0–15 s wait, which used to swamp the provisioning gains in the totals.

## 5. Failure handling

| Situation | Behaviour with all four changes |
|---|---|
| `ray start` fails in a pod | The step fails through `wait $SKY_RAY_START_PID`; the pod exits. Same as master. |
| The uv dry run fails (no PyPI) | A warning is logged and the step continues. The wheel install resolves on a cold cache, or fails on its own. |
| A worker never joins within 90 s | The wait returns a partial count; workers are started with `no_restart=True`. Same as master. |
| `kubectl exec` drops during the wait | Missing done line, so the call is retried. It doesn't trigger a Ray restart. |
| Ray is already running when a start command runs | The guard finds it and runs `ray stop`, as before. |
| No `pgrep` in a custom image | The guard always runs `ray stop`, as before. |
| Skylet fails to start | The launch fails once the wait returns. |
| A pod is deleted during setup | `ej-prj-bonus` fails the launch in about 5–11 s instead of about 100 s; managed jobs recover sooner. |
| The combined status fetch errors | Same transient/fatal classification as `get_job_status`; the wait retries. |
| The cluster runtime lacks the submitted-at payload | `submitted_at` is None, and the old separate call is made. |
| The launch saw the job RUNNING | The jittered first check is unchanged. |
| A controller restart resumes jobs | No launch status is remembered, so the jittered first check is unchanged. |

## 6. Compatibility

- **API, client and server:** no change, so there is no `API_VERSION` bump. Everything is server-side or in the cluster's startup script.
- **Existing clusters:** pod arguments apply only to new pods. The controller's combined fetch uses functions every supported runtime has, and falls back if the timestamp is missing.
- **Custom images:**
  - Their dependencies now install in `setup_commands` (`ej-prj-0`), off Ray's critical path.
  - Without the pins pre-installed, they still download setuptools at pod start, which is no worse than before.
  - The docs snippet now suggests installing the pins.
- **HA controllers:** keep the PyPI install before `ray start`.
- **gRPC (`SKYPILOT_ENABLE_GRPC=1`):** unaffected by the combined fetch, which keeps two cheap gRPC calls. gRPC separately saves about 3.5 s per launch in job submission (measured in `launch_latency_analysis_4_nodes.md` §6).

## 7. Results

All measurements are on `kind-skypilot` (one kind node, 12 host cores), with `ej/job.yaml` on 4 nodes. Medians of 5 runs unless stated.

### 7.1 `sky launch` A/B (`bench_launch_latency.py`)

| Candidate | Baseline | Runtime setup end → launched | Provisioning total | Verdict |
|---|---|---|---|---|
| `ej-prj-1` (`532cced15`) | `1f9136510` | 12.47 → **9.89 s** (−21%, p=0.004) | 20.38 → **17.38 s** (−15%) | VALID |
| `ej-prj-0` alone (`9c42bf108`) | `5ef55efa4` | 12.73 → 12.43 s (−2%) | 20.56 → 20.86 s | NOT PROVEN (masked by the old polling) |
| `ej-prj-0` + `ej-prj-1` (`a291dec2a`) | `5ef55efa4` | 12.49 → **9.43 s** (−24%, p=0.004) | 20.85 → **16.77 s** (−20%) | VALID |

### 7.2 `ej-prj-3` on top (pod-side timing, 3 runs against 5)

| Metric | `ej-prj-bonus` (0+1+bonus) | `ej-prj-3` | Δ |
|---|---|---|---|
| Head container start → GCS up | 7.31 s | 5.89 s | −1.4 s |
| Head container start → whole cluster in Ray | 12.95 s | 11.37 s | −1.6 s |
| Client total, runs without a slow pod | 33.4–35.0 s | 32.0 / 32.4 s | about −2 s |

The image pins are not in these numbers: kind pulls the published image.

### 7.3 `sky jobs launch` (consolidation mode, `sky api start --deploy`)

| Phase (median) | master | master (re-run) | job0 only | ej-prj-3 (0+1+bonus+3) | **combined** |
|---|---|---|---|---|---|
| Provisioning → job submitted | 26.8 | 27.2 | 25.5 | 19.2 | **18.7** |
| Job finished → RUNNING | 3.0 | 2.9 | 1.3 | 3.1 | **1.8** |
| RUNNING → SUCCEEDED | 7.7 | 15.8 | 3.4 | 14.5 | **3.4** |
| SUCCEEDED → client exits | 2.8 | 2.6 | 4.6 | 4.5 | 6.7 (2.6–12.3) |
| **Client start → SUCCEEDED** | 60.4 | 70.1 | 55.0 | 59.6 | **45.3** |
| Client total | 62.9 | 72.4 | 58.9 | 63.0 | **57.6** (46.4–67.1) |

- **The provisioning changes save about 8 s per managed job.**
- **`ej-prj-job0` saves 1.2–1.7 s while confirming the job started.** It also reduces RUNNING → SUCCEEDED to a steady 3.4 s, against a random 7.7–15.8 s, so about 7.5 s on average.
- **The controller has the job done by 45 s, against 60–70 s on master.** The client's exit now varies the most: two combined runs exited quickly and finished in 46.4 s and 46.9 s end to end.
- **The first combined run was a warm-up after a server restart:** its submit request took 13.4 s and the controller's launch 9.0 s.

### 7.4 Unit tests

- **New tests by change:**
  - `ej-prj-0`: `test_pod_starts_ray_before_preparing_skypilot`.
  - `ej-prj-1`: 12 tests in `test_provisioner_ray_wait.py`.
  - `ej-prj-3`: 6 tests in `test_ray_stop_if_running.py`.
  - `ej-prj-job0`: 14 tests, including one that runs the generated codegen against a real job table.
- **On `ej-prj-3-job1`:** the jobs, skylet, provision and Kubernetes suites ran 3141 passed and 2 failed.
  - `test_details_is_none_when_no_state_or_failure` also fails on master.
  - `test_pending_volume_probe::...one_cluster_event_per_second` is timing-sensitive; it passes when run alone.
- **Lint:** pylint scores 10.00/10 on the changed source files. The mypy errors that remain are also on master.

## 8. Risks

- **One long-lived exec per launch.** The Ray wait holds a `kubectl exec` open for up to 90 s, and some proxies drop idle streams. Drops are retried; if they turn out to be common, add a heartbeat line.
- **More concurrent work on the head during startup.** Skylet, the uv dry run and Ray all start together. That's fine on the clusters tested, but it could slow pods with less than 1 CPU.
- **PyPI still matters.** The uv dry run is the remaining PyPI step on the pod's path, and the main source of slow pods (one run in five).
- **Package changes on custom images while Ray runs.** Dependencies the wheel adds are now installed after Ray has started. Ray's lazy imports could see a changed package.
- **The guard depends on Ray's process paths.** If Ray renames `gcs_server` or `raylet`, the guard sees no Ray and skips `ray stop` on a restart. A test pins the pattern, and on a fresh pod skipping is correct anyway.
- **The image change needs a rebuild and publish.** Until then it has no effect.
- **The first-check skip is for short jobs only.** A job still running when the launch returns still waits up to 15 s past its end. For long jobs that's a small fraction of the run.

## 9. Remaining work, by expected gain

1. **Client exit after SUCCEEDED (2.6–12.3 s).** Find out whether the client's 5 s status poll or its wait for the persisted logs dominates, then shorten it.
2. **Steady RUNNING → SUCCEEDED (3.4 s).** Reuse the terminal status the launch saw, and fetch the end time with it, so only one head command remains.
3. ~~**Repeated storage checks in `jobs.launch`.**~~ Done in `ej-prj-job1` (§12), about 0.2 s.
3a. **The first storage check in each `jobs.launch` (4.8 s, §13).** Request caches are cleared after every request, so every launch re-runs it. Cache "no storage cloud enabled" across requests with a TTL; it is 70% of phase A.
3b. **Slow runtime setup in about half the runs (§13).** Find why the step from "Setting up SkyPilot runtime" to "Cluster successfully provisioned" sometimes takes 15–18 s instead of about 11 s.
4. **Skylet start (3–4 s).** Now nearly the critical path; start it earlier or make it cheaper.
5. **Drop the uv dry run.** Install the wheel offline, which removes the remaining PyPI step from the pod.
6. **Bake the Ray patches into the image,** with a version stamp (about 1 s).
7. **Turn on gRPC by default for Kubernetes clusters** (about 3.5 s per launch).
8. **Publish the rebuilt image,** then measure the pin change.

## 10. Reproducing

```bash
# sky launch A/B (builds two worktrees, alternates runs)
python ej/bench_launch_latency.py run --candidate-ref ej-prj-3-job1 \
    --task ej/job.yaml --runs 5

# sky jobs launch: restart the server on the code under test, then
sky api stop && sky api start --deploy
for i in 1 2 3 4 5; do sky jobs launch -y -c ej-prj-3 ej/job.yaml; done
# Per-phase times: spot.created_at/start_at/end_at and job_events in
# ~/.sky/spot_jobs.db; cluster_events in ~/.sky/state.db.
```

## 11. Update: client exit fix (`ej-prj-job0` `7c26bd708`; on `ej-prj-3-job1` `f506c2f1c`)

**The problem.** While a job isn't RUNNING, the client's log follower (`_stream_logs_by_id`) re-checks the job's status only every 15 s. After §3.4 shortened RUNNING to about 3.4 s, the client usually missed it, and noticed the job had finished up to 15 s late. That made the total either about 46 s or about 57 s, depending on a sub-second race (`launch_latency_managed_jobs_ej-prj-3.md` §9).

**The fix.** The follower already reads the controller's per-job log every second. It now also re-checks the status as soon as the controller writes a line of its own there. The controller writes such lines right after each status change, and only about 15 while a cluster is being set up, so the extra checks are few. The 15 s check stays as a fallback.

**Result on `ej-prj-3-job1`** (6 clean runs):
- **Client exit:** 2.4–3.0 s after SUCCEEDED, against 1.6–12.6 s before.
- **`sky jobs launch` total:** median 49.0 s; 48.8 s without the warm-up run after the server restart. Master took 62.9–72.4 s.

This removes item 1 of §9.

## 12. Update: storage-cloud lookup once per request (`ej-prj-job1` `d37133a6f`; on `ej-prj-3-job1` `7aeec8ede`)

**The problem.**
- `storage_lib.get_cached_enabled_storage_cloud_names_or_refresh()` lists the clouds whose object storage can be used. It reads the cached enabled clouds for the STORAGE capability, then checks the storage-only providers (R2, CoreWeave, VAST Data, Hugging Face).
- With no storage cloud enabled (here, Kubernetes only), the cached list is empty. `get_cached_enabled_clouds_or_refresh` treats empty as "not checked yet", so every call re-ran the storage check: 0.3–0.7 s warm, about 4.8 s cold. The storage-only provider checks take about 0 ms.
- One `jobs.launch` calls it three times:
  - `sky/jobs/server/core.py`, to decide whether file mounts go through a bucket;
  - `sky/utils/controller_utils.py`, to pick the controller's extra packages;
  - `sky/task.py`, for the preferred store.

**The fix** (`sky/data/storage.py`).
- The lookup moved into `_enabled_storage_cloud_names(workspace, config_fingerprint)`, cached with `annotations.ttl_cache(scope='request', maxsize=16, ttl=60, lock=...)`.
- **Empty answers are cached too.** They are the slow case.
- **Each request still looks again.** Request-scoped caches are cleared when a request ends, so newly added storage credentials are seen by the next request.
- **Workspaces and configs are kept apart.** The key is the active workspace and `check.current_config_fingerprint()` (renamed from `_config_fingerprint`). That matters in a jobs controller, which serves several and never clears request caches. There the 60 s TTL bounds how long a newly enabled storage cloud can go unseen.
- **The re-check is still seen.** `_is_storage_cloud_enabled` runs `sky check` for a missing cloud, then looks again; it now clears the cache in between.
- **Unchanged behaviour:** callers still get their own list, and `raise_if_no_cloud_access` still raises `NoCloudAccessError`. Failed lookups (e.g. an unreadable workspace) are not cached.
- `annotations.ttl_cache` takes an optional `lock`, since cachetools caches aren't thread-safe on their own.

**Tests.** 11 new tests (`tests/unit_tests/test_sky/data/test_enabled_storage_clouds_cache.py`) and 3 for `ttl_cache`. Full unit suite: 10,704 passed; the 20 failures also fail on master or pass when run alone.

**Correction (§13).** The table below compared two sessions, not an A/B, and overstates the gain. A probe of the lookups in an interleaved A/B shows that `jobs.launch` in consolidation mode looks up the storage clouds twice, not three times: 4.8 s, then 0.2 s on master; 4.8 s, then 0.001 s with the cache. The change saves about 0.2 s, which is within phase A's noise. The 7.3 → 6.5 s below was drift between sessions.

**First measurement on `ej-prj-3-job1`** (6 clean runs, jobs 73–78; medians without the warm-up run; superseded by §13):

| Phase | before (`ej-prj-3-job1` without `ej-prj-job1`) | **with `ej-prj-job1`** |
|---|---|---|
| **Client start → job row created** | 7.3 s (6.8–8.5) | **6.5 s (6.5–7.2)** |
| Row → STARTING | 3.1 s | 2.4 s |
| STARTING → RUNNING | 31.6 s | 31.4 s |
| RUNNING → SUCCEEDED | 3.4 s | 3.4 s |
| SUCCEEDED → client exits | 2.6 s | 2.2 s |
| **Client total** | 48.8 s | **46.9 s** |

- That session's request time dropped about 0.8 s, but §13 shows most of it was drift.

This removes item 3 of §9. The first lookup (item 3a) is the bigger target.

## 13. A/B against master (all five changes)

**Method.**
- **Arms:** master `815e61a76` against `ej-prj-3-job1` `7aeec8ede` (0 + 1 + 3 + bonus + job0 + job1).
- **Order:** 4 alternating blocks (master, branch, master, branch). Each block switched the code, restarted the API server with `sky api start --deploy`, then ran 1 warm-up run (dropped) and 4 measured runs. That gives 8 measured runs per arm (jobs 79–98).
- **Runs:** `sky jobs launch -y -c ej-prj-3 ej/job.yaml`, one at a time, each verified as the only managed job (`trace_job2.py`). All 20 succeeded.
- **Phases:** from `spot.created_at`/`start_at`/`end_at`, `job_events`, the cluster's "Provisioning on" event, and the client's exit time.

**Result** (median, with the range of the 8 runs):

| Phase | master | **`ej-prj-3-job1`** | Δ | Changed by |
|---|---|---|---|---|
| A: client start → job row created | 6.8 (6.4–7.0) | 6.8 (6.5–7.2) | 0 | `ej-prj-job1` (≈0.2 s, see below) |
| B: row → STARTING | 2.7 (2.3–3.1) | 2.5 (2.1–4.5) | −0.2 | — |
| C: STARTING → provisioning | 3.0 (2.4–3.4) | 3.1 (2.4–3.4) | 0 | — |
| D: provisioning → job submitted | 24.9 (24.0–27.8) | **22.5** (19.1–25.4) | −2.4 | `ej-prj-0`, `-1`, `-3` |
| E: job runs | 7.6 (7.4–8.2) | 7.7 (7.3–8.2) | 0 | — |
| F: job ended → RUNNING | 3.2 (2.4–3.6) | **1.4** (1.0–1.9) | −1.8 | `ej-prj-job0` |
| G: RUNNING → SUCCEEDED | 11.1 (3.7–16.1) | **3.5** (3.4–3.6) | −7.6 | `ej-prj-job0` |
| H: SUCCEEDED → client exits | 2.8 (1.4–6.7) | **2.5** (1.3–2.9) | −0.3 | `ej-prj-job0` (client fix) |
| **Client start → SUCCEEDED** | 59.0 (51.4–64.4) | **48.4** (43.7–51.1) | −10.6 | |
| **Client total** | 61.5 (58.1–67.4) | **50.1** (46.1–54.0) | **−11.4** | |

**Findings.**
- **The controller changes give the most, and the steadiest, gain.** RUNNING → SUCCEEDED went from a random 3.7–16.1 s to 3.4–3.6 s in every run. Every branch run beat every master run on F and G.
- **The client exit no longer varies:** 1.3–2.9 s against 1.4–6.7 s.
- **Provisioning gains show in only half the runs today.**
  - The 4 fast branch runs (jobs 85, 95, 97, 98) submitted the job 19.1–20.1 s after provisioning started.
  - The other 4 (jobs 86, 87, 88, 96) took 24.9–25.4 s, like master.
  - In the slow runs, the step from "Setting up SkyPilot runtime" to "Cluster successfully provisioned" took 15–18 s, against about 11 s in the fast runs and 16 s on master. Not root-caused yet (§9 item 3b). Earlier sessions saw this in about 1 run in 5, attributed to the pod's PyPI dry run.
- **`ej-prj-job1` doesn't show in phase A.** A probe that logged each storage lookup (3 runs per arm, jobs 99–104; temporary, not committed) found:

  | | 1st lookup (`jobs/server/core.py`) | 2nd lookup (`controller_utils.py`) |
  |---|---|---|
  | master | 4.76–4.85 s | 0.18–0.25 s |
  | `ej-prj-3-job1` | 4.73–4.80 s | 0.001–0.002 s |

  - In consolidation mode, `task.py`'s preferred-store lookup isn't reached, so there are 2 lookups, not 3. The change saves about 0.2 s.
  - **The first lookup costs 4.8 s of phase A's 6.8 s on both arms,** because request caches are cleared after every request. That is the larger target (§9 item 3a).
- **Totals against earlier sessions.** Master took 61.5 s here, against 62.9 s and 72.4 s in earlier sessions; G's randomness explains most of that spread. The branch took 50.1 s, against 46.9–49.0 s earlier; the slow provisioning runs explain the difference.

