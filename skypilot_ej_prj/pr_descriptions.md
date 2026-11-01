# PR descriptions

One section per branch on `fork` (https://github.com/siliconej/skypilot). Copy everything under a branch's heading into its PR.

**Stacking.** `ej-prj-bonus` contains `ej-prj-0` and a rebased `ej-prj-1`, and `ej-prj-3` is built on `ej-prj-bonus`. Open them in this order, or rebase each one after the previous one merges:

`ej-prj-0` → `ej-prj-1` → `ej-prj-bonus` → `ej-prj-3`

`ej-prj-job0`, `ej-prj-job1` and `ej-k8s-pool-warning` branch directly off master and are independent.

**Test environment for every number below:**
- **Cluster:** `kind-skypilot`, one kind node with 12 cores.
- **Task:** `ej/job.yaml`, a one-line `echo` on 4 nodes unless stated.
- **Managed jobs:** consolidation mode (`sky api start --deploy`).

---

## `ej-prj-0`: [Kubernetes] Start Ray before preparing SkyPilot in the pod

Commit: `9c42bf108`

### Summary

- **Problem:**
  - The Kubernetes pod's startup script installed `skypilot[kubernetes,remote]` from PyPI before `ray start`. The head started Ray late, and every worker, which waits for the head's GCS port, started late with it.
  - Nothing before `ray start` imports `sky`: the Ray patches and the hostNetwork probe are shipped inline. `setup_commands` also replaced that copy with the cluster's own wheel later.
- **Fix:**
  - Drop the PyPI install from the non-HA path and start Ray in the background.
  - Meanwhile, warm uv's cache with `uv pip install --dry-run`. It changes nothing in the environment, but without it the later wheel install resolves on a cold cache, about 5 s instead of about 20 ms.
- **Unchanged:**
  - The marker `setup_commands` waits on is still written before the wheel install.
  - The step still waits for Ray (`wait $SKY_RAY_START_PID`) and fails if Ray fails.
  - HA controllers keep the PyPI install, because their recovery path imports `sky` on a restarted pod.

Files: `sky/templates/kubernetes-ray.yml.j2`, `sky/provision/instance_setup.py` (docstrings), and a new template test.

### Tested

- [x] Code formatting: `yapf --diff` and `isort --check` are clean on the changed files. Pylint on `sky/` adds no messages over master. mypy has no errors beyond master's 2 in `fetch_nebius.py`.
- [x] Any manual or new tests for this PR (please specify below)
  - **New unit test:** `test_pod_starts_ray_before_preparing_skypilot` in `tests/unit_tests/test_sky/clouds/test_kubernetes_ray_template_snapshot.py`, for the `base_cpu`, `multinode` and `high_availability` cases. It checks:
    - the PyPI install is there only for HA, and the dry run only for non-HA;
    - the order: Ray install → patches → `ray start` → dry run → marker → `wait`.
    - It fails on master (2 of 3 cases) and passes here. The snapshot test file passes: 88 passed.
  - **Launch latency A/B** (`tests/kubernetes/launching_benchmarks/bench_launch_latency.py`, from `ej-prj-1`; interleaved runs with a permutation test). Runtime setup end → "Cluster launched":

    | Variant | Before | After | Change | Result |
    |---|---|---|---|---|
    | This change alone | 12.73 s | 12.43 s | −2% | Not significant: the old 1 s polling hides the earlier Ray |
    | With `ej-prj-1` | 12.49 s | **9.43 s** | **−24%, p = 0.004** | Significant. Provisioning total 20.85 → 16.77 s |
- [ ] All smoke tests: `/smoke-test` (CI) or `pytest tests/test_smoke.py` (local)
- [x] Relevant individual tests: `/smoke-test -k test_name` (CI) or `pytest tests/test_smoke.py::test_name` (local)
  - These ran locally with `--kubernetes` on kind, on the combined branch (this PR plus `ej-prj-1`, `ej-prj-bonus`, `ej-prj-3` and `ej-prj-job0`). All passed:
    - `test_basic.py::test_minimal`, `test_launch_fast`
    - `test_cluster_job.py::test_multi_hostname`, `test_multi_node_failure`, `test_kubernetes_non_debian_image[rockylinux:9-dnf-2]`, `test_kubernetes_non_debian_image[redhat/ubi9-dnf-2]`
    - `test_kubernetes_host_network.py::test_kubernetes_host_network_multi_node_same_node`
    - `test_images.py::test_kubernetes_default_image_no_conda`, `test_kubernetes_docker_image_and_ssh`
    - `test_setup.py::test_kubernetes_post_provision_runcmd`
- [ ] Backward compatibility: `/quicktest-core` (CI) or `pytest tests/smoke_tests/test_backward_compat.py` (local)
  - Not run. Only new pods get the new script; there's no API change.

---

## `ej-prj-1`: [Kubernetes] Wait for Ray on the head in one call and start skylet meanwhile

Commits: `532cced15`, plus the benchmark `0392f886f` and `be7a199f8`. In the stacked branches these are rebased as `709688366`, `b889d71af` and `a291dec2a`.

### Summary

- **Problem:**
  - On a freshly booted Kubernetes cluster, the provisioner polled the head once a second until every pod's Ray had joined. Each poll was a `kubectl exec` that started three Python interpreters (2–4 s), so it could see readiness up to a whole poll late.
  - The skylet only started after the wait, although it needs only the head.
- **Fix:**
  - **One blocking wait.** `ray_wait_for_alive_nodes_command` makes one remote call that keeps a single `GcsClient` open on the head and checks every 0.2 s, until exactly `num_nodes` are alive or the deadline passes. Output keeps the old probe's format, plus a done marker.
  - **A dropped call is retried.** A call that ends without the done marker means the exec dropped, not that Ray is down, so it is retried for the time left. It doesn't trigger a Ray restart.
  - **Skylet starts meanwhile.** It runs in a background thread during the wait (`_run_alongside`), carrying over the thread-local provision log path and the context variables. It is joined before returning, and its errors are re-raised.
- **Unchanged:**
  - the 90 s fallback;
  - `no_restart` for workers;
  - re-launches of existing clusters and other clouds, which keep the single probe.
- **Also:** adds `tests/kubernetes/launching_benchmarks/bench_launch_latency.py` with a README.
  - It launches one task from two SkyPilot builds (own worktree, venv and API server each), interleaved A B B A.
  - It compares `provision.log` segments with a permutation test, and reports VALID only when the targeted segment is significantly faster.

Files: `sky/provision/instance_setup.py`, `sky/provision/provisioner.py`, a new unit test file, and the benchmark.

### Tested

- [x] Code formatting: `yapf --diff` and `isort --check` are clean on the changed files. Pylint on `sky/` adds no messages over master. mypy has no errors beyond master's 2 in `fetch_nebius.py`.
  - Outside CI's pylint scope, `bench_launch_latency.py` has 3 pylint warnings: inconsistent quotes ×2, and `subprocess.run` without `check`.
- [x] Any manual or new tests for this PR (please specify below)
  - **New unit tests:** 12 in `tests/unit_tests/test_sky/provision/test_provisioner_ray_wait.py`; 12 passed. They run the real wait script against a local socket and a fake `ray._raylet`:
    - returns once every node has joined;
    - retries while GCS comes up;
    - reports the partial count at the deadline;
    - reports "no cluster" when nothing listens;
    - falls back to the default port without a port file;
    - ships the script as one shell word.

    They also cover the provisioner:
    - a ready cluster is waited for once;
    - skylet starts while workers are still joining;
    - a dropped wait is retried, not treated as dead Ray;
    - workers that never join are started without a head restart;
    - a head without Ray is started after the wait;
    - a skylet failure still fails the launch.
  - **Launch latency A/B** (this benchmark, 4-node task on kind). Runtime setup end → "Cluster launched": 12.47 → **9.89 s (−21%, p = 0.004)**, VALID. Provisioning total 20.38 → 17.38 s.
- [ ] All smoke tests: `/smoke-test` (CI) or `pytest tests/test_smoke.py` (local)
- [x] Relevant individual tests: `/smoke-test -k test_name` (CI) or `pytest tests/test_smoke.py::test_name` (local)
  - These ran locally with `--kubernetes` on kind, on the combined branch. All passed:
    - `test_cluster_job.py::test_multi_hostname`, `test_multi_node_failure`, `test_kubernetes_recovery`
    - `test_provision_logs.py::test_provision_logs_streaming`, `test_worker_provision_logs_streaming` (skylet logs from its thread)
    - `test_setup.py::test_kubernetes_provision_command_timeout`
- [ ] Backward compatibility: `/quicktest-core` (CI) or `pytest tests/smoke_tests/test_backward_compat.py` (local)
  - Not run. The change is server-side, and the wait script uses only the standard library and `ray`.

---

## `ej-prj-bonus`: [Provision] Fail setup fast when a pod is lost

Commits: `24c6d876b`, `17411838d`, `6db5eba63`, `7aed7c84e`. Stacked on `ej-prj-0` and `ej-prj-1`.

### Summary

- **Problem:**
  - A pod deleted after it started (`kubectl delete pod`, an eviction) never comes back: SkyPilot pods use `restartPolicy: Never`.
  - The launch only noticed after about 106 s. The head-side Ray wait ran its whole 90 s timeout, because a worker that never joined is invisible to it. The fallback then tried to start Ray on the missing pod, retrying six times.
  - The error didn't say a pod was gone.
- **Fix:**
  - **Don't retry an exec whose pod is gone.** `command_runner.is_exec_target_gone()` matches the kubectl errors that are permanent: NotFound, a completed pod, phase Failed or Succeeded. `instance_setup._auto_retry` stops on those. Errors that can be transient, such as "unable to upgrade connection", are still retried.
  - **Liveness check.** `provision.query_instances(non_terminated_only=False)` lists the cluster's instances. One that is missing, terminated or stopped in two listings 2 s apart raises the new `InstancesLostDuringSetupError` (a `ClusterSetUpError`) naming it. A check that fails never fails the launch.
  - **Time-boxed Ray wait.** The Kubernetes wait runs in calls of at most 10 s, with the check between calls. When Ray is ready within the first call, nothing changes.
  - **Clearer errors.** When any setup stage fails, the check runs before the error is re-raised, so the error names a lost pod and keeps the stage's error as context.
- **New smoke test:** `test_kubernetes_pod_loss.py` with `scripts/delete_pod_mid_launch.sh`. It force-deletes the worker or head pod mid-setup and asserts the launch fails within 45 s with `InstancesLostDuringSetupError` naming the pod.

### Tested

- [x] Code formatting: `yapf --diff` and `isort --check` are clean on the changed files. Pylint on `sky/` adds no messages over master. mypy has no errors beyond master's 2 in `fetch_nebius.py`. `7aed7c84e` adds `is_exec_target_gone` to `sky/utils/command_runner.pyi`; without it, mypy failed on its use in `instance_setup._auto_retry`.
- [x] Any manual or new tests for this PR (please specify below)
  - **New unit tests** in `tests/unit_tests/test_sky/provision/test_provisioner_ray_wait.py`:
    - a worker deleted mid-wait fails at the next chunk;
    - a deleted head fails without retrying the wait;
    - a pod missing from one listing doesn't fail the launch;
    - a failing instance check doesn't fail the launch;
    - a stage failing on a deleted pod names the pod;
    - a stage failing with every pod present keeps its own error;
    - which exec errors mean the pod is gone;
    - a stage isn't retried once its pod is gone.

    Together with the template tests: 112 passed.
  - **Manual, on kind:** a deleted worker now fails the launch about 11 s after the deletion, and a deleted head about 5 s after, instead of about 106 s.
- [ ] All smoke tests: `/smoke-test` (CI) or `pytest tests/test_smoke.py` (local)
- [x] Relevant individual tests: `/smoke-test -k test_name` (CI) or `pytest tests/test_smoke.py::test_name` (local)
  - These ran locally with `--kubernetes` on kind, on the combined branch. All passed:
    - `test_kubernetes_pod_loss.py::test_launch_fails_fast_when_pod_deleted[worker]` (42 s) and `[head]` (45 s), both new
    - `test_cluster_job.py::test_kubernetes_recovery`, `test_kubernetes_pod_failure_detection`
- [ ] Backward compatibility: `/quicktest-core` (CI) or `pytest tests/smoke_tests/test_backward_compat.py` (local)
  - Not run. There's no API change; the new exception subclasses `ClusterSetUpError`.

---

## `ej-prj-3`: [Kubernetes] Start the head's Ray sooner

Commits: `8cce4953b`, `b4b668cd7`. Stacked on `ej-prj-bonus`, and also carries that branch's stub fix (cherry-picked as `aeb9e96f7`).

### Summary

- **Skip `ray stop` when no Ray is running.**
  - Both start commands ran `ray stop` before `ray start`. On a fresh pod there's nothing to stop, but it still costs a Python start: 1.4–1.5 s idle, 2.9–3.8 s while the pod is busy setting up. On the head that delays every worker.
  - `RAY_STOP_IF_RUNNING` now stops Ray only if `pgrep` finds a GCS, raylet or dashboard process, so restarts still stop Ray first.
  - The pattern brackets one letter of each name (`[g]cs_server`), so it can't match the shell running it.
  - Without `pgrep`, Ray is stopped as before.
- **Pre-install the runtime pins in the image.**
  - The pod pins `setuptools<70` and `click<8.3.0` at startup, but the image ships setuptools 84, so every pod downloaded setuptools from PyPI before `ray start`: 0.6 s usually, 12 s in one traced launch.
  - `Dockerfile_k8s`, `Dockerfile_k8s_gpu` and the docs' custom-image snippet now install both pins, so a pod only checks them (about 2 ms).
  - This takes effect once the image is rebuilt and published.

### Tested

- [x] Code formatting: `yapf --diff` and `isort --check` are clean on the changed files. Pylint on `sky/` adds no messages over master. mypy has no errors beyond master's 2 in `fetch_nebius.py`.
- [x] Any manual or new tests for this PR (please specify below)
  - **New unit tests:** 3 in `tests/unit_tests/test_sky/provision/test_ray_stop_if_running.py`. The `test_ray_is_stopped_only_when_running` cases cover three situations: `pgrep` finds Ray, finds nothing, or isn't installed. The others check that the real `pgrep` doesn't match the guard's own command line, and that both start commands stop Ray only through the guard, before `ray start`. With the provision and template tests: 118 passed.
  - **In a live pod:** with Ray running, the guard stopped it; after Ray stopped, it skipped; it never matched its own `bash -c`.
  - **Image:** `docker build -f Dockerfile_k8s` succeeds.
  - **On kind** (4 nodes; 3 runs against 5 for the baseline):
    - head container start → GCS up: 7.31 → 5.89 s;
    - head container start → whole cluster in Ray: 12.95 → 11.37 s;
    - client total, for runs without a slow pod: about 2 s faster.
- [ ] All smoke tests: `/smoke-test` (CI) or `pytest tests/test_smoke.py` (local)
- [x] Relevant individual tests: `/smoke-test -k test_name` (CI) or `pytest tests/test_smoke.py::test_name` (local)
  - These ran locally with `--kubernetes` on kind, on the combined branch. All passed:
    - `test_kubernetes_host_network.py::test_kubernetes_host_network_relaunch_after_head_ray_stopped` and `test_basic.py::test_launch_fast`: relaunches where Ray must still be stopped.
    - `test_cluster_job.py::test_kubernetes_non_debian_image` (both cases), with images that may lack `pgrep`.
    - `test_images.py::test_kubernetes_docker_image_and_ssh`, with a custom image.
  - **Not run:** `test_kubernetes_host_network_relaunch_after_ray_stopped` needs 2 schedulable nodes, and kind has 1.
- [ ] Backward compatibility: `/quicktest-core` (CI) or `pytest tests/smoke_tests/test_backward_compat.py` (local)
  - Not run. Only new pods get the new commands; custom images without the pins download them at pod start, as before.

---

## `ej-prj-job0`: [Jobs] Notice a short managed job's start and end sooner

Commits: `30337ed94`, `7c26bd708`. Based on master `815e61a76`.

### Summary

- **Status and submitted-at in one head command.**
  - After a job's cluster is launched, the controller confirmed the job had started with two commands on the head: status, then submitted-at. Without gRPC, each is a `kubectl exec` or SSH round trip of about 1.6 s.
  - `JobLibCodeGen.get_job_status_and_submitted_timestamp` now returns both. It calls only the `job_lib` functions the two commands already call, so older cluster runtimes support it.
  - Over gRPC, or if the time is missing, the old second call is still made. Error classification is shared with `get_job_status`.
- **No first-check sleep when the launch already saw the job finish.**
  - The monitor loop slept a random 0–15 s before its first status check, even when the launch had just seen the job finish.
  - The executor now remembers the status its launch (or recovery) saw, and the loop checks at once if that status is terminal.
  - A job still running keeps the jittered first gap, so loops that start together still spread out.
- **The client follower checks the status when the controller logs.**
  - `sky jobs launch` and `sky jobs logs` follow a not-yet-RUNNING job by reading the controller's per-job log every second, but re-checked the status only every 15 s. A short job is RUNNING for only a few seconds, so the follower usually missed it, then found the job finished up to 15 s late.
  - The controller writes a plain line of its own around each status change, and only about 15 while a cluster is set up. The follower now re-checks as soon as it sees one, and keeps the 15 s check as a fallback.

### Tested

- [x] Code formatting: `yapf --diff` and `isort --check` are clean on the changed files. Pylint on `sky/` adds no messages over master. mypy has no errors beyond master's 2 in `fetch_nebius.py`.
- [x] Any manual or new tests for this PR (please specify below)
  - **New unit tests:**
    - `TestJobStatusAndSubmittedAt` in `tests/unit_tests/test_sky/jobs/test_utils.py`: one command returns both; a missing timestamp gives None; gRPC fetches only the status; the codegen runs against a real job table.
    - The follower tests: reports whether the controller logged a line; checks the status as soon as it does; otherwise every 15 s.
    - `TestFirstCheckAfterLaunch` in `test_controller.py`.
    - Submit-wait tests in `test_recovery_strategy.py`.

    `tests/unit_tests/test_sky/jobs`: 1367 passed, 1 failed. The failure, `test_details_is_none_when_no_state_or_failure`, also fails on master.
  - **Managed-job benchmark on kind** (5 runs each, median):
    - job finished → RUNNING: 3.0 → 1.3 s;
    - RUNNING → SUCCEEDED: 7.7 s (random 4.9–10.5) → 3.4 s (3.3–3.4).
    - The client exits 2.1–2.9 s after SUCCEEDED in 6/6 runs, against 1.6–12.6 s before; before, it missed RUNNING in 4 of 7 runs.
  - **Interleaved A/B with all launch-latency changes against master** (8 runs per side): RUNNING → SUCCEEDED 11.1 s (random 3.7–16.1) → 3.5 s (3.4–3.6) in every run. `sky jobs launch` total 61.5 → 50.1 s on 4 nodes, and 70.8 → 54.8 s on 6 nodes.
- [ ] All smoke tests: `/smoke-test` (CI) or `pytest tests/test_smoke.py` (local)
- [x] Relevant individual tests: `/smoke-test -k test_name` (CI) or `pytest tests/test_smoke.py::test_name` (local)
  - These ran locally with `--kubernetes` on kind, consolidation mode, on the combined branch. All passed:
    - `test_managed_job.py`:
      - `test_managed_jobs_basic`, `test_job_pipeline`
      - recovery: `test_managed_jobs_recovery_kubernetes_multinode`, `test_managed_jobs_recovery_default_resources`, `test_managed_jobs_exit_code_recovery_single`, `test_managed_jobs_exit_code_recovery_multinode`, `test_managed_jobs_retry_logs`
      - `test_managed_jobs_wait_success`, `test_managed_job_node_names_multi_node`
      - follower: `test_managed_jobs_logs_tail`, `test_managed_jobs_cli_exit_codes`, `test_managed_jobs_cancelled_job_logs`
    - `test_basic.py::test_jobs_launch_and_logs`
  - **Not run here:**
    - `test_managed_jobs_log_tail_cleanup` is skipped in consolidation mode.
    - The `ha_kill_*` tests need an HA controller.
    - The non-consolidation (controller cluster) path still needs a CI run.
- [ ] Backward compatibility: `/quicktest-core` (CI) or `pytest tests/smoke_tests/test_backward_compat.py` (local)
  - Not run; recommended. The codegen uses only `job_lib` payload functions that existing cluster runtimes have, and falls back when the timestamp is missing.

---

## `ej-prj-job1`: [Data] Look up the enabled storage clouds once per request

Commit: `d37133a6f`. Based on master `815e61a76`.

### Summary

- **Problem:**
  - With no storage cloud enabled, the cached list of enabled storage clouds is empty, and `get_cached_enabled_clouds_or_refresh` treats empty as "not checked yet". Every `get_cached_enabled_storage_cloud_names_or_refresh()` call re-ran the storage check.
  - A `jobs.launch` request in consolidation mode looks twice: the file-mount translation (4.8 s), then the controller's packages (0.18–0.25 s).
- **Fix:** reuse the answer, including an empty one, for the rest of the request.
  - **Cache:** `annotations.ttl_cache(scope='request', ttl=60, lock=...)`, keyed by workspace and config fingerprint.
  - **A process that never clears request caches,** such as a jobs controller, keeps workspaces and configs apart, and the 60 s TTL limits how stale its answer can get.
  - **`_is_storage_cloud_enabled` clears the cache after its `sky check` re-check,** so the re-check is still seen.
  - **Failed lookups aren't cached,** and callers still get their own list.
- **Supporting changes:**
  - `annotations.ttl_cache` takes an optional `lock`.
  - `check._config_fingerprint` is renamed to the public `current_config_fingerprint`.

### Tested

- [x] Code formatting: `yapf --diff` and `isort --check` are clean on the changed files. Pylint on `sky/` adds no messages over master. mypy has no errors beyond master's 2 in `fetch_nebius.py`.
- [x] Any manual or new tests for this PR (please specify below)
  - **New unit tests:**
    - 11 in `tests/unit_tests/test_sky/data/test_enabled_storage_clouds_cache.py`:
      - an empty answer is looked up once per request, and enabled clouds and storage-only clouds are reused;
      - `raise_if_no_cloud_access` still raises from the reused answer, and callers get their own list;
      - a new request looks again; workspaces and configs are kept apart;
      - a failed lookup isn't reused;
      - a `sky check` fix is seen in the same request, and a cloud still disabled after it stays disabled.
    - 3 for `ttl_cache` in `test_annotations.py`: request scope, lock and no lock.

    With the check, annotation and `controller_utils` tests: 113 passed. Full unit suite: 10,704 passed; the 20 failures also fail on master or pass when run alone.
  - **Probe on kind** (temporary logging around the lookup, 3 `sky jobs launch` runs per side):

    | | 1st lookup | 2nd lookup |
    |---|---|---|
    | master | 4.76–4.85 s | 0.18–0.25 s |
    | this change | 4.73–4.80 s | 0.001–0.002 s |

    - The saving is about 0.2 s per launch, which is within phase A's noise in end-to-end runs.
    - The first lookup's 4.8 s remains. Caching across requests would be a follow-up.
- [ ] All smoke tests: `/smoke-test` (CI) or `pytest tests/test_smoke.py` (local)
- [ ] Relevant individual tests: `/smoke-test -k test_name` (CI) or `pytest tests/test_smoke.py::test_name` (local)
  - Not yet run with this change. Suggested:
    - `test_managed_job.py::test_managed_jobs_basic`
    - `test_managed_job.py::test_managed_jobs_storage`, which needs a storage cloud
    - `test_mount_and_storage.py`, which needs a storage cloud
    - `test_basic.py::test_jobs_launch_and_logs`
  - The 20 managed-job A/B runs on kind (all five changes) succeeded with it.
- [ ] Backward compatibility: `/quicktest-core` (CI) or `pytest tests/smoke_tests/test_backward_compat.py` (local)
  - Not run. The change is server-side only, with no API change.

---

## `ej-k8s-pool-warning`: [Kubernetes] Hide urllib3 pool-full warning from capped API clients

Commit: `5bb46b39a`. Based on master `5ef55efa4`.

### Summary

- **Problem:**
  - SkyPilot's Kubernetes API clients cap their connection pool at `CONNECTION_POOL_MAXSIZE` (4). Going past it is by design, for example when creating or inspecting more than 4 pods in parallel.
  - urllib3 then warns "Connection pool is full, discarding connection" each time it closes an extra connection.
  - `_api_logging_decorator` keeps the urllib3 logger at ERROR during API calls. But any `kubernetes.client.Configuration` built elsewhere, or in another thread mid-call, resets the logger to WARNING, so the warning still reached users.
- **Fix:**
  - A `logging.Filter` on `urllib3.connectionpool` drops exactly this warning when the pool size in it equals `CONNECTION_POOL_MAXSIZE`. Pool-full warnings from other pools still show.
  - A level reset doesn't undo a filter, and a filter takes no lock per record.

### Tested

- [x] Code formatting: `yapf --diff` and `isort --check` are clean on the changed files. Pylint on `sky/` adds no messages over master. mypy has no errors beyond master's 2 in `fetch_nebius.py`.
- [x] Any manual or new tests for this PR (please specify below)
  - **New unit test:** `test_pool_full_warning_hidden_for_capped_clients` in `tests/unit_tests/test_sky/adaptors/test_kubernetes_adaptor.py`. The file passes: 28 passed.
- [ ] All smoke tests: `/smoke-test` (CI) or `pytest tests/test_smoke.py` (local)
- [ ] Relevant individual tests: `/smoke-test -k test_name` (CI) or `pytest tests/test_smoke.py::test_name` (local)
  - Not run. Suggested: a multi-node Kubernetes launch with more than 4 pods, e.g. `test_cluster_job.py::test_multi_hostname` with `--num-nodes 6`, checking that no "Connection pool is full" warning is printed.
- [ ] Backward compatibility: `/quicktest-core` (CI) or `pytest tests/smoke_tests/test_backward_compat.py` (local)
  - Not run. The change only affects logging.
