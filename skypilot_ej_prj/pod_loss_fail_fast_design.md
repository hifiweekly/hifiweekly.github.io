# Design: fail fast when a pod is deleted during launch

- **Branch:** `ej-prj-bonus` (fork), based on `ej-proj-0-1`, which has
  `ej-prj-0` (pods start Ray first) and `ej-prj-1` (one blocking Ray wait,
  skylet started meanwhile).
- **Status:** components A, B and D are implemented on `ej-prj-bonus`
  (uncommitted); component C is not. See section 9 for what was built, how it
  differs from this design, and the measured results.
- **Reproduction:** `ej/sim_pod_delete.sh [worker|head]` (supersedes `ej/sim_ray_timeout.sh`). Head-side wait without a cluster:
  `ej/sim_ray_wait_timeout.py`.

## 1. Background

If a pod is deleted during `sky launch` (by `kubectl delete pod`, an eviction
or a queue controller), the cluster can never become healthy: job pods use
`restartPolicy: Never`, so nothing recreates them. SkyPilot still takes far
too long to notice.

The run with `ej/sim_ray_timeout.sh` on kind (6 nodes, `worker1` deleted right
after runtime setup) went like this:

| Time | What happened |
|---|---|
| 19:47:37.2 | `End: setup_runtime_on_cluster`; the head-side Ray wait starts |
| 19:47:39 | `worker1` deleted |
| 19:49:07.5 | The Ray wait gives up after **90.3 s** (`_K8S_RAY_READY_TIMEOUT_SECONDS`): 5 of 6 nodes alive |
| 19:49:07–23 | `start_ray_on_worker_nodes` retries 6 times (`_auto_retry`, back-off 1 s → 5 s); every attempt fails with `Error from server (NotFound): pods "...-worker1" not found` |
| 19:49:23.6 | `RuntimeError: Failed to start ray on the worker node`; the cluster is left in INIT |

That's about **106 s** spent on a launch that was lost from 19:47:39 on.
Under `sky jobs launch`, the controller only starts its teardown, back-off and
relaunch after this. So every lost pod adds about 106 s to the job's recovery.

### Where the time goes

1. **The worker side of the Ray wait can't see the loss.** The wait runs on the
   head and counts alive nodes in Ray's GCS. A worker deleted before it joined
   never appears there, so the head just waits out the full 90 s. (Kubernetes
   knows at once.)
2. **A deleted head is retried until the deadline.**
   `wait_for_ray_cluster_healthy` treats a call without the done marker as a
   dropped `kubectl exec`, and retries every 1 s until the 90 s deadline. A
   deleted head looks exactly like that.
3. **`_auto_retry` retries a pod that no longer exists.** The fallback
   (`start_ray_on_worker_nodes`) and every other `@_auto_retry` stage (runtime
   setup, file mounts, skylet start) retry up to 6 times. A `NotFound` is
   permanent, so all 6 attempts are wasted (about 16 s, plus 6 exec round
   trips).
4. **Nothing in the provisioner checks the pods after "Pods are up".** From
   `_post_provision_setup` on, the code assumes the instances from
   `get_cluster_info` still exist.

The scheduling stage already handles this well.
`_wait_for_pods_to_schedule` / `_wait_for_pods_to_run` fail with a
`_deleted_pods_error` that explains the cause, after a 10 s grace period
(`_MISSING_POD_GRACE_SECONDS`). This design brings the same behaviour to the
runtime-setup stage.

## 2. Goals and non-goals

### Goals

- Fail a launch whose pod is deleted during post-provision setup within about
  **10 s** of the deletion, instead of about 106 s.
- Give an error that names the lost pod, with the deletion reason when the
  API server has one. Reuse the wording of the existing scheduling-stage
  error.
- No extra calls on the happy path when Ray is ready within the first check
  window (the usual case: 4–10 s on kind).
- Don't fail a healthy launch because of one stale or partial pod list (a
  caching proxy or virtual-cluster syncer). The existing 10 s grace period
  exists for the same reason.
- Stay cloud-agnostic where that's cheap: VMs terminated during setup have
  the same problem.

### Non-goals

- Recreating the lost pod or replacing it in place. A pod with the same name
  belongs to a new launch (see the comment on `_MISSING_POD_GRACE_SECONDS`).
  Recovery remains the caller's job: failover in `sky launch`, and
  teardown + relaunch in the jobs controller.
- Lost pods after the launch (job running). The jobs controller already
  detects those through `refresh_cluster_status_handle`.
- Changing `_K8S_RAY_READY_TIMEOUT_SECONDS`. 90 s stays the limit for Ray
  being slow; this design is about Ray never coming up because its pod is
  gone.

## 3. Design

### 3.1 Overview

```
wait_for_ray_cluster_healthy (provisioner, K8s just-booted path)
  loop:
    head_runner.run(wait for N alive, timeout = min(remaining, CHUNK))  # one exec
    if SKY_ALIVE_NODES == N: done                       <- happy path, 1 call
    if no done marker (exec dropped) or chunk ended without N:
        _raise_if_instances_lost()                      <- new, 1 LIST call
    if deadline passed: break

_raise_if_instances_lost()          (new, cloud-agnostic)
    provision.query_instances(...) vs. cluster_info.instances
    missing / terminating instance  -> confirm once more after a short pause
    still lost -> raise exceptions.InstancesLostDuringSetupError(reason)

_auto_retry(should_retry=...)       (instance_setup)
    never retry an error whose stderr says the exec target is gone
```

There are four components. Each is useful on its own, so they can land as
separate commits.

### 3.2 Component A: a liveness check for the cluster's instances

A new helper inside `_post_provision_setup` (it has `cloud_name`,
`cluster_name`, `provider_config` and `cluster_info` already):

```python
def raise_if_instances_lost() -> None:
    """Raise if an instance of this launch is gone or going away."""
    lost = _lost_instances()                 # one provision.query_instances
    if not lost:
        return
    time.sleep(_LOST_INSTANCE_CONFIRM_SECONDS)   # 2 s
    lost = _lost_instances()                 # must be lost twice in a row
    if lost:
        raise exceptions.InstancesLostDuringSetupError(...)
```

- **What counts as lost.** `_lost_instances()` calls
  `provision.query_instances(cloud_name, cluster_name.display_name,
  cluster_name.name_on_cloud, provider_config, non_terminated_only=True)`. It
  returns the instances in `cluster_info.instances` that are missing from the
  result, or whose status is not UP. For Kubernetes, `query_instances` already
  flags pods that are being deleted (`is_terminating`) and pods in the
  Failed phase.
- **Why a second look.** A single list that omits a pod mustn't fail the
  launch, which is the partial-view case the 10 s grace period guards against
  in the scheduling stage. Here a second query 2 s later is enough: the pods
  were seen running moments before, so a real deletion won't reappear.
  Waiting the full 10 s would defeat the purpose.
- **The reason in the error.** For Kubernetes, build the message with the
  existing `_deleted_pods_error`, which looks up the deletion reason from the
  pod and its events. That gives the same text as the scheduling stage, for
  example `Pod ...-worker1 was deleted: <reason>`. Other clouds use the reason
  string from `query_instances`.
- **New exception.** `exceptions.InstancesLostDuringSetupError` subclasses
  `ClusterSetUpError`, so existing handlers that catch `ClusterSetUpError` keep
  working. It also gives the jobs controller and failover a clear type to act
  on (section 3.6).
- **Cost.** One LIST call with a label selector, made only off the happy path
  (see 3.3).

### 3.3 Component B: wait for Ray in time-boxed calls

`wait_for_ray_cluster_healthy` currently makes one call that can block for the
whole 90 s. Instead, cap each call at `_RAY_WAIT_CHUNK_SECONDS = 10` and check
the instances between calls:

```python
deadline = time.time() + timeout
while True:
    remaining = max(deadline - time.time(), 0.0)
    chunk = min(remaining, _RAY_WAIT_CHUNK_SECONDS)
    returncode, stdout, stderr = head_runner.run(
        ray_wait_for_alive_nodes_command(n, chunk), ...,
        timeout=int(chunk) + _RAY_WAIT_CALL_SLACK_SECONDS)
    probe = parse_ray_probe(returncode, stdout)
    if healthy(probe) or time.time() >= deadline:
        return probe
    raise_if_instances_lost()          # dropped exec or N not reached yet
    if RAY_WAIT_DONE_MARKER not in stdout:
        time.sleep(1)                  # dropped exec: brief pause, as today
```

- **Happy path unchanged.** Ray is usually ready in 4–10 s, so the first call
  returns as soon as every node is alive (0.2 s polling), as it does today. No
  extra exec and no LIST call.
- **Worker deleted.** It's detected at the next chunk boundary: at most 10 s,
  plus the 2 s confirmation. That's about 12 s instead of 90 s.
- **Head deleted.** The exec fails right away with no done marker, so the check
  runs immediately: about 2–3 s instead of 90 s.
- **Cost of the time box.** Only when Ray is slow but healthy: one extra exec
  (about 0.3–1 s, including starting the Python wait script) every 10 s. Over
  a full 90 s timeout that's 8 extra calls, which is acceptable for a path
  that's already slow.
- **Chunk size.** 10 s, with the same reasoning as
  `_MISSING_POD_GRACE_SECONDS`. 5 s halves detection time but doubles the
  extra calls on slow-but-healthy clusters. Make it a module constant so the
  benchmark can tune it.

**Alternative considered: watch the pods in a background thread and kill the
exec.** This detects loss at once, but `CommandRunner.run` has no
cancellation handle. Killing the local `kubectl exec` would also leave the
wait script running on the head until its own timeout. Time-boxed calls get
within about 10 s with no new concurrency. Revisit if 10 s turns out to be
too slow.

**Alternative considered: a Kubernetes watch instead of LIST.** It's lower
latency, but needs watch connection handling (timeouts, reconnects,
`resourceVersion`) for a check that only runs a few times per launch. Not
worth it.

### 3.4 Component C: the head reports nodes that died after joining

This is an optional, cheap addition to `_WAIT_ALIVE_NODES_PY`. It already
reads every node's state from the GCS. If a node that was ALIVE earlier in
the wait is now DEAD, the script can stop and print
`SKY_DEAD_NODES <count>` before `SKY_RAY_WAIT_DONE`. The provisioner then
runs `raise_if_instances_lost()` straight away instead of waiting for the
chunk boundary.

- This catches a worker deleted after it joined Ray, as soon as Ray's health
  check marks it dead. Component B alone would catch it within 10 s.
- It can't catch a worker deleted before it joined Ray, which is the case in
  the reproduction. Component B handles that one.
- A DEAD node isn't proof that the pod is gone (a raylet can crash and
  restart), so the script only reports it and the decision stays with
  component A.

### 3.5 Component D: don't retry a pod that is gone

`_auto_retry` decides by exception type (`should_retry`), and most stages use
the default, which retries everything. Two changes:

1. **Recognise the error.** Add `error from server (notfound): pods` to
   `KubernetesCommandRunner._POD_GONE_EXEC_SIGNATURES`. Today that list
   catches a completed or failed pod, but not a deleted one. Add a public
   helper, `command_runner.is_exec_target_gone(text) -> bool`, built on the
   same list.
2. **Stop retrying.** In `_auto_retry`, stop when
   `command_runner.is_exec_target_gone(str(e))` is true. The stage errors
   already include stderr in their message (`Detailed Error: ... =====
   stderr ===== ...`). Re-raise, and let the provisioner turn it into
   `InstancesLostDuringSetupError` (next point).

In `post_provision_runtime_setup`'s `except Exception`, run
`raise_if_instances_lost()` before re-raising. Any stage that fails because an
instance is gone (runtime setup, file mounts, skylet start, the Ray fallback)
then reports the lost pod, not just the exec error. This costs one LIST call,
only on failure.

### 3.6 Effect on callers

- **`sky launch`:** the error comes about 100 s sooner and names the pod and
  the reason. The cluster stays in INIT, as now; `sky launch` again or
  `sky down` works as before.
- **The provisioning failover loop** (`RetryingVmProvisioner`): unchanged. A
  setup error is not a resources-unavailable error, so it doesn't fail over to
  another region today. Whether a lost pod should trigger failover is a
  separate question; see 7.2.
- **`sky jobs launch`:** the controller's `_launch` loop already treats any
  launch exception as "tear down, back off, retry" (`recovery_strategy.py`).
  It now gets there about 100 s sooner per lost pod. No controller change is
  needed. Optionally, it could classify `InstancesLostDuringSetupError` with
  its own retry code (`_error_kind`), so the job events say "pod deleted
  during setup" rather than a generic error.

## 4. Failure handling

| Case | Today | With this design |
|---|---|---|
| Worker deleted before joining Ray | 90 s wait + ~16 s retries → `RuntimeError` | Next chunk boundary (≤ 10 s) + 2 s confirmation → `InstancesLostDuringSetupError` |
| Worker deleted after joining | 90 s, or until Ray drops the node | Component C: as soon as GCS marks it DEAD; otherwise ≤ 10 s + 2 s |
| Head deleted during the wait | Retried every 1 s for 90 s | First failed exec → check → ~2–3 s |
| Pod deleted during runtime setup or file mounts | Up to 6 `_auto_retry` attempts | First failure, no retry → check → error naming the pod |
| One stale list misses a pod | (not checked) | Second query 2 s later sees it → wait continues |
| `query_instances` itself errors (API server blip) | (not checked) | Log at debug and continue the wait (as for a transient error). Never fail a launch on a failed check |
| Ray slow but every pod alive | Single call up to 90 s | Same outcome; one extra exec + LIST every 10 s |

## 5. Compatibility

- **Provisioner only.** No API, template or skylet changes, so no
  server/client version issue. The wait script change (C) is additive: older
  output without `SKY_DEAD_NODES` parses as before.
- **Cloud-agnostic.** Component A uses `provision.query_instances`, which
  every provisioner implements. B and C only apply to the Kubernetes
  just-booted path, where the head-side wait exists. D applies wherever
  `_auto_retry` runs, but the signature is a kubectl phrasing, so it changes
  nothing for SSH.
- **RBAC.** The provisioner already lists pods for the cluster (scheduling
  waits, `query_instances`), so no new permissions are needed.

## 6. Testing

### Unit tests (`tests/unit_tests/test_sky/provision/`)

- **`raise_if_instances_lost`:**
  - all present → returns;
  - one missing twice → raises with the pod name;
  - missing once then present → returns;
  - `query_instances` raises → returns (logged).
- **Time-boxed wait:**
  - Ray ready in the first chunk → exactly one exec and zero
    `query_instances` calls;
  - never ready and nothing lost → ceil(timeout / chunk) execs, then the
    fallback;
  - a worker lost after chunk 1 → raises after 2 execs;
  - the head exec fails with `NotFound` → raises before any retry sleep.
- **`_auto_retry`:** a `RuntimeError` containing
  `Error from server (NotFound): pods` is not retried. Other `CommandError`s
  still are.
- **Wait script (C):** with a fake `ray._raylet` whose node goes ALIVE → DEAD,
  the script prints `SKY_DEAD_NODES 1` and exits before the timeout. (Same
  harness as `test_provisioner_ray_wait.py`.)

### End to end on kind

- `ej/sim_ray_timeout.sh ray-timeout-sim ej/job.yaml`: expect the launch to fail
  about 12 s after the deletion (today it's about 106 s), with an error naming
  `...-worker1`.
- A variant that deletes the head instead: expect about 3 s.
- `sky jobs launch` with a pod deleted during STARTING: check that the
  controller log shows the new error and that the relaunch starts sooner.
- **Happy-path regression:** run the A/B benchmark
  (`tests/kubernetes/launching_benchmarks/bench_launch_latency.py`) with
  `ej-proj-0-1` as the baseline and `ej-prj-bonus` as the candidate.
  `post_runtime_s` must not get worse; expect no difference.

### Smoke test (candidate)

`test_launch_fails_fast_when_pod_deleted` (Kubernetes only):
- Launch a 2-node cluster with `--async`.
- Wait for `End:   setup_runtime_on_cluster` in `sky logs --provision`.
- Delete the worker pod.
- Assert the launch fails within 30 s and the provision log names the pod.

This timing-dependent setup works reliably in `ej/sim_ray_timeout.sh` because
the window between runtime setup and Ray being ready is several seconds.

## 7. Risks and open questions

1. **False positives** are the main risk: failing a healthy launch is worse
   than a slow failure. Mitigations:
   - the check only runs off the happy path;
   - a pod must be lost in two queries;
   - errors in the check never fail the launch.
2. **Should a lost pod trigger failover?** If a queue controller (Kueue)
   evicts pods on purpose, retrying the same context may just be evicted
   again. Out of scope here, but the new exception type makes that policy easy
   to add later.
3. **Chunk size.** 10 s is a guess; measure detection time against the extra
   calls on a large cluster (for example 32 nodes, where Ray takes longer to
   assemble).
4. **The 2 s confirmation pause** sits on the failure path only. It could
   instead be one direct GET of the pod (authoritative 404) for Kubernetes.
   That's faster, but adds a Kubernetes-only branch to component A.

## 8. Implementation plan (commits on `ej-prj-bonus`)

1. `[Kubernetes] Do not retry an exec whose target pod is gone`: component D
   (signature, helper, `_auto_retry`), with unit tests. This alone saves about
   16 s on every lost pod.
2. `[Provision] Fail setup fast when an instance is lost`: component A, the
   new exception, the check in `post_provision_runtime_setup`'s
   `except Exception`, with unit tests.
3. `[Kubernetes] Check pod liveness between time-boxed Ray waits`: component B,
   with unit tests and the kind reproduction (expect about 12 s).
4. *(optional)* `[Kubernetes] Report Ray nodes that die during the wait`:
   component C.
5. Benchmark on the happy path against `ej-proj-0-1`, then add the smoke test.


## 9. Implementation and results

### 9.1 What was built

| Component | Where | Notes |
|---|---|---|
| D: don't retry a gone pod | `sky/utils/command_runner.py`: `_EXEC_TARGET_GONE_SIGNATURES`, `is_exec_target_gone()`; `sky/provision/instance_setup.py`: `_auto_retry` | Matches only errors that are permanent: `Error from server (NotFound): pods`, a completed pod, phase Failed or Succeeded. It leaves out `unable to upgrade connection` and `container not found`, which `_POD_GONE_EXEC_SIGNATURES` matches but which can be transient |
| A: liveness check | `sky/provision/provisioner.py`: `_lost_instances()`, `_raise_if_instances_lost()`, `_fail_fast_on_lost_instances()` | Calls `provision.query_instances(..., non_terminated_only=False)` and checks it twice, `_LOST_INSTANCE_CONFIRM_SECONDS = 2` apart. Any error in the check is logged at debug level and ignored |
| A: new exception | `sky/exceptions.py`: `InstancesLostDuringSetupError(ClusterSetUpError)` | |
| A: check when a setup stage fails | `_post_provision_setup` | `_fail_fast_on_lost_instances` is added to the existing `with rich_utils.safe_status(...)` statement, so no code is re-indented. Any failing setup stage is checked, and the stage's own error is kept as the cause |
| B: time-boxed Ray wait | `wait_for_ray_cluster_healthy` | `_RAY_WAIT_CHUNK_SECONDS = 10`. Each call times out at chunk + 30 s slack. The check runs after every call that ends without all nodes alive |
| C: head reports dead nodes | — | Not implemented. Without it, the worst case for a worker that dies after joining Ray is 10 s + 2 s |

### 9.2 Where it differs from the design

- **Where the reason comes from.** The reason is the string `query_instances`
  returns, not `_deleted_pods_error`. `_deleted_pods_error` is private to the
  Kubernetes provisioner and takes scheduling-stage arguments. For a
  force-deleted pod with no events the reason is `not found`. Where the
  deleter left an event (for example an eviction), `query_instances` already
  reads it through `_get_pod_missing_reason`.
- **Pods still terminating.** A pod with a deletion timestamp that still
  reports `Running` maps to UP in `query_instances`. It is caught only once it
  is gone, after its grace period, which is 30 s by default for
  `kubectl delete pod` without `--force`. Catching it earlier would need a
  Kubernetes-only check of `deletion_timestamp`.
- **The jobs-controller follow-ups are not implemented.** These are the short
  back-off and the separate retry code from the `recovery_strategy.py`
  analysis.

### 9.3 Tests

- **Unit tests,** in
  `tests/unit_tests/test_sky/provision/test_provisioner_ray_wait.py`. All 24
  pass, 12 of them new:
  - a worker lost mid-wait fails at the next chunk, after one exec and two
    listings;
  - a deleted head fails without retrying the wait;
  - a pod missing from one listing doesn't fail the launch;
  - a failing check doesn't fail the launch;
  - a failing stage names the lost pod and keeps its own error as the cause;
  - a failing stage with every pod present keeps its own error;
  - `is_exec_target_gone` for 5 kinds of stderr;
  - `_auto_retry` doesn't retry `NotFound` but still retries a dropped stream.

  The test harness now mocks `query_instances` and shortens the Ray timeout,
  because the fake head answers instantly where the real one blocks.
- **Broader unit tests:** the provision, Kubernetes, adaptor and utils tests
  give 3060 passed and 2 failed. The 2 are
  `test_skypilot_config_configmap_sync`, which fails the same way without this
  change.
- **pylint:** 10.00/10. mypy shows no new errors.
- **Smoke test:** `tests/smoke_tests/test_kubernetes_pod_loss.py::test_launch_fails_fast_when_pod_deleted[worker|head]`
  (Kubernetes only). The logic is in
  `tests/smoke_tests/scripts/delete_pod_mid_launch.sh`:
  - It launches a 2-node cluster.
  - It polls the head pod (`kubectl exec ... pgrep -f 'import base64; exec'`,
    about 0.2 s per check) until the head is running the Ray wait. The wait
    only starts once runtime setup has finished on every node.
  - It force-deletes the worker or head pod.
  - It asserts that the launch exits non-zero within 45 s, with
    `InstancesLostDuringSetupError` naming the pod.

  If the launch gets past the window first (Ray was already ready), it
  retries with a fresh cluster, up to 3 times. On kind, both cases pass on
  the first attempt: worker after 12 s, head after 6 s, 41 s in total. With
  the fix stashed, the worker case fails as intended:
  `took 105s to notice the deletion (max 45s)`.

### 9.4 Results on kind (6-node `ej/job.yaml`)

| Scenario | Before | After |
|---|---|---|
| Worker deleted right after runtime setup | Failed after ~106 s: 90 s wait + 6 futile worker restarts, ending in `RuntimeError: Failed to start ray on the worker node` | **Failed 10.9 s after the deletion** (12.5 s after the wait started): `InstancesLostDuringSetupError: ... rts-202010-48017c75-worker1: not found` |
| Head deleted right after runtime setup | ~90 s of retrying the wait, then the fallback | **Failed 5.0 s after runtime setup ended**: `... podloss-204306-48017c75-head: not found` |
| Nothing deleted | Launches | Launches, 6/6 ranks. Ray ready within the first chunk; no instance checks run |

`sky launch` exits with code 1 in the failure cases. An earlier note that it
exited 0 was caused by the old reproduction script, not by SkyPilot.

### 9.5 Next steps

1. Commit as described in section 8 (D, then A, then B), then run the A/B
   benchmark against `ej-proj-0-1` to confirm the happy path is unchanged.
2. Jobs controller (`recovery_strategy.py`): give
   `InstancesLostDuringSetupError` its own retry code, and a short first
   back-off instead of 60 s, since a deleted pod isn't a capacity problem.
   With this change, back-off becomes the largest part of a recovery.
3. Optional: component C, and a Kubernetes-only check for pods that are
   terminating but still Running.
