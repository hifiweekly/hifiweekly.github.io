# Design: start Ray before preparing SkyPilot in the Kubernetes pod

- **Branch:** `ej-prj-0`, cut from `origin/master` at `5ef55efa4`
- **Commit:** `9c42bf108` "[Kubernetes] Start Ray before preparing SkyPilot in the pod"
- **Pushed to:** https://github.com/siliconej/skypilot/tree/ej-prj-0
- **Origin:** idea #1 in `scheduling_latency_analysis.md`
- **Related:** `ray_wait_and_skylet_overlap_design.md` (ideas #2 and #3, branch `ej-prj-1`)

## 1. Background

Each pod of a SkyPilot cluster on Kubernetes runs its own setup script (the
container `args` in `sky/templates/kubernetes-ray.yml.j2`). Three steps run in
parallel in the background:

1. **apt-ssh-setup:** installs the system packages and starts sshd.
2. **runtime-setup:** installs the runtime and starts Ray.
3. **env-setup:** sets up the environment variables.

On master, step 2 ran these commands one after another:

```
conda / uv / ray install        (setuptools<70, click<8.3.0, ray==2.56.1)
uv pip install skypilot[kubernetes,remote]     <- from PyPI
wait for `patch`, apply Ray patches
touch /tmp/ray_skypilot_installation_complete  <- marker for setup_commands
ray start --head            (head)
wait for head port, ray start  (workers)
```

Meanwhile the provisioner waits for the pods to run, then runs
`setup_runtime_on_cluster` (the template's `setup_commands`). That step waits
for the marker, then installs the cluster's own SkyPilot wheel with
`SKYPILOT_WHEEL_INSTALLATION_COMMANDS`. Finally the provisioner waits until
Ray reports all nodes alive.

### The problem

The PyPI install sits on Ray's critical path, but it isn't needed there:

- **Nothing before `ray start` imports `sky`.** The Ray patches
  (`ray_patches_cmd`) and the hostNetwork probe (`_host_network_probe_cmd`)
  are already shipped inline in the pod spec. They were made inline precisely
  because the `sky` on the node then is an unrelated release.
- **The copy is thrown away.** The wheel install uninstalls it again:
  ```
  - skypilot==0.14.2
  + skypilot==1.0.0.dev0 (from file:///home/sky/.sky/wheels/.../skypilot-1.0.0.dev0-py3-none-any.whl)
  ```
- **It delays every worker.** Workers wait for the head's GCS port before they
  run `ray start`, so a late head means late workers too.

On the default image the dependencies are already baked in, so the install
only resolves 107 packages and downloads the skypilot package. That takes
about 1.3 s (`Resolved 107 packages in 1.07s`, plus a 3.9 MiB download). On a
custom image without the dependencies, it takes much longer.

### A side effect it was providing

The PyPI install also warmed uv's cache. Because of that, the wheel install
that follows resolved in about 20 ms. Without the warm cache, the same resolve
takes 4.5–6 s. This was found during implementation (section 6.2). It is why
the design replaces the install with a dry run instead of just deleting it.

## 2. Goals and non-goals

### Goals

- Start `ray start --head` earlier, so the workers can join earlier.
- Don't move the cost onto `setup_runtime_on_cluster`. Keep the wheel's
  dependency resolve fast.
- Keep the guarantee that no two installs change the runtime environment at
  the same time.
- Keep failure behaviour: if Ray fails to start, the step fails and the pod
  exits.
- No change for HA controllers.

### Non-goals

- Building the `setuptools<70` pin and the Ray patches into the default image.
  That needs an image rebuild, and together they cost under 1 s today.
- Changing `imagePullPolicy: Always`. The default image uses a `:latest` tag,
  so `IfNotPresent` would leave pods on a stale image.
- Changing VM clouds or Slurm. They install Ray and SkyPilot through
  `RAY_SKYPILOT_INSTALLATION_COMMANDS` in `setup_commands`, not in pod args.
- Changing how the provisioner waits for Ray. That is `ej-prj-1`.

## 3. Design

### 3.1 New step 2 order

```
conda / uv / ray install        (unchanged)
[HA only] uv pip install skypilot[kubernetes,remote]
wait for `patch`, apply Ray patches          (unchanged)
( ray start --head | wait for head port + ray start ) &   <- background
[non-HA] uv pip install --dry-run 'skypilot[kubernetes,remote]'   <- warms uv cache
touch /tmp/ray_skypilot_installation_complete
wait $SKY_RAY_START_PID                       <- step still fails if Ray fails
```

```
time ─────────────────────────────────────────────────────────────▶
master   [ray install][PyPI install ~1.3s][patches][ray start ......]
                                          marker ▲
ej-prj-0 [ray install][patches][ray start ..........]
                               [dry-run ~1s]
                                    marker ▲
```

Ray starts about one PyPI-install earlier on the head, and so do the workers.
The marker is written at about the same time as before, so
`setup_runtime_on_cluster` starts the wheel install at about the same time,
and still with a warm cache.

### 3.2 Removing the PyPI install

The install line is now wrapped in `{% if high_availability %}`. A comment in
the template explains why SkyPilot isn't installed before `ray start`.

### 3.3 Starting Ray in the background

The existing head/worker `if ... fi` block (with its `set +e`, the
`ray_head_start_command`, and the worker's port wait plus
`ray_worker_start_command`) is wrapped in `( ... ) &` and its PID is kept in
`SKY_RAY_START_PID`.

- **Output** goes to the same `/tmp/runtime-setup.log` as before, because the
  subshell inherits the step's redirection.
- **Failure propagation:** the step ends with `wait $SKY_RAY_START_PID`. Under
  the step's `set -e`, a non-zero exit from Ray's subshell fails the step. The
  `|| exit 1` in the start commands therefore has the same effect as before:
  `/tmp/runtime-setup.failed` is written and the pod exits.

### 3.4 Warming uv's cache with a dry run

```
VIRTUAL_ENV=${SKY_RUNTIME_DIR:-$HOME}/skypilot-runtime UV_SYSTEM_PYTHON=false \
  {{sky_unset_pythonpath_and_set_cwd}} ~/.local/bin/uv pip install --dry-run \
  'skypilot[kubernetes,remote]' \
  || echo "WARNING: could not warm the uv cache; continuing."
```

- **It's safe next to a starting Ray.** `--dry-run` resolves (fetching index
  pages and metadata into `~/.cache/uv`) but changes nothing in the
  environment. A real install running at the same time as `ray start` could
  replace packages Ray is importing (for example, a `click` downgrade on custom
  images).
- **It runs before the marker,** so the wheel install in `setup_commands` finds
  the cache warm.
- **It's best-effort.** A failure, such as no PyPI access, leaves only a cold
  cache. The old install had no `|| true`, so on master a PyPI failure failed
  the whole step.
- **It's skipped for HA,** which does the real install instead.

### 3.5 The marker

`/tmp/ray_skypilot_installation_complete` keeps its name and its role: it
means "setup_commands may now install the wheel". It is written after
everything in step 2 that changes the environment (the Ray install, the
patches, and for HA the PyPI install), and before `wait` on Ray. As on master,
the marker is written before Ray has finished starting. The status message
changed from "Ray and skypilot installation completed" to "Ray installation
completed".

### 3.6 HA controllers

HA recovery (the `{% if high_availability %}` block after the steps) runs
`{{sky_python_cmd}} -c 'from sky.provision import instance_setup; ...'` and
`from sky.jobs import state` on a restarted pod, before `setup_commands` run
again. HA controllers therefore keep the PyPI install, in its old position
before `ray start`.

## 4. Failure handling

| Case | Behaviour |
|---|---|
| `ray start` fails (head or worker) | Ray's subshell exits 1, `wait` fails the step under `set -e`, `/tmp/runtime-setup.failed` is written and the pod exits. Same as master. |
| Dry run fails (no network, PyPI down) | Warning logged, step continues. The wheel install resolves on a cold cache (slower) or fails on its own, as it would on master. |
| No `skypilot` in the env when the wheel install runs | `uv pip uninstall skypilot` prints "Skipping skypilot as it is not installed" and exits 0 (checked with uv). The install then proceeds. |
| Worker waits for the head port | Unchanged. The wait just ends sooner, because the head starts sooner. |

## 5. Compatibility

- **Existing clusters and relaunches:** the pod args only run when a pod is
  created, so existing pods are unaffected. Pods created later get the new
  args.
- **Custom `image_id` images:** these now get their dependencies from the
  wheel install in `setup_commands`, instead of from the pod.
  - The download time moves off Ray's path onto runtime setup.
  - The total is about the same, and Ray starts earlier.
  - The dependency install now runs while Ray is running. On master the wheel
    install already ran then, but typically installed a single package.
- **Server/client versions:** the change is only in the server-side template,
  so there is no API change.
- **Other code reading the template:** the snapshot goldens skip the pod
  `args`, so they didn't need regenerating.

## 6. Testing

### 6.1 Unit tests

New test in `tests/unit_tests/test_sky/clouds/test_kubernetes_ray_template_snapshot.py`:
`test_pod_starts_ray_before_preparing_skypilot`, run for the `base_cpu`,
`multinode` and `high_availability` cases.

- **For every case:** the PyPI install appears only for HA, and the dry run
  only for non-HA.
- **Ordering:** `RAY_INSTALLATION_COMMANDS` < `RAY_PATCHES_CMD` <
  `RAY_HEAD_START_COMMAND` < (dry run) < marker < `wait $SKY_RAY_START_PID`.
- **For HA:** the PyPI install comes before `ray start`.

Results:
- The new test fails on master (2 of 3 cases) and passes on `ej-prj-0`.
- The snapshot and pod-startup test files pass in full: 121 passed.
- The rendered pod `args` for `base_cpu`, `multinode` and `high_availability`
  all pass `bash -n`.

The 5 failures in `test_kubernetes.py` and `test_backend_utils.py` also happen
on master. They depend on the local `~/.sky` contexts.

### 6.2 End to end on kind

Task: `ej/job.yaml`, 4 nodes on `kind-skypilot`.

**First version (PyPI install deleted, no warm-up):**
- The launch worked, but the wheel install took
  `Resolved 107 packages in 6.05s`.
- Runtime setup went from about 1.6 s to 10 s.
- The provision total was about 22 s, against about 20 s on master.

**Cache experiment on the head pod:**

| Setup | Resolve time for the wheel's dependencies |
|---|---|
| Fresh cache, wheel resolve | 4.55 s |
| Fresh cache, PyPI dry run | 1.28 s |
| Wheel resolve right after that dry run | 14 ms |

**Final version (Ray in the background, dry-run warm-up):**

| Run | Pods ready | Runtime setup | Runtime → launched | Total |
|---|---|---|---|---|
| 1 | 20 s (slow pod start) | 5.3 s | 6.1 s | 32.6 s |
| 2 | 4.6 s | 1.6 s | 12.7 s | 20.1 s |

In both runs the wheel resolve took 22–27 ms, all 4 ranks ran, and the head
ran `sky` 1.0.0-dev0.

### 6.3 Open: A/B benchmark

The single launches are too noisy to show a gain. Run 2 matches master's
median (about 12.5 s runtime → launched). This is expected on master: the old
polling checks for Ray only every 3–5 s, so a Ray that comes up earlier is
often seen on the same check (see the interaction notes in section 8).

To measure it:

```bash
python ej/bench_launch_latency.py run --candidate-ref ej-prj-0 \
    --task ej/job.yaml --runs 5            # baseline: merge-base with origin/master
```

**Expected result:**
- `post_runtime_s` and `provision_total_s` lower, by up to about 1.3 s.
- `pods_ready_s` unchanged.
- `runtime_setup_s` about the same.

Repeat the run with this change rebased on `ej-prj-1`. There, the blocking
wait should make the earlier Ray visible.

## 7. Risks

- **Small expected gain.** On the default image the removed install costs
  about 1.3 s, not the 3–5 s estimated in the analysis. That estimate assumed
  a full dependency install. The gain is larger on custom images without the
  dependencies.
- **Network dependency is unchanged.** The dry run still contacts PyPI. Where
  PyPI is unreachable, the launch now proceeds and fails later in the wheel
  install (if it needs to download anything), instead of failing in the pod.
- **Running package changes on custom images.** Dependencies the wheel brings
  in are now installed while Ray runs. Already-imported modules are
  unaffected. Ray's lazy imports could see a changed package.
- **Gain hidden by the old polling.** On master, part of the gain is hidden by
  the polling interval (section 6.3).

## 8. Interaction with `ej-prj-1`

- **The two change different code and merge independently.** This change is
  in the template; `ej-prj-1` changes the provisioner.
- **They complement each other.** `ej-prj-1`'s blocking wait (0.2 s polling)
  is what turns an earlier Ray into an earlier launch.
- **The gains won't fully add up.** `ej-prj-1` starts skylet while it waits
  for Ray. Once Ray comes up quickly enough, the launch is bounded by the
  skylet start (about 1.7–3.7 s on kind) instead of by Ray.

## 9. Follow-up work

1. Run the A/B benchmark: `ej-prj-0` against master, and #1 + #2 + #3 against
   `ej-prj-1`.
2. Bake `setuptools<70` and the Ray patches into the default image. Keep the
   runtime steps as fallbacks for custom images.
3. Consider `imagePullPolicy: IfNotPresent` for images pinned by digest or
   version, as opposed to `:latest`.
4. For HA, consider installing the cluster's wheel from the persistent volume
   on recovery instead of PyPI. Recovery then gets the matching version, and
   HA pods can start Ray first too.
