# Launch latency analysis: `ej/job.yaml` on `kind-skypilot`

## Scope

- **Task:** `ej/job.yaml` (`multinode-bench`): 4 nodes, `cpus: 1+`, `memory: 1+`, `infra: k8s/kind-skypilot`.
- **Log analysed:** `ej/oct27_1039.log`, the run starting at 10:37:38. There is no `oct27_1037.log`; this is the closest run. The other runs in `ej/` were used for comparison.
- **What the log covers:** provisioning only, up to `✓ Cluster launched`. It doesn't include job submission or how long the `run:` step took.
- **Code version:** SkyPilot `1.0.0-dev0`, commit `1f9136510d0c` (dirty), Ray 2.56.1.

## Where the time goes (23.1 s total)

| Stage | Wall time | Share |
|---|---|---|
| K8s services/RBAC, `create_namespaced_pod` x4, waiting for pods to reach Running | 4.1 s | 18% |
| `internal_file_mounts` | 1.0 s | 4% |
| `setup_runtime_on_cluster` (dev wheel install) | 2.1 s | 9% |
| **Waiting for all 4 Ray nodes to join** | **14.0 s** | **61%** |
| `start_skylet_on_head_node` | 1.6 s | 7% |

The other runs show the same pattern:

| Log | Ray wait | Total | Ray share |
|---|---|---|---|
| `oct27_1022.log` | 17.6 s | 27.1 s | 65% |
| `oct27_1033.log` | 20.1 s | 29.9 s | 67% |
| `oct27_1039.log` | 14.0 s | 23.1 s | 61% |
| `oct30_1022.log` | 20.2 s | 30.2 s | 67% |

## Root causes

### 1. Ray starts late inside the pods
The pod startup script (`sky/templates/kubernetes-ray.yml.j2`, around lines 1457–1500) runs these steps one after another before it starts Ray:

1. `uv pip install 'setuptools<70'`, which replaces the setuptools 84.0 already in the image.
2. `uv pip install 'skypilot[kubernetes,remote]'`, which resolves 107 packages and downloads **skypilot 0.14.2 from PyPI**.
3. Applies 6 Ray source patches (`apply_patches.py`).
4. `ray stop`, then `ray start --head`, followed by `RAY_HEAD_WAIT_INITIALIZED_COMMAND`.

The PyPI install in step 2 is wasted work. About 2 s later, `setup_runtime_on_cluster` uninstalls it and installs the dev wheel:

```
- skypilot==0.14.2
+ skypilot==1.0.0.dev0 (from file:///home/sky/.sky/wheels/.../skypilot-1.0.0.dev0-py3-none-any.whl)
```

The workers loop until the head's GCS port is reachable (`kubernetes-ray.yml.j2`, around line 1536). Only then do they run their own `ray start`, so any delay on the head pushes back every worker.

### 2. Each readiness check is slow
`check_ray_port_and_cluster_healthy()` (`sky/provision/provisioner.py:790`) runs `RAY_ALIVE_NODE_COUNT_COMMAND` (`sky/provision/instance_setup.py:360`) through `kubectl exec`. That command starts three separate Python interpreters:

1. one to read the Ray port file,
2. one that imports `sky.utils.message_utils` to encode the payload,
3. one that imports `ray._raylet.GcsClient` to count alive nodes.

The checks were logged at 45.67 → 49.67 → 53.33 → 56.68 → 59.67, so each probe costs **about 2.3–4 s**, and the loop also sleeps 1 s between probes (`provisioner.py:834-851`). The launch therefore finds out the cluster is ready up to a full probe cycle after it actually is.

### 3. The setup stages run one after another
In `sky/provision/provisioner.py:760-910`, these steps run in order:

`internal_file_mounts` → `setup_runtime_on_cluster` → Ray wait → `start_skylet_on_head_node`

Skylet startup, and the head's runtime setup, only need the head's Ray up, not all the workers.

### 4. A smaller cost: `imagePullPolicy: Always`
`kubernetes-ray.yml.j2:612` sets `imagePullPolicy: Always`. Every pod therefore checks the registry (the "Pulling image" lines in the log), even on kind where the image is already cached. This step takes about 3.8 s of the pod-wait phase.

## Top 3 refactoring ideas

### 1. Make the pod bootstrap Ray-first and stop installing SkyPilot from PyPI in it
- Remove `pip install skypilot[kubernetes,remote]` from the template's Ray step. Nothing in the pod needs SkyPilot before the dev wheel arrives. The host-network probe is already shipped inline for this exact reason (`instance_setup.py:285-292`).
- Build the `setuptools<70` pin and the Ray patches into the default image. Keep the runtime step only as a fallback that runs when they're missing, for custom `image_id` images.
- Start `ray start --head` as early as possible, in parallel with any remaining installs, so workers can join sooner.

**Expected gain:** about 3–5 s on the critical path, and it removes a PyPI round-trip from every launch.

### 2. Replace the polling loop with one blocking wait on the head
- Replace the `while True: probe(); sleep(1)` loop with a single `kubectl exec` that runs one Python process on the head. That process reads the port, opens one `GcsClient`, and checks `get_all_node_info()` every 0.2 s until `alive == num_nodes` or a timeout. It then prints the payload once.
- If you keep the loop instead, at least merge the three Python calls in `RAY_ALIVE_NODE_COUNT_COMMAND` into one interpreter.

**Expected gain:** about 3–6 s. The wait ends within about 0.2 s of the last worker joining.

### 3. Run head-only setup in parallel with waiting for the workers
- Wait for the head GCS first. Then run `start_skylet_on_head_node`, and the head's `setup_runtime`, in a thread while the workers are still joining. Gate only job submission on the full node count.
- Use `imagePullPolicy: IfNotPresent` for pinned, versioned image tags.

**Expected gain:** about 1.6 s from overlapping skylet startup, plus up to about 3 s from the pull-policy change on local clusters.

## Expected outcome

Together, these changes could bring this 4-node kind launch from **about 23 s to roughly 8–11 s**. This is an estimate from the log timings and hasn't been measured.

## Risks and testing

- **Idea 1** changes the pod bootstrap, which every Kubernetes launch uses. Test with custom `image_id` images (non-Debian, no Ray installed), with host-network mode, and in offline or minimal environments.
- **Idea 3** changes the order of steps in `sky/provision/provisioner.py`, which is shared with cluster restarts and recovery. Run `/smoke-test --kubernetes` on any change.
- To also measure job submission and the `run:` step, rerun with `SKYPILOT_DEBUG=1` and capture `sky logs` output, in addition to `sky logs --provision`.
