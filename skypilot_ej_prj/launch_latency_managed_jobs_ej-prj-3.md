# Managed-job launch latency on `ej-prj-3`

- **Command:** `sky jobs launch -y -c ej-prj-3 ej/job.yaml` (4 nodes on kind).
- **Setup:** local API server in `--deploy` mode, so the jobs controller runs on the server (consolidation mode).
- **Runs:** 5 in a row (managed jobs 4–8), one at a time. Each run waited until the previous job's pods were gone.
- **Sources:**
  - client wall clock;
  - `spot` rows (`created_at`, `start_at`, `end_at`) and `job_events` in `~/.sky/spot_jobs.db`;
  - `cluster_events` in `~/.sky/state.db` (1 s resolution);
  - one extra run traced with `SKYPILOT_TIMELINE_FILE_PATH` (job 9).

## 1. Result

| | Median | Range |
|---|---|---|
| `sky jobs launch` client total | **63.0 s** | 57.4–64.9 |
| `sky launch` client total, same branch and YAML (from `ray_start_sooner_ej-prj-3.md`) | 32.0–32.4 s | |

The managed job takes about **31 s longer** than a plain launch. Almost all of that is controller and polling overhead, not provisioning.

## 2. Where the time goes (median of 5)

| Phase | Median | Range | Managed-job only? |
|---|---|---|---|
| A. Client start → job row created (client start ~1.6 s, then the server's `jobs.launch` request) | 7.5 | 6.8–8.7 | yes |
| B. Row created → controller picks it up (STARTING) | 2.5 | 2.3–3.7 | yes |
| C. STARTING → provisioning begins | 2.8 | 2.7–3.8 | mostly |
| D1. Provisioning → pods up | 5.0 | 4.0–5.0 | no |
| D2. Pods up → runtime set up (Ray, skylet) | 11.0 | 10.0–16.0 | no |
| D3. → job submitted on the cluster | 4.2 | 3.9–5.1 | no |
| E. Job runs on the cluster (an `echo` on 4 nodes) | 7.7 | 7.4–7.8 | no |
| F. Job finished → controller marks it RUNNING | 3.1 | 2.8–3.5 | yes |
| **G. RUNNING → SUCCEEDED** | **14.5** | 9.0–17.9 | yes |
| H. SUCCEEDED → client exits | 4.5 | 1.9–6.5 | yes |
| **Total** | **63.0** | 57.4–64.9 | |

- **The shared launch path is about the same as a plain launch.** D1+D2+D3+E is 27.9 s here, against about 28 s inside `sky launch`. So the ej-prj-0/1/3 changes carry over to managed jobs.
- **About 35 s is managed-job overhead:** A+B+C+F+G+H.
- **Phase F means the controller is late.** On every run, the job had already finished on the cluster before the controller even marked it RUNNING.

## 3. Causes, from code and trace

### G (14.5 s): the monitor loop's first status check

`controller.py::_status_check_gap_seconds(first_check=True)` sleeps `random.uniform(0, JOB_STATUS_CHECK_GAP_SECONDS=15)` before the first status check. The check itself is a network check plus 1–2 `kubectl exec` calls to the head (about 1.6–3.4 s in the trace). Only then does it see SUCCEEDED.

- **Effect:** every managed job finishes 0–15 s (7.5 s on average) after its job ends.
- **Why it's there:** the jitter spreads out monitor loops that start together.
- **Why it's wasted on this path:** the controller already polled the job's status during F (`_wait_until_job_starts_on_cluster`) and saw it past INIT. It throws that status away, then sleeps before asking again.

### F (3.1 s): waiting for the job to start

After `sky.launch` returns, `_wait_until_job_starts_on_cluster` makes one status call, then one `get_job_timestamp` call, each a `kubectl exec` of about 1.6 s.

### A (7.5 s): the server's `jobs.launch` request

- **No timeline spans for most of it.** In the traced run, about 5.7 s of `sky.jobs.server.core.launch` had no instrumented spans.
- **One confirmed contributor: three `sky check` runs for storage.** The client shows "Checking infra choices..." 3 times.
  - `storage.get_cached_enabled_storage_cloud_names_or_refresh()` is called 3 times per launch: `jobs/server/core.py:191`, and `controller_utils.py:438` and `:1188`.
  - With only Kubernetes enabled, the cached storage-cloud list is empty, and `get_cached_enabled_clouds_or_refresh` treats empty as "refresh". So every call runs a full check of every allowed context, including the remote GKE one.
  - Measured directly: 1.5 s in a cold process, 0.22 s warm, every time.
- **The rest is unaccounted for.** It's probably the optimizer, DAG dumps and admin policy. A profiler (py-spy) would pin it down.

### B (2.5 s) and C (2.8 s): handoffs

- **B:** the scheduler's handoff to a controller process.
- **C:** the controller sends the launch as its own API request. In the trace, `write_cluster_config` took 3.1 s here, against 0.5–0.8 s in a plain `sky launch`, probably because the server process is busy with other work.

### H (4.5 s): the client's log stream

The client follows the job through status polls, with 2–5 s gaps (`_LOG_STREAM_CHECK_CONTROLLER_GAP_SECONDS=5`), so it notices SUCCEEDED late.

## 4. Optimization ideas, by expected gain

| # | Idea | Expected gain | Risk |
|---|---|---|---|
| 1 | Skip the first-check sleep when the launch already observed the job. Pass the status seen in `_wait_until_job_starts_on_cluster` into the monitor loop. If it's terminal, handle it at once; otherwise make the first check sooner (e.g. `uniform(0, 3)`). | **~7–12 s** for short jobs; up to 15 s off every job's end | Low. Jitter is still kept for resumed controllers (the thundering-herd case) |
| 2 | Return the job's submitted-at time with the status in F, so it takes one `kubectl exec` instead of two. Gets cheaper again with gRPC (`SKYPILOT_ENABLE_GRPC=1`), as measured earlier. | ~1.6 s | Low |
| 3 | Cache "no storage cloud enabled" for the rest of the request (`lru_cache(scope='request')`) instead of refreshing on each of the 3 calls. | 0.5–4 s | Low. A storage cloud enabled mid-request is only seen by the next request |
| 4 | Make the client notice the end sooner: a shorter status gap once the job is RUNNING, or a push from the controller. | ~2–4 s | Low |
| 5 | Profile the remaining ~4 s of `jobs.launch`, and the 3 s `write_cluster_config` inside the controller's launch. | unknown | — |

Ideas 1–4 together could bring a short managed job from about 63 s to about 45 s. None of them touches the provisioning path.

## 5. Caveats

- **Only one trace.** Job 9 was traced, but at the time a stray jobs controller cluster existed on kind (see §6). That added a 4 s controller status refresh plus a 1.7 s head command to `jobs.launch` (72 s total). The 5 untraced runs had no such cluster, and their numbers above are clean.
- **The pod-image change in `ej-prj-3` isn't measured.** It needs a published image (see `ray_start_sooner_ej-prj-3.md`).

## 6. Environment changes left behind

- **A stray jobs controller cluster.** While enabling tracing, the API server was restarted once without `--deploy`. That turned consolidation mode off, so a test job launched a real jobs controller cluster, `sky-jobs-controller-48017c75` (4 CPU, 16 GB), on kind. It is still up.
- **The API server runs on a temporary config.** It is the user's config plus `jobs.controller.consolidation_mode: true`, at `/tmp/claude-1000/-run-media-ejiang-2T2-Projects-skypilot/17cc7554-b867-4c70-ae9c-b1ee77b2fc34/scratchpad/consolidated_config.yaml`, with timeline tracing on.
- **To restore:**

  ```bash
  sky down sky-jobs-controller-48017c75
  sky api stop && sky api start --deploy
  ```

## 7. Ideas 1 and 2 implemented: branch `ej-prj-job0`

- **Branch:** `ej-prj-job0`, from `origin/master` `815e61a76`. Commit `30337ed94`, pushed to the fork.
- **Idea 2:** `JobLibCodeGen.get_job_status_and_submitted_timestamp` and `managed_job_utils.get_job_status_and_submitted_at` return the status and the submitted-at time in one head command.
  - It calls only existing job_lib functions, so it works on older cluster runtimes.
  - With gRPC, or if the time is missing, it falls back to the old second call.
- **Idea 1:** `StrategyExecutor.pop_job_status_seen_at_launch()` hands the monitor loop the status that the launch (or a recovery) saw.
  - If that status is terminal, the first check runs immediately.
  - Otherwise the jittered first gap is unchanged.

A/B on kind: 4 nodes, the same `ej/job.yaml`, 5 runs each, alternated by restarting `sky api start --deploy` (master: jobs 10–14; branch: jobs 15–19).

| Phase | master | `ej-prj-job0` | Δ |
|---|---|---|---|
| F. Job finished → RUNNING | 3.0 (2.6–3.5) | 1.3 (0.9–1.8) | −1.7 |
| G. RUNNING → SUCCEEDED | 7.7 (4.9–10.5) | 3.4 (3.3–3.4) | −4.3 |
| H. SUCCEEDED → client exits | 2.8 (2.4–3.6) | 4.6 (3.6–7.9) | +1.8 (client poll alignment, noise) |
| Total | 62.9 (61.7–66.1) | 58.9 (58.0–64.0) | **−4.0** |

- **The new path ran in all 5 branch runs.** The controller logged "already finished when its launch returned" each time.
- **The master runs were lucky.** On master, G's expected value is about 7.5 s of random wait plus 3.4 s of checks, so about 11 s. The 5 master runs drew short random waits, which makes the −4.3 s measured gap smaller than the long-run gain (about −7.5 s on average).
- **About 3.4 s of G remains:** a network check plus two head commands (status and end time).
- **Possible next step:** reuse the terminal status the launch saw and fetch the end time along with it. That would cut G to about one command.

## 8. Combined branch `ej-prj-3-job1`

- **Branch:** `ej-prj-3` with `ej-prj-job0` cherry-picked on top (`81bd4cf57`, a clean cherry-pick).
- **Code it contains:** ej-prj-0, ej-prj-1, ej-prj-bonus, ej-prj-3 (minus the image change, which needs a published image) and job0.
- **Unit tests:** 3141 passed, 2 failed.
  - `test_details_is_none_when_no_state_or_failure` also fails on master.
  - `test_pending_volume_probe::...one_cluster_event_per_second` is timing-sensitive; it passes when run alone.
- **Runs:** 5 runs (jobs 20–24), then 5 fresh master runs in the same session (jobs 25–29, "master2").

| Phase (median) | master | master2 | job0 | ej-prj-3 | **combined** |
|---|---|---|---|---|---|
| D. Provisioning → job submitted | 26.8 | 27.2 | 25.5 | 19.2 | **18.7** |
| F. Job finished → RUNNING | 3.0 | 2.9 | 1.3 | 3.1 | **1.8** |
| G. RUNNING → SUCCEEDED | 7.7 | 15.8 | 3.4 | 14.5 | **3.4** |
| H. SUCCEEDED → client exits | 2.8 | 2.6 | 4.6 | 4.5 | 6.7 (2.6–12.3) |
| **Client start → SUCCEEDED** | 60.4 | 70.1 | 55.0 | 59.6 | **45.3** |
| Client total | 62.9 | 72.4 | 58.9 | 63.0 | **57.6** (46.4–67.1) |

- **The gains add up as expected.** The controller has the job SUCCEEDED by 45.3 s, against 60–70 s on master: the provisioning gain (~8 s) plus job0 (~1.5 s in F, and G reduced to a steady 3.4 s).
- **Run 20 was a warm-up** right after the API server restart: A took 13.4 s and C 9.0 s.
- **The client's exit is now the largest source of variance** (H: 2.6–12.3 s).
  - The two runs where H was short finished in 46.4 s and 46.9 s end to end.
  - Likely cause, not yet confirmed: the client's 5 s status poll (`_LOG_STREAM_CHECK_CONTROLLER_GAP_SECONDS`), and its wait for the controller to save the job's logs after SUCCEEDED (`_PERSISTED_LOG_WAIT_TIMEOUT_SECONDS`).
  - This is idea 4, the next target.

## 9. Why the client's exit varies (confirmed with clean traces)

**Cause.** While a managed job isn't RUNNING, the client's log follower re-checks the job's status only every `JOB_STATUS_CHECK_GAP_SECONDS` (15 s). This is in `_stream_logs_by_id`, `sky/jobs/utils.py:2975-3010`. Meanwhile it reads the controller log every 1 s, but only for the spinner text.

**Two outcomes:**
- **The check catches RUNNING:** the client streams the logs from the cluster, then polls the status every 1 s and exits about 2 s after SUCCEEDED.
- **RUNNING falls between checks:** the next check, up to 15 s later, already sees SUCCEEDED, and the client prints the saved logs then.

**Why job0 made it worse.** `ej-prj-job0` shortened the RUNNING window to about 3.4 s, so it now usually falls between two checks.

**Method.** 7 runs on `ej-prj-3-job1` (jobs 36–42), using `trace_job2.py`. Each run was verified to be the only managed job in progress; `jobs_seen` had one id. The client ran with `SKYPILOT_DEBUG=1`, so every status check shows up as a server-timestamped "The log is not ready yet" line.

| Run | Client status checks (s) | RUNNING | SUCCEEDED | Caught RUNNING? | Exit after SUCCEEDED | Total |
|---|---|---|---|---|---|---|
| probe | 10.6, 25.6, 40.6 | 42.3 | 45.6 | no | 11.7 | 57.2 |
| r1 | 10.8, 25.8, 40.8 | 41.4 | 44.6 | no | 12.6 | 57.2 |
| r2 | 10.6, 25.6, 40.6 | 41.2 | 44.5 | no | 12.6 | 57.1 |
| r3 | 10.7, 25.7, (40.7 → RUNNING) | 40.7 | 44.2 | **yes** | 1.6 | **45.7** |
| r4 | 10.9, 25.9, 40.9 | 42.1 | 45.5 | no | 11.8 | 57.3 |
| r5 | 10.8, 25.8, (40.8 → RUNNING) | 40.5 | 44.1 | **yes** | 2.2 | **46.3** |
| r6 | 11.1, 26.1, (41.1 → RUNNING) | 41.1 | 44.5 | **yes** | 2.1 | **46.5** |

**What the traces show:**
- **The client's checks run every 15.02 s exactly,** starting about 10.8 s after the CLI starts.
- **Provisioning is so steady now** that RUNNING always lands within about 1.5 s of the third check (40.5–42.3 s).
- **A sub-second race decides the result.** If the check comes first, the run takes 57 s; if RUNNING comes first, 46 s.
- **The 11–12 s penalty is the rest of the 15 s gap, less the 3.4 s RUNNING window.** The "missed" exits came 0.1–0.4 s after the next check, so the wait for the saved logs is negligible.

**Fix idea.** Re-check the status as soon as the controller log, which the loop already reads every second, has new lines. The controller writes there just before it marks the job RUNNING and when the job finishes. Expected result: every run takes about 46 s, with no extra database queries while nothing changes.

## 10. Fix implemented: check the status when the controller logs (`ej-prj-job0`, uncommitted)

**What changed** (`sky/jobs/utils.py`):
- `_read_controller_log` does what `read_provision_status_from_log` did, and also reports whether a plain (non-payload) line was read. `read_provision_status_from_log` now wraps it.
- The wait loop in `_stream_logs_by_id` re-checks the job's status as soon as the controller has logged a line of its own. The 15 s check is still there as a fallback.
- The controller writes only about 15 plain lines while a cluster is being set up, so the change adds few database queries.

**Tests:**
- 3 new unit tests: the reader reports plain lines, including when a line is still being written; the loop checks right after the controller logs; and it still checks every 15 s otherwise.
- 2 existing tests were repointed to patch the new reader.
- Jobs and skylet suites: 1532 passed, 1 failed. The failure (`test_details_is_none_when_no_state_or_failure`) also fails on master.

**On kind:** 6 clean runs (jobs 43–48), no `SKYPILOT_DEBUG`, `ej-prj-job0` with the fix.

| Run | RUNNING | SUCCEEDED | First log line | Streamed before SUCCEEDED | Exit after SUCCEEDED |
|---|---|---|---|---|---|
| r1 | 59.8 | 63.2 | 61.4 | yes | 2.1 |
| r2 | 47.3 | 50.9 | 49.3 | yes | 2.3 |
| r3 | 47.2 | 50.7 | 49.0 | yes | 2.3 |
| r4 | 47.7 | 57.9 | 50.4 | yes | 2.9 |
| r5 | 49.8 | 53.3 | 51.8 | yes | 2.3 |
| r6 | 46.4 | 49.9 | 48.6 | yes | 2.4 |

**Reading the results:**
- **The client caught RUNNING every time,** so it streamed the logs as the job ran. It exited 2.1–2.9 s after SUCCEEDED, against 1.6–12.6 s before (7 traced runs, §9).
- **r1** was the first run after the server restart.
- **r4's** launch saw the job still RUNNING, so the controller kept its jittered first check, as designed.
- **These totals don't include the provisioning changes,** because `ej-prj-job0` is based on master.

## 11. Combined branch with the client fix

- **Branch:** `ej-prj-3-job1`, now with the client fix cherry-picked on top (`f506c2f1c`; on `ej-prj-job0` it is `7c26bd708`, pushed).
- **Runs:** 6 clean runs (jobs 49–54) after an API server restart, without `SKYPILOT_DEBUG`.

| Phase (median) | master | master (re-run) | combined | **combined + client fix** |
|---|---|---|---|---|
| Provisioning → job submitted | 26.8 | 27.2 | 18.7 | 19.0 |
| Job finished → RUNNING | 3.0 | 2.9 | 1.8 | 1.7 |
| RUNNING → SUCCEEDED | 7.7 | 15.8 | 3.4 | 3.4 |
| SUCCEEDED → client exits | 2.8 | 2.6 | 6.7 (2.6–12.3) | **2.7 (2.4–3.0)** |
| Client start → SUCCEEDED | 60.4 | 70.1 | 45.3 | 46.4 |
| **Client total** | 62.9 | 72.4 | 57.6 (46.4–67.1) | **49.0 (46.4–58.0)** |

- **The exit spread is gone:** every run exited 2.4–3.0 s after SUCCEEDED.
- **The 58.0 s run was a warm-up:** it was the first after the restart, and its submit and launch handoff took 10.4 s and 9.4 s.
- **Without it:** 46.4–51.6 s, median 48.8 s. That's about 14 s faster than master (62.9 s), and 23 s faster than today's re-run of master (72.4 s).

## 12. Storage-cloud lookup once per request (`ej-prj-job1`)

Idea 3 of §4 (the storage check in A). Committed as `d37133a6f` on `ej-prj-job1` (pushed) and cherry-picked onto `ej-prj-3-job1` (renamed from `ej-prj-3-job0`) as `7aeec8ede`. Design: `launch_latency_consolidated_design.md` §12.

- **Cause:** with no storage cloud enabled, each of the three lookups in `jobs.launch` re-ran the storage check (0.3–0.7 s warm).
- **Fix:** reuse the answer, including an empty one, for the rest of the request, keyed by workspace and config.
- **First result** (jobs 73–78, against an earlier session): A 7.3 → 6.5 s; client total 48.8 → 46.9 s. **Superseded by §13:** that comparison mixed sessions.
- **Probed gain:** `jobs.launch` in consolidation mode looks up storage twice, not three times: 4.8 s, then 0.2 s on master; 4.8 s, then 0.001 s with the cache. The saving is about 0.2 s. The first lookup (4.8 s of A) remains; caching it across requests is the next step.

## 13. A/B against master (interleaved, all changes)

Master `815e61a76` against `ej-prj-3-job1` `7aeec8ede`, 4 alternating blocks of 1 warm-up + 4 measured runs (8 measured runs per arm, jobs 79–98). Details: `launch_latency_consolidated_design.md` §13.

| Phase (median) | master | **`ej-prj-3-job1`** |
|---|---|---|
| A | 6.8 | 6.8 |
| B | 2.7 | 2.5 |
| C | 3.0 | 3.1 |
| D | 24.9 | **22.5** (19.1–20.1 in 4 runs, about 25 in 4) |
| E | 7.6 | 7.7 |
| F | 3.2 | **1.4** |
| G | 11.1 (3.7–16.1) | **3.5** (3.4–3.6) |
| H | 2.8 (1.4–6.7) | **2.5** (1.3–2.9) |
| Client start → SUCCEEDED | 59.0 | **48.4** |
| **Client total** | 61.5 (58.1–67.4) | **50.1** (46.1–54.0) |

