# SkyPilot Take-Home Task for System Engineering: Design Doc

---

### **Requirement 1: Your prototype on a branch:**

**List:** [https\://github.com/siliconej/skypilot/branches](https://github.com/siliconej/skypilot/branches)  
**Consolidated single PR:** [https\://github.com/skypilot-org/skypilot/compare/master...siliconej:skypilot:ej-prj-3-job1?expand=1](https://github.com/skypilot-org/skypilot/compare/master...siliconej:skypilot:ej-prj-3-job1?expand=1) 

### **Requirement 2: Your design doc (markdown, in-repo or linked):**

This Doc, see details at Project Write-up below,  
Markdown version: [https\://hifiweekly.github.io/skypilot\_ej\_prj/overview.md](https://hifiweekly.github.io/skypilot_ej_prj/overview.md) 

### **Requirement 3: Your benchmark script and a result table:**

**Benchmark script:**  
[https\://hifiweekly.github.io/skypilot\_ej\_prj/reproduce\_benchmarks.py](https://hifiweekly.github.io/skypilot_ej_prj/reproduce_benchmarks.py)

**Read Me:**  
[https\://hifiweekly.github.io/skypilot\_ej\_prj/README.html](https://hifiweekly.github.io/skypilot_ej_prj/README.html) 

**\- Sample benchmark run:**  
python ej/reproduce\_benchmarks.py jobs-ab \--nodes 6      \# \~22 min, 8 measured runs per side  
python ej/reproduce\_benchmarks.py pod-probe \--nodes 6    \# \~2 min  
python ej/reproduce\_benchmarks.py pull-policy \--nodes 6  \# \~5 min  
python ej/reproduce\_benchmarks.py storage-probe          \# \~6 min  
python ej/reproduce\_benchmarks.py sky-launch-ab \-- run \--candidate-ref ej-prj-3-job1 \--task ej/job.yaml \--runs 5

**Result Table:** [https\://hifiweekly.github.io/skypilot\_ej\_prj/report.html](https://hifiweekly.github.io/skypilot_ej_prj/report.html)

**TL;DR: 6 nodes, medians of 8 interleaved runs per side (range), seconds:**  
Consult the UML Sequence diagram in SVG for Phase definition and transition: [https\://hifiweekly.github.io/skypilot\_ej\_prj/sky\_jobs\_launch\_sequence\_i2.svg](https://hifiweekly.github.io/skypilot_ej_prj/sky_jobs_launch_sequence_6nodes.svg)    
┌─────────────────────────────┬───────────────────┬──────────────────┬────────┐  
│ Phase (s)                   │ master            │ ej-prj-i2-5      │ Change │  
├─────────────────────────────┼───────────────────┼──────────────────┼────────┤  
│ A: client start → job row   │  6.7 (6.4–7.1)    │  7.0 (6.9–7.4)   │  \+0.2  │  
│ B: row → STARTING           │  2.6 (2.2–4.4)    │  2.5 (2.1–4.4)   │  −0.1  │  
│ C: STARTING → provisioning  │  3.4 (3.1–3.9)    │  2.9 (2.5–4.0)   │  −0.4  │  
│ **D: provisioning → submitted │ 37.4 (29.5–124.8) │ 17.8 (13.8–21.9) │ −19.6**  │ **\<\< \-52.4%**  
│ E: job runs                 │  8.5 (8.2–8.5)    │  6.7 (6.4–6.8)   │  −1.8  │  
│ F: job ended → RUNNING      │  2.8 (1.8–3.6)    │  2.6 (2.0–3.1)   │  −0.2  │  
│ **G: RUNNING → SUCCEEDED      │ 14.5 (10.3–17.6)  │  0.1 (0.1–1.6)   │ −14.4**  │ \<\< **\-99.3%**  
│ H: SUCCEEDED → client exits │  2.8 (2.3–5.4)    │  3.9 (2.6–5.1)   │  \+1.2  │  
├─────────────────────────────┼───────────────────┼──────────────────┼────────┤  
│ **Client total                │ 79.1 (70.0–166.7) │ 45.0 (39.1–49.2) │ −34.0  │ \<\< \-43.0%**  
└─────────────────────────────┴───────────────────┴──────────────────┴────────┘

Against master, ej-prj-i2-5 run diff: sky launch by 14.5 s (33%) and sky jobs launch by 34.0 s (43%). Every ej-prj-i2-5 launch beat every master launch on both benchmarks, and all 40 launches succeeded. All run done on Ubuntu 26.04, i7-8700K, 32GB memory

### **Requirement 4: A short README explaining how to reproduce your numbers from a clean machine.**

**Read Me:**  
[https\://hifiweekly.github.io/skypilot\_ej\_prj/README\_i2.html](https://hifiweekly.github.io/skypilot_ej_prj/README_i2.html)

**Support Files:**  
[https\://hifiweekly.github.io/skypilot\_ej\_prj/job.yaml](https://hifiweekly.github.io/skypilot_ej_prj/job.yaml) (same as the file in the pdf given)  
[https\://hifiweekly.github.io/skypilot\_ej\_prj/task\_bytecode.yaml](https://hifiweekly.github.io/skypilot_ej_prj/task_bytecode.yaml)  
[https\://hifiweekly.github.io/skypilot\_ej\_prj/task\_published\_ifnotpresent.yaml](https://hifiweekly.github.io/skypilot_ej_prj/task_published_ifnotpresent.yaml)  
[https\://hifiweekly.github.io/skypilot\_ej\_prj/bench\_launch\_latency.py](https://hifiweekly.github.io/skypilot_ej_prj/bench_launch_latency.py) (a pending utility script in branch)

**Sample Run Result \#1: Ubuntu 26.04: [See Appendix C](#bookmark=id.ywa1lch5zkw3)**  
**Sample Run Result \#2: MacOS Tahoe 26.7.1: [See Appendix D](#bookmark=id.2t62vgupmpad)**

---

## Project Write-up

The refactoring in these branches ([https\://github.com/siliconej/skypilot/branches/](https://github.com/siliconej/skypilot/branches/yours)) is a set of optimizations stacked in a progressive fashion. The svg diagrams linked here (4 nodes test \[[https\://hifiweekly.github.io/skypilot\_ej\_prj/sky\_jobs\_launch\_sequence.svg](https://hifiweekly.github.io/skypilot_ej_prj/sky_jobs_launch_sequence.svg)\], 6 nodes test \[[https\://hifiweekly.github.io/skypilot\_ej\_prj/sky\_jobs\_launch\_sequence\_6nodes.svg](https://hifiweekly.github.io/skypilot_ej_prj/sky_jobs_launch_sequence_6nodes.svg)\]) show the different updates that are prepared for your review. Overall there are two parts of the refactoring, first set focuses on section D (cycles between provision and job submission), second set focuses on the rest of the whole workflow. I also implemented the bonus feature and fixed a warning on linux. See more detailed below.

**Oct 30 \~ Oct 31: First Iteration**

The priority of these optimizations is analyzed and determined based on the delta time reduction that could be achieved theoretically according to first trace analysis on the master HEAD. I also spent some time on a fast code walkthrough of the current job launch workflow ([https\://hifiweekly.github.io/skypilot\_ej\_prj/sky\_jobs\_launch\_workflow.svg](https://hifiweekly.github.io/skypilot_ej_prj/sky_jobs_launch_workflow.svg)). More than 5 optimization strategies picked during the first cut, only the benchmark results that show statistical significance were pushed to branches.

Notice that all the analysis is based on the tracing and profiling on a Linux box running Ubuntu 26.04. I did notice the subtle differences of the benchmark results between Linux and MacOS. For example, on Linux during each pod’s Runtime setup, Ray init is slower than the MacOS, so we see less effect of optimizing slow PyPI access using “uv pip install \--offline” on Linux, but potentially it could benefit MacOS’s Runtime setup speed.

A more lengthy consolidated design doc of all the optimization included so far in this project is at: [https\://hifiweekly.github.io/skypilot\_ej\_prj/launch\_latency\_consolidated\_design.html](https://hifiweekly.github.io/skypilot_ej_prj/launch_latency_consolidated_design.html).

**Nov 1 \~ Nov 2: Second Iteration**

The [second iteration](#bookmark=id.ksoj6pws011n) focused on more aggressive code refactoring on Phase D and Phase G. I also profiled all the thread waiting time. And was able to expand the increase reduction to over 40% in total.

During the refactoring, I also found a gRPC data conversion bug, causing some of the timestamps to be malformed. As the current gRPC is not enabled by default, and there are other fixes pending according to the commit logs. I didn’t enable gRPC for the benchmark. According to the profiling, by enabling the gRPC, it can also cut down the latency in Phase D, E, and F. (see [https\://hifiweekly.github.io/skypilot\_ej\_prj/grpc\_latency\_analysis.html](https://hifiweekly.github.io/skypilot_ej_prj/grpc_latency_analysis.html).

See more details in [https\://hifiweekly.github.io/skypilot\_ej\_prj/launch\_latency\_design\_after\_job2.html](https://hifiweekly.github.io/skypilot_ej_prj/launch_latency_design_after_job2.html).

### **Interaction 1 \- First Set: ej-prj-0, ej-prj-1, and ej-prj-3**

These three changes focus on the job preparation and per node job submission, and will have impact on unmanaged job scheduling by ‘sky launch’ as well.

**\* Branch ej-prj-0: Start ray in the pod before preparing SkyPilot** [https\://github.com/skypilot-org/skypilot/compare/master...siliconej:skypilot:ej-prj-0?expand=1](https://github.com/skypilot-org/skypilot/compare/master...siliconej:skypilot:ej-prj-0?expand=1)  
PR description: [https\://hifiweekly.github.io/skypilot\_ej\_prj/pr\_descriptions.html\#ej-prj-0](https://hifiweekly.github.io/skypilot_ej_prj/pr_descriptions.html#ej-prj-0) 

**\* TL;DR**

**Problem.** Each pod's startup script installed skypilot\[kubernetes,remote\] from PyPI before ray start. That install was on Ray's critical path, but nothing there needs it:  
\- Nothing before ray start imports sky, because the Ray patches and the hostNetwork probe are shipped inline.  
\- The installed copy is replaced by the cluster's own wheel later.  
\- Workers wait for the head's port, so a late head delays every worker.

**Change (kubernetes-ray.yml.j2, step 2):**  
\- Drop the PyPI install, except for HA controllers, whose recovery imports sky.  
\- Start Ray in the background right after the patches.  
\- Meanwhile, run uv pip install \--dry-run to warm uv's cache. Without it, the wheel install's resolve takes 4.5–6 s instead of about 20 ms. The dry run changes nothing in the environment and is best-effort.  
\- The marker for setup\_commands keeps its meaning.  
\- wait \$SKY\_RAY\_START\_PID keeps a Ray failure fatal.

**Gain.** About 1.3s earlier Ray on the default image, more on custom images. Alone, the old 3–5 s polling hid it: an A/B showed −2%, not significant. Combined with ej-prj-1's blocking wait, the A/B was significant: runtime setup end → launched went from 12.49s to 9.43s (−24%, p \= 0.004).

**Risks.**  
\- The dry run still depends on PyPI.  
\- On custom images, dependencies now install while Ray runs.

**Tests.**  
A new template test checks the order, including the HA case, and the rendered pod args pass bash \-n.

**Design Doc.** [https\://hifiweekly.github.io/skypilot\_ej\_prj/ray\_first\_pod\_bootstrap\_design.html](https://hifiweekly.github.io/skypilot_ej_prj/ray_first_pod_bootstrap_design.html) 

---

**\* Branch ej-prj-1: Wait for Ray in one blocking call; start skylet during the wait**  
[https\://github.com/skypilot-org/skypilot/compare/master...siliconej:skypilot:ej-prj-1?expand=1](https://github.com/skypilot-org/skypilot/compare/master...siliconej:skypilot:ej-prj-1?expand=1)  
PR description: [https\://hifiweekly.github.io/skypilot\_ej\_prj/pr\_descriptions.html\#ej-prj-1](https://hifiweekly.github.io/skypilot_ej_prj/pr_descriptions.html#ej-prj-1) 

**\* TL;DR**

**Problem.** On a fresh Kubernetes cluster, waiting for every pod's Ray to join took 61–67% of a launch. Two causes:  
\- The poll was slow. The provisioner checked once a second, but each check was a kubectl exec that started three Python interpreters (2.3–4s), so it noticed readiness up to a whole probe late.  
\- Skylet started late. It only started after the wait, although it needs only the head.

**Change (provisioner.py, instance\_setup.py):**  
\- One blocking call. A small script on the head (standard library plus ray, sent base64-encoded) keeps one GcsClient open and checks every 0.2s until exactly num\_nodes are alive or the deadline passes.  
  \- Its output matches the old probe's, plus a final SKY\_RAY\_WAIT\_DONE line.  
  \- A call that ends without that line means the exec dropped, so it is retried for the time left; it doesn't restart Ray.  
  \- A local timeout bounds a hung exec.  
\- Skylet during the wait. \_run\_alongside starts skylet in a thread, carrying over the provision log path and the request context. The thread is joined before returning, and its errors are re-raised.

**Unchanged:** the exact-count rule, the 90 s fallback, no\_restart for workers, re-launches of existing clusters, and other clouds.

**Results (kind, 4 nodes):**  
\- Runtime installed → launched: about 19.6 s → 14.7 s.  
\- A later A/B confirmed it: 12.47 s → 9.89 s (−21%, p \= 0.004).  
\- 12 new unit tests.

**Risks:** an exec held open for up to 90 s can be dropped by proxies, and the port-reading logic now exists in two copies.

**Design Doc.** [https\://hifiweekly.github.io/skypilot\_ej\_prj/ray\_wait\_and\_skylet\_overlap\_design.html](https://hifiweekly.github.io/skypilot_ej_prj/ray_wait_and_skylet_overlap_design.html) 

---

**\* Branch ej-prj-3: Skip the needless ray stop; pre-install the runtime pins in the image**  
[https\://github.com/skypilot-org/skypilot/compare/master...siliconej:skypilot:ej-prj-3?expand=1](https://github.com/skypilot-org/skypilot/compare/master...siliconej:skypilot:ej-prj-3?expand=1)   
PR description: [https\://hifiweekly.github.io/skypilot\_ej\_prj/pr\_descriptions.html\#ej-prj-3](https://hifiweekly.github.io/skypilot_ej_prj/pr_descriptions.html#ej-prj-3) 

**\* TL;DR**

**Measured first.** A timestamped trace of the pod's startup script found two waste steps before ray start:  
\- ray stop on a pod that never ran Ray: 1.4–1.5s idle, 2.9–3.8s while the pod was busy setting up.  
\- Downloading setuptools\<70 from PyPI: the image ships setuptools 84, so every pod fetched the pin. This took 0.6s usually and 12.1s in one traced run.

Because the workers wait for the head's GCS port, a late head delays the whole cluster.

**Changes:**  
1\. RAY\_STOP\_IF\_RUNNING. ray stop now runs only if pgrep finds a GCS, raylet or dashboard process. Bracketed patterns (\[g\]cs\_server) keep it from matching its own command line. Without pgrep, Ray is stopped as before, and restarts still stop Ray first.  
2\. Pins in the image. Dockerfile\_k8s, Dockerfile\_k8s\_gpu and the docs' custom-image snippet pre-install setuptools\<70 and click\<8.3.0, so pods only check them (1–2 ms). This takes effect only once the image is rebuilt and published.

**Not changed:**  
\- Turning off the dashboard: it saves 0.9 s, but not on the GCS port the workers wait for.  
\- Baking the Ray patches into the image: about 1 s, but it needs a version stamp.

**Results (kind, 4 nodes):**  
\- Head GCS up 7.31 → 5.89 s after container start.  
\- Whole cluster in Ray 12.95 → 11.37 s, every run.  
\- Launch about 2 s faster.  
\- 6 new unit tests. The image builds, but the pin change hasn't been measured on kind.

**Design Doc.** [https\://hifiweekly.github.io/skypilot\_ej\_prj/ray\_start\_sooner\_ej-prj-3.html](https://hifiweekly.github.io/skypilot_ej_prj/ray_start_sooner_ej-prj-3.html)

---

### **Iteration 1 \- Second Set: ej-prj-job0 and ej-prj-job1**

These two changes focus on the skypilot tasks before and after the job preparation and job submission, including whole job request processing, submission, job monitoring, and job exit.

**\* Branch ej-prj-job0: Fetch status and submitted-at in one call; skip the first status-check sleep; re-check status on controller logs change**  
[https\://github.com/skypilot-org/skypilot/compare/master...siliconej:skypilot:ej-prj-job0?expand=1](https://github.com/skypilot-org/skypilot/compare/master...siliconej:skypilot:ej-prj-job0?expand=1)   
PR description: [https\://hifiweekly.github.io/skypilot\_ej\_prj/pr\_descriptions.html\#ej-prj-job0](https://hifiweekly.github.io/skypilot_ej_prj/pr_descriptions.html#ej-prj-job0) 

**\* TL;DR**

**Problem.** After provisioning, a managed job lost about 15–35s to polling:  
\- Confirming the job started took two kubectl exec round trips to the head, status and then submitted-at time, about 1.6s each.  
\- The monitor loop slept a random 0–15s before its first status check, although the launch had just seen the job finish. It threw that status away.  
\- The client's log follower re-checked the status only every 15s. It usually missed the job's few seconds in RUNNING, so it noticed the end up to 15s late.

**Changes:**  
1\. One head command (JobLibCodeGen.get\_job\_status\_and\_submitted\_timestamp) returns both values. It calls only existing job\_lib functions, so older cluster runtimes work, and it falls back to the second call on gRPC or a missing time.  
2\. The first check runs at once. pop\_job\_status\_seen\_at\_launch() hands the monitor loop the status the launch saw. If it's terminal, the loop checks immediately; a running job keeps the jittered gap.  
3\. The follower re-checks the status as soon as the controller writes a plain log line (about 15 per launch). The 15s check stays as a fallback.

**Results (kind, 4 nodes):**  
\- Job finished → RUNNING: 3.0 → 1.3 s.  
\- RUNNING → SUCCEEDED: random 4.9–10.5 s → steady 3.4 s.  
\- The client exits 2.1–2.9 s after SUCCEEDED, against 1.6–12.6 s before.  
\- 14 new unit tests.

**Design Doc.** [https\://hifiweekly.github.io/skypilot\_ej\_prj/launch\_latency\_managed\_jobs\_ej-prj-3.html](https://hifiweekly.github.io/skypilot_ej_prj/launch_latency_managed_jobs_ej-prj-3.html) (Section 3, 7, and 10\)

---

**\* Branch ej-prj-job1: Look up the enabled storage clouds once per request**  
[https\://github.com/skypilot-org/skypilot/compare/master...siliconej:skypilot:ej-prj-job1?expand=1](https://github.com/skypilot-org/skypilot/compare/master...siliconej:skypilot:ej-prj-job1?expand=1)  
PR description: [https\://hifiweekly.github.io/skypilot\_ej\_prj/pr\_descriptions.html\#ej-prj-job1](https://hifiweekly.github.io/skypilot_ej_prj/pr_descriptions.html#ej-prj-job1) 

**\* TL;DR**

**Problem.**  
\- Every call ran a full check. get\_cached\_enabled\_storage\_cloud\_names\_or\_refresh() lists the clouds whose object storage can be used. With no storage cloud enabled (here, Kubernetes only), the cached list is empty, and get\_cached\_enabled\_clouds\_or\_refresh treats empty as "not checked yet". So every call re-ran the full storage check.  
\- jobs.launch repeats it. The doc assumed 3 calls per jobs.launch; the probe in §12's correction found 2 in consolidation mode.

**Change (sky/data/storage.py):**  
\- The answer, including an empty one, is cached for the rest of the request in \_enabled\_storage\_cloud\_names, through annotations.ttl\_cache(scope='request', ttl=60, lock=...).  
\- The key is the workspace and the config fingerprint. A process that never clears request caches, such as a jobs controller, keeps them apart, and the 60 s TTL limits how stale its answer can get.  
\- The cache is cleared after \_is\_storage\_cloud\_enabled runs sky check, so a cloud that check enables is seen.  
\- Unchanged: callers still get their own list, raise\_if\_no\_cloud\_access still raises, and failed lookups aren't cached.  
\- Supporting changes: annotations.ttl\_cache takes an optional lock, and check.\_config\_fingerprint became the public current\_config\_fingerprint.

**Results:**  
\- 14 new unit tests. Full unit suite: 10,704 passed; the 20 failures also fail on master or pass when run alone.  
\- The probe on kind measured the second lookup at 0.18–0.25 s on master and 0.001s with the change, so the gain is about 0.2 s, within phase A's noise.  
\- The earlier 7.3 → 6.5s comparison mixed sessions and is superseded.

**Design Doc.** Section 12 of [https\://hifiweekly.github.io/skypilot\_ej\_prj/launch\_latency\_consolidated\_design.html](https://hifiweekly.github.io/skypilot_ej_prj/launch_latency_consolidated_design.html) 

---

### **Iteration 2**

**\* Branch ej-prj-i2-2: \[Fix\] gRPC timestamps format bug**  
[https\://github.com/skypilot-org/skypilot/compare/master...siliconej:skypilot:ej-prj-i2-2?expand=1](https://github.com/skypilot-org/skypilot/compare/master...siliconej:skypilot:ej-prj-i2-2?expand=1)   
**PR Description:** [https\://hifiweekly.github.io/skypilot\_ej\_prj/pr\_descriptions\_i2.html\#ej-prj-i2-2](https://hifiweekly.github.io/skypilot_ej_prj/pr_descriptions_i2.html#ej-prj-i2-2) 

**\* TL;DR**

**Problem:**  
GetJobSubmittedTimestamp and GetJobEndedTimestamp return the time in a protobuf float, which can hold a current Unix time only to the nearest 128 s.  
With SKYPILOT\_ENABLE\_GRPC on, a managed job's start and end times were off by up to about a minute, and a short job's duration was recorded as 0 s.

**Fix:**  
Add optional double timestamp\_double to both responses. A float can't be widened in place without breaking the wire format.  
Keep setting the float, so older clients still work.  
When a skylet doesn't set the new field, read the exact value through the existing head command instead of accepting the rounded one.  
Bump SKYLET\_VERSION to 44, so existing clusters pick this up.

**Not a speedup: gRPC is off by default, so launch times don't change.**

**Design Doc.**  
[https\://hifiweekly.github.io/skypilot\_ej\_prj/grpc\_latency\_analysis.html](https://hifiweekly.github.io/skypilot_ej_prj/grpc_latency_analysis.html) 

**\* Branch ej-prj-i2-3: \[refactoring\] focusing on Phase D and Phase E**  
[https\://github.com/skypilot-org/skypilot/compare/master...siliconej:skypilot:ej-prj-i2-3?expand=1](https://github.com/skypilot-org/skypilot/compare/master...siliconej:skypilot:ej-prj-i2-3?expand=1)  
**PR Description:** [https\://hifiweekly.github.io/skypilot\_ej\_prj/pr\_descriptions\_i2.html\#ej-prj-i2-3](https://hifiweekly.github.io/skypilot_ej_prj/pr_descriptions_i2.html#ej-prj-i2-3) 

**\* TL;DR**

**Problem**  
On 6 nodes on kind, sky launch drops from 43.1 s to 26.9 s (−38%), and sky jobs launch from 71.8 s to 41.8 s (−42%).

\- Pods and Ray startup: Ray comes up sooner, which matters because every worker waits for the head.  
  \- Start Ray before preparing SkyPilot in the pod, and skip ray stop when no Ray is running.  
  \- Pre-install the runtime pins in the image, so pods don't download them at startup.  
  \- New on this branch:  
    \- Precompile runtime bytecode in the k8s images (cab8b302a).  
    \- Stop importing Ray Serve in ray start (0038650ac).  
    \- Watch pods between polls, so the last running pod is noticed sooner (16738238e).  
\- Provisioner: wait for Ray in one blocking call on the head instead of polling, and start skylet during that wait.  
\- Managed jobs:  
  \- Fetch a job's status and submitted-at time in one head command.  
  \- Skip the random first-check sleep when the launch already saw the job finish.  
  \- Re-check the status as soon as the controller writes a log line.  
  \- Look up the enabled storage clouds once per request.  
\- Bug fixes (no speedup):  
  \- Keep job timestamps exact over gRPC, which adds optional proto fields and bumps SKYLET\_VERSION to 44\.  
  \- Fail fast when a pod is lost during setup: about 5–11 s instead of about 106 s.  
\- Benchmark: add bench\_launch\_latency.py, an A/B of sky launch between two builds.

Compatibility: no API change. Generated head commands call only functions older cluster runtimes already have. The image changes take effect only once a new image is published.

This branch also includes couple of low risk optimization in Phase D.

**Tested**

A/B against 5ef55efa4 on kind-skypilot (6 nodes, 8 runs per arm, medians). The base uses the published image and the candidate a locally built bytecode image, both with IfNotPresent.

┌───────────────────────────────┬───────────┬─────────────┬─────────┬────────┐  
│            Measure            │ 5ef55efa4 │ ej-prj-i2-3 │ Change  │   p    │  
├───────────────────────────────┼───────────┼─────────────┼─────────┼────────┤  
│ sky launch wall time          │ 43.1 s    │ 26.9 s      │ −16.2 s │ 0.0002 │  
├───────────────────────────────┼───────────┼─────────────┼─────────┼────────┤  
│ sky launch provisioning total │ 25.5 s    │ 12.0 s      │ −13.4 s │ 0.0002 │  
├───────────────────────────────┼───────────┼─────────────┼─────────┼────────┤  
│ sky jobs launch client total  │ 71.8 s    │ 41.8 s      │ −30.1 s │ 0.0002 │  
├───────────────────────────────┼───────────┼─────────────┼─────────┼────────┤  
│ Provisioning → job submitted  │ 30.4 s    │ 14.6 s      │ −15.9 s │ 0.0002 │  
├───────────────────────────────┼───────────┼─────────────┼─────────┼────────┤  
│ Job ended → RUNNING           │ 3.3 s     │ 2.4 s       │ −0.9 s  │ 0.003  │  
├───────────────────────────────┼───────────┼─────────────┼─────────┼────────┤  
│ RUNNING → SUCCEEDED           │ 9.9 s     │ 3.3 s       │ −6.7 s  │ 0.07   │  
└───────────────────────────────┴───────────┴─────────────┴─────────┴────────┘

**Design Doc.**  
[https\://hifiweekly.github.io/skypilot\_ej\_prj/precompile\_runtime\_bytecode\_design.html](https://hifiweekly.github.io/skypilot_ej_prj/precompile_runtime_bytecode_design.html)   
[https\://hifiweekly.github.io/skypilot\_ej\_prj/pod\_watch\_between\_polls\_design.md](https://hifiweekly.github.io/skypilot_ej_prj/pod_watch_between_polls_design.md)

**\* Branch ej-prj-i2-4: \[refactoring\] focusing on Phase G**  
[https\://github.com/skypilot-org/skypilot/compare/master...siliconej:skypilot:ej-prj-i2-4?expand=1](https://github.com/skypilot-org/skypilot/compare/master...siliconej:skypilot:ej-prj-i2-4?expand=1)  
**PR Description:** [https\://hifiweekly.github.io/skypilot\_ej\_prj/pr\_descriptions\_i2.html\#ej-prj-i2-4](https://hifiweekly.github.io/skypilot_ej_prj/pr_descriptions_i2.html#ej-prj-i2-4)

**Summary**

Over two A/Bs on 6 nodes on kind, sky jobs launch drops by about 30 s, and RUNNING → SUCCEEDED falls to about 0.1 s. sky launch drops from 43.1 s to 26.9 s, measured on ej-prj-i2-3; the new commit only changes managed jobs code.

\- Pods and Ray startup:  
  \- Start Ray before preparing SkyPilot in the pod, and skip ray stop when no Ray is running.  
  \- Pre-install the runtime pins and precompile bytecode in the k8s images.  
  \- Stop importing Ray Serve in ray start.  
  \- Watch pods between polls, so the last running pod is noticed sooner.  
\- Provisioner: wait for Ray in one blocking call on the head instead of polling, and start skylet during that wait.  
\- Managed jobs:  
  \- Fetch a job's status and submitted-at time in one head command.  
  \- Skip the random first-check sleep when the launch already saw the job finish.  
  \- Re-check the status as soon as the controller writes a log line.  
  \- Look up the enabled storage clouds once per request.  
  \- New on this branch (b848c9c1e): get a finished job's end time with its status.  
    \- A launch that saw the job succeed hands its status and end time to the monitor loop, which then needs no head command.  
    \- The monitor loop's own status check also returns the end time.  
    \- The log follower trusts a tail exit code of SUCCEEDED, and reads the recorded status every 0.2 s until the controller writes it.  
\- Bug fixes (no speedup):  
  \- Keep job timestamps exact over gRPC, which adds optional proto fields and bumps SKYLET\_VERSION to 44\.  
  \- Fail fast when a pod is lost during setup: about 5–11 s instead of about 106 s.  
\- Benchmark: add bench\_launch\_latency.py, an A/B of sky launch between two builds.

Compatibility: no API change. Generated head commands call only functions older cluster runtimes already have. The image changes take effect only once a new image is published.

**Tested**

All A/Bs ran on kind-skypilot (6 nodes, 8 runs per arm, medians). There's no direct A/B of this branch against 5ef55efa4, so these are the two closest.

New commit, ej-prj-i2-3 vs ej-prj-i2-4 (bytecode image in both arms):

┌──────────────────────────┬─────────────┬─────────────┬────────┬────────────────────────┐  
│         Measure          │ ej-prj-i2-3 │ ej-prj-i2-4 │ Change │           p            │  
├──────────────────────────┼─────────────┼─────────────┼────────┼────────────────────────┤  
│ RUNNING → SUCCEEDED      │ 3.35 s      │ 0.12 s      │ −3.2 s │ 0.0001                 │  
├──────────────────────────┼─────────────┼─────────────┼────────┼────────────────────────┤  
│ SUCCEEDED → client exits │ 2.48 s      │ 3.84 s      │ \+1.4 s │ 0.0001                 │  
├──────────────────────────┼─────────────┼─────────────┼────────┼────────────────────────┤  
│ Client total             │ 36.07 s     │ 34.31 s     │ −1.8 s │ rank 0.038; means 0.38 │  
└──────────────────────────┴─────────────┴─────────────┴────────┴────────────────────────┘

The wait after SUCCEEDED grows because the controller saves the logs after SUCCEEDED, and the client now reaches that wait sooner. The p-value on means is weak because of one 41.6 s run.

Everything before it, 5ef55efa4 vs ej-prj-i2-3 (published image vs bytecode image, both IfNotPresent):

┌──────────────────────────────┬───────────┬─────────────┬─────────┬────────┐  
│           Measure            │ 5ef55efa4 │ ej-prj-i2-3 │ Change  │   p    │  
├──────────────────────────────┼───────────┼─────────────┼─────────┼────────┤  
│ sky launch wall time         │ 43.1 s    │ 26.9 s      │ −16.2 s │ 0.0002 │  
├──────────────────────────────┼───────────┼─────────────┼─────────┼────────┤  
│ sky jobs launch client total │ 71.8 s    │ 41.8 s      │ −30.1 s │ 0.0002 │  
└──────────────────────────────┴───────────┴─────────────┴─────────┴────────┘

**Design Doc.**  
[https\://hifiweekly.github.io/skypilot\_ej\_prj/job\_end\_time\_with\_status\_design.html](https://hifiweekly.github.io/skypilot_ej_prj/pod_watch_between_polls_design.md)

### ---

### **Bonus: 90s timeout handling** 

**\* Branch ej-prj-bonus: skip the 90s wait when a pod is deleted.**  
[https\://github.com/skypilot-org/skypilot/compare/master...siliconej:skypilot:ej-prj-bonus?expand=1](https://github.com/skypilot-org/skypilot/compare/master...siliconej:skypilot:ej-prj-bonus?expand=1)  
PR description: [https\://hifiweekly.github.io/skypilot\_ej\_prj/pr\_descriptions.html\#ej-prj-bonus](https://hifiweekly.github.io/skypilot_ej_prj/pr_descriptions.html#ej-prj-bonus) 

**\* TL;DR**

**Problem.**  
\- The pod never comes back. A pod deleted after it started, by kubectl delete pod or an eviction, stays gone: SkyPilot pods use restartPolicy: Never.  
\- The launch noticed only after about 106s.  
  \- The head's Ray wait ran its whole 90s timeout, because a worker that never joined is invisible to it.  
  \- The fallback then tried to start Ray on the missing pod, retrying six times.  
\- The error never said a pod was gone.

**Changes:**  
1\. Don't retry an exec whose pod is gone (command\_runner.is\_exec\_target\_gone). \_auto\_retry stops on permanent kubectl errors: NotFound, a completed pod, phase Failed or Succeeded. Transient errors such as "unable to upgrade connection" are still retried.  
2\. Check the instances. provision.query\_instances lists the cluster's pods. One missing, terminated or stopped in two listings 2s apart raises the new InstancesLostDuringSetupError (a ClusterSetUpError) naming it. A failing check never fails the launch.  
3\. Time-box the Ray wait into calls of at most 10s, checking the instances between calls. Nothing changes when Ray is ready in the first call.  
4\. Name the pod in errors. Any failing setup stage checks the instances before re-raising, keeping the stage's error as context.

**Results (kind):** a deleted worker fails the launch in about 11s and a deleted head in about 5s, instead of about 106s. Managed jobs recover sooner.

**Tests:**  
\- 9 new unit tests.  
\- A new smoke test, test\_launch\_fails\_fast\_when\_pod\_deleted (worker and head), deletes a pod mid-setup. Both cases passed on kind in 42–45 s, against a 45 s limit.  
\- 7aed7c84e adds the function to command\_runner.pyi; without it, CI's mypy check fails.  
\- verified both unmanaged job case (sky launch) and managed job case (sky jobs launch).

**Design Doc**: [https\://hifiweekly.github.io/skypilot\_ej\_prj/pod\_loss\_fail\_fast\_design.html](https://hifiweekly.github.io/skypilot_ej_prj/pod_loss_fail_fast_design.html) 

---

### **Fix: k8s-pool-warning suppression**

**\* Branch ej-k8s-pool-warning: suppress pool warning on linux.**  
[https\://github.com/skypilot-org/skypilot/compare/master...siliconej:skypilot:ej-k8s-pool-warning?expand=1](https://github.com/skypilot-org/skypilot/compare/master...siliconej:skypilot:ej-k8s-pool-warning?expand=1)  
PR description: [https\://hifiweekly.github.io/skypilot\_ej\_prj/pr\_descriptions.html\#ej-k8s-pool-warning](https://hifiweekly.github.io/skypilot_ej_prj/pr_descriptions.html#ej-k8s-pool-warning) 

**Problem.**  
\- The pool overflows by design. SkyPilot's Kubernetes API clients cap their connection pool at CONNECTION\_POOL\_MAXSIZE (4). Going past it is normal, for example when creating or inspecting more than 4 pods in parallel.  
\- Each extra connection prints a warning. When urllib3 closes one, it logs WARNING:urllib3.connectionpool:Connection pool is full, discarding connection: ... Connection pool size: 4\.  
\- The existing guard leaks. \_api\_logging\_decorator keeps the urllib3 logger at ERROR during API calls. But any kubernetes.client.Configuration built elsewhere, or in another thread mid-call, resets it to WARNING, so the noise reached users.  
\- It's visible today on linux, not reproducible on macOS. Each 6-node managed job's controller log from today's benchmark has 6–7 of these warnings (jobs 105–124). Neither master nor ej-prj-3-job1 includes this fix.

**Change** (sky/adaptors/kubernetes.py, commit 5bb46b39a):  
\- A filter on the urllib3.connectionpool logger drops exactly this message, and only when its pool size equals CONNECTION\_POOL\_MAXSIZE.  
\- Other pools' pool-full warnings still show, as do all other urllib3 warnings.  
\- Why a filter: a level reset doesn't undo a filter, and a filter takes no lock per record.

**Tests:** 1 new parametrized test (test\_pool\_full\_warning\_hidden\_for\_capped\_clients) checks the warning is hidden at the capped size, still shown at other sizes, and that other warnings pass through. The adaptor test file passes: 28 passed.

Design doc: [https\://hifiweekly.github.io/skypilot\_ej\_prj/k8s\_pool\_full\_warning\_design.html](https://hifiweekly.github.io/skypilot_ej_prj/k8s_pool_full_warning_design.html) 

## Appendix A: Other references

Zip file of other markdown version of the linked design docs: [https\://hifiweekly.github.io/skypilot\_ej\_prj/references.zip](https://hifiweekly.github.io/skypilot_ej_prj/references.zip)

## Appendix B: User’s Perceived Time

### **ej-prj-3-job0: sky jobs launch**

### **Linux**

(skypilot) ejiang@benjamin:\~/2T2/Projects/skypilot/ej\$ time sky jobs launch \-y \-c ej-prj-3-job0  ej/job.yaml  
'ej/job.yaml' looks like a yaml path but does not exist. Please check if the path is correct.  
It will be treated as a command to be run remotely. Continue? \[y/N\]: ^CAborted\!

real	0m13.373s  
user	0m2.104s  
sys	0m0.211s  
(skypilot) ejiang@benjamin:\~/2T2/Projects/skypilot/ej\$ time sky jobs launch \-y \-c ej-prj-3-job0  ej/job.yaml^C  
(skypilot) ejiang@benjamin:\~/2T2/Projects/skypilot/ej\$ cd ..  
(skypilot) ejiang@benjamin:\~/2T2/Projects/skypilot\$ time sky jobs launch \-y \-c ej-prj-3-job0  ej/job.yaml  
YAML to run: ej/job.yaml  
Launching managed job 'ej-prj-3-job0'...  
├── Waiting for task resources on 4 nodes.  
└── Job started. Streaming logs... (Ctrl-C to exit log streaming; job will not be killed)  
(head, rank=0, pid=1317) \[rank 0\] started at 1793467500.240161246 on ej-prj-3-job0-55-48017c75-head  
(worker1, rank=1, pid=652, ip=10.244.0.227) \[rank 1\] started at 1793467502.030684619 on ej-prj-3-job0-55-48017c75-worker1  
(worker2, rank=2, pid=699, ip=10.244.0.225) \[rank 2\] started at 1793467502.093509092 on ej-prj-3-job0-55-48017c75-worker2  
(worker3, rank=3, pid=610, ip=10.244.0.228) \[rank 3\] started at 1793467502.080429595 on ej-prj-3-job0-55-48017c75-worker3  
✓ Job finished (status: SUCCEEDED).  
✓ Managed job finished: 55 (status: SUCCEEDED).

📋 Useful Commands  
Managed Job ID: 55  
├── To cancel the job:		sky jobs cancel 55  
├── To stream job logs:		sky jobs logs 55  
├── To stream controller logs:	sky jobs logs \--controller 55  
└── To view all managed jobs:	sky jobs queue

**real	0m47.945s**  
user	0m2.568s  
sys	0m0.266s

(skypilot) ejiang@benjamin:\~/2T2/Projects/skypilot\$ time sky jobs launch \-y \-c ej-prj-3-job0  ej/job.yaml  
YAML to run: ej/job.yaml  
Launching managed job 'ej-prj-3-job0'...  
├── Waiting for task resources on 4 nodes.  
└── Job started. Streaming logs... (Ctrl-C to exit log streaming; job will not be killed)  
(head, rank=0, pid=1315) \[rank 0\] started at 1793467988.581994637 on ej-prj-3-job0-56-48017c75-head  
(worker2, rank=2, pid=565, ip=10.244.0.230) \[rank 2\] started at 1793467990.358667314 on ej-prj-3-job0-56-48017c75-worker2  
(worker3, rank=3, pid=562, ip=10.244.0.231) \[rank 3\] started at 1793467990.438768959 on ej-prj-3-job0-56-48017c75-worker3  
(worker1, rank=1, pid=563, ip=10.244.0.232) \[rank 1\] started at 1793467990.447552834 on ej-prj-3-job0-56-48017c75-worker1  
✓ Job finished (status: SUCCEEDED).  
✓ Managed job finished: 56 (status: SUCCEEDED).

📋 Useful Commands  
Managed Job ID: 56  
├── To cancel the job:		sky jobs cancel 56  
├── To stream job logs:		sky jobs logs 56  
├── To stream controller logs:	sky jobs logs \--controller 56  
└── To view all managed jobs:	sky jobs queue

**real	0m46.723s**  
user	0m2.684s  
sys	0m0.258s

(skypilot) ejiang@benjamin:\~/2T2/Projects/skypilot\$ time sky jobs launch \-y \-c ej-prj-3-job0  ej/job.yaml  
YAML to run: ej/job.yaml  
Launching managed job 'ej-prj-3-job0'...  
├── Waiting for task resources on 4 nodes.  
└── Job started. Streaming logs... (Ctrl-C to exit log streaming; job will not be killed)  
(head, rank=0, pid=1305) \[rank 0\] started at 1793468130.556993036 on ej-prj-3-job0-57-48017c75-head  
(worker1, rank=1, pid=633, ip=10.244.0.234) \[rank 1\] started at 1793468132.477528286 on ej-prj-3-job0-57-48017c75-worker1  
(worker3, rank=3, pid=595, ip=10.244.0.236) \[rank 3\] started at 1793468132.585763930 on ej-prj-3-job0-57-48017c75-worker3  
(worker2, rank=2, pid=572, ip=10.244.0.235) \[rank 2\] started at 1793468132.645533742 on ej-prj-3-job0-57-48017c75-worker2  
✓ Job finished (status: SUCCEEDED).  
✓ Managed job finished: 57 (status: SUCCEEDED).

📋 Useful Commands  
Managed Job ID: 57  
├── To cancel the job:		sky jobs cancel 57  
├── To stream job logs:		sky jobs logs 57  
├── To stream controller logs:	sky jobs logs \--controller 57  
└── To view all managed jobs:	sky jobs queue

**real	0m48.028s**  
user	0m2.512s  
sys	0m0.283s

---

### **MacOS**

(skypilot-bench) ejiang@kodaly skypilot-bench % time sky jobs launch \-y \-c last\_test ej/job.yaml  
YAML to run: ej/job.yaml  
Launching managed job 'last\_test'...  
├── Waiting for task resources on 4 nodes.  
└── Job started. Streaming logs... (Ctrl-C to exit log streaming; job will not be killed)  
(head, rank=0, pid=1435) \[rank 0\] started at 1793553909.024757212 on last-test-8-a83a3d8f-head  
(worker3, rank=3, pid=666, ip=10.244.0.166) \[rank 3\] started at 1793553910.141855254 on last-test-8-a83a3d8f-worker3  
(worker2, rank=2, pid=584, ip=10.244.0.168) \[rank 2\] started at 1793553910.141324462 on last-test-8-a83a3d8f-worker2  
(worker1, rank=1, pid=656, ip=10.244.0.167) \[rank 1\] started at 1793553910.142494629 on last-test-8-a83a3d8f-worker1  
✓ Job finished (status: SUCCEEDED).  
✓ Managed job finished: 8 (status: SUCCEEDED).

📋 Useful Commands  
Managed Job ID: 8  
├── To cancel the job:		sky jobs cancel 8  
├── To stream job logs:		sky jobs logs 8  
├── To stream controller logs:	sky jobs logs \--controller 8  
└── To view all managed jobs:	sky jobs queue  
sky jobs launch \-y \-c last\_test ej/job.yaml  **0.89s user 0.26s system 3% cpu 33.986 total**

(skypilot-bench) ejiang@kodaly skypilot-bench % time sky jobs launch \-y \-c last\_test ej/job.yaml  
YAML to run: ej/job.yaml  
Launching managed job 'last\_test'...  
├── Waiting for task resources on 4 nodes.  
└── Job started. Streaming logs... (Ctrl-C to exit log streaming; job will not be killed)  
(head, rank=0, pid=1433) \[rank 0\] started at 1793553948.910087217 on last-test-9-a83a3d8f-head  
(worker3, rank=3, pid=669, ip=10.244.0.170) \[rank 3\] started at 1793553950.038120550 on last-test-9-a83a3d8f-worker3  
(worker2, rank=2, pid=578, ip=10.244.0.172) \[rank 2\] started at 1793553950.027564592 on last-test-9-a83a3d8f-worker2  
(worker1, rank=1, pid=640, ip=10.244.0.169) \[rank 1\] started at 1793553950.042590384 on last-test-9-a83a3d8f-worker1  
✓ Job finished (status: SUCCEEDED).  
✓ Managed job finished: 9 (status: SUCCEEDED).

📋 Useful Commands  
Managed Job ID: 9  
├── To cancel the job:		sky jobs cancel 9  
├── To stream job logs:		sky jobs logs 9  
├── To stream controller logs:	sky jobs logs \--controller 9  
└── To view all managed jobs:	sky jobs queue  
sky jobs launch \-y \-c last\_test ej/job.yaml  **0.90s user 0.26s system 3% cpu 30.486 total**

##   Appendix C: Ubuntu Benchmark Result: sky-launch-ab

This run is valid: on a 6-node sky launch, ej-prj-3-job1 (5cb8dace3) reached "Cluster launched" about 6 s (24%) faster than baseline 5ef55efa4, where it branched from master. All 10 measured launches succeeded on all 6 ranks, with 1 warm-up launch per build excluded, and no launch fell back to restarting Ray. The first table is your summary.md as a real ASCII table.

\+--------------------------+--------------------------+--------------------------+---------------+--------+  
\| Metric                   \| Baseline median          \| Candidate median         \| Change in     \| p (one \|  
\|                          \| (mean ± sd, n)           \| (mean ± sd, n)           \| median        \| sided) \|  
\+--------------------------+--------------------------+--------------------------+---------------+--------+  
\| post\_runtime\_s (primary) \| 14.50s (15.05 ± 1.38, 5\) \| 8.38s (7.41 ± 2.44, 5\)   \| \-6.12s (-42%) \| 0.0041 \|  
\| provision\_total\_s        \| 24.48s (24.88 ± 1.51, 5\) \| 18.52s (19.54 ± 2.49, 5\) \| \-5.96s (-24%) \| 0.0087 \|  
\| pods\_ready\_s             \| 5.39s (5.62 ± 0.57, 5\)   \| 6.11s (5.87 ± 0.55, 5\)   \| \+0.72s (+13%) \| 0.7539 \|  
\| runtime\_setup\_s          \| 4.05s (4.22 ± 0.45, 5\)   \| 4.35s (6.27 ± 4.58, 5\)   \| \+0.31s (+8%)  \| 0.7682 \|  
\+--------------------------+--------------------------+--------------------------+---------------+--------+

\+-----------------------------------------------------+--------+  
\| Check                                               \| Result \|  
\+-----------------------------------------------------+--------+  
\| Each build ran its own, distinct commit             \| Pass   \|  
\| All launches succeeded and ran on every rank        \| Pass   \|  
\| No candidate launch fell back to restarting Ray     \| Pass   \|  
\| Candidate never ran the old polling loop            \| Pass   \|  
\| Primary metric lower with p \< 0.05                  \| Pass   \|  
\+-----------------------------------------------------+--------+  
\| Verdict                                             \| VALID  \|  
\+-----------------------------------------------------+--------+

These are the per-launch times in seconds, from results.csv:

\+-----------+-----+-----------+--------------+-----------+------+---------------+  
\| Build     \| Run \| Launch    \| post\_runtime \| provision \| pods \| runtime\_setup \|  
\|           \|     \| wall time \|              \| total     \|      \|               \|  
\+-----------+-----+-----------+--------------+-----------+------+---------------+  
\| baseline  \|  0  \|  41.2     \|  14.7        \|  24.9     \| 6.3  \|   3.9         \|  
\| baseline  \|  1  \|  41.3     \|  14.5        \|  24.5     \| 6.2  \|   3.8         \|  
\| baseline  \|  2  \|  43.8     \|  17.5        \|  27.4     \| 5.0  \|   4.9         \|  
\| baseline  \|  3  \|  40.2     \|  14.2        \|  24.1     \| 5.4  \|   4.4         \|  
\| baseline  \|  4  \|  39.6     \|  14.3        \|  23.5     \| 5.2  \|   4.0         \|  
\+-----------+-----+-----------+--------------+-----------+------+---------------+  
\| candidate \|  0  \|  36.4     \|   9.7        \|  19.0     \| 6.1  \|   3.2         \|  
\| candidate \|  1  \|  35.5     \|   8.5        \|  18.4     \| 6.2  \|   3.7         \|  
\| candidate \|  2  \|  35.4     \|   8.4        \|  18.5     \| 5.8  \|   4.4         \|  
\| candidate \|  3  \|  34.2     \|   7.1        \|  17.9     \| 4.9  \|   5.8         \|  
\| candidate \|  4  \|  40.6     \|   3.3        \|  23.9     \| 6.3  \|  14.3         \|  
\+-----------+-----+-----------+--------------+-----------+------+---------------+

\- Where the gain is: all of it is in post\_runtime\_s (Ray wait \+ skylet start). Pod start and runtime setup show no significant change (p \= 0.75 and 0.77). The client's median launch wall time fell from 41.2 s to 35.5 s.  
\- The slowest candidate run: candidate run 4 spent 14.3 s in runtime setup but only 3.3 s after it, for a 23.9 s total. This is the same pattern as the \~10 s single-pod stall we traced to PyPI earlier, but I didn't check this run's logs to confirm the cause. Even so, it took about the same as the fastest baseline run (23.5 s).  
\- Report title: it still says "baseline vs. ej-prj-1" because that text is hardcoded in the script. The candidate here is ej-prj-3-job1, so the title understates what was tested.

The results are in ej/bench\_results/20261031-234001-sky-launch-ab/, including summary.md, results.csv, meta.json and logs/.

## Appendix D: MacOS Benchmark Result: sky-launch-ab

This run is valid, and the gain is clearer than in the last one: ej-prj-3-job1 reached "Cluster launched" 6.5 s (32%) faster than baseline 5ef55efa4. All 10 measured launches succeeded on all 6 ranks. Every candidate run also beat every baseline run on both post\_runtime\_s and total provisioning (12.6–15.7 s against 18.1–21.4 s), so p \= 0.0041 is about the lowest a 5-against-5 test can give.

It seems to come from a different machine than the earlier runs: the paths are /Volumes/CRU2T1/... and /private/tmp/..., which suggests macOS. The pods also report about 14 CPUs, against 12 on the Linux box. That makes the comparison below one across two environments.

Medians, seconds     Linux box (23:40 run)       This run (likely Mac)  
                     baseline  candidate         baseline  candidate  
\+------------------+---------+---------+       \+---------+---------+  
\| Pods ready       \|     5.4 \|     6.1 \|       \|     5.6 \|     4.8 \|  
\| File mounts      \|     1.4 \|     1.5 \|       \|     0.7 \|     0.8 \|  
\| Runtime setup    \|     2.3 \|     2.6 \|       \|     8.0 \|     6.4 \|  
\| Ray \+ skylet     \|    14.5 \|     8.4 \|       \|     5.9 \|     2.0 \|  
\| Skylet start \*   \|     1.6 \|     4.8 \|       \|     0.8 \|     2.0 \|  
\+------------------+---------+---------+       \+---------+---------+  
\| Provision total  \|    24.5 \|    18.5 \|       \|    20.7 \|    14.1 \|  
\| Client wall time \|    41.2 \|    35.5 \|       \|    35.0 \|    28.3 \|  
\+------------------+---------+---------+       \+---------+---------+

\* On the candidate, skylet starts while Ray is being waited on, so it overlaps the  
  row above. On the baseline it runs after that wait. The Ray \+ skylet row includes it either way.

What the numbers show

\- The Ray wait change is still the main gain (−3.9 s). The baseline polled 1 time here and 2–3 times on Linux, which is why its Ray \+ skylet step is so much shorter here.  
\- The Ray wait has nearly disappeared here. In 4 of 5 candidate runs, Ray \+ skylet equals the skylet time (1.9–2.2 s), so all nodes had already joined when runtime setup ended.  
\- Runtime setup is now the biggest stage on this machine (6.4 s). On Linux the Ray wait is what holds launches back, and faster setup just moves time into it. Here, faster setup should shorten the launch directly.  
\- Runtime setup is 1.6 s faster on the candidate (p \= 0.045). This is plausible because the candidate no longer does a real PyPI install of SkyPilot during pod startup, but the Linux run didn't show the gain.  
\- The pod-start gain is probably noise. It's 0.8 s faster here (p \= 0.045), but was 0.7 s slower on the same code in the last run, and no change targets it.

## Appendix E: Ubuntu Benchmark Result: jobs-ab –nodes 6

ej-prj-3-job1 is clearly faster: its median client total is 54.6 s against 69.5 s on master, a 14.9 s (21%) cut. All 16 measured runs succeeded, and every candidate run (50.5–60.3 s) beat every master run (61.5–74.9 s). It also replicates §14 of the design doc, which had 54.8 s against 70.8 s.

The report's p \= 0.0000 isn't literally zero. I computed the exact value at p \= 0.00016, the lowest possible for 8 runs against 8\.

Where the time goes

These are means, because medians don't add up across phases.

┌─────────────────────────────────┬────────┬───────────┬────────┬────────────────────────┐  
│              Phase              │ Master │ Candidate │ Change │        Exact p         │  
├─────────────────────────────────┼────────┼───────────┼────────┼────────────────────────┤  
│ G: RUNNING → SUCCEEDED          │ 9.74   │ 3.48      │ −6.25  │ 0.00016                │  
├─────────────────────────────────┼────────┼───────────┼────────┼────────────────────────┤  
│ D: provisioning → job submitted │ 30.77  │ 25.82     │ −4.95  │ 0.013                  │  
├─────────────────────────────────┼────────┼───────────┼────────┼────────────────────────┤  
│ F: job ended → RUNNING          │ 3.16   │ 1.12      │ −2.05  │ 0.00016                │  
├─────────────────────────────────┼────────┼───────────┼────────┼────────────────────────┤  
│ H: SUCCEEDED → client exits     │ 3.32   │ 2.51      │ −0.81  │ 0.16 (not significant) │  
├─────────────────────────────────┼────────┼───────────┼────────┼────────────────────────┤  
│ A: client start → job row       │ 7.12   │ 6.87      │ −0.25  │ 0.036                  │  
├─────────────────────────────────┼────────┼───────────┼────────┼────────────────────────┤  
│ E: job runs                     │ 8.38   │ 8.96      │ \+0.58  │ 0.0065                 │  
├─────────────────────────────────┼────────┼───────────┼────────┼────────────────────────┤  
│ B \+ C: hand-off to controller   │ 6.14   │ 6.26      │ \+0.12  │ not tested             │  
├─────────────────────────────────┼────────┼───────────┼────────┼────────────────────────┤  
│ Client total                    │ 68.64  │ 55.02     │ −13.62 │ 0.00016                │  
└─────────────────────────────────┴────────┴───────────┴────────┴────────────────────────┘

\- Most of the gain is after the job finishes: F, G and H account for about 9.1 s of the 13.6 s. G is now 3.4–3.6 s in every run, compared with 3.8–16.8 s on master.  
\- Provisioning contributes about 5 s, but that average hides the main finding below.

Main finding: the candidate has fast and slow runs

The report's median D of 25.1s doesn't match any actual run, because the runs fall into two groups:

┌─────────────────┬──────┬───────────────────┬───────────┬──────────────────┐  
│      Group      │ Runs │ Runtime setup (s) │   D (s)   │ Client total (s) │  
├─────────────────┼──────┼───────────────────┼───────────┼──────────────────┤  
│ Candidate, fast │ 4    │ 1.8–2.5           │ 21.8–22.1 │ 50.5–52.9        │  
├─────────────────┼──────┼───────────────────┼───────────┼──────────────────┤  
│ Candidate, slow │ 4    │ 10.4–16.8         │ 28.1–32.8 │ 56.3–60.3        │  
├─────────────────┼──────┼───────────────────┼───────────┼──────────────────┤  
│ Master          │ 8    │ 2.2–2.9           │ 28.6–33.4 │ 61.5–74.9        │  
└─────────────────┴──────┴───────────────────┴───────────┴──────────────────┘

\- In slow runs, most of the provisioning gain disappears: the median D is about 28.9 s against master's 30.6 s. In fast runs it's about 22 s, a gain of about 8.6 s.  
\- The stall probably affects master too, but it's hidden: master's runtime setup stays short because its delay shows up in the Ray wait instead.  
\- The cause isn't confirmed by this run: worker logs aren't saved. It looks like the roughly 10 s single-pod stall I traced earlier to resolving setuptools\<70, but I can't confirm these four runs are the same thing.  
\- §15 of the design doc needs revisiting: it blames the PyPI dry run and proposes \--offline, but ej-prj-3-job2 tested that and showed no gain.

Smaller things to note

\- Small slowdowns: E is up 0.58s, and §14 also showed it slightly slower (+0.3 s). The submit step is up about 0.45 s, but §14 had it going the other way, so I'd treat that one as unproven. Neither is diagnosed, and these p-values aren't adjusted for testing many metrics.  
\- Skylet start is slower (1.63 → 4.17s): this is expected, since it now overlaps the Ray wait, so it doesn't add its full time to the total.  
\- Pod start is unchanged (5.47 vs 5.60s): the IfNotPresent gain is still available.  
\- First launch after a restart isn't counted: warm-ups are dropped, and they ran slower in A and C. The very first one took 143.9 s, with A at 68 s, while later warm-ups had A at about 9 s. I haven't looked into why.

What to try next

┌───────────────────────────────────────────┬──────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────┐  
│                 Candidate                 │                       Likely gain                        │                                  Evidence                                  │  
├───────────────────────────────────────────┼──────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┤  
│ Rebuilt image with the pins (611a9432b)   │ Up to about 3.9 s on average, and removes the slow group │ Inferred from the two groups, and assumes the setuptools\<70 cause is right │  
├───────────────────────────────────────────┼──────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┤  
│ IfNotPresent for the default image        │ Up to about 3 s                                          │ Pod start only, from the pull-policy test                                  │  
├───────────────────────────────────────────┼──────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┤  
│ Avoid the first storage lookup in phase A │ Up to about 4.8 s                                        │ §13 probe; the fix isn't designed yet                                      │  
└───────────────────────────────────────────┴──────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────────────┘

I'd start with the rebuilt image, because it targets the variability as well as the average, and the code change is already on this branch. The test is the same A/B with the rebuilt image, ideally also saving worker logs so a remaining stall can be identified. Do you want me to set that up?

## Appendix F: Nov 2 Benchmark Run

(skypilot) ejiang@benjamin:\~/2T2/Projects/skypilot\$ python ej/reproduce\_benchmarks.py jobs-ab \--nodes 6 \\  
    \--base 5ef55efa4 \--candidate ej-prj-i2-5 \\  
    \--base-task ej/task\_published\_ifnotpresent.yaml \\  
    \--candidate-task ej/task\_bytecode.yaml \\  
    \--blocks 4 \--runs 4 \--warmup 1  
\[21:40:05\] jobs-ab: 5ef55efa4 (5ef55efa4) vs ej-prj-i2-5 (7e488f558), 6 nodes, 4 blocks x (1 warm-up \+ 4 measured). Out: /run/media/ejiang/2T2/Projects/skypilot/ej/bench\_results/20261102-214005-jobs-ab  
\[21:40:17\] Block 1/4: base at 5ef55efa4  
\[21:41:53\]   base\_b1\_w: job 454 rc=0 CLEAN total=88.6  
\[21:43:21\]   base\_b1\_1: job 455 rc=0 CLEAN total=81.0  
\[21:44:46\]   base\_b1\_2: job 456 rc=0 CLEAN total=78.1  
\[21:46:00\]   base\_b1\_3: job 457 rc=0 CLEAN total=67.2  
\[21:47:17\]   base\_b1\_4: job 458 rc=0 CLEAN total=70.1  
\[21:47:30\] Block 2/4: candidate at 7e488f558  
\[21:48:26\]   candidate\_b2\_w: job 459 rc=0 CLEAN total=51.6  
\[21:49:11\]   candidate\_b2\_1: job 460 rc=0 CLEAN total=41.8  
\[21:50:04\]   candidate\_b2\_2: job 461 rc=0 CLEAN total=48.7  
\[21:50:48\]   candidate\_b2\_3: job 462 rc=0 CLEAN total=40.9  
\[21:51:33\]   candidate\_b2\_4: job 463 rc=0 CLEAN total=40.5  
\[21:51:47\] Block 3/4: base at 5ef55efa4  
\[21:53:25\]   base\_b3\_w: job 464 rc=0 CLEAN total=93.9  
\[21:54:53\]   base\_b3\_1: job 465 rc=0 CLEAN total=80.8  
\[21:56:11\]   base\_b3\_2: job 466 rc=0 CLEAN total=72.9  
\[21:57:32\]   base\_b3\_3: job 467 rc=0 CLEAN total=74.5  
\[21:58:41\]   base\_b3\_4: job 468 rc=0 CLEAN total=61.6  
\[21:58:53\] Block 4/4: candidate at 7e488f558  
\[21:59:49\]   candidate\_b4\_w: job 469 rc=0 CLEAN total=50.6  
\[22:00:33\]   candidate\_b4\_1: job 470 rc=0 CLEAN total=39.7  
\[22:01:18\]   candidate\_b4\_2: job 471 rc=0 CLEAN total=40.6  
\[22:02:01\]   candidate\_b4\_3: job 472 rc=0 CLEAN total=38.1  
\[22:02:47\]   candidate\_b4\_4: job 473 rc=0 CLEAN total=42.9  
\[22:02:47\] Restoring ej-prj-i2-6 and restarting the API server on it  
\# sky jobs launch A/B: 5ef55efa4 (5ef55efa4) vs ej-prj-i2-5 (7e488f558)

\- 6 nodes on \`kind-skypilot\`, consolidation mode.  
\- 4 alternating blocks of 1 warm-up \+ 4 measured runs. Measured, clean and successful runs: base 8, candidate 8\.

\#\# Phases (seconds, median and range)

\| Phase \| base \| candidate \| Δ median \|  
\|---\|---\|---\|---\|  
\| A: client start \-\> job row created \| 7.1 (6.5–8.1) \| 6.7 (6.4–7.5) \| \-0.3 \|  
\| B: row created \-\> STARTING \| 2.8 (2.6–5.2) \| 2.4 (2.1–4.8) \| \-0.4 \|  
\| C: STARTING \-\> provisioning starts \| 3.4 (2.6–4.0) \| 3.1 (2.5–3.9) \| \-0.3 \|  
\| D: provisioning \-\> job submitted \| 32.6 (28.6–38.1) \| 14.7 (14.1–21.2) \| \-18.0 \|  
\| E: job runs on the cluster \| 8.6 (8.4–13.0) \| 6.7 (6.4–6.9) \| \-1.9 \|  
\| F: job ended \-\> RUNNING \| 3.0 (-2.8–3.4) \| 2.6 (1.6–3.2) \| \-0.4 \|  
\| G: RUNNING \-\> SUCCEEDED \| 13.8 (3.5–16.9) \| 0.1 (0.1–0.2) \| \-13.7 \|  
\| H: SUCCEEDED \-\> client exits \| 2.8 (2.6–4.0) \| 4.2 (3.4–4.6) \| \+1.5 \|  
\| to\_SUCCEEDED: client start \-\> SUCCEEDED \| 70.2 (58.5–78.4) \| 36.8 (34.7–44.6) \| \-33.5 \|  
\| total: client total \| 73.7 (61.6–81.0) \| 40.8 (38.1–48.7) \| \-32.9 \|

Permutation test on the client total (difference of means, two-sided): p \= 0.0000.

\#\# Phase D stages (seconds, median and range)

\| Stage \| base \| candidate \|  
\|---\|---\|---\|  
\| 'Provisioning on' event \-\> provision log starts \| 0.28 (0.0–0.9) \| 0.41 (0.2–1.0) \|  
\| create pods, wait until Running \| 2.76 (2.6–3.1) \| 1.79 (1.6–2.2) \|  
\| SSH readiness \| 0.25 (0.2–0.3) \| 0.32 (0.3–0.4) \|  
\| internal\_file\_mounts \| 1.21 (1.0–2.2) \| 2.15 (1.5–2.8) \|  
\| setup\_runtime\_on\_cluster \| 9.89 (3.3–19.4) \| 1.85 (1.5–11.6) \|  
\| Ray wait \+ skylet start \| 14.69 (8.4–19.9) \| 4.50 (2.2–5.0) \|  
\| cluster launched \-\> job submitted \| 3.63 (3.4–3.9) \| 3.32 (3.3–3.8) \|  
\| phase D total \| 32.64 (28.6–38.1) \| 14.67 (14.1–21.2) \|  
\| skylet start on its own \| 1.58 (1.6–1.8) \| 2.46 (2.2–2.8) \|

\#\# Runs

\| Run \| Job \| Clean \| rc \| Total \|  
\|---\|---\|---\|---\|---\|  
\| base\_b1\_w \| 454 \| True \| 0 \| 88.6 \|  
\| base\_b1\_1 \| 455 \| True \| 0 \| 81.0 \|  
\| base\_b1\_2 \| 456 \| True \| 0 \| 78.1 \|  
\| base\_b1\_3 \| 457 \| True \| 0 \| 67.2 \|  
\| base\_b1\_4 \| 458 \| True \| 0 \| 70.1 \|  
\| candidate\_b2\_w \| 459 \| True \| 0 \| 51.6 \|  
\| candidate\_b2\_1 \| 460 \| True \| 0 \| 41.8 \|  
\| candidate\_b2\_2 \| 461 \| True \| 0 \| 48.7 \|  
\| candidate\_b2\_3 \| 462 \| True \| 0 \| 40.9 \|  
\| candidate\_b2\_4 \| 463 \| True \| 0 \| 40.5 \|  
\| base\_b3\_w \| 464 \| True \| 0 \| 93.9 \|  
\| base\_b3\_1 \| 465 \| True \| 0 \| 80.8 \|  
\| base\_b3\_2 \| 466 \| True \| 0 \| 72.9 \|  
\| base\_b3\_3 \| 467 \| True \| 0 \| 74.5 \|  
\| base\_b3\_4 \| 468 \| True \| 0 \| 61.6 \|  
\| candidate\_b4\_w \| 469 \| True \| 0 \| 50.6 \|  
\| candidate\_b4\_1 \| 470 \| True \| 0 \| 39.7 \|  
\| candidate\_b4\_2 \| 471 \| True \| 0 \| 40.6 \|  
\| candidate\_b4\_3 \| 472 \| True \| 0 \| 38.1 \|  
\| candidate\_b4\_4 \| 473 \| True \| 0 \| 42.9 \|  
