### **SkyPilot Take-Home Task for System Engineering: Design Doc**

---

**\* Your prototype on a branch:**  
**List:** [https\://github.com/siliconej/skypilot/branches](https://github.com/siliconej/skypilot/branches)  
**Consolidated single PR:** [https\://github.com/skypilot-org/skypilot/compare/master...siliconej:skypilot:ej-prj-3-job1?expand=1](https://github.com/skypilot-org/skypilot/compare/master...siliconej:skypilot:ej-prj-3-job1?expand=1) 

**\* Your design doc (markdown, in-repo or linked):**  
This Doc, see details at Project Write-up below, markdown:

**\* Your benchmark script and a result table:**

**Benchmark script:**  
[https\://hifiweekly.github.io/skypilot\_ej\_prj/reproduce\_benchmarks.py](https://hifiweekly.github.io/skypilot_ej_prj/reproduce_benchmarks.py)

**\- Sample benchmark run:**  
python ej/reproduce\_benchmarks.py jobs-ab \--nodes 6      \# \~22 min, 8 measured runs per side  
python ej/reproduce\_benchmarks.py pod-probe \--nodes 6    \# \~2 min  
python ej/reproduce\_benchmarks.py pull-policy \--nodes 6  \# \~5 min  
python ej/reproduce\_benchmarks.py storage-probe          \# \~6 min  
python ej/reproduce\_benchmarks.py sky-launch-ab \-- run \--candidate-ref ej-prj-3-job1 \--task ej/job.yaml \--runs 5

**Result Table:** [https\://hifiweekly.github.io/skypilot\_ej\_prj/report.html](https://hifiweekly.github.io/skypilot_ej_prj/report.html)  
**TL;DR: 6 nodes, medians of 8 interleaved runs per side (range), seconds:**  
┌───────────────────────────────────┬──────────────────┬──────────────────┬────────┬───────────────┐  
│               Phase               │      master      │  ej-prj-3-job1   │ Change │ 4-node change │  
├───────────────────────────────────┼──────────────────┼──────────────────┼────────┼───────────────┤  
│ A: client start → job row created │ 6.8 (6.4–7.2)    │ 7.1 (6.5–7.2)    │ \+0.3   │ 0             │  
├───────────────────────────────────┼──────────────────┼──────────────────┼────────┼───────────────┤  
│ B+C: hand-off to controller       │ 5.6              │ 5.4              │ −0.2   │ −0.1          │  
├───────────────────────────────────┼──────────────────┼──────────────────┼────────┼───────────────┤  
│ D: provisioning → job submitted   │ 32.7 (29.6–41.5) │ 26.3 (22.3–28.3) │ −6.4   │ −2.4          │  
├───────────────────────────────────┼──────────────────┼──────────────────┼────────┼───────────────┤  
│ E: job runs                       │ 8.5              │ 8.8              │ \+0.3   │ 0             │  
├───────────────────────────────────┼──────────────────┼──────────────────┼────────┼───────────────┤  
│ F: job ended → RUNNING            │ 2.8 (2.2–3.3)    │ 1.2 (0.8–1.8)    │ −1.6   │ −1.8          │  
├───────────────────────────────────┼──────────────────┼──────────────────┼────────┼───────────────┤  
│ G: RUNNING → SUCCEEDED            │ 8.4 (3.9–14.0)   │ 3.5 (3.4–3.6)    │ −4.9   │ −7.6          │  
├───────────────────────────────────┼──────────────────┼──────────────────┼────────┼───────────────┤  
│ H: SUCCEEDED → client exits       │ 3.0 (2.4–11.0)   │ 2.9 (2.6–3.3)    │ −0.1   │ −0.3          │  
├───────────────────────────────────┼──────────────────┼──────────────────┼────────┼───────────────┤  
│ Client start → SUCCEEDED          │ 64.7             │ 51.7             │ −13.0  │ −10.6         │  
├───────────────────────────────────┼──────────────────┼──────────────────┼────────┼───────────────┤  
│ Client total                      │ 70.8 (61.1–77.5) │ 54.8 (51.5–58.3) │ −16.0  │ −11.4         │  
└───────────────────────────────────┴──────────────────┴──────────────────┴────────┴───────────────┘  
Consult the UML Sequence diagram in SVG for Phase definition and transition: [https\://hifiweekly.github.io/skypilot\_ej\_prj/sky\_jobs\_launch\_sequence\_6nodes.svg](https://hifiweekly.github.io/skypilot_ej_prj/sky_jobs_launch_sequence_6nodes.svg)   
All run done on Ubuntu 26.04, i7-8700K, 32GB memory

**\* A short README explaining how to reproduce your numbers from a clean machine.**

**Read Me:**

---

**\* Project Write-up**

The refactoring in these branches ([https\://github.com/siliconej/skypilot/branches/yours](https://github.com/siliconej/skypilot/branches/yours)) is a set of optimizations stacked in a progressive fashion. The svg linked here (4 nodes \[[https\://hifiweekly.github.io/skypilot\_ej\_prj/sky\_jobs\_launch\_sequence.svg](https://hifiweekly.github.io/skypilot_ej_prj/sky_jobs_launch_sequence.svg)\], 6 nodes \[[https\://hifiweekly.github.io/skypilot\_ej\_prj/sky\_jobs\_launch\_sequence\_6nodes.svg](https://hifiweekly.github.io/skypilot_ej_prj/sky_jobs_launch_sequence_6nodes.svg)\]) shows the different updates that are prepared for your review. Overall there are two parts of the refactoring, first set focuses on section D (cycles between provision and job submission), second set focuses on the rest of the whole workflow. I also implemented the bonus feature and fixed a warning. See more detailed below.

The priority of these optimizations is analyzed and determined based on the delta time reduction that could be achieved theoretically according to first trace analysis on the master HEAD.

A more length consolidated design doc of all the optimization included so far this project is at: [https\://hifiweekly.github.io/skypilot\_ej\_prj/launch\_latency\_consolidated\_design.html](https://hifiweekly.github.io/skypilot_ej_prj/launch_latency_consolidated_design.html) 

### **First Set: ej-prj-0, ej-prj-1, and ej-prj-3**

These three changes focus on the job preparation and per node job submission, and will have impact on unmanaged job scheduling by ‘sky launch’ as well.

**\* Branch ej-prj-0: Start ray in the pod before preparing SkyPilot** [https\://github.com/skypilot-org/skypilot/compare/master...siliconej:skypilot:ej-prj-0?expand=1](https://github.com/skypilot-org/skypilot/compare/master...siliconej:skypilot:ej-prj-0?expand=1)

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

**Gain.** About 1.3 s earlier Ray on the default image, more on custom images. Alone, the old 3–5 s polling hid it: an A/B showed −2%, not significant. Combined with ej-prj-1's blocking wait, the A/B was significant: runtime setup end → launched went from 12.49 s to 9.43 s (−24%, p \= 0.004).

**Risks.**  
\- The dry run still depends on PyPI.  
\- On custom images, dependencies now install while Ray runs.

**Tests.**  
A new template test checks the order, including the HA case, and the rendered pod args pass bash \-n.

**Design Doc.** [https\://hifiweekly.github.io/skypilot\_ej\_prj/ray\_first\_pod\_bootstrap\_design.html](https://hifiweekly.github.io/skypilot_ej_prj/ray_first_pod_bootstrap_design.html) 

---

**\* Branch ej-prj-1: Wait for Ray in one blocking call; start skylet during the wait**  
[https\://github.com/skypilot-org/skypilot/compare/master...siliconej:skypilot:ej-prj-1?expand=1](https://github.com/skypilot-org/skypilot/compare/master...siliconej:skypilot:ej-prj-1?expand=1)

**\* TL;DR**

**Problem.** On a fresh Kubernetes cluster, waiting for every pod's Ray to join took 61–67% of a launch. Two causes:  
\- The poll was slow. The provisioner checked once a second, but each check was a kubectl exec that started three Python interpreters (2.3–4 s), so it noticed readiness up to a whole probe late.  
\- Skylet started late. It only started after the wait, although it needs only the head.

**Change (provisioner.py, instance\_setup.py):**  
\- One blocking call. A small script on the head (standard library plus ray, sent base64-encoded) keeps one GcsClient open and checks every 0.2 s until exactly num\_nodes are alive or the deadline passes.  
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

**\* TL;DR**

**Measured first.** A timestamped trace of the pod's startup script found two waste steps before ray start:  
\- ray stop on a pod that never ran Ray: 1.4–1.5 s idle, 2.9–3.8 s while the pod was busy setting up.  
\- Downloading setuptools\<70 from PyPI: the image ships setuptools 84, so every pod fetched the pin. This took 0.6 s usually and 12.1 s in one traced run.

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

### **Second Set: ej-prj-job0 and ej-prj-job1**

These two changes focus on the skypilot tasks before and after the job preparation and job submission, including whole job request processing, submission, job monitoring, and job exit.

**\* Branch ej-prj-job0: Fetch status and submitted-at in one call; skip the first status-check sleep; re-check status on controller logs change**  
[https\://github.com/skypilot-org/skypilot/compare/master...siliconej:skypilot:ej-prj-job0?expand=1](https://github.com/skypilot-org/skypilot/compare/master...siliconej:skypilot:ej-prj-job0?expand=1) 

**\* TL;DR**

**Problem.** After provisioning, a managed job lost about 15–35 s to polling:  
\- Confirming the job started took two kubectl exec round trips to the head, status and then submitted-at time, about 1.6 s each.  
\- The monitor loop slept a random 0–15 s before its first status check, although the launch had just seen the job finish. It threw that status away.  
\- The client's log follower re-checked the status only every 15 s. It usually missed the job's few seconds in RUNNING, so it noticed the end up to 15 s late.

**Changes:**  
1\. One head command (JobLibCodeGen.get\_job\_status\_and\_submitted\_timestamp) returns both values. It calls only existing job\_lib functions, so older cluster runtimes work, and it falls back to the second call on gRPC or a missing time.  
2\. The first check runs at once. pop\_job\_status\_seen\_at\_launch() hands the monitor loop the status the launch saw. If it's terminal, the loop checks immediately; a running job keeps the jittered gap.  
3\. The follower re-checks the status as soon as the controller writes a plain log line (about 15 per launch). The 15 s check stays as a fallback.

**Results (kind, 4 nodes):**  
\- Job finished → RUNNING: 3.0 → 1.3 s.  
\- RUNNING → SUCCEEDED: random 4.9–10.5 s → steady 3.4 s.  
\- The client exits 2.1–2.9 s after SUCCEEDED, against 1.6–12.6 s before.  
\- 14 new unit tests.

**Design Doc.** [https\://hifiweekly.github.io/skypilot\_ej\_prj/launch\_latency\_managed\_jobs\_ej-prj-3.html](https://hifiweekly.github.io/skypilot_ej_prj/launch_latency_managed_jobs_ej-prj-3.html) (Section 3, 7, and 10\)

---

**\* Branch ej-prj-job1: Look up the enabled storage clouds once per request**  
[https\://github.com/skypilot-org/skypilot/compare/master...siliconej:skypilot:ej-prj-job1?expand=1](https://github.com/skypilot-org/skypilot/compare/master...siliconej:skypilot:ej-prj-job1?expand=1)

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
\- The probe on kind measured the second lookup at 0.18–0.25 s on master and 0.001 s with the change, so the gain is about 0.2 s, within phase A's noise.  
\- The earlier 7.3 → 6.5 s comparison mixed sessions and is superseded.

**Design Doc.** Section 12 of [https\://hifiweekly.github.io/skypilot\_ej\_prj/launch\_latency\_consolidated\_design.html](https://hifiweekly.github.io/skypilot_ej_prj/launch_latency_consolidated_design.html) 

---

### **Bonus: 90s timeout handling** 

[https\://github.com/skypilot-org/skypilot/compare/master...siliconej:skypilot:ej-prj-bonus?expand=1](https://github.com/skypilot-org/skypilot/compare/master...siliconej:skypilot:ej-prj-bonus?expand=1)

**\* TL;DR**

**Problem.**  
\- The pod never comes back. A pod deleted after it started, by kubectl delete pod or an eviction, stays gone: SkyPilot pods use restartPolicy: Never.  
\- The launch noticed only after about 106 s.  
  \- The head's Ray wait ran its whole 90 s timeout, because a worker that never joined is invisible to it.  
  \- The fallback then tried to start Ray on the missing pod, retrying six times.  
\- The error never said a pod was gone.

**Changes:**  
1\. Don't retry an exec whose pod is gone (command\_runner.is\_exec\_target\_gone). \_auto\_retry stops on permanent kubectl errors: NotFound, a completed pod, phase Failed or Succeeded. Transient errors such as "unable to upgrade connection" are still retried.  
2\. Check the instances. provision.query\_instances lists the cluster's pods. One missing, terminated or stopped in two listings 2 s apart raises the new InstancesLostDuringSetupError (a ClusterSetUpError) naming it. A failing check never fails the launch.  
3\. Time-box the Ray wait into calls of at most 10 s, checking the instances between calls. Nothing changes when Ray is ready in the first call.  
4\. Name the pod in errors. Any failing setup stage checks the instances before re-raising, keeping the stage's error as context.

**Results (kind):** a deleted worker fails the launch in about 11 s and a deleted head in about 5 s, instead of about 106 s. Managed jobs recover sooner.

**Tests:**  
\- 9 new unit tests.  
\- A new smoke test, test\_launch\_fails\_fast\_when\_pod\_deleted (worker and head), deletes a pod mid-setup. Both cases passed on kind in 42–45 s, against a 45 s limit.  
\- 7aed7c84e adds the function to command\_runner.pyi; without it, CI's mypy check fails.

**Design Doc**: [https\://hifiweekly.github.io/skypilot\_ej\_prj/pod\_loss\_fail\_fast\_design.html](https://hifiweekly.github.io/skypilot_ej_prj/pod_loss_fail_fast_design.html) 

---

### **Fix: k8s-pool-warning suppression**

[https\://github.com/skypilot-org/skypilot/compare/master...siliconej:skypilot:ej-k8s-pool-warning?expand=1](https://github.com/skypilot-org/skypilot/compare/master...siliconej:skypilot:ej-k8s-pool-warning?expand=1)

**Problem.**  
\- The pool overflows by design. SkyPilot's Kubernetes API clients cap their connection pool at CONNECTION\_POOL\_MAXSIZE (4). Going past it is normal, for example when creating or inspecting more than 4 pods in parallel.  
\- Each extra connection prints a warning. When urllib3 closes one, it logs WARNING:urllib3.connectionpool:Connection pool is full, discarding connection: ... Connection pool size: 4\.  
\- The existing guard leaks. \_api\_logging\_decorator keeps the urllib3 logger at ERROR during API calls. But any kubernetes.client.Configuration built elsewhere, or in another thread mid-call, resets it to WARNING, so the noise reached users.  
\- It's visible today. Each 6-node managed job's controller log from today's benchmark has 6–7 of these warnings (jobs 105–124). Neither master nor ej-prj-3-job1 includes this fix.

**Change** (sky/adaptors/kubernetes.py, commit 5bb46b39a):  
\- A filter on the urllib3.connectionpool logger drops exactly this message, and only when its pool size equals CONNECTION\_POOL\_MAXSIZE.  
\- Other pools' pool-full warnings still show, as do all other urllib3 warnings.  
\- Why a filter: a level reset doesn't undo a filter, and a filter takes no lock per record.

**Tests:** 1 new parametrized test (test\_pool\_full\_warning\_hidden\_for\_capped\_clients) checks the warning is hidden at the capped size, still shown at other sizes, and that other warnings pass through. The adaptor test file passes: 28 passed.

Design doc: [https\://hifiweekly.github.io/skypilot\_ej\_prj/k8s\_pool\_full\_warning\_design.html](https://hifiweekly.github.io/skypilot_ej_prj/k8s_pool_full_warning_design.html) 