# SkyPilot Take-Home Task for System Engineering: Design Doc

Latest update: Nov 4, 2026  
---

### **Requirement 1: Your prototype on a branch:**

**List:** [https\://github.com/siliconej/skypilot/branches](https://github.com/siliconej/skypilot/branches)  
**Consolidated single PR:** [https\://github.com/skypilot-org/skypilot/compare/master...siliconej:skypilot:ej-prj-i3-2?expand=1](https://github.com/skypilot-org/skypilot/compare/master...siliconej:skypilot:ej-prj-i3-2?expand=1%20)   
([See Appendix H: commit list](#bookmark=id.cd6o0eo96un4))

### **Requirement 2: Your design doc (markdown, in-repo or linked):**

This Doc, see details at Project Write-up below,  
Markdown version: [https\://hifiweekly.github.io/skypilot\_ej\_prj/overview.md](https://hifiweekly.github.io/skypilot_ej_prj/overview.md) 

### **Requirement 3: Your benchmark script and a result table:**

**Benchmark script:**  
[https\://hifiweekly.github.io/skypilot\_ej\_prj/reproduce\_benchmarks.py](https://hifiweekly.github.io/skypilot_ej_prj/reproduce_benchmarks.py)

**Read Me:**  
[https\://hifiweekly.github.io/skypilot\_ej\_prj/README\_i4.html](https://hifiweekly.github.io/skypilot_ej_prj/README_i4.html)   
(Iteration 1: [https\://hifiweekly.github.io/skypilot\_ej\_prj/README.html](https://hifiweekly.github.io/skypilot_ej_prj/README.html),  
 Iteration 2: [https\://hifiweekly.github.io/skypilot\_ej\_prj/README\_i2.html](https://hifiweekly.github.io/skypilot_ej_prj/README_i2.html),  
 Iteration 3: [https\://hifiweekly.github.io/skypilot\_ej\_prj/README\_i3.html](https://hifiweekly.github.io/skypilot_ej_prj/README_i3.html)) 

**TL;DR:**

How to run the benchmark

**One-time setup (README section 2):**

1\. Get the code: fetch the fork and create local branches named ej-prj-i4-5 and the others. The scripts look refs up by local name.  
2\. Install: create the venv with uv pip install \-e ".\[kubernetes,aws\]". The aws extra matters, because phases A and C depend on the aws CLI being on PATH.  
3\. Copy the scripts in: ej/reproduce\_benchmarks.py, run.sh and sky\_launch\_no\_tips.py aren't in git, so copy them in, keep them untracked, and run sky local up.  
4\. Build the bytecode image: build it from Dockerfile\_k8s, load it into kind, and check that it has a uv cache. Without the cache, the offline wheel install falls back to the network.  
5\. Warm up: do one throwaway launch per task with sky api start \--deploy.

**Before each run:**  
\- The working tree is clean.  
\- No pods, clusters or managed jobs are left.  
\- which aws finds the CLI.  
\- BASE=\$(git merge-base origin/master ej-prj-i4-5) prints 796c5bcc8.

**Run (about 40 minutes in all):**  
OUT=\$PWD/ej/bench\_results/\$(date \+%Y%m%d-%H%M%S)-i4-5-vs-master; mkdir \-p "\$OUT"

**\# sky launch: isolated servers and worktrees, no speedup settings**  
python ej/bench\_results/20261105-i4-5b-vs-master/sky\_launch\_no\_tips.py \\  
    run \--baseline-ref "\$BASE" \--candidate-ref ej-prj-i4-5 \\  
    \--task "\$PWD/ej/task\_published\_ifnotpresent.yaml" \\  
    \--candidate-task "\$PWD/ej/task\_bytecode.yaml" \\  
    \--runs 8 \--warmup 1 \--context kind-skypilot \--out "\$OUT/sky-launch"

**\# sky jobs launch: switches your checkout and restarts your server per block**  
python ej/reproduce\_benchmarks.py jobs-ab \--nodes 6 \\  
    \--base "\$BASE" \--candidate ej-prj-i4-5 \\  
    \--base-task "\$PWD/ej/task\_published\_ifnotpresent.yaml" \\  
    \--candidate-task "\$PWD/ej/task\_bytecode.yaml" \\  
    \--blocks 4 \--runs 4 \--warmup 1 \--out "\$OUT/jobs"

Or run bash ej/bench\_results/20261105-i4-5b-vs-master/run.sh, which does both exactly as for the reference results.

**Read the results:**  
\- sky-launch/summary.md. Its Checks list must be all ticked. Wall time is the launch\_wall\_s column in results.csv.  
\- jobs/report.md. Every run must show Clean True and rc 0, with 8 measured runs per arm.  
\- p-values: summary.md gives one-sided p-values, and report.md rounds them, showing 0.0000 when every run of one arm beats the other. With 8 runs per arm, the exact two-sided value is then 0.0002.

**Result Table: (Iteration 4\)** [https\://hifiweekly.github.io/skypilot\_ej\_prj/report\_1105.html](https://hifiweekly.github.io/skypilot_ej_prj/report_1105.html)

**Support Files:**  
[https\://hifiweekly.github.io/skypilot\_ej\_prj/job.yaml](https://hifiweekly.github.io/skypilot_ej_prj/job.yaml) (same as the file in the pdf given)  
[https\://hifiweekly.github.io/skypilot\_ej\_prj/task\_bytecode.yaml](https://hifiweekly.github.io/skypilot_ej_prj/task_bytecode.yaml)  
[https\://hifiweekly.github.io/skypilot\_ej\_prj/task\_published\_ifnotpresent.yaml](https://hifiweekly.github.io/skypilot_ej_prj/task_published_ifnotpresent.yaml)  
[https\://hifiweekly.github.io/skypilot\_ej\_prj/bench\_launch\_latency.py](https://hifiweekly.github.io/skypilot_ej_prj/bench_launch_latency.py) (a pending utility script in branch)

**TL;DR: 6 nodes, medians of 8 interleaved runs per side (range), seconds:**  
Consult the workflow diagram in SVG for Phase definition and transition:  
[https\://hifiweekly.github.io/skypilot\_ej\_prj/sky\_ray\_time\_workflow\_i4\_5\_v4.svg](https://hifiweekly.github.io/skypilot_ej_prj/sky_ray_time_workflow_i4_5_v4.svg)   

Summary  
┌──────────────────────────────────┬─────────────────────┬─────────────────────┬─────────────────┬────────┐  
│ Median (range)                   │ Master 796c5bcc8    │ i4-5 fa974f21e      │ Change          │ p      │  
├──────────────────────────────────┼─────────────────────┼─────────────────────┼─────────────────┼────────┤  
│ **sky launch wall time             │ 45.67 s (43.3–55.3) │ 22.44 s (20.9–25.2) │ −23.23 s (−51%) │ 0.0002** │  
│ **sky jobs launch client total     │ 74.95 s (70.8–83.4) │ 30.84 s (30.5–32.9) │ −44.11 s (−59%) │ 0.0002** │  
├──────────────────────────────────┼─────────────────────┼─────────────────────┼─────────────────┼────────┤  
│ Phase D: provisioning → submit   │ 33.08 s             │ 10.43 s             │ −22.64 s        │ 0.0002 │  
│   Ray wait \+ skylet start        │ 16.84 s             │  1.06 s             │ −15.78 s        │ 0.0002 │  
│   Runtime setup                  │  7.98 s (2.5–24.3)  │  2.00 s (1.8–2.3)   │  −5.99 s        │ 0.0002 │  
│ Phase G: RUNNING → SUCCEEDED     │ 12.58 s             │  0.22 s             │ −12.36 s        │ 0.0002 │  
└──────────────────────────────────┴─────────────────────┴─────────────────────┴─────────────────┴────────┘  
Also notice that iteration 4 also fixed the uv installation’s stall issue that’s causing the big variation of the runtime.

┌─────────────────────────────────────┬─────────────────────┬─────────────────────┬─────────┬────────┐  
│ Phase                               │ Master              │ ej-prj-i4-5         │ Change  │ p      │  
├─────────────────────────────────────┼─────────────────────┼─────────────────────┼─────────┼────────┤  
│ A: client start → job row           │  7.03 (6.4–7.2)     │  2.31 (1.8–2.7)     │  −4.72  │ 0.0002 │ \<\< iter 3  
│ B: row → STARTING                   │  2.81 (2.3–5.5)     │  2.37 (1.9–3.7)     │  −0.44  │ 0.090  │  
│ C: STARTING → provisioning          │  3.20 (2.5–3.6)     │  3.21 (2.4–3.8)     │  \+0.01  │ 0.86   │  
│ D: provisioning → job submitted     │ 33.08 (30.8–42.9)   │ 10.43 (9.9–11.4)    │ −22.64  │ 0.0002 │ \<\< iter 1&2&4  
│ E: job runs                         │  8.75 (8.2–10.0)    │  6.76 (6.5–6.9)     │  −1.99  │ 0.0002 │ \<\< iter 1  
│ F: job ended → RUNNING              │  2.94 (1.6–3.4)     │  2.45 (2.1–2.9)     │  −0.49  │ 0.13   │  
│ G: RUNNING → SUCCEEDED              │ 12.58 (3.8–20.7)    │  0.22 (0.1–1.1)     │ −12.36  │ 0.0002 │ \<\< iter 2  
│ H: SUCCEEDED → client exits         │  4.76 (2.6–6.0)     │  2.78 (2.6–3.9)     │  −1.98  │ 0.027  │ \<\< iter 1  
├─────────────────────────────────────┼─────────────────────┼─────────────────────┼─────────┼────────┤  
│ Client start → SUCCEEDED            │ 70.17 (66.9–80.7)   │ 27.94 (27.2–29.0)   │ −42.22  │ 0.0002 │  
│ Client total                        │ 74.95 (70.8–83.4)   │ 30.84 (30.5–32.9)   │ −44.11  │ 0.0002 │ **\<\< \-59%**  
└─────────────────────────────────────┴─────────────────────┴─────────────────────┴─────────┴────────┘

Phase D breaks down as follows:

┌─────────────────────────────────────┬─────────────────────┬─────────────────────┬─────────┬────────┐  
│ Phase D stage                       │ Master              │ ej-prj-i4-5         │ Change  │ p      │  
├─────────────────────────────────────┼─────────────────────┼─────────────────────┼─────────┼────────┤  
│ Create pods, wait until Running     │  2.72 (2.6–3.2)     │  1.89 (1.7–2.4)     │  −0.83  │ 0.0002 │  
│ SSH readiness                       │  0.25 (0.2–0.3)     │  0.33 (0.3–0.4)     │  \+0.08  │ 0.0002 │  
│ internal\_file\_mounts                │  1.19 (1.1–2.9)     │  1.23 (1.2–1.3)     │  \+0.04  │ 0.45   │  
│ setup\_runtime\_on\_cluster            │  7.98 (2.5–24.3)    │  2.00 (1.8–2.3)     │  −5.99  │ 0.0002 │ \<\< iter 4  
│ Ray wait \+ skylet start             │ 16.84 (7.8–20.0)    │  1.06 (0.7–1.4)     │ −15.78  │ 0.0002 │ \<\< iter 1&2&4  
│ Cluster launched → job submitted    │  3.57 (3.4–5.6)     │  3.52 (3.4–3.6)     │  −0.05  │ 0.12   │  
├─────────────────────────────────────┼─────────────────────┼─────────────────────┼─────────┼────────┤  
│ Phase D total                       │ 33.08 (30.8–42.9)   │ 10.43 (9.9–11.4)    │ −22.64  │ 0.0002 │  
│ Skylet start on its own             │  1.69 (1.6–1.8)     │  0.47 (0.3–0.6)     │  −1.21  │ 0.0002 │ \<\< iter 4  
└─────────────────────────────────────┴─────────────────────┴─────────────────────┴─────────┴────────┘

Notice that almost all running time has been optimized.  
Against master, ej-prj-i3-2, All run done on Ubuntu 26.04, i7-8700K, 32GB memory

### **Requirement 4: A short README explaining how to reproduce your numbers from a clean machine.**

**Read Me:**  
[https\://hifiweekly.github.io/skypilot\_ej\_prj/README\_i4.html](https://hifiweekly.github.io/skypilot_ej_prj/README_i4.html)

**Support Files:**  
[https\://hifiweekly.github.io/skypilot\_ej\_prj/20261105-i4-5b-vs-master.zip](https://hifiweekly.github.io/skypilot_ej_prj/20261105-i4-5b-vs-master.zip) (.py and .sh required)  
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

The [second iteration](#bookmark=id.ksoj6pws011n) focused on more aggressive code refactoring on Phase D and Phase G. I also profiled all the thread waiting time. And was able to expand the latency reduction to over 40% in total.

During the refactoring, I found a gRPC data conversion bug, causing some of the timestamps to be malformed. As the current gRPC is not enabled by default, and there are other fixes pending according to the commit logs. I didn’t enable gRPC for the benchmark. According to the profiling, by enabling the gRPC, it can further cut down the latency in Phase D, E, and F. (see [https\://hifiweekly.github.io/skypilot\_ej\_prj/grpc\_latency\_analysis.html](https://hifiweekly.github.io/skypilot_ej_prj/grpc_latency_analysis.html).

Updated workflow svg: [https\://hifiweekly.github.io/skypilot\_ej\_prj/sky\_jobs\_launch\_workflow\_i2\_6.svg](https://hifiweekly.github.io/skypilot_ej_prj/sky_jobs_launch_workflow_i2_6.svg)  
See more details in [https\://hifiweekly.github.io/skypilot\_ej\_prj/launch\_latency\_design\_after\_job2.html](https://hifiweekly.github.io/skypilot_ej_prj/launch_latency_design_after_job2.html).

**Nov 2 \~ Nov 3: Third Iteration**

In this iteration, I focused on Phase A, and optimized the way SkyPilot checked the cloud storage configuration. AWS is the hotspot here, and SkyPilot did check twice. On my Ubuntu, even though I don’t have AWS configured, it took about 5 to 6 seconds to perform the whole check in Phase A. The optimization idea is to use the existing fingerprinting mechanism to report the first check’s status and do not repeat the check.

Updated workflow svg: [https\://hifiweekly.github.io/skypilot\_ej\_prj/sky\_jobs\_launch\_workflow\_i3\_2.svg](https://hifiweekly.github.io/skypilot_ej_prj/sky_jobs_launch_workflow_i3_2.svg)  
(I also put all the important optimizations with their commit number at the bottom)

See more details in the design write-up iteration 3 section below ([https\://docs.google.com/document/d/1bFl6dX6-Yxn0bbFsV72ajOLRLsWutcu8CikZCKVhpUQ/edit?tab=t.0\#bookmark=id.wb1ickkdm5hc](https://docs.google.com/document/d/1bFl6dX6-Yxn0bbFsV72ajOLRLsWutcu8CikZCKVhpUQ/edit?tab=t.0#bookmark=id.wb1ickkdm5hc)) 

**Nov 4 \~ Nov 5: Fourth Iteration**

In this iteration, I focused on the Phase D and Phase E, particularly on the provision time, runtime loading time and the synchronization of cluster head node and worker node. In this investigation, I identified the dashboard loading is in the critical path of loading and cost the longest delay. In the latest branch, I removed the dashboard loading using Ray patch. The workflow then further unveiled a new 1s wait time in the C++, which I also optimized.

**Before Iteration 4:**  
![][image1]  
A more detailed analysis of the wait time in the chart above: [https\://docs.google.com/document/d/1qSaU-\_E6NFg3casAC-K5Ti-kH\_sHWn0Z6eaKMfhKjPc/edit?tab=t.0](https://docs.google.com/document/d/1qSaU-_E6NFg3casAC-K5Ti-kH_sHWn0Z6eaKMfhKjPc/edit?tab=t.0) 

**After Iteration 4:**  
     ![][image2]

### **Iteration 1 \- First Set: ej-prj-0, ej-prj-1, and ej-prj-3**

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

### **Iteration 2 \- Nov 1 \~ Nov 2**

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

**\* Branch ej-prj-i2-6: \[refactoring\] focusing on Phase G**  
[https\://github.com/skypilot-org/skypilot/compare/master...siliconej:skypilot:ej-prj-i2-4?expand=1](https://github.com/skypilot-org/skypilot/compare/master...siliconej:skypilot:ej-prj-i2-4?expand=1)  
**PR Description:** [https\://hifiweekly.github.io/skypilot\_ej\_prj/pr\_descriptions\_i2.html\#ej-prj-i2-4](https://hifiweekly.github.io/skypilot_ej_prj/pr_descriptions_i2.html#ej-prj-i2-4)

**Summary**

**\- Problem:** after a managed job ended, the controller ran separate head commands for its status and its end time, and the log follower ran its own status check. Without gRPC, each takes about 1.6 s.  
**\- Fix:**  
  \- One head command now returns the status, submitted time and end time together.  
  \- If the launch already saw SUCCEEDED with an end time, the monitor loop marks the job done with no head command.  
  \- The follower trusts a tail exit code of SUCCEEDED, and checks for the final status every 0.2 s instead of every 1 s.  
\- Unchanged: other terminal statuses, such as FAILED, still get a fresh check. gRPC, runtimes and missing end times fall back to the old fetch, and there's no API change.

**Result (6 nodes, 8 runs per arm, medians):**

┌──────────────────────────┬─────────┬────────────────┬────────────────────────┐  
│         Measure          │ Before  │     After      │           p            │  
├──────────────────────────┼─────────┼────────────────┼────────────────────────┤  
│ RUNNING → SUCCEEDED      │ 3.35 s  │ 0.12 s         │ 0.0001                 │  
├──────────────────────────┼─────────┼────────────────┼────────────────────────┤  
│ SUCCEEDED → client exits │ 2.48 s  │ 3.84 s (worse) │ 0.0001                 │  
├──────────────────────────┼─────────┼────────────────┼────────────────────────┤  
│ Client total             │ 36.07 s │ 34.31 s        │ rank 0.038; means 0.38 │  
└──────────────────────────┴─────────┴────────────────┴────────────────────────┘

\- Why the client gains less: SUCCEEDED arrives sooner, but the controller saves the logs after it, so the client waits longer there. ej-prj-i2-6 then cuts that wait from 4.3 s to 2.6 s.

**Risks:**  
\- Skipped checks: the new path skips the checks that normally run before marking success, in critical controller code, so it needs close review.  
\- Log display: RUNNING now lasts about 0.1 s, so a short job's logs usually appear in one burst at the end.

**Not tested:**  
\- smoke tests, including recovery  
\- an older cluster runtime  
\- the controller-cluster path  
\- multi-task pipelines and pools

Before a PR: b848c9c1e's message has the wrong explanation for the longer wait, so use 7db718aa0's wording.

**Design Doc.**  
[https\://hifiweekly.github.io/skypilot\_ej\_prj/job\_end\_time\_with\_status\_design.html](https://hifiweekly.github.io/skypilot_ej_prj/job_end_time_with_status_design.html)

### ---

### **Iteration 3 \- Nov 3 \~ Nov 4**

**\* Branch ej-prj-i3-1: This is rather an observation, could be documented.**  
During Phase A, several seconds were spent on AWS probing for either S3 storage or AWS credential. These time are wasted for local job launches if these jobs don’t have needs to store data in the cloud object storages. There are two configurations KV could be set in the job yaml files that could make the launch much faster.

See Appendix I for the benchmark result, but here is the TL;DR:  
┌────────────────────────────────┬─────────────────────┬─────────────────────┬─────────────────┬────────┐  
│ Benchmark (median, with range) │  Master 796c5bcc8   │     ej-prj-i3-1     │     Change      │   p    │  
├────────────────────────────────┼─────────────────────┼─────────────────────┼─────────────────┼────────┤  
│ sky launch wall time           │ 46.23 s (42.4–50.3) │ 27.37 s (24.6–28.8) │ −18.86 s (−41%) │ 0.0002 │  
├────────────────────────────────┼─────────────────────┼─────────────────────┼─────────────────┼────────┤  
│ **sky jobs launch client total   │ 67.5 s (57.2–70.5)  │ 34.7 s (33.3–39.4)  │ −32.8 s (−49%)  │ 0.0002** │  
└────────────────────────────────┴─────────────────────┴─────────────────────┴─────────────────┴────────┘

See this HTML for more details:   
[https\://hifiweekly.github.io/skypilot\_ej\_prj/local\_run\_speedup\_tips.html](https://hifiweekly.github.io/skypilot_ej_prj/local_run_speedup_tips.html).

**\* Branch ej-prj-i3-2: Reuse an empty storage check result for an unchanged config**  
[https\://github.com/skypilot-org/skypilot/compare/master...siliconej:skypilot:ej-prj-i3-2?expand=1](https://github.com/skypilot-org/skypilot/compare/master...siliconej:skypilot:ej-prj-i3-2?expand=1%20)   
**PR Description:** [https\://hifiweekly.github.io/skypilot\_ej\_prj/pr\_descriptions\_i3.html\#ej-prj-i3-2](https://hifiweekly.github.io/skypilot_ej_prj/pr_descriptions_i3.html#ej-prj-i3-2%20) 

**Problem:** based on the observation in Branch ej-prj-i3-1 above. **W**ith no storage cloud enabled, the stored list of enabled clouds is empty, which looks the same as "never checked". So every sky jobs launch reran the storage check and its aws and gcloud commands, even for a Kubernetes-only task. That took about 4.75 s of phase A's 7s.

**Fix:** after a storage check finds nothing, store the config fingerprint and the time in the existing config table. Later launches skip the check while the fingerprint matches and the record is under 10 minutes old.

The check still reruns when:  
\- the config changes (new fingerprint)  
\- the allowed clouds list changes  
\- the record reaches 10 minutes, which reusing it doesn't extend

Results (6 nodes, 8 measured runs per side, launches about 40 s apart, patch on vs. off):

┌──────────────────────┬──────────────┬────────────┬─────────┬────────┐  
│ Measure              │ Without      │ With       │ Change  │ p      │  
├──────────────────────┼──────────────┼────────────┼─────────┼────────┤  
│ Phase A              │  6.85 s      │  2.20 s    │ −4.65 s │ 0.0047 │  
│ sky jobs launch      │ 41.31 s      │ 37.06 s    │ −4.25 s │ 0.013  │  
└──────────────────────┴──────────────┴────────────┴─────────┴────────┘

No other phase changed significantly. sky launch doesn't appear to benefit, because it reaches this check only for tasks with storage mounts.

**Trade-offs:**  
\- New credentials added without running sky check can go unnoticed for up to 10 minutes.  
\- A temporary check failure can also stick for up to 10 minutes. sky check clears both as soon as it finds an enabled cloud.  
\- Launches more than 10 minutes apart get no saving.

**Scope:** storage checks only, with no API change or migration. Compute checks are unchanged. Phase C's own aws configure list call stays, since it's a separate path that only NO\_UPLOAD removes.

**Design Doc.**  
[https\://hifiweekly.github.io/skypilot\_ej\_prj/empty\_storage\_check\_reuse\_design.html](https://hifiweekly.github.io/skypilot_ej_prj/empty_storage_check_reuse_design.html)

### ---

### **Iteration 4 \- Nov 4 \~ Nov 5**

**\* Branch ej-prj-i4-1 thru Branch ej-prj-i4-5:** the ej-prj-i4-\* branches all build on ej-prj-i3-2 (d771518f1). Together they add five launch-speed changes, all inside provisioning, phase D. ej-prj-i4-5 contains all five.  
[https\://github.com/skypilot-org/skypilot/compare/master...siliconej:skypilot:ej-prj-i4-5?expand=1](https://github.com/skypilot-org/skypilot/compare/master...siliconej:skypilot:ej-prj-i4-5?expand=1)   
**PR Description:** [https\://hifiweekly.github.io/skypilot\_ej\_prj/pr\_descriptions\_i4.html](https://hifiweekly.github.io/skypilot_ej_prj/pr_descriptions_i4.html) 

┌─────────────┬────────────────────┬──────────────────────────────────────────────────────────┬────────────────┐  
│ Branch      │ Based on           │ Adds                                                     │ On the fork?   │  
├─────────────┼────────────────────┼──────────────────────────────────────────────────────────┼────────────────┤  
│ ej-prj-i4-1 │ ej-prj-i3-2        │ 925abd467  0.2 s head-node poll                          │ yes            │  
│ ej-prj-i4-2 │ ej-prj-i4-1        │ afd94a698  Lean skylet launcher                          │ yes            │  
│ ej-prj-i4-3 │ ej-prj-i4-1        │ 8bfc6e4c6  Offline wheel install                         │ yes            │  
│ ej-prj-i4-4 │ ej-prj-i4-3        │ d6fa30e94  Lean skylet launcher (from i4-2, lint fix)    │ yes            │  
│ ej-prj-i4-5 │ ej-prj-i4-4        │ 1266cccd2  No Ray dashboard                              │ yes            │  
│             │                    │ fa974f21e  0.2 s Ray local connect retry                 │ no (local)     │  
└─────────────┴────────────────────┴──────────────────────────────────────────────────────────┴────────────────┘

**The changes**

**1\. 0.2 s head-node poll (925abd467)**  
   \- What: a new Ray patch (utils.py.patch) makes a worker's ray start check every 0.2 s, instead of every 2 s, whether the head has registered. The number of checks rises tenfold, so the time before giving up stays the same.  
   \- Measured: sky launch −0.93 s (p \= 0.0048).  
**2\. Lean skylet launcher (afd94a698, then d6fa30e94 on i4-4)**  
   \- What: skylet starts without import sky. A new launch\_constants.py has no SkyPilot imports, and attempt\_skylet.py has its own copies of the few helpers it needs.  
   \- Measured: skylet start fell from about 3.0 s to 0.3 s. Launch totals didn't change until the dashboard was removed, because Ray's own wait was longer than skylet's start.  
**3\. Offline wheel install (8bfc6e4c6)**  
   \- What: pods install the SkyPilot wheel with uv pip install \--offline and fall back to the network only if that fails. The PyPI dry run is skipped once the uv cache is warm.  
   \- Measured: runtime setup 3.05 → 1.64 s, and setup stalls 3/24 → 0/24 launches. sky launch −0.63 s (p \= 0.0073).  
**4\. No Ray dashboard (1266cccd2)**  
   \- What: SKYPILOT\_RAY\_NO\_DASHBOARD=1 and two Ray patches (services.py.patch and agent.py.patch). ray start no longer starts or waits for the dashboard, and the dashboard agent loads no modules.  
   \- Measured: sky jobs launch −3.57 s and sky launch −3.12 s (both p \< 0.0001).  
   \- Cost: Ray's dashboard UI, metrics export and Ray job and state APIs are gone on SkyPilot clusters.  
**5\. 0.2 s Ray local connect retry (fa974f21e)**  
   \- What: RAY\_raylet\_client\_connect\_timeout\_milliseconds=200 and RAY\_raylet\_client\_num\_connect\_attempts=50 on ray start. Ray now retries after 0.2 s instead of 1 s when a raylet's first connect to its plasma store fails, which happened on about a third of raylets. Ray still gives up after about 10 s, as before.  
   \- Measured: the last worker registers about 0.46 s sooner, and the alive-nodes wait fell from 1.83 to 0.50 s. Neither launch total moved significantly, because file mounts and runtime setup now decide the length of the launch.

**Together**

Against master, ej-prj-i4-5 cuts sky jobs launch from 74.95 to 30.84 s (−59%) and sky launch from 45.67 to 22.44 s (−51%). Every branch run beat every master run. These numbers include the bytecode image, which isn't published.

**State**  
All changes committed and pushed to fork.

**Design doc.**  
[https\://hifiweekly.github.io/skypilot\_ej\_prj/launch\_latency\_design\_i4.html](https://hifiweekly.github.io/skypilot_ej_prj/launch_latency_design_i4.html) 

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

Appendix C: Ubuntu Benchmark Result: sky-launch-ab  
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

## Appendix G: Nov 2 User Perceived Time

(skypilot) ejiang@benjamin:\~/2T2/Projects/skypilot\$ time sky jobs launch \-y \-c i2 ej/task\_bytecode.yaml   
YAML to run: ej/task\_bytecode.yaml  
Launching managed job 'i2'...  
├── Waiting for task resources on 6 nodes.  
└── Job started. Streaming logs... (Ctrl-C to exit log streaming; job will not be killed)  
(head, rank=0, pid=1353) \[rank 0\] started at 1793688490.979418040 on i2-475-48017c75-head  
(worker3, rank=3, pid=616, ip=10.244.0.12) \[rank 3\] started at 1793688491.373891838 on i2-475-48017c75-worker3  
(worker2, rank=2, pid=621, ip=10.244.0.13) \[rank 2\] started at 1793688491.332624540 on i2-475-48017c75-worker2  
(worker4, rank=4, pid=618, ip=10.244.0.11) \[rank 4\] started at 1793688491.531573109 on i2-475-48017c75-worker4  
(worker5, rank=5, pid=611, ip=10.244.0.14) \[rank 5\] started at 1793688491.529177662 on i2-475-48017c75-worker5  
(worker1, rank=1, pid=620, ip=10.244.0.10) \[rank 1\] started at 1793688491.856921367 on i2-475-48017c75-worker1  
✓ Job finished (status: SUCCEEDED).  
✓ Managed job finished: 475 (status: SUCCEEDED).

📋 Useful Commands  
Managed Job ID: 475  
├── To cancel the job:		sky jobs cancel 475  
├── To stream job logs:		sky jobs logs 475  
├── To stream controller logs:	sky jobs logs \--controller 475  
└── To view all managed jobs:	sky jobs queue

**real	0m46.066s**  
user	0m2.755s  
sys	0m0.259s  
(skypilot) ejiang@benjamin:\~/2T2/Projects/skypilot\$ time sky jobs launch \-y \-c i2 ej/task\_bytecode.yaml   
YAML to run: ej/task\_bytecode.yaml  
Launching managed job 'i2'...  
├── Waiting for task resources on 6 nodes.  
└── Job started. Streaming logs... (Ctrl-C to exit log streaming; job will not be killed)  
(head, rank=0, pid=1318) \[rank 0\] started at 1793688650.792021163 on i2-476-48017c75-head  
(worker2, rank=2, pid=609, ip=10.244.0.17) \[rank 2\] started at 1793688651.771657470 on i2-476-48017c75-worker2  
(worker1, rank=1, pid=618, ip=10.244.0.19) \[rank 1\] started at 1793688651.739417433 on i2-476-48017c75-worker1  
(worker3, rank=3, pid=613, ip=10.244.0.18) \[rank 3\] started at 1793688651.813173027 on i2-476-48017c75-worker3  
(worker5, rank=5, pid=617, ip=10.244.0.20) \[rank 5\] started at 1793688651.865371093 on i2-476-48017c75-worker5  
(worker4, rank=4, pid=609, ip=10.244.0.16) \[rank 4\] started at 1793688652.188471140 on i2-476-48017c75-worker4  
✓ Job finished (status: SUCCEEDED).  
✓ Managed job finished: 476 (status: SUCCEEDED).

📋 Useful Commands  
Managed Job ID: 476  
├── To cancel the job:		sky jobs cancel 476  
├── To stream job logs:		sky jobs logs 476  
├── To stream controller logs:	sky jobs logs \--controller 476  
└── To view all managed jobs:	sky jobs queue

**real	0m41.092s**  
user	0m2.638s  
sys	0m0.290s  
(skypilot) ejiang@benjamin:\~/2T2/Projects/skypilot\$ time sky jobs launch \-y \-c i2 ej/task\_bytecode.yaml   
YAML to run: ej/task\_bytecode.yaml  
Launching managed job 'i2'...  
├── Waiting for task resources on 6 nodes.  
└── Job started. Streaming logs... (Ctrl-C to exit log streaming; job will not be killed)  
(raylet, ip=10.244.0.24) \[2026-11-03 06:52:43,394 W 604 604\] rpc\_client.h:638: Failed to connect to the observability pubsub at address i2-477-48017c75-head.default.svc.cluster.local:6380 within 5 seconds.  
(raylet, ip=10.244.0.23) \[2026-11-03 06:52:43,395 W 612 612\] rpc\_client.h:638: Failed to connect to the observability pubsub at address i2-477-48017c75-head.default.svc.cluster.local:6380 within 5 seconds.  
(worker2, rank=2, pid=612, ip=10.244.0.25) \[rank 2\] started at 1793688765.703214143 on i2-477-48017c75-worker2  
(worker4, rank=4, pid=605, ip=10.244.0.22) \[rank 4\] started at 1793688765.703171127 on i2-477-48017c75-worker4  
(worker1, rank=1, pid=604, ip=10.244.0.24) \[rank 1\] started at 1793688765.784297502 on i2-477-48017c75-worker1  
(worker3, rank=3, pid=612, ip=10.244.0.23) \[rank 3\] started at 1793688765.847164266 on i2-477-48017c75-worker3  
(head, rank=0, pid=1439) \[rank 0\] started at 1793688765.873922832 on i2-477-48017c75-head  
(worker5, rank=5, pid=608, ip=10.244.0.26) \[rank 5\] started at 1793688766.281949504 on i2-477-48017c75-worker5  
✓ Job finished (status: SUCCEEDED).  
✓ Managed job finished: 477 (status: SUCCEEDED).

📋 Useful Commands  
Managed Job ID: 477  
├── To cancel the job:		sky jobs cancel 477  
├── To stream job logs:		sky jobs logs 477  
├── To stream controller logs:	sky jobs logs \--controller 477  
└── To view all managed jobs:	sky jobs queue

**real	0m50.865s**  
user	0m2.601s  
sys	0m0.254s  
(skypilot) ejiang@benjamin:\~/2T2/Projects/skypilot\$ time sky jobs launch \-y \-c i2 ej/task\_bytecode.yaml   
YAML to run: ej/task\_bytecode.yaml  
Launching managed job 'i2'...  
├── Waiting for task resources on 6 nodes.  
└── Job started. Streaming logs... (Ctrl-C to exit log streaming; job will not be killed)  
(head, rank=0, pid=1329) \[rank 0\] started at 1793688846.109701606 on i2-478-48017c75-head  
(worker5, rank=5, pid=657, ip=10.244.0.32) \[rank 5\] started at 1793688847.115384906 on i2-478-48017c75-worker5  
(worker3, rank=3, pid=665, ip=10.244.0.29) \[rank 3\] started at 1793688847.231560147 on i2-478-48017c75-worker3  
(worker1, rank=1, pid=655, ip=10.244.0.30) \[rank 1\] started at 1793688847.321857649 on i2-478-48017c75-worker1  
(worker2, rank=2, pid=655, ip=10.244.0.28) \[rank 2\] started at 1793688847.399336759 on i2-478-48017c75-worker2  
(worker4, rank=4, pid=652, ip=10.244.0.31) \[rank 4\] started at 1793688847.598413756 on i2-478-48017c75-worker4  
✓ Job finished (status: SUCCEEDED).  
✓ Managed job finished: 478 (status: SUCCEEDED).

📋 Useful Commands  
Managed Job ID: 478  
├── To cancel the job:		sky jobs cancel 478  
├── To stream job logs:		sky jobs logs 478  
├── To stream controller logs:	sky jobs logs \--controller 478  
└── To view all managed jobs:	sky jobs queue

**real	0m45.767s**  
user	0m2.671s  
sys	0m0.257s

## Appendix H: Iteration 1 and Iteration 2: Commit List baselined on master

ej-prj-i2-6 has 23 commits on top of master at 796c5bcc8, all aimed at faster Kubernetes launches for sky launch and sky jobs launch. It's on fork at 407784c57 and isn't merged.

┌──────────────────────┬───────────┬────────────────────────────────────────────────────────────────────────────┐  
│         Area         │  Commit   │                                   Change                                   │  
├──────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────────┤  
│ Pods and Ray startup │ 7251606a1 │ Start Ray before preparing SkyPilot in the pod                             │  
├──────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────────┤  
│                      │ ab543979c │ Wait for Ray on the head in one call, and start skylet meanwhile           │  
├──────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────────┤  
│                      │ 863c66883 │ Skip ray stop when no Ray is running                                       │  
├──────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────────┤  
│                      │ c727790d3 │ Pre-install the runtime pins in the image                                  │  
├──────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────────┤  
│                      │ 0c576e860 │ Precompile runtime bytecode in the k8s images                              │  
├──────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────────┤  
│                      │ 1ddcc3240 │ Don't import Ray Serve in ray start                                        │  
├──────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────────┤  
│                      │ 71b3c9213 │ Watch pods between polls while waiting for them to run                     │  
├──────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────────┤  
│ Managed jobs         │ 04040ba00 │ Notice a short managed job's end sooner                                    │  
├──────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────────┤  
│                      │ 96019e28a │ Check a followed job's status when its controller logs                     │  
├──────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────────┤  
│                      │ 30890119e │ Keep job timestamps exact over gRPC                                        │  
├──────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────────┤  
│                      │ 7db718aa0 │ Get a finished job's end time with its status                              │  
├──────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────────┤  
│                      │ 680e03581 │ Check for a finished job's saved logs every 0.2 s                          │  
├──────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────────┤  
│                      │ d080ff076 │ Get a finished job's log directory with its status                         │  
├──────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────────┤  
│ Storage              │ dc6c01a92 │ Look up the enabled storage clouds once per request                        │  
├──────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────────┤  
│ Lost pods            │ b7f871174 │ Don't retry an exec whose target pod is gone                               │  
├──────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────────┤  
│                      │ ee4e69d1a │ Fail setup fast when an instance is lost                                   │  
├──────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────────┤  
│                      │ fab8e57ed │ Smoke test a pod deleted while the cluster is set up                       │  
├──────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────────┤  
│                      │ cd9a21dcc │ Add is\_exec\_target\_gone to the command\_runner stub                         │  
├──────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────────┤  
│ Benchmark            │ e08dabd78 │ Add a Kubernetes launch latency A/B benchmark                              │  
├──────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────────┤  
│                      │ 6de470672 │ Trim the benchmark's README                                                │  
├──────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────────┤  
│                      │ 08a101ba1 │ Purge stale clusters in the benchmark, and count only good runs            │  
├──────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────────┤  
│                      │ 7e488f558 │ Let the benchmark use a separate task for the candidate                    │  
├──────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────────┤  
│                      │ 407784c57 │ Add ej/bench\_launch\_latency.py (an old copy) and four ej/\*.yaml task files │  
├──────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────────┤  
│                      │ 650eb5a67 │ \[Kubernetes\] Create a file mount's target in the same call as rsync        │  
├──────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────────┤  
│                      │ d771518f1 │ \[Core\] Reuse an empty storage check result for an unchanged config         │  
└──────────────────────┴───────────┴────────────────────────────────────────────────────────────────────────────┘

## Appendix I: Iteration 3: Cloud Storage Configuration Benchmark

ej-prj-i3-1 is much faster than master on both benchmarks with both speedup settings on: sky launch drops from 46.2 s to 27.4 s, and sky jobs launch from 67.5 s to 34.7 s. All 16 measured runs of each benchmark succeeded, and every branch run beat every master run.

┌────────────────────────────────┬─────────────────────┬─────────────────────┬─────────────────┬────────┐  
│ Benchmark (median, with range) │  Master 796c5bcc8   │     ej-prj-i3-1     │     Change      │   p    │  
├────────────────────────────────┼─────────────────────┼─────────────────────┼─────────────────┼────────┤  
│ sky launch wall time           │ 46.23 s (42.4–50.3) │ 27.37 s (24.6–28.8) │ −18.86 s (−41%) │ 0.0002 │  
├────────────────────────────────┼─────────────────────┼─────────────────────┼─────────────────┼────────**┤**  
**│ sky jobs launch client total   │ 67.5 s (57.2–70.5)  │ 34.7 s (33.3–39.4)  │ −32.8 s (−49%)  │ 0.0002** │  
└────────────────────────────────┴─────────────────────┴─────────────────────┴─────────────────┴────────┘

I worked out both p-values from the raw runs (two-sided exact test), because the jobs report rounds its value to 0.0000.

Where sky jobs launch saves time:

┌─────────────────────────────────┬────────┬─────────────┬─────────┐  
│              Phase              │ Master │ ej-prj-i3-1 │ Change  │  
├─────────────────────────────────┼────────┼─────────────┼─────────┤  
│ A: client start → job row       │ 2.3 s  │ 2.4 s       │ 0.0 s   │  
├─────────────────────────────────┼────────┼─────────────┼─────────┤  
│ B: row → STARTING               │ 2.4 s  │ 2.7 s       │ \+0.3 s  │  
├─────────────────────────────────┼────────┼─────────────┼─────────┤  
│ C: STARTING → provisioning      │ 1.2 s  │ 1.1 s       │ 0.0 s   │  
├─────────────────────────────────┼────────┼─────────────┼─────────┤  
│ D: provisioning → job submitted │ 33.2 s │ 15.3 s      │ −17.9 s │  
├─────────────────────────────────┼────────┼─────────────┼─────────┤  
│ E: job runs                     │ 9.4 s  │ 7.3 s       │ −2.1 s  │  
├─────────────────────────────────┼────────┼─────────────┼─────────┤  
│ F: job ended → RUNNING          │ 2.9 s  │ 2.7 s       │ −0.2 s  │  
├─────────────────────────────────┼────────┼─────────────┼─────────┤  
│ G: RUNNING → SUCCEEDED          │ 7.1 s  │ 0.1 s       │ −7.0 s  │  
├─────────────────────────────────┼────────┼─────────────┼─────────┤  
│ H: SUCCEEDED → client exits     │ 3.5 s  │ 2.6 s       │ −0.8 s  │  
└─────────────────────────────────┴────────┴─────────────┴─────────┘

Phase changes are differences of medians, so they don't add up exactly to the total. Inside phase D, the largest savings are:  
\- Ray wait \+ skylet start: 18.04 → 4.70 s  
**\- Runtime setup: 8.60 → 2.05 s**

How to read these numbers:  
\- The settings took effect: phases A and C are about 2.3 s and 1.2 s on both sides, matching the earlier tips results. They're also why the gap is smaller than the earlier 79.1 → 45.0 s, which ran without these settings.  
\- Image and code are measured together: master uses the published image and the branch uses the bytecode image, so this isn't the code change alone. The bytecode image also isn't published, so users wouldn't get this result today.  
\- This says nothing about the ej-prj-i3-1 rsync change on its own: file mounts were actually 1.76 s on master and 2.26 s on the branch, with overlapping ranges (1.1–2.2 vs 1.2–3.5 s). The direct test against ej-prj-i2-6 is the better evidence, and it showed 1.45 → 1.08 s with no change in total time.  
\- The sky launch comparison is less affected by the settings: that benchmark always sets allowed\_clouds, so only NO\_UPLOAD was new there.

## Appendix J: Nov 4 User Perceived Time

(skypilot) ejiang@benjamin:\~/2T2/Projects/skypilot\$ time sky jobs launch \-y ej/task\_bytecode.yaml  
YAML to run: ej/task\_bytecode.yaml  
Launching managed job 'sky-106e-ejiang'...  
├── Waiting for task resources on 6 nodes.  
└── Job started. Streaming logs... (Ctrl-C to exit log streaming; job will not be killed)  
(head, rank=0, pid=1328) \[rank 0\] started at 1793818068.185642809 on sky-106e-ejiang-636-48017c75-head  
(worker1, rank=1, pid=673, ip=10.244.0.50) \[rank 1\] started at 1793818068.585237383 on sky-106e-ejiang-636-48017c75-worker1  
(worker5, rank=5, pid=671, ip=10.244.0.53) \[rank 5\] started at 1793818068.639009291 on sky-106e-ejiang-636-48017c75-worker5  
(worker3, rank=3, pid=672, ip=10.244.0.51) \[rank 3\] started at 1793818068.728010902 on sky-106e-ejiang-636-48017c75-worker3  
(worker4, rank=4, pid=666, ip=10.244.0.49) \[rank 4\] started at 1793818068.807059254 on sky-106e-ejiang-636-48017c75-worker4  
(worker2, rank=2, pid=659, ip=10.244.0.52) \[rank 2\] started at 1793818069.076884678 on sky-106e-ejiang-636-48017c75-worker2  
✓ Job finished (status: SUCCEEDED).  
✓ Managed job finished: 636 (status: SUCCEEDED).

📋 Useful Commands  
Managed Job ID: 636  
├── To cancel the job:		sky jobs cancel 636  
├── To stream job logs:		sky jobs logs 636  
├── To stream controller logs:	sky jobs logs \--controller 636  
└── To view all managed jobs:	sky jobs queue

**real	0m49.501s**  
user	0m2.693s  
sys	0m0.260s  
(skypilot) ejiang@benjamin:\~/2T2/Projects/skypilot\$ time sky jobs launch \-y ej/task\_bytecode.yaml  
YAML to run: ej/task\_bytecode.yaml  
Launching managed job 'sky-a9ad-ejiang'...  
├── Waiting for task resources on 6 nodes.  
└── Job started. Streaming logs... (Ctrl-C to exit log streaming; job will not be killed)  
(head, rank=0, pid=1324) \[rank 0\] started at 1793818107.411045081 on sky-a9ad-ejiang-637-48017c75-head  
(worker4, rank=4, pid=607, ip=10.244.0.58) \[rank 4\] started at 1793818108.152072607 on sky-a9ad-ejiang-637-48017c75-worker4  
(worker1, rank=1, pid=603, ip=10.244.0.56) \[rank 1\] started at 1793818108.110732960 on sky-a9ad-ejiang-637-48017c75-worker1  
(worker3, rank=3, pid=606, ip=10.244.0.59) \[rank 3\] started at 1793818108.180162061 on sky-a9ad-ejiang-637-48017c75-worker3  
(worker2, rank=2, pid=615, ip=10.244.0.57) \[rank 2\] started at 1793818108.588622946 on sky-a9ad-ejiang-637-48017c75-worker2  
(worker5, rank=5, pid=612, ip=10.244.0.55) \[rank 5\] started at 1793818108.590081974 on sky-a9ad-ejiang-637-48017c75-worker5  
✓ Job finished (status: SUCCEEDED).  
✓ Managed job finished: 637 (status: SUCCEEDED).

📋 Useful Commands  
Managed Job ID: 637  
├── To cancel the job:		sky jobs cancel 637  
├── To stream job logs:		sky jobs logs 637  
├── To stream controller logs:	sky jobs logs \--controller 637  
└── To view all managed jobs:	sky jobs queue

**real	0m36.300s**  
user	0m2.678s  
sys	0m0.261s  
(skypilot) ejiang@benjamin:\~/2T2/Projects/skypilot\$ time sky jobs launch \-y ej/task\_bytecode.yaml  
YAML to run: ej/task\_bytecode.yaml  
Launching managed job 'sky-a759-ejiang'...  
├── Waiting for task resources on 6 nodes.  
└── Job started. Streaming logs... (Ctrl-C to exit log streaming; job will not be killed)  
(head, rank=0, pid=1328) \[rank 0\] started at 1793818228.209299101 on sky-a759-ejiang-638-48017c75-head  
(worker2, rank=2, pid=584, ip=10.244.0.65) \[rank 2\] started at 1793818228.991223431 on sky-a759-ejiang-638-48017c75-worker2  
(worker3, rank=3, pid=592, ip=10.244.0.61) \[rank 3\] started at 1793818229.045513140 on sky-a759-ejiang-638-48017c75-worker3  
(worker4, rank=4, pid=584, ip=10.244.0.63) \[rank 4\] started at 1793818229.032756622 on sky-a759-ejiang-638-48017c75-worker4  
(worker1, rank=1, pid=588, ip=10.244.0.62) \[rank 1\] started at 1793818229.620459745 on sky-a759-ejiang-638-48017c75-worker1  
(worker5, rank=5, pid=583, ip=10.244.0.64) \[rank 5\] started at 1793818229.620436025 on sky-a759-ejiang-638-48017c75-worker5  
✓ Job finished (status: SUCCEEDED).  
✓ Managed job finished: 638 (status: SUCCEEDED).

📋 Useful Commands  
Managed Job ID: 638  
├── To cancel the job:		sky jobs cancel 638  
├── To stream job logs:		sky jobs logs 638  
├── To stream controller logs:	sky jobs logs \--controller 638  
└── To view all managed jobs:	sky jobs queue

**real	0m35.944s**  
user	0m2.581s  
sys	0m0.277s  
(skypilot) ejiang@benjamin:\~/2T2/Projects/skypilot\$ time sky jobs launch \-y ej/task\_bytecode.yaml  
YAML to run: ej/task\_bytecode.yaml  
Launching managed job 'sky-544d-ejiang'...  
├── Waiting for task resources on 6 nodes.  
└── Job started. Streaming logs... (Ctrl-C to exit log streaming; job will not be killed)  
(head, rank=0, pid=1324) \[rank 0\] started at 1793818293.633702581 on sky-544d-ejiang-639-48017c75-head  
(worker4, rank=4, pid=597, ip=10.244.0.69) \[rank 4\] started at 1793818294.390290492 on sky-544d-ejiang-639-48017c75-worker4  
(worker1, rank=1, pid=597, ip=10.244.0.71) \[rank 1\] started at 1793818294.338773376 on sky-544d-ejiang-639-48017c75-worker1  
(worker2, rank=2, pid=597, ip=10.244.0.67) \[rank 2\] started at 1793818294.356252429 on sky-544d-ejiang-639-48017c75-worker2  
(worker3, rank=3, pid=601, ip=10.244.0.70) \[rank 3\] started at 1793818294.435096352 on sky-544d-ejiang-639-48017c75-worker3  
(worker5, rank=5, pid=590, ip=10.244.0.68) \[rank 5\] started at 1793818294.635159965 on sky-544d-ejiang-639-48017c75-worker5  
✓ Job finished (status: SUCCEEDED).  
✓ Managed job finished: 639 (status: SUCCEEDED).

📋 Useful Commands  
Managed Job ID: 639  
├── To cancel the job:		sky jobs cancel 639  
├── To stream job logs:		sky jobs logs 639  
├── To stream controller logs:	sky jobs logs \--controller 639  
└── To view all managed jobs:	sky jobs queue

**real	0m36.479s**  
user	0m2.592s  
sys	0m0.245s  


[image1]: <data:image/png;base64,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>

[image2]: <data:image/png;base64,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>