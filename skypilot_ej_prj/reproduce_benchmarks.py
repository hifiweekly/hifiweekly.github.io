#!/usr/bin/env python3
"""Reproduces the Kubernetes launch-latency benchmarks in ej/.

Subcommands (see `--help` of each):

  jobs-ab        Interleaved `sky jobs launch` A/B of two git refs (default:
                 origin/master vs ej-prj-3-job1). Per-phase times A-H, the
                 phase-D stage breakdown, and a permutation test.
                 -> launch_latency_consolidated_design.md §13, §14, §15.
  storage-probe  Logs every storage-cloud lookup inside `jobs.launch` for two
                 refs (temporary instrumentation, restored afterwards).
                 -> launch_latency_consolidated_design.md §13.
  pod-probe      Launches one throwaway cluster and measures, inside the pods:
                 the uv dry-run resolve per pod, the marker waits, offline vs
                 online wheel resolve, and cold vs warm `attempt_skylet`
                 import. -> launch_latency_consolidated_design.md §15.
  pull-policy    Pod start time with imagePullPolicy IfNotPresent (through
                 pod_config) against the template's Always.
                 -> launch_latency_consolidated_design.md §15.
  sky-launch-ab  Runs the `sky launch` A/B,
                 tests/kubernetes/launching_benchmarks/bench_launch_latency.py.
                 For `run`, it supplies --task (from --nodes), --context and
                 --out unless they are given.
                 -> launch_latency_consolidated_design.md §7.1.

Requirements:
  - Run from the repo's venv (`source .venv/bin/activate`), with a local API
    server and a Kubernetes context (default `kind-skypilot`).
  - jobs-ab and storage-probe switch the checkout (detached) and restart the
    API server with `sky api start --deploy` (consolidation mode). They refuse
    to run with uncommitted changes to tracked files, and always restore the
    original ref and restart the server on it. Untracked files (such as ej/)
    are left alone.
  - Only clusters this script creates (named `bench-*`, or `<job-name>-<id>`
    for managed jobs) are launched or torn down.

Results go to ej/bench_results/<timestamp>-<subcommand>/ (report.md,
summary.json, and raw traces).
"""
import argparse
import contextlib
import datetime
import glob
import json
import os
import pathlib
import random
import re
import shutil
import sqlite3
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

REPO = pathlib.Path(__file__).resolve().parent.parent
HOME = pathlib.Path.home()
JOBS_DB = HOME / '.sky' / 'spot_jobs.db'
STATE_DB = HOME / '.sky' / 'state.db'
SKY_LOGS = HOME / 'sky_logs'
TERMINAL = ('SUCCEEDED', 'FAILED', 'FAILED_SETUP', 'FAILED_PRECHECKS',
            'FAILED_NO_RESOURCE', 'FAILED_CONTROLLER', 'CANCELLED')
PHASES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'to_SUCCEEDED', 'total']
PHASE_NAMES = {
    'A': 'client start -> job row created',
    'B': 'row created -> STARTING',
    'C': 'STARTING -> provisioning starts',
    'D': 'provisioning -> job submitted',
    'E': 'job runs on the cluster',
    'F': 'job ended -> RUNNING',
    'G': 'RUNNING -> SUCCEEDED',
    'H': 'SUCCEEDED -> client exits',
    'to_SUCCEEDED': 'client start -> SUCCEEDED',
    'total': 'client total',
}
D_STAGES = [
    'pre', 'pods', 'ssh', 'file_mounts', 'runtime_setup', 'ray_skylet',
    'submit', 'D'
]
D_STAGE_NAMES = {
    'pre': "'Provisioning on' event -> provision log starts",
    'pods': 'create pods, wait until Running',
    'ssh': 'SSH readiness',
    'file_mounts': 'internal_file_mounts',
    'runtime_setup': 'setup_runtime_on_cluster',
    'ray_skylet': 'Ray wait + skylet start',
    'submit': 'cluster launched -> job submitted',
    'D': 'phase D total',
}


# ----------------------------------------------------------------- helpers
def log(msg: str) -> None:
    print(f'[{datetime.datetime.now():%H:%M:%S}] {msg}', flush=True)


def run(cmd: List[str],
        check: bool = True,
        timeout: Optional[float] = None,
        env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd,
                          cwd=REPO,
                          check=check,
                          timeout=timeout,
                          env=env,
                          text=True,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)


def sky_cmd() -> str:
    candidate = pathlib.Path(sys.executable).parent / 'sky'
    return str(candidate) if candidate.exists() else 'sky'


def quiet_env() -> Dict[str, str]:
    env = dict(os.environ)
    env['SKYPILOT_DEBUG'] = '0'
    return env


def new_out_dir(name: str, out: Optional[str]) -> pathlib.Path:
    path = pathlib.Path(out) if out else (
        REPO / 'ej' / 'bench_results' /
        f'{datetime.datetime.now():%Y%m%d-%H%M%S}-{name}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_task(path: pathlib.Path,
               nodes: int,
               context: str,
               extra_config: Optional[str] = None) -> None:
    text = (f'resources:\n'
            f'  infra: k8s/{context}\n'
            f'  cpus: 1+\n'
            f'  memory: 1+\n'
            f'num_nodes: {nodes}\n'
            f'run: |\n'
            f'  echo "[rank ${{SKYPILOT_NODE_RANK}}] started at '
            f'$(date +%s.%N) on $(hostname)"\n')
    if extra_config:
        text += extra_config
    path.write_text(text)


def describe(values: List[float]) -> Dict[str, float]:
    return dict(median=statistics.median(values),
                mean=statistics.mean(values),
                min=min(values),
                max=max(values),
                n=len(values))


def permutation_p(a: List[float], b: List[float], rounds: int = 20000) -> float:
    """Two-sided permutation test on the difference of means."""
    rng = random.Random(0)
    observed = abs(statistics.mean(a) - statistics.mean(b))
    pooled = a + b
    hits = 0
    for _ in range(rounds):
        rng.shuffle(pooled)
        diff = abs(
            statistics.mean(pooled[:len(a)]) - statistics.mean(pooled[len(a):]))
        if diff >= observed - 1e-12:
            hits += 1
    return (hits + 1) / (rounds + 1)


# ------------------------------------------------------------- git / server
def current_ref() -> str:
    branch = run(['git', 'branch', '--show-current']).stdout.strip()
    return branch or run(['git', 'rev-parse', 'HEAD']).stdout.strip()


def short_sha(ref: str) -> str:
    return run(['git', 'rev-parse', '--short', ref]).stdout.strip()


def ensure_clean() -> None:
    dirty = run(['git', 'status', '--porcelain',
                 '--untracked-files=no']).stdout.strip()
    if dirty:
        sys.exit('Tracked files have uncommitted changes; commit or stash '
                 f'them first:\n{dirty}')


def checkout(ref: str) -> None:
    branch = run(
        ['git', 'rev-parse', '--verify', '--quiet', f'refs/heads/{ref}'],
        check=False).returncode == 0
    if branch:
        run(['git', 'switch', '-q', ref])
    else:
        run(['git', 'switch', '-q', '--detach', ref])


def restart_server() -> str:
    """Restarts the local API server on the checked-out code."""
    run([sky_cmd(), 'api', 'stop'], check=False, timeout=300)
    run([sky_cmd(), 'api', 'start', '--deploy'], timeout=600)
    info = run([sky_cmd(), 'api', 'info'], check=False, timeout=120).stdout
    match = re.search(r'commit: ([0-9a-f]+)', info)
    head = run(['git', 'rev-parse', 'HEAD']).stdout.strip()
    if not match or not head.startswith(match.group(1)[:9]):
        raise RuntimeError(f'API server is not on HEAD {head[:9]}:\n{info}')
    return head[:9]


@contextlib.contextmanager
def restore_checkout() -> Iterator[None]:
    original = current_ref()
    try:
        yield
    finally:
        log(f'Restoring {original} and restarting the API server on it')
        run(['git', 'checkout', '--', 'sky/data/storage.py'], check=False)
        checkout(original)
        restart_server()


# ------------------------------------------------------------ jobs tracing
def busy_jobs() -> List[Tuple[int, str]]:
    db = sqlite3.connect(JOBS_DB)
    return [
        (jid, status)
        for jid, status in db.execute('select spot_job_id, status from spot')
        if status not in TERMINAL
    ]


def trace_jobs_launch(out_path: pathlib.Path, task: pathlib.Path,
                      job_name: str) -> Dict[str, Any]:
    """Runs one `sky jobs launch`, recording client output and statuses."""
    busy = busy_jobs()
    if busy:
        raise RuntimeError(f'Another managed job is in progress: {busy}')
    db = sqlite3.connect(JOBS_DB)
    max_id = db.execute('select max(spot_job_id) from spot').fetchone()[0] or 0
    events: List[Tuple[float, str, str]] = []
    seen: set = set()
    stop = threading.Event()
    t0 = time.time()

    def sample() -> None:
        conn = sqlite3.connect(JOBS_DB)
        last: Dict[int, str] = {}
        while not stop.is_set():
            for jid, status in conn.execute(
                    'select spot_job_id, status from spot '
                    'where spot_job_id > ?', (max_id,)):
                seen.add(jid)
                if last.get(jid) != status:
                    events.append(
                        (time.time() - t0, 'STATUS', f'job {jid} {status}'))
                    last[jid] = status
            time.sleep(0.1)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    proc = subprocess.Popen(
        [sky_cmd(), 'jobs', 'launch', '-y', '-c', job_name,
         str(task)],
        cwd=REPO,
        env=quiet_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        line = re.sub(r'\x1b\[[0-9;]*m', '', line).rstrip()
        if line:
            events.append((time.time() - t0, 'OUT', line[:200]))
    proc.wait()
    exit_t = time.time()
    events.append((exit_t - t0, 'EXIT', str(proc.returncode)))
    time.sleep(0.3)
    stop.set()
    sampler.join(timeout=2)
    events.sort()
    clean = len(seen) == 1
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f'# t0={t0} jobs_seen={sorted(seen)} '
                f'{"CLEAN" if clean else "CONTAMINATED"}\n')
        for t, kind, text in events:
            f.write(f'{t:7.2f} {kind:6s} {text}\n')
    return dict(job_id=min(seen) if seen else None,
                clean=clean,
                t0=t0,
                exit_t=exit_t,
                rc=proc.returncode)


def wait_cluster_gone(context: str, cluster: str, timeout: float = 300) -> bool:
    """Waits until the controller has torn down a job's cluster.

    The next run, or an API server restart, would otherwise overlap the
    teardown: a restart in consolidation mode can interrupt it, and a
    teardown in progress slows the next launch.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        names = run(
            ['kubectl', '--context', context, 'get', 'pods', '-o', 'name'],
            check=False,
            timeout=60).stdout.split()
        if not any(n.startswith(f'pod/{cluster}-') for n in names):
            return True
        time.sleep(2)
    log(f'WARNING: pods of {cluster} still exist after {timeout:.0f} s')
    return False


def local_ts(text: str) -> float:
    return datetime.datetime.strptime(text, '%Y-%m-%d %H:%M:%S.%f').timestamp()


def job_phases(job_id: int, cluster: str, t0: float,
               exit_t: float) -> Optional[Dict[str, float]]:
    jobs = sqlite3.connect(JOBS_DB)
    state = sqlite3.connect(STATE_DB)
    row = jobs.execute(
        'select created_at, start_at, end_at from spot where spot_job_id=?',
        (job_id,)).fetchone()
    events: Dict[str, float] = {}
    for status, ts in jobs.execute(
            'select new_status, timestamp from job_events '
            'where spot_job_id=? order by id', (job_id,)):
        events.setdefault(status, local_ts(ts))
    prov = state.execute(_PROVISIONING_STARTED_SQL, (cluster,)).fetchone()[0]
    if (not row or None in row or prov is None or
            not {'STARTING', 'RUNNING', 'SUCCEEDED'} <= set(events)):
        return None
    created, start_at, end_at = row
    return dict(A=created - t0,
                B=events['STARTING'] - created,
                C=prov - events['STARTING'],
                D=start_at - prov,
                E=end_at - start_at,
                F=events['RUNNING'] - end_at,
                G=events['SUCCEEDED'] - events['RUNNING'],
                H=exit_t - events['SUCCEEDED'],
                to_SUCCEEDED=events['SUCCEEDED'] - t0,
                total=exit_t - t0)


_PROVISIONING_STARTED_SQL = (
    'select min(transitioned_at) from cluster_events where name=? and '
    'reason like \'Provisioning on%\'')

_LOG_TS = re.compile(r'^[DIWE] (\d\d-\d\d \d\d:\d\d:\d\d\.\d+) ')


def find_provision_log(cluster: str, since: float) -> Optional[pathlib.Path]:
    needle = f"Provisioning '{cluster}' took"
    for path in sorted(SKY_LOGS.glob('sky-*/provision.log'),
                       key=lambda p: p.stat().st_mtime,
                       reverse=True):
        if path.stat().st_mtime < since - 60:
            break
        if needle in path.read_text(errors='replace'):
            return path
    return None


def provision_marks(path: pathlib.Path, cluster: str) -> Dict[str, float]:
    year = datetime.datetime.now().year
    marks: Dict[str, float] = {}
    for line in path.read_text(errors='replace').splitlines():
        match = _LOG_TS.match(line)
        if not match:
            continue
        t = datetime.datetime.strptime(f'{year}-{match.group(1)}',
                                       '%Y-%m-%d %H:%M:%S.%f').timestamp()
        marks.setdefault('start', t)
        line = re.sub(r'(Start|End):\s+', lambda m: m.group(1) + ': ', line)
        if f"Provisioning '{cluster}' took" in line:
            marks['pods_up'] = t
        elif 'Start: internal_file_mounts' in line:
            marks['fm_start'] = t
        elif 'End: internal_file_mounts' in line:
            marks['fm_end'] = t
        elif 'Start: setup_runtime_on_cluster' in line:
            marks['rt_start'] = t
        elif 'End: setup_runtime_on_cluster' in line:
            marks['rt_end'] = t
        elif 'Start: start_skylet_on_head_node' in line:
            marks.setdefault('skylet_start', t)
        elif 'End: start_skylet_on_head_node' in line:
            marks['skylet_end'] = t
        elif 'Cluster launched' in line:
            marks['launched'] = t
    return marks


def d_breakdown(job_id: int, cluster: str,
                since: float) -> Optional[Dict[str, float]]:
    path = find_provision_log(cluster, since)
    if path is None:
        return None
    marks = provision_marks(path, cluster)
    need = ('pods_up', 'fm_start', 'fm_end', 'rt_start', 'rt_end', 'launched')
    if any(k not in marks for k in need):
        return None
    jobs = sqlite3.connect(JOBS_DB)
    state = sqlite3.connect(STATE_DB)
    start_at = jobs.execute('select start_at from spot where spot_job_id=?',
                            (job_id,)).fetchone()[0]
    prov = state.execute(_PROVISIONING_STARTED_SQL, (cluster,)).fetchone()[0]
    return dict(pre=marks['start'] - prov,
                pods=marks['pods_up'] - marks['start'],
                ssh=marks['fm_start'] - marks['pods_up'],
                file_mounts=marks['fm_end'] - marks['fm_start'],
                runtime_setup=marks['rt_end'] - marks['rt_start'],
                ray_skylet=marks['launched'] - marks['rt_end'],
                submit=start_at - marks['launched'],
                D=start_at - prov,
                skylet_alone=(marks.get('skylet_end', 0) -
                              marks.get('skylet_start', 0)),
                provision_log=str(path))


# ---------------------------------------------------------------- jobs-ab
def cmd_jobs_ab(args: argparse.Namespace) -> None:
    ensure_clean()
    out = new_out_dir('jobs-ab', args.out)
    task = out / 'task.yaml'
    write_task(task, args.nodes, args.context)
    arms = {'base': args.base, 'candidate': args.candidate}
    shas = {arm: short_sha(ref) for arm, ref in arms.items()}
    log(f'jobs-ab: {args.base} ({shas["base"]}) vs {args.candidate} '
        f'({shas["candidate"]}), {args.nodes} nodes, {args.blocks} blocks x '
        f'({args.warmup} warm-up + {args.runs} measured). Out: {out}')
    runs: List[Dict[str, Any]] = []
    with restore_checkout():
        for block in range(args.blocks):
            arm = 'base' if block % 2 == 0 else 'candidate'
            checkout(arms[arm])
            sha = restart_server()
            log(f'Block {block + 1}/{args.blocks}: {arm} at {sha}')
            for i in range(args.warmup + args.runs):
                warmup = i < args.warmup
                tag = f'{arm}_b{block + 1}_{"w" if warmup else i}'
                trace = trace_jobs_launch(out / f'{tag}.txt', task,
                                          args.job_name)
                rec: Dict[str, Any] = dict(arm=arm,
                                           block=block + 1,
                                           tag=tag,
                                           warmup=warmup,
                                           **trace)
                if trace['job_id'] is not None:
                    cluster = f'{args.job_name}-{trace["job_id"]}'
                    rec['phases'] = job_phases(trace['job_id'], cluster,
                                               trace['t0'], trace['exit_t'])
                    rec['d_stages'] = d_breakdown(trace['job_id'], cluster,
                                                  trace['t0'])
                    wait_cluster_gone(args.context, cluster)
                total = (rec.get('phases') or {}).get('total')
                log(f'  {tag}: job {trace["job_id"]} rc={trace["rc"]} '
                    f'{"CLEAN" if trace["clean"] else "CONTAMINATED"} '
                    f'total={total if total is None else round(total, 1)}')
                runs.append(rec)
    report_jobs_ab(out, args, shas, runs)


def report_jobs_ab(out: pathlib.Path, args: argparse.Namespace,
                   shas: Dict[str, str], runs: List[Dict[str, Any]]) -> None:
    usable = [
        r for r in runs
        if not r['warmup'] and r['clean'] and r['rc'] == 0 and r.get('phases')
    ]
    by_arm = {
        arm: [r for r in usable if r['arm'] == arm
             ] for arm in ('base', 'candidate')
    }
    summary: Dict[str, Any] = dict(base=args.base,
                                   candidate=args.candidate,
                                   shas=shas,
                                   nodes=args.nodes,
                                   runs=runs,
                                   phases={},
                                   d_stages={})
    lines = [
        f'# sky jobs launch A/B: {args.base} ({shas["base"]}) vs '
        f'{args.candidate} ({shas["candidate"]})', '',
        f'- {args.nodes} nodes on `{args.context}`, consolidation mode.',
        f'- {args.blocks} alternating blocks of {args.warmup} warm-up + '
        f'{args.runs} measured runs. Measured, clean and successful runs: '
        f'base {len(by_arm["base"])}, candidate {len(by_arm["candidate"])}.',
        '', '## Phases (seconds, median and range)', '',
        '| Phase | base | candidate | Δ median |', '|---|---|---|---|'
    ]
    for key in PHASES:
        vals = {
            arm: [r['phases'][key] for r in rows
                 ] for arm, rows in by_arm.items()
        }
        if not all(vals.values()):
            continue
        desc = {arm: describe(v) for arm, v in vals.items()}
        summary['phases'][key] = desc
        lines.append(
            f'| {key}: {PHASE_NAMES[key]} | '
            f'{desc["base"]["median"]:.1f} ({desc["base"]["min"]:.1f}–'
            f'{desc["base"]["max"]:.1f}) | '
            f'{desc["candidate"]["median"]:.1f} '
            f'({desc["candidate"]["min"]:.1f}–{desc["candidate"]["max"]:.1f})'
            f' | {desc["candidate"]["median"] - desc["base"]["median"]:+.1f} |')
    if all(by_arm.values()):
        p = permutation_p([r['phases']['total'] for r in by_arm['base']],
                          [r['phases']['total'] for r in by_arm['candidate']])
        summary['total_permutation_p'] = p
        lines += [
            '', f'Permutation test on the client total (difference of '
            f'means, two-sided): p = {p:.4f}.'
        ]
    lines += [
        '', '## Phase D stages (seconds, median and range)', '',
        '| Stage | base | candidate |', '|---|---|---|'
    ]
    for key in D_STAGES + ['skylet_alone']:
        vals = {
            arm: [r['d_stages'][key] for r in rows if r.get('d_stages')
                 ] for arm, rows in by_arm.items()
        }
        if not all(vals.values()):
            continue
        desc = {arm: describe(v) for arm, v in vals.items()}
        summary['d_stages'][key] = desc
        name = D_STAGE_NAMES.get(key, 'skylet start on its own')
        lines.append(
            f'| {name} | {desc["base"]["median"]:.2f} '
            f'({desc["base"]["min"]:.1f}–{desc["base"]["max"]:.1f}) | '
            f'{desc["candidate"]["median"]:.2f} '
            f'({desc["candidate"]["min"]:.1f}–{desc["candidate"]["max"]:.1f}) |'
        )
    lines += [
        '', '## Runs', '', '| Run | Job | Clean | rc | Total |',
        '|---|---|---|---|---|'
    ]
    for r in runs:
        total = (r.get('phases') or {}).get('total')
        lines.append(f'| {r["tag"]} | {r["job_id"]} | {r["clean"]} | {r["rc"]} '
                     f'| {"" if total is None else f"{total:.1f}"} |')
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print('\n'.join(lines))
    log(f'Report: {out / "report.md"}')


# ----------------------------------------------------------- storage-probe
_PROBE_MARK = 'STORAGE_PROBE'


def instrument_storage(log_path: pathlib.Path) -> None:
    path = REPO / 'sky' / 'data' / 'storage.py'
    src = path.read_text()
    name = 'get_cached_enabled_storage_cloud_names_or_refresh'
    marker = f'def {name}('
    if marker not in src or _PROBE_MARK in src:
        raise RuntimeError(f'Cannot instrument {path}')
    src = src.replace(marker, f'def _probe_{name}(', 1)
    src += f'''

def {name}(*args, **kwargs):  # {_PROBE_MARK}
    import os as _os, time as _time, traceback as _tb
    _t = _time.time()
    try:
        return _probe_{name}(*args, **kwargs)
    finally:
        _caller = _tb.extract_stack(limit=3)[0]
        with open({str(log_path)!r}, 'a') as _f:
            _f.write(f'{{_t:.3f}} pid={{_os.getpid()}} '
                     f'dt={{_time.time() - _t:.4f}} '
                     f'caller={{_caller.filename.split("sky/")[-1]}}:'
                     f'{{_caller.lineno}}\\n')
'''
    path.write_text(src)


def cmd_storage_probe(args: argparse.Namespace) -> None:
    ensure_clean()
    out = new_out_dir('storage-probe', args.out)
    task = out / 'task.yaml'
    write_task(task, args.nodes, args.context)
    probe_log = out / 'probe.log'
    results: Dict[str, List[Dict[str, Any]]] = {}
    with restore_checkout():
        for arm, ref in (('base', args.base), ('candidate', args.candidate)):
            checkout(ref)
            instrument_storage(probe_log)
            try:
                # restart_server checks the commit, which ignores the
                # uncommitted instrumentation.
                restart_server()
                log(f'storage-probe: {arm} ({ref}) at {short_sha(ref)}')
                for i in range(args.warmup + args.runs):
                    tag = f'{arm}_{"w" if i < args.warmup else i}'
                    with open(probe_log, 'a', encoding='utf-8') as f:
                        f.write(f'--- {tag}\n')
                    trace = trace_jobs_launch(out / f'{tag}.txt', task,
                                              args.job_name)
                    log(f'  {tag}: job {trace["job_id"]} rc={trace["rc"]}')
                    if trace['job_id'] is not None:
                        wait_cluster_gone(args.context,
                                          f'{args.job_name}-{trace["job_id"]}')
            finally:
                run(['git', 'checkout', '--', 'sky/data/storage.py'])
    # Parse calls per run.
    current = None
    for line in probe_log.read_text().splitlines():
        if line.startswith('--- '):
            current = line[4:]
            results[current] = []
            continue
        match = re.search(r'dt=([\d.]+) caller=(\S+)', line)
        if match and current:
            results[current].append(
                dict(seconds=float(match.group(1)), caller=match.group(2)))
    lines = [
        f'# Storage-cloud lookups in `jobs.launch`: {args.base} vs '
        f'{args.candidate}', '', '| Run | Calls (seconds, caller) |',
        '|---|---|'
    ]
    for tag, calls in results.items():
        lines.append(f'| {tag} | ' + '; '.join(
            f'{c["seconds"]:.3f} s ({c["caller"]})' for c in calls) + ' |')
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    (out / 'summary.json').write_text(json.dumps(results, indent=1))
    print('\n'.join(lines))
    log(f'Report: {out / "report.md"}')


# --------------------------------------------------------------- pod-probe
def kubectl(context: str, *argv: str, timeout: float = 300) -> str:
    return run(['kubectl', '--context', context, *argv],
               check=False,
               timeout=timeout).stdout


def cluster_pods(context: str, cluster: str) -> List[str]:
    names = kubectl(context, 'get', 'pods', '-o', 'name').split()
    return sorted(
        n.split('/', 1)[1] for n in names if n.startswith(f'pod/{cluster}-'))


def pod_exec(context: str, pod: str, script: str, timeout: float = 300) -> str:
    return kubectl(context,
                   'exec',
                   pod,
                   '--',
                   'bash',
                   '-c',
                   script,
                   timeout=timeout)


_POD_TIMES = r'''
for f in /tmp/env_setup_complete /tmp/apt_ssh_setup_complete \
         /tmp/ray_skypilot_installation_complete; do
  [ -e "$f" ] && echo "MARK $(basename $f) $(stat -c %Y "$f")"
done
grep -oE 'Resolved [0-9]+ packages? in [0-9.]+m?s' /tmp/runtime-setup.log \
  | sed 's/^/RESOLVE /'
'''

_HEAD_TESTS = r'''
cd ~
export VIRTUAL_ENV=~/skypilot-runtime UV_SYSTEM_PYTHON=false
UV=~/.local/bin/uv; PY=~/skypilot-runtime/bin/python
W=$(ls ~/.sky/wheels/*/skypilot-*.whl 2>/dev/null | head -1)
ms() { echo $(( ($2 - $1) / 1000000 )); }
if [ -n "$W" ]; then
  for mode in offline online; do
    rm -rf /tmp/bench_uvc; mkdir -p /tmp/bench_uvc
    flag=""; [ $mode = offline ] && flag="--offline"
    s=$(date +%s%N)
    out=$(env -u PYTHONPATH UV_CACHE_DIR=/tmp/bench_uvc $UV pip install \
          --dry-run $flag --reinstall-package skypilot "$W[kubernetes, remote]" 2>&1)
    rc=$?; e=$(date +%s%N)
    echo "WHEEL $mode rc=$rc wall_ms=$(ms $s $e) $(echo "$out" | grep -oE 'Resolved [0-9]+ packages? in [0-9.]+m?s')"
  done
  rm -rf /tmp/bench_uvc
fi
for target in sky.skylet.attempt_skylet sky.skylet.skylet; do
  rm -rf /tmp/bench_pyc
  s=$(date +%s%N); PYTHONPYCACHEPREFIX=/tmp/bench_pyc env -u PYTHONPATH $PY -c "import $target" >/dev/null 2>&1; e=$(date +%s%N)
  cold=$(ms $s $e)
  s=$(date +%s%N); PYTHONPYCACHEPREFIX=/tmp/bench_pyc env -u PYTHONPATH $PY -c "import $target" >/dev/null 2>&1; e=$(date +%s%N)
  echo "IMPORT $target cold_ms=$cold warm_ms=$(ms $s $e)"
done
rm -rf /tmp/bench_pyc
'''


def container_started(context: str, pod: str) -> Optional[float]:
    text = kubectl(
        context, 'get', 'pod', pod, '-o',
        'jsonpath={.status.containerStatuses[0].state.running'
        '.startedAt}')
    if not text:
        return None
    return datetime.datetime.strptime(
        text,
        '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=datetime.timezone.utc).timestamp()


def cmd_pod_probe(args: argparse.Namespace) -> None:
    out = new_out_dir('pod-probe', args.out)
    task = out / 'task.yaml'
    write_task(task, args.nodes, args.context)
    cluster = args.cluster
    if not cluster.startswith('bench-'):
        sys.exit('--cluster must start with "bench-" (this script only '
                 'touches clusters it owns).')
    log(f'pod-probe: launching {cluster} ({args.nodes} nodes) on '
        f'{current_ref()} ({short_sha("HEAD")}); the API server must run this '
        'code.')
    t0 = time.time()
    launch = run([sky_cmd(), 'launch', '-y', '-c', cluster,
                  str(task)],
                 check=False,
                 timeout=1200,
                 env=quiet_env())
    (out / 'launch.log').write_text(launch.stdout)
    result: Dict[str, Any] = dict(cluster=cluster, rc=launch.returncode)
    try:
        if launch.returncode != 0:
            raise RuntimeError(f'sky launch failed; see {out / "launch.log"}')
        prov = find_provision_log(cluster, t0)
        marks = provision_marks(prov, cluster) if prov else {}
        stages = {}
        if marks:
            pairs = [('pods', 'start', 'pods_up'),
                     ('file_mounts', 'fm_start', 'fm_end'),
                     ('runtime_setup', 'rt_start', 'rt_end'),
                     ('skylet', 'skylet_start', 'skylet_end')]
            stages = {
                name: marks[b] - marks[a]
                for name, a, b in pairs
                if a in marks and b in marks
            }
        result['provision_stages'] = stages
        # Per-pod: marker times and the uv resolves.
        pods = cluster_pods(args.context, cluster)
        per_pod = {}
        for pod in pods:
            started = container_started(args.context, pod)
            text = pod_exec(args.context, pod, _POD_TIMES)
            info: Dict[str, Any] = dict(resolves=[])
            for line in text.splitlines():
                if line.startswith('MARK ') and started:
                    _, name, epoch = line.split()
                    info[name] = int(epoch) - started
                elif line.startswith('RESOLVE '):
                    info['resolves'].append(line[len('RESOLVE '):])
            per_pod[pod.rsplit('-', 1)[-1]] = info
        result['pods'] = per_pod
        # Per-pod waits reported by setup_runtime_on_cluster.
        waits = {}
        for f in glob.glob(
                str(HOME / '.sky' / 'metadata' / 'v1' / 'clusters' /
                    f'{cluster}-*' / 'instances' / '*' / 'logs' /
                    'setup_runtime_on_cluster.log')):
            node = pathlib.Path(f).parent.parent.name
            waits[node] = re.findall(
                r'(\S+ \S+ installation completed in '
                r'\d+ secs)',
                pathlib.Path(f).read_text(encoding='utf-8', errors='replace'))
        result['setup_runtime_waits'] = waits
        head = next((p for p in pods if p.endswith('-head')), None)
        if head:
            text = pod_exec(args.context, head, _HEAD_TESTS, timeout=900)
            result['head_tests'] = [
                line for line in text.splitlines()
                if line.startswith(('WHEEL ', 'IMPORT '))
            ]
    finally:
        if not args.keep:
            log(f'Tearing down {cluster}')
            run([sky_cmd(), 'down', '-y', cluster], check=False, timeout=600)
    lines = [
        f'# Pod-side probe: {cluster} ({args.nodes} nodes) on '
        f'{current_ref()} ({short_sha("HEAD")})', '',
        '## Provision stages (seconds)', ''
    ]
    for name, secs in result.get('provision_stages', {}).items():
        lines.append(f'- {name}: {secs:.2f}')
    lines += [
        '', '## Per pod', '',
        '| Pod | Marker written (s after container start) | uv resolves |',
        '|---|---|---|'
    ]
    for pod, info in result.get('pods', {}).items():
        marker = info.get('ray_skypilot_installation_complete')
        lines.append(f'| {pod} | '
                     f'{"" if marker is None else f"{marker:.0f}"} | '
                     f'{"; ".join(info["resolves"])} |')
    lines += ['', '## setup_runtime_on_cluster, per worker', '']
    for node, items in sorted(result.get('setup_runtime_waits', {}).items()):
        lines.append(f'- {node}: {"; ".join(items)}')
    lines += ['', '## Head pod tests', '']
    lines += [f'- `{line}`' for line in result.get('head_tests', [])]
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    (out / 'summary.json').write_text(json.dumps(result, indent=1))
    print('\n'.join(lines))
    log(f'Report: {out / "report.md"}')


# ------------------------------------------------------------- pull-policy
_PULL_CONFIG = '''config:
  kubernetes:
    pod_config:
      spec:
        containers:
          - imagePullPolicy: IfNotPresent
'''


def cmd_pull_policy(args: argparse.Namespace) -> None:
    out = new_out_dir('pull-policy', args.out)
    rows = []
    for policy, count in (('Always', args.baseline_runs), ('IfNotPresent',
                                                           args.runs)):
        task = out / f'task_{policy}.yaml'
        write_task(task, args.nodes, args.context,
                   _PULL_CONFIG if policy == 'IfNotPresent' else None)
        for i in range(count):
            cluster = f'bench-pull-{policy.lower()}'
            t0 = time.time()
            launch = run([sky_cmd(), 'launch', '-y', '-c', cluster,
                          str(task)],
                         check=False,
                         timeout=1200,
                         env=quiet_env())
            row: Dict[str, Any] = dict(policy=policy,
                                       run=i + 1,
                                       rc=launch.returncode)
            try:
                pods = cluster_pods(args.context, cluster)
                if pods:
                    row['actual_policy'] = kubectl(
                        args.context, 'get', 'pod', pods[0], '-o',
                        'jsonpath={.spec.containers[0].imagePullPolicy}')
                prov = find_provision_log(cluster, t0)
                if prov:
                    match = re.search(
                        rf"Provisioning '{re.escape(cluster)}' took "
                        r'([\d.]+) seconds', prov.read_text(errors='replace'))
                    if match:
                        row['pods_seconds'] = float(match.group(1))
            finally:
                run([sky_cmd(), 'down', '-y', cluster],
                    check=False,
                    timeout=600)
            log(f'pull-policy: {row}')
            rows.append(row)
    lines = [
        f'# Pod start: imagePullPolicy Always vs IfNotPresent '
        f'({args.nodes} nodes, {current_ref()} {short_sha("HEAD")})', '',
        '| Policy | Run | rc | Policy on the pod '
        '| Pods created → Running (s) |', '|---|---|---|---|---|'
    ]
    for r in rows:
        lines.append(f'| {r["policy"]} | {r["run"]} | {r["rc"]} | '
                     f'{r.get("actual_policy", "")} | '
                     f'{r.get("pods_seconds", "")} |')
    for policy in ('Always', 'IfNotPresent'):
        vals = [
            r['pods_seconds']
            for r in rows
            if r['policy'] == policy and 'pods_seconds' in r
        ]
        if vals:
            d = describe(vals)
            lines.append(f'\n{policy}: median {d["median"]:.2f} s '
                         f'({d["min"]:.2f}–{d["max"]:.2f}), n={d["n"]}')
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    (out / 'summary.json').write_text(json.dumps(rows, indent=1))
    print('\n'.join(lines))
    log(f'Report: {out / "report.md"}')


# ----------------------------------------------------------- sky-launch-ab
BENCH_LAUNCH_LATENCY = (REPO / 'tests' / 'kubernetes' / 'launching_benchmarks' /
                        'bench_launch_latency.py')


def cmd_sky_launch_ab(args: argparse.Namespace) -> None:
    """Runs tests/kubernetes/launching_benchmarks/bench_launch_latency.py.

    For `run`, fills in what that script needs but this one already knows:
    a task YAML (it has no default) built from --nodes and --context, the
    context, and an output directory under ej/bench_results/.
    """
    if not BENCH_LAUNCH_LATENCY.exists():
        sys.exit(f'{BENCH_LAUNCH_LATENCY.relative_to(REPO)} not found: check '
                 'out a branch that has it (ej-prj-1 or later).')
    # argparse.REMAINDER keeps the `--` that separates our arguments from
    # the passthrough ones; bench_launch_latency.py would take it as its
    # subcommand.
    passthrough = list(args.passthrough)
    if passthrough[:1] == ['--']:
        passthrough = passthrough[1:]
    if not passthrough:
        sys.exit('sky-launch-ab needs bench_launch_latency.py arguments, '
                 'e.g.: sky-launch-ab -- run --candidate-ref ej-prj-3-job1')
    if passthrough[0] == 'run':

        def given(flag: str) -> bool:
            return any(
                a == flag or a.startswith(flag + '=') for a in passthrough)

        out = (None
               if given('--out') else new_out_dir('sky-launch-ab', args.out))
        if not given('--task'):
            task = (out or new_out_dir('sky-launch-ab', args.out)) / 'task.yaml'
            write_task(task, args.nodes, args.context)
            passthrough += ['--task', str(task)]
        if not given('--context'):
            passthrough += ['--context', args.context]
        if out is not None:
            passthrough += ['--out', str(out)]
    cmd = [sys.executable, str(BENCH_LAUNCH_LATENCY), *passthrough]
    log(f'Running: {" ".join(cmd)}')
    sys.exit(subprocess.call(cmd, cwd=REPO))


# --------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    def common(p: argparse.ArgumentParser, nodes: int = 6) -> None:
        p.add_argument('--nodes', type=int, default=nodes)
        p.add_argument('--context', default='kind-skypilot')
        p.add_argument('--out',
                       help='output directory '
                       '(default: ej/bench_results/<timestamp>-<command>)')

    p = sub.add_parser('jobs-ab', help='interleaved sky jobs launch A/B')
    common(p)
    p.add_argument('--base', default='origin/master')
    p.add_argument('--candidate', default='ej-prj-3-job1')
    p.add_argument('--blocks',
                   type=int,
                   default=4,
                   help='alternating blocks, base first (default 4)')
    p.add_argument('--runs',
                   type=int,
                   default=4,
                   help='measured runs per block (default 4)')
    p.add_argument('--warmup',
                   type=int,
                   default=1,
                   help='warm-up runs per block, dropped (default 1)')
    p.add_argument('--job-name', default='benchab')
    p.set_defaults(func=cmd_jobs_ab)

    p = sub.add_parser('storage-probe',
                       help='time each storage lookup in jobs.launch')
    common(p)
    p.add_argument('--base', default='origin/master')
    p.add_argument('--candidate', default='ej-prj-3-job1')
    p.add_argument('--runs', type=int, default=2)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--job-name', default='benchprobe')
    p.set_defaults(func=cmd_storage_probe)

    p = sub.add_parser('pod-probe',
                       help='pod-side timings on one throwaway cluster')
    common(p)
    p.add_argument('--cluster', default='bench-pod-probe')
    p.add_argument('--keep',
                   action='store_true',
                   help='keep the cluster up afterwards')
    p.set_defaults(func=cmd_pod_probe)

    p = sub.add_parser('pull-policy',
                       help='pod start with IfNotPresent vs Always')
    common(p)
    p.add_argument('--runs',
                   type=int,
                   default=3,
                   help='IfNotPresent launches (default 3)')
    p.add_argument('--baseline-runs',
                   type=int,
                   default=3,
                   help='Always launches (default 3; 0 to skip)')
    p.set_defaults(func=cmd_pull_policy)

    p = sub.add_parser('sky-launch-ab',
                       help='run tests/kubernetes/launching_benchmarks/'
                       'bench_launch_latency.py with the given args')
    common(p)
    p.add_argument('passthrough', nargs=argparse.REMAINDER)
    p.set_defaults(func=cmd_sky_launch_ab)

    args = parser.parse_args()
    if shutil.which('kubectl') is None and args.command != 'sky-launch-ab':
        sys.exit('kubectl not found on PATH')
    args.func(args)


if __name__ == '__main__':
    main()
