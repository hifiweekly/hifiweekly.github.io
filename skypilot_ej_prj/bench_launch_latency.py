#!/usr/bin/env python3
"""A/B benchmark: K8s launch latency before vs. after the ej-prj-1 change.

Launches the same task repeatedly from two SkyPilot builds and compares
the provisioning timings parsed from each launch's provision.log:

  baseline   the branch's merge-base with origin/master (polling Ray wait,
             skylet started after the wait)
  candidate  ej-prj-1 (blocking Ray wait, skylet started during the wait)

Each build gets its own git worktree, virtualenv and isolated local API
server (own port, SKY_RUNTIME_DIR, HOME and config), so the developer's
working tree, ~/.sky state and main API server are never touched. Runs
alternate A/B/B/A... so drift on the K8s cluster (image cache, node load)
hits both builds equally.

The optimization is reported VALID when all of these hold:
  * every launch of both builds succeeded and ran the task on every rank;
  * no candidate launch fell back to restarting Ray;
  * the candidate's median of the targeted segment (runtime installed ->
    cluster launched) is lower, with a one-sided permutation test p < alpha.

Usage:
  # Full A/B run (needs an idle K8s context; ~3-4 min per launch pair):
  python ej/bench_launch_latency.py run --runs 5

  # Re-analyze provision logs you already have:
  python ej/bench_launch_latency.py analyze \\
      --baseline ej/oct27_10*.log ej/oct30_1022.log \\
      --candidate ej/after_ray_wait_run*.log
"""
import argparse
import csv
import datetime
import json
import os
import pathlib
import random
import re
import signal
import statistics
import subprocess
import sys
import time
from typing import Dict, List, Tuple

REPO = pathlib.Path(__file__).resolve().parent.parent
ANSI = re.compile(r'\x1b\[[0-9;]*m')
LOG_LINE = re.compile(r'^[DIWE] (\d\d-\d\d \d\d:\d\d:\d\d\.\d+) PID=\d+ ')

# Metrics, each the time between two provision.log markers. The start is the
# provisioner's first line, not the log's first timestamp: `sky logs
# --provision` under SKYPILOT_DEBUG prints its own, later, client lines first.
MARKERS = {
    'start': 'provisioner.py',
    'pods_running': 'all pods are scheduled and running',
    'runtime_done': 'End:   setup_runtime_on_cluster',
    'launched': 'Cluster launched',
}
METRICS = {
    # The segment ej-prj-1 targets: Ray wait + skylet start.
    'post_runtime_s': ('runtime_done', 'launched'),
    'provision_total_s': ('start', 'launched'),
    'pods_ready_s': ('start', 'pods_running'),
    'runtime_setup_s': ('pods_running', 'runtime_done'),
}
PRIMARY = 'post_runtime_s'
RAY_RESTART = 'Starting Ray on the entire cluster'
OLD_POLL = 'Ray cluster is not ready yet'
WAIT_RETRY = 'Ray wait on the head ended early'


# ---------------------------------------------------------------- parsing


def parse_provision_log(text: str) -> Dict[str, object]:
    """Returns the metrics and health flags of one provision.log."""
    stamps: Dict[str, datetime.datetime] = {}
    for raw in text.splitlines():
        line = ANSI.sub('', raw)
        m = LOG_LINE.match(line)
        if not m:
            continue
        ts = datetime.datetime.strptime(m.group(1), '%m-%d %H:%M:%S.%f')
        if MARKERS['start'] in line:
            stamps.setdefault('start', ts)
        for name, marker in MARKERS.items():
            if name != 'start' and marker in line:
                # Last occurrence wins: a retried stage ends at its last End.
                stamps[name] = ts
    result: Dict[str, object] = {}
    for metric, (a, b) in METRICS.items():
        result[metric] = ((stamps[b] - stamps[a]).total_seconds()
                          if a in stamps and b in stamps else None)
    clean = ANSI.sub('', text)
    commit = re.search(r'SkyPilot version: [^;]*; commit: ([0-9a-f]{7,40})',
                       clean)
    result['commit'] = commit.group(1)[:12] if commit else ''
    result['launched'] = 'launched' in stamps
    result['ray_restarted'] = RAY_RESTART in clean
    result['old_poll_rounds'] = clean.count(OLD_POLL)
    result['wait_retries'] = clean.count(WAIT_RETRY)
    return result


# ------------------------------------------------------------- statistics


def describe(values: List[float]) -> Dict[str, float]:
    return {
        'n': len(values),
        'mean': statistics.mean(values),
        'median': statistics.median(values),
        'stdev': statistics.stdev(values) if len(values) > 1 else 0.0,
        'min': min(values),
        'max': max(values),
    }


def permutation_p_value(baseline: List[float],
                        candidate: List[float],
                        rounds: int = 20000,
                        seed: int = 0) -> float:
    """One-sided p-value that the candidate's mean is lower by chance.

    Exact when the number of labelings is small, Monte Carlo otherwise.
    No scipy needed.
    """
    observed = statistics.mean(baseline) - statistics.mean(candidate)
    pooled = baseline + candidate
    k = len(baseline)
    rng = random.Random(seed)
    hits = 0
    for _ in range(rounds):
        rng.shuffle(pooled)
        diff = statistics.mean(pooled[:k]) - statistics.mean(pooled[k:])
        if diff >= observed - 1e-12:
            hits += 1
    return (hits + 1) / (rounds + 1)


def compare(rows: List[Dict[str, object]],
            alpha: float) -> Tuple[str, bool]:
    """Returns the markdown report and the verdict from per-launch rows."""
    by_variant: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_variant.setdefault(str(row['variant']), []).append(row)
    base = by_variant.get('baseline', [])
    cand = by_variant.get('candidate', [])
    out = ['# Launch latency A/B: baseline vs. ej-prj-1', '']
    out.append('| Metric | Baseline median (mean ± sd, n) | '
               'Candidate median (mean ± sd, n) | Δ median | p (one-sided) |')
    out.append('|---|---|---|---|---|')
    primary_ok = False
    for metric in METRICS:
        b = [float(r[metric]) for r in base if r.get(metric) is not None]
        c = [float(r[metric]) for r in cand if r.get(metric) is not None]
        if not b or not c:
            out.append(f'| {metric} | n/a | n/a | n/a | n/a |')
            continue
        db, dc = describe(b), describe(c)
        delta = dc['median'] - db['median']
        pct = 100 * delta / db['median'] if db['median'] else 0.0
        p = permutation_p_value(b, c) if len(b) > 1 and len(c) > 1 else 1.0
        if metric == PRIMARY:
            primary_ok = delta < 0 and p < alpha
        out.append(f"| {metric}{' (primary)' if metric == PRIMARY else ''} | "
                   f"{db['median']:.2f}s ({db['mean']:.2f} ± "
                   f"{db['stdev']:.2f}, {db['n']}) | "
                   f"{dc['median']:.2f}s ({dc['mean']:.2f} ± "
                   f"{dc['stdev']:.2f}, {dc['n']}) | "
                   f'{delta:+.2f}s ({pct:+.0f}%) | {p:.4f} |')

    failures = [r for r in rows if not r['ok']]
    cand_restarts = [r for r in cand if r['ray_restarted']]
    cand_old_poll = [r for r in cand if int(r['old_poll_rounds']) > 0]
    commits = {
        v: sorted({str(r.get('commit', '')) for r in rs})
        for v, rs in by_variant.items()
    }
    distinct_builds = (len(commits.get('baseline', [])) == 1 and
                       len(commits.get('candidate', [])) == 1 and
                       commits['baseline'] != commits['candidate'])
    out += ['', 'Commits seen in provision logs: ' + ', '.join(
        f'{v}={"/".join(c) or "?"}' for v, c in sorted(commits.items()))]
    checks = [
        ('Each build ran its own, distinct commit', distinct_builds),
        ('All launches succeeded and ran on every rank', not failures),
        ('No candidate launch fell back to restarting Ray',
         not cand_restarts),
        ('Candidate never ran the old polling loop', not cand_old_poll),
        (f'Primary metric lower with p < {alpha}', primary_ok),
    ]
    out += ['', '## Checks', '']
    out += [f"- [{'x' if ok else ' '}] {name}" for name, ok in checks]
    verdict = all(ok for _, ok in checks)
    out += [
        '', f"**Verdict: {'VALID' if verdict else 'NOT PROVEN'}**", '',
        f'Primary metric `{PRIMARY}` is the segment ej-prj-1 changes: from '
        'the end of `setup_runtime_on_cluster` to "Cluster launched" (Ray '
        'wait + skylet start). p is a one-sided permutation test on the '
        'means.'
    ]
    if failures:
        out += ['', '## Failed launches', '']
        out += [f"- {r['variant']} run {r['run']}: {r.get('error', '')}"
                for r in failures]
    return '\n'.join(out) + '\n', verdict


# ------------------------------------------------------------ environment


def sh(cmd: List[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, text=True, **kwargs)


def run_in(variant: 'Variant', cmd: List[str],
           **kwargs) -> subprocess.CompletedProcess:
    """Runs a command of `variant` from its own neutral directory.

    Never from the repo root: `python -m sky.server.server` puts the cwd
    first on sys.path, so a server started there imports the working tree's
    `sky` instead of the build it was installed from.
    """
    kwargs.setdefault('check', False)
    return subprocess.run(cmd,
                          env=variant.env(),
                          cwd=str(variant.runtime),
                          text=True,
                          **kwargs)


class Variant:
    """One SkyPilot build with its own worktree, venv and API server."""

    def __init__(self, name: str, ref: str, root: pathlib.Path, port: int,
                 context: str):
        self.name, self.ref, self.port, self.context = name, ref, port, context
        self.dir = root / name
        self.worktree = self.dir / 'src'
        self.venv = self.dir / 'venv'
        self.runtime = self.dir / 'runtime'
        self.commit = ''

    @property
    def sky(self) -> str:
        return str(self.venv / 'bin' / 'sky')

    def env(self) -> Dict[str, str]:
        env = dict(os.environ)
        for key in list(env):
            # Never inherit another server's endpoint or a test's config.
            if key.startswith(('SKYPILOT_', 'PYTEST')) or key in (
                    'SKY_RUNTIME_DIR', 'VIRTUAL_ENV'):
                env.pop(key)
        env.update({
            'SKY_RUNTIME_DIR': str(self.runtime),
            'HOME': str(self.runtime),
            'SKYPILOT_API_SERVER_LOCAL_PORT': str(self.port),
            'SKYPILOT_GLOBAL_CONFIG': str(self.runtime / 'config.yaml'),
            'KUBECONFIG': os.environ.get(
                'KUBECONFIG', os.path.expanduser('~/.kube/config')),
            'SKYPILOT_DISABLE_USAGE_COLLECTION': '1',
            'SKYPILOT_DEBUG': '1',
            'PATH': f"{self.venv / 'bin'}:{os.environ['PATH']}",
        })
        return env

    def setup(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        if not self.worktree.exists():
            sh(['git', '-C', str(REPO), 'worktree', 'add', '--detach',
                str(self.worktree), self.ref])
        self.commit = sh(
            ['git', '-C', str(self.worktree), 'rev-parse', 'HEAD'],
            capture_output=True).stdout.strip()
        print(f'[{self.name}] {self.ref} = {self.commit[:12]}', flush=True)
        if not self.venv.exists():
            sh(['uv', 'venv', '--seed', '--python', '3.11', str(self.venv)])
        sh(['uv', 'pip', 'install', '--python',
            str(self.venv / 'bin' / 'python'), '-q', '-e',
            f'{self.worktree}[kubernetes,server]'])
        self.runtime.mkdir(parents=True, exist_ok=True)
        (self.runtime / 'config.yaml').write_text(
            'allowed_clouds:\n  - kubernetes\n'
            f'kubernetes:\n  allowed_contexts:\n    - {self.context}\n')
        # Fail before any launch if this build would import another `sky`.
        found = run_in(self, [
            str(self.venv / 'bin' / 'python'), '-c',
            'import sky; print(sky.__file__)'
        ], capture_output=True).stdout.strip()
        # Last line: under SKYPILOT_DEBUG, importing sky logs config first.
        found = found.splitlines()[-1] if found else ''
        if not found.startswith(str(self.worktree)):
            raise RuntimeError(f'[{self.name}] imports sky from {found}, '
                               f'not from {self.worktree}')

    def start_server(self) -> None:
        run_in(self, [self.sky, 'api', 'start'],
               capture_output=True,
               check=True)
        print(f'[{self.name}] API server on 127.0.0.1:{self.port}',
              flush=True)

    def stop_server(self) -> None:
        run_in(self, [self.sky, 'api', 'stop'], capture_output=True)

    def launch_once(self, cluster: str, task: str, num_nodes: int,
                    out_dir: pathlib.Path) -> Dict[str, object]:
        row: Dict[str, object] = {'variant': self.name, 'cluster': cluster}
        start = time.monotonic()
        launch = run_in(self, [
            self.sky, 'launch', '-y', '-c', cluster, '--infra',
            f'k8s/{self.context}', task
        ],
                        capture_output=True)
        row['launch_wall_s'] = round(time.monotonic() - start, 3)
        (out_dir / f'{cluster}.launch.log').write_text(launch.stdout +
                                                       launch.stderr)
        prov = run_in(self, [self.sky, 'logs', '--provision', cluster],
                      capture_output=True)
        (out_dir / f'{cluster}.provision.log').write_text(prov.stdout)
        row.update(parse_provision_log(prov.stdout))
        ranks = set(
            re.findall(r'\[rank (\d+)\] started',
                       ANSI.sub('', launch.stdout + launch.stderr)))
        row['ranks_seen'] = len(ranks)
        right_build = str(row['commit']) == self.commit[:12]
        row['ok'] = (launch.returncode == 0 and bool(row['launched']) and
                     len(ranks) == num_nodes and right_build)
        if not row['ok']:
            row['error'] = (f'rc={launch.returncode}, ranks '
                            f'{len(ranks)}/{num_nodes}, launched='
                            f"{row['launched']}, commit={row['commit']} "
                            f'(expected {self.commit[:12]})')
        run_in(self, [self.sky, 'down', '-y', cluster], capture_output=True)
        return row


def busy_pods(context: str) -> List[str]:
    """SkyPilot pods already on the context: they would skew the timings."""
    proc = subprocess.run(
        ['kubectl', '--context', context, 'get', 'pods', '-A', '-l',
         'parent=skypilot', '-o', 'name'],
        capture_output=True, text=True, check=False)
    return [p for p in proc.stdout.split() if p]


def run(args: argparse.Namespace) -> int:
    import yaml  # pylint: disable=import-outside-toplevel
    task_path = str(pathlib.Path(args.task).resolve())
    num_nodes = int(yaml.safe_load(open(task_path)).get('num_nodes', 1))
    busy = busy_pods(args.context)
    if busy and not args.force:
        print(f'{len(busy)} SkyPilot pod(s) already on {args.context!r} '
              f'(e.g. {busy[0]}). They would skew the timings; wait for '
              'them to finish or pass --force.', file=sys.stderr)
        return 2
    baseline_ref = args.baseline_ref or sh(
        ['git', '-C', str(REPO), 'merge-base', args.candidate_ref,
         'origin/master'], capture_output=True).stdout.strip()
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    out_dir = pathlib.Path(args.out or REPO / 'ej' / f'bench-{stamp}')
    logs_dir = out_dir / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    work = pathlib.Path(args.workdir).resolve()
    variants = {
        'baseline': Variant('baseline', baseline_ref, work, args.port,
                            args.context),
        'candidate': Variant('candidate', args.candidate_ref, work,
                             args.port + 1, args.context),
    }
    rows: List[Dict[str, object]] = []
    live: List[tuple] = []

    def cleanup(*_):
        for variant, cluster in live:
            run_in(variant, [variant.sky, 'down', '-y', cluster],
                   capture_output=True)
        for v in variants.values():
            v.stop_server()
        if not args.keep_worktrees:
            for v in variants.values():
                subprocess.run(['git', '-C', str(REPO), 'worktree', 'remove',
                                '--force', str(v.worktree)],
                               capture_output=True, check=False)

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(130))
    try:
        for v in variants.values():
            v.setup()
            v.start_server()
        for i in range(args.warmup + args.runs):
            # A B B A ...: alternate who goes first to cancel drift.
            order = (['baseline', 'candidate'] if i % 2 == 0 else
                     ['candidate', 'baseline'])
            for name in order:
                v = variants[name]
                cluster = f'bench-{name[:4]}-{i}'
                live.append((v, cluster))
                tag = 'warmup' if i < args.warmup else f'run {i - args.warmup}'
                print(f'[{name}] {tag}: launching {cluster} ...', flush=True)
                row = v.launch_once(cluster, task_path, num_nodes, logs_dir)
                live.remove((v, cluster))
                row['run'] = i - args.warmup
                print(f"[{name}] {tag}: ok={row['ok']} "
                      f"{PRIMARY}={row.get(PRIMARY)} "
                      f"total={row.get('provision_total_s')}", flush=True)
                if i >= args.warmup:
                    rows.append(row)
    except KeyboardInterrupt:
        print('Interrupted; cleaning up.', file=sys.stderr)
    finally:
        cleanup()

    fields = ['variant', 'run', 'cluster', 'ok', 'launch_wall_s',
              *METRICS, 'ranks_seen', 'ray_restarted', 'old_poll_rounds',
              'wait_retries', 'launched', 'commit', 'error']
    with open(out_dir / 'results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / 'meta.json').write_text(json.dumps({
        'baseline_ref': baseline_ref, 'candidate_ref': args.candidate_ref,
        'context': args.context, 'task': task_path, 'runs': args.runs,
        'warmup': args.warmup, 'num_nodes': num_nodes}, indent=2))
    report, verdict = (compare(rows, args.alpha) if rows else
                       ('No completed runs.\n', False))
    (out_dir / 'summary.md').write_text(report)
    print(report)
    print(f'Results: {out_dir}')
    return 0 if verdict else 1


def analyze(args: argparse.Namespace) -> int:
    rows = []
    for variant, paths in (('baseline', args.baseline),
                           ('candidate', args.candidate)):
        for i, path in enumerate(paths):
            row = parse_provision_log(pathlib.Path(path).read_text(
                errors='replace'))
            row.update(variant=variant, run=i, cluster=path,
                       ok=bool(row['launched']))
            if not row['ok']:
                row['error'] = 'no "Cluster launched" in log'
            rows.append(row)
            print(f"{variant:9} {path}: {PRIMARY}={row[PRIMARY]} "
                  f"total={row['provision_total_s']} "
                  f"restart={row['ray_restarted']}")
    report, verdict = compare(rows, args.alpha)
    print()
    print(report)
    return 0 if verdict else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run', help='Launch both builds and compare.')
    r.add_argument('--candidate-ref', default='ej-prj-1')
    r.add_argument('--baseline-ref', default=None,
                   help='Default: merge-base of candidate and origin/master.')
    r.add_argument('--context', default='kind-skypilot')
    r.add_argument('--task', default=str(REPO / 'ej' / 'job.yaml'))
    r.add_argument('--runs', type=int, default=5)
    r.add_argument('--warmup', type=int, default=1,
                   help='Untimed launches per build first (image pulls).')
    r.add_argument('--port', type=int, default=46601,
                   help='Baseline server port; candidate uses port + 1.')
    r.add_argument('--workdir', default='/tmp/sky-bench-ab')
    r.add_argument('--out', default=None)
    r.add_argument('--alpha', type=float, default=0.05)
    r.add_argument('--keep-worktrees', action='store_true')
    r.add_argument('--force', action='store_true',
                   help='Run even if SkyPilot pods are already present.')
    a = sub.add_parser('analyze', help='Compare existing provision logs.')
    a.add_argument('--baseline', nargs='+', required=True)
    a.add_argument('--candidate', nargs='+', required=True)
    a.add_argument('--alpha', type=float, default=0.05)
    args = parser.parse_args()
    return run(args) if args.cmd == 'run' else analyze(args)


if __name__ == '__main__':
    sys.exit(main())
