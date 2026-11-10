#!/usr/bin/env python3
"""Evaluate frozen JSON criteria against explicit lab attempts; stdout is verdict/1."""
import argparse
import datetime as dt
import hashlib
import json
import math
import operator
import os
from pathlib import Path
import re
import statistics
import sys


class EvidenceError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    def reject(value):
        raise EvidenceError(f'non-finite JSON constant: {value}')
    return json.loads(Path(path).read_text(), parse_constant=reject)


def number(value):
    require(type(value) in (int, float) and math.isfinite(value), f'invalid numeric value: {value!r}')
    return value


def timestamp(value):
    parsed = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(parsed.tzinfo is not None, 'timestamp must include timezone')
    return parsed.timestamp()


class Records:
    def __init__(self, lab, root):
        self.lab, self.root, self.cache = Path(lab), Path(root), {}

    def job(self, job, attempt):
        require(re.fullmatch(r'job-[a-zA-Z0-9_-]+', job) is not None, 'invalid job ID')
        require(re.fullmatch(r'att-[1-9][0-9]*', attempt) is not None, 'explicit attempt required')
        key = (job, attempt)
        if key in self.cache:
            return self.cache[key]
        directory = self.lab / 'jobs' / job / attempt
        jp = directory.parent / 'job.json'
        rp = directory / 'result.json'
        spec, result = read_json(jp), read_json(rp)
        require(spec['job'] == result['job'] == job and result['attempt'] == attempt, 'job/attempt identity mismatch')
        require(spec['spec']['image_id'] == result['image_id'], 'image identity mismatch')
        tests = spec['spec']['tests']
        require(bool(tests), 'empty job')
        expected = {f'{test}#{i}' for i, test in enumerate(tests)}
        rows = result['quality']['tests']
        require(set(rows) == expected, 'result tests differ from submitted order')
        cov = result['evidence']['coverage']
        require(all(type(cov.get(k)) is int and cov[k] == len(tests) for k in
                    ('tests_expected', 'tests_started', 'tests_completed_cleanly')), 'incomplete job coverage')
        require(cov.get('interrupted') is False and result['execution']['status'] == 'completed', 'interrupted/incomplete job')
        require(result['quality']['all_ok'] is True, 'job assessment failed')
        for name, row in rows.items():
            require(row.get('ok') is True and type(row.get('exit_code')) is int and row['exit_code'] == 0
                    and row.get('partial') is False, f'unclean test: {name}')
            for suffix in ('.stdout', '.stderr'):
                filename = name + suffix
                require(Path(filename).name == filename, 'invalid evidence filename')
                want = result['evidence']['files'].get(filename)
                require(bool(want), f'missing evidence hash: {filename}')
                require(digest((directory / filename).read_bytes()) == want, f'evidence hash mismatch: {filename}')
        info = {'job': job, 'attempt': attempt, 'result': str(rp), 'result_sha256': digest(rp.read_bytes()),
                'job_sha256': digest(jp.read_bytes())}
        self.cache[key] = (directory, result, tests, info)
        return self.cache[key]

    def select(self, ref):
        directory, result, tests, info = self.job(ref['job'], ref['attempt'])
        test = ref['test']
        selected = [(i, f'{test}#{i}', result['quality']['tests'][f'{test}#{i}'])
                    for i, name in enumerate(tests) if name == test]
        require(type(ref['count']) is int and ref['count'] > 0 and len(selected) == ref['count'],
                f'{test}: expected {ref["count"]} samples, found {len(selected)}')
        return directory, selected, dict(info, test=test)

    def metric(self, ref, metric):
        directory, rows, evidence = self.select(ref)
        values = []
        for _, key, row in rows:
            # Port of frozen-experiment/agg.py bench(): last row per tg/pp kind,
            # then median of each invocation's avg_ts (never pool samples_ts).
            if metric in ('tg_ts', 'pp_ts'):
                bench = {}
                for sample in row.get('samples') or []:
                    if 'avg_ts' in sample:
                        bench['tg_ts' if sample.get('n_gen', 0) > 0 else 'pp_ts'] = sample['avg_ts']
                value = bench.get(metric)
            elif metric in ('gpu_delta_mib', 'host_cgroup_peak_mib'):
                value = row.get('mem', {}).get(metric)
            else:
                value = row.get(metric)
            values.append(number(value))
        aggregate = max(values) if metric in ('gpu_delta_mib', 'host_cgroup_peak_mib') else statistics.median(values)
        return aggregate, dict(evidence, samples=[key for _, key, _ in rows], values=values,
                               aggregate=aggregate, minimum=min(values), maximum=max(values))

    def text(self, refs, files=()):
        evidence, hashes = [], []
        for ref in refs:
            directory, rows, info = self.select(ref)
            samples = []
            for _, key, _ in rows:
                value = digest((directory / (key + '.stdout')).read_bytes())
                hashes.append(value)
                samples.append({'sample': key, 'sha256': value})
            evidence.append(dict(info, samples=samples))
        for source in files:
            _, info = self.sidecar(source)
            hashes.append(info['sha256'])
            evidence.append(info)
        require(len(hashes) >= 2, 'text equality requires at least two samples')
        return len(set(hashes)) == 1, evidence

    def sidecar(self, criterion):
        # External legacy evidence is pinned, not silently treated as lab output.
        path = Path(criterion['path'])
        path = path if path.is_absolute() else self.root / path
        data = path.read_bytes()
        require(digest(data) == criterion['sha256'], f'sidecar hash mismatch: {path}')
        return data.decode('utf-8'), {'path': str(path), 'sha256': digest(data), 'source': 'legacy sidecar'}

    def host(self, criterion):
        _, _, _, info = self.job(criterion['job'], criterion['attempt'])
        directory = self.lab / 'jobs' / criterion['job'] / criterion['attempt']
        state_path = directory / 'state.jsonl'
        states = [json.loads(line) for line in state_path.read_text().splitlines()]
        require(all(s.get('job') == criterion['job'] and s.get('attempt') == criterion['attempt'] for s in states),
                'state identity mismatch')
        start = min(timestamp(s['ts']) for s in states if s['state'] == 'running')
        end = min(timestamp(s['ts']) for s in states if s['state'] == 'assessing')
        require(end > start, 'invalid run interval')
        log, ev = self.sidecar(criterion)
        points = []
        for line in log.splitlines():
            fields = line.split()
            when = timestamp(fields[0])
            points.append((when, fields))
        require(points and all(a[0] < b[0] for a, b in zip(points, points[1:])), 'loadmon timestamps not strictly increasing')
        # Include samples bracketing the entire execution, with bounded gaps.
        before = [p for p in points if p[0] <= start]
        after = [p for p in points if p[0] >= end]
        require(before and after, 'loadmon does not cover full run interval')
        selected = [p for p in points if before[-1][0] <= p[0] <= after[0][0]]
        gap = number(criterion['max_gap_s'])
        require(gap > 0 and all(b[0] - a[0] <= gap for a, b in zip(selected, selected[1:])), 'loadmon sampling gap')
        limits = criterion['counters']
        require(bool(limits), 'host_quiet needs explicit monitored counters')
        maxima = {key: 0 for key in limits}
        loads = []
        for _, fields in selected:
            loads.append(number(float(fields[1])))
            counters = dict(item.split('=', 1) for item in fields[4:])
            for key in limits:
                require(key in counters, f'loadmon missing counter: {key}')
                value = int(counters[key])
                require(value >= 0, 'negative host counter')
                maxima[key] = max(maxima[key], value)
        quiet = all(maxima[k] <= number(v) for k, v in limits.items())
        if 'max_load_1m' in criterion:
            quiet = quiet and max(loads) <= number(criterion['max_load_1m'])
        return quiet, [dict(info, **ev, state_sha256=digest(state_path.read_bytes()),
                            start=start, end=end, sample_count=len(selected), counter_maxima=maxima,
                            load_1m_range=[min(loads), max(loads)])]


OPS = {'>=': operator.ge, '<=': operator.le, '==': operator.eq, '>': operator.gt, '<': operator.lt}
METRICS = {'decode_tps', 'prefill_tps', 'ppl', 'kld', 'top1_pct', 'tg_ts', 'pp_ts',
           'gpu_delta_mib', 'host_cgroup_peak_mib'}


def check(c, records):
    metric = c['metric']
    ev = []
    if metric == 'text_sha':
        value, ev = records.text(c['groups'], c.get('files', ()))
    elif metric == 'host_quiet':
        value, ev = records.host(c)
    elif metric == 'backend_ops':
        records.job(c['job'], c['attempt'])
        log, info = records.sidecar(c)
        plain = re.sub(r'\x1b\[[0-9;]*m', '', log)
        backend = re.escape(c['backend'])
        block = re.search(r'Backend \d+/\d+: ' + backend + r'\n(.*?)(?=\nBackend \d+/\d+:|\Z)', plain, re.S)
        require(block is not None, 'requested backend absent')
        counts = re.findall(r'(\d+)/(\d+) tests passed', block[1])
        require(len(counts) == 1, 'missing or ambiguous backend pass count')
        passed, total = map(int, counts[0])
        operation_lines = re.findall(r'^\s+([A-Z_0-9]+)\(.*\):\s*(\S+)\s*$', block[1], re.M)
        require(total > 0 and len(operation_lines) == total, 'backend operation coverage mismatch')
        value = passed == total and all(op == c['operation'] and status == 'OK' for op, status in operation_lines)
        value = value and bool(re.search(r'Backend ' + backend + r': OK', block[1]))
        ev = [dict(info, job=c['job'], attempt=c['attempt'], backend=c['backend'], operation=c['operation'],
                   passed=passed, total=total, association='legacy sidecar associated by criteria; not a lab test')]
    else:
        require(metric in METRICS, f'unknown metric: {metric}')
        value, sample = records.metric(c['candidate'], metric)
        ev.append(sample)
        mode = c.get('mode', 'value')
        if mode != 'value':
            baseline, sample = records.metric(c['baseline'], metric)
            ev.append(sample)
            if mode in ('delta_pct', 'abs_delta_pct'):
                require(baseline > 0, 'percentage baseline must be positive')
                value = 100 * (value / baseline - 1)
                if mode == 'abs_delta_pct':
                    value = abs(value)
            elif mode == 'abs_delta':
                value = abs(value - baseline)
            else:
                raise EvidenceError(f'unknown mode: {mode}')
        if 'alternating' in c:
            schedule = c['alternating']
            require(c['candidate']['job'] == c['baseline']['job'] and
                    c['candidate']['attempt'] == c['baseline']['attempt'], 'alternating comparison must use same attempt')
            _, _, tests, _ = records.job(c['candidate']['job'], c['candidate']['attempt'])
            group = schedule['group']
            require(len(group) == len(set(group)) and len(group) >= 2, 'invalid alternating group')
            require(c['candidate']['test'] in group and c['baseline']['test'] in group, 'comparison absent from alternating group')
            rounds, repeat = schedule['rounds'], schedule.get('repeat', 1)
            require(type(rounds) is int and rounds >= 2 and type(repeat) is int and repeat >= 1, 'invalid rounds/repeat')
            # Same forward/reverse semantics as make_order.py, filtered to this group.
            expected = [name for _ in range(repeat) for i in range(rounds) for name in (group if i % 2 == 0 else group[::-1])]
            observed = [name for name in tests if name in group]
            require(observed == expected, 'alternating order mismatch')
            ev.append({'job': c['candidate']['job'], 'attempt': c['candidate']['attempt'], 'alternating_order': observed})
    op = c['op']
    require(op in OPS, f'unknown comparator: {op}')
    threshold = c['threshold']
    if type(value) is bool:
        require(op == '==' and type(threshold) is bool, 'boolean criterion requires == boolean')
    else:
        number(value)
        number(threshold)
    return {'id': c['id'], 'status': 'PASS' if OPS[op](value, threshold) else 'FAIL',
            'metric': metric, 'op': op, 'threshold': threshold, 'value': value, 'evidence': ev}


def evaluate(protocol, lab, root):
    records = Records(lab, root)
    try:
        require(protocol['schema'] == 'criteria/1', 'unsupported criteria schema')
        criteria = protocol['criteria']
        require(isinstance(criteria, list) and bool(criteria), 'empty criteria')
        ids = [c['id'] for c in criteria]
        require(all(isinstance(i, str) and i for i in ids) and len(set(ids)) == len(ids), 'invalid/duplicate criterion IDs')
    except (KeyError, TypeError, ValueError) as exc:
        return {'schema': 'verdict/1', 'status': 'FAIL', 'criteria': [], 'error': str(exc), 'job_ids': []}
    results = []
    for c in criteria:
        try:
            results.append(check(c, records))
        except (OSError, ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
            results.append({'id': c['id'], 'metric': c.get('metric'), 'status': 'FAIL',
                            'error': f'{type(exc).__name__}: {exc}', 'evidence': []})
    # Include requested IDs even when records are missing.
    def job_ids(value):
        if isinstance(value, dict):
            return ({value['job']} if isinstance(value.get('job'), str) else set()).union(*(job_ids(v) for v in value.values()))
        if isinstance(value, list):
            return set().union(*(job_ids(v) for v in value))
        return set()
    return {'schema': 'verdict/1', 'protocol': protocol.get('id'), 'scope': protocol.get('scope'),
            'status': 'PASS' if all(r['status'] == 'PASS' for r in results) else 'FAIL',
            'criteria': results, 'job_ids': sorted(job_ids(criteria))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('criteria', type=Path)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2], help='root for legacy sidecar paths')
    parser.add_argument('--lab', type=Path, default=None, help='defaults to LAB or ROOT/lab')
    args = parser.parse_args()
    try:
        raw = args.criteria.read_bytes()
        protocol = read_json(args.criteria)
        result = evaluate(protocol, args.lab or os.environ.get('LAB') or args.root / 'lab', args.root)
        result.update(criteria_file=str(args.criteria), criteria_sha256=digest(raw))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        result = {'schema': 'verdict/1', 'status': 'FAIL', 'criteria': [], 'job_ids': [], 'error': str(exc)}
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
