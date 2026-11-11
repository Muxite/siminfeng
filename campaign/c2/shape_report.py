#!/usr/bin/env python3
"""Read-only campaign accounting. All intervals and calendar days are UTC."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import statistics

TASK_CLASSES = {
    r'^J-OM': 'over-memory',
    r'^(J-T|H)': 'tooling',
    r'^(J-E|J-P)': 'granite-experiments',
    r'^(J-BQ|build)': 'build',
    r'^(J-KS|KS)': 'kernel-screen',
    r'^J-SPEC': 'speculation',
    r'^J-FIO$': 'storage',
    r'^J-ISO': 'isolation',
    r'^J-Q$': 'quality',
    r'^J-UP$': 'build',
    r'^J1(?:-|$)': 'tooling',
    r'^J2-(?:build|image-qual|repro)$': 'build',
    r'^J2-model$': 'model-preparation',
    r'^J5(?:-|$)': 'granite-experiments',
    r'^codex-': 'tooling',
    r'^(?:sleep|true)$': 'test-noise',
}
GPU_PATTERN = re.compile(r'om_run\.sh|om_spec\.py|llama-bench|kernel_screen|abrun\.py|llama-perplexity')
FAILURE_PATTERNS = {
    'flag_error': r'invalid argument|unknown argument|error: invalid|unrecognized (?:argument|option)|unknown option|(?:^|\n)\s*usage:',
    'oom/watchdog': r'out of memory|\boom\b|oom.kill|watchdog|cuda.*alloc.*fail|std::bad_alloc',
    'docker_start': r'cannot connect.*docker|docker.*permission denied|permission denied.*docker|error response from daemon|failed to (?:create|start).*container|docker:|nvidia-container-cli',
    'build_fail': r'build_engine\.py|ninja: build stopped|cmake error|compilation terminated|fatal error:|undefined reference|build.*fail',
    'timeout': r'timeout|timed out|deadline exceeded',
}


def timestamp(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value / 1e9 if value > 1e14 else float(value)
    parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def iso(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def union(intervals):
    result = []
    for start, end in sorted((a, b) for a, b in intervals if b > a):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def intersection(left, right):
    left, right = union(left), union(right)
    result, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        a, b = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if b > a:
            result.append((a, b))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return result


def duration(intervals):
    return sum(b - a for a, b in union(intervals))


def classify_failure(text):
    for category, pattern in FAILURE_PATTERNS.items():
        if re.search(pattern, text, re.I):
            return category
    return 'other'


def task_class(task):
    return next((value for pattern, value in TASK_CLASSES.items() if re.search(pattern, task)), 'other')


def jsonlines(path, warnings):
    if not path.exists():
        return
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            warnings.append(f'{path}:{n}: malformed JSON skipped')


def aggregate_worklog(rows, since=None):
    tasks, lead_events = {}, []
    for row in rows:
        if not row.get('worker') and row.get('event') in ('contamination', 'result', 'failure'):
            if since is None or (timestamp(row.get('ts')) or 0) >= since:
                lead_events.append({**row, 'worker': 'opus'})
            continue
        task = row.get('task', 'unknown')
        item = tasks.setdefault(task, dict(task=task, worker='unknown', wall_s=0, tokens=0,
                                          verdict=None, issues_count=0, rework_rounds=0,
                                          opus_fix=False, end_events=0, checkins=0, events=[]))
        item['events'].append(row)
        if row.get('worker'):
            item['worker'] = row['worker']
        if row.get('event') == 'end':
            item['end_events'] += 1
            for field in ('wall_s', 'tokens'):
                item[field] += float(row.get(field) or 0)
        if row.get('event') == 'checkin':
            item['checkins'] += 1
            item['verdict'] = row.get('verdict')
            issues = row.get('issues', [])
            item['issues_count'] += len(issues) if isinstance(issues, (list, dict)) else int(issues or 0)
            # Rework rounds are cumulative snapshots, not additive check-in counts.
            item['rework_rounds'] = max(item['rework_rounds'], int(row.get('rework_rounds') or 0))
            item['opus_fix'] |= bool(row.get('opus_fix'))
    selected = []
    workers = {}
    for item in tasks.values():
        if since is not None and not any((timestamp(r.get('ts')) or 0) >= since for r in item['events']):
            continue
        selected.append(item)
        worker = workers.setdefault(item['worker'], dict(tasks=0, wall_s=0, tokens=0, issues_count=0,
                                                        rework_rounds=0, opus_fixes=0))
        worker['tasks'] += 1
        for field in ('wall_s', 'tokens', 'issues_count', 'rework_rounds'):
            worker[field] += item[field]
        worker['opus_fixes'] += int(item['opus_fix'])
    return dict(tasks=sorted(selected, key=lambda r: r['task']), workers=workers, lead_events=lead_events)


def read_json(path, warnings):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append(f'{path}: {exc}')
        return None


def generate(root, since=None, now=None):
    now = now or datetime.now(timezone.utc).timestamp()
    warnings, joined, tools = [], {}, defaultdict(Counter)
    for path in sorted((root / 'evidence/jetson-arm').glob('*.jsonl')):
        for row in jsonlines(path, warnings):
            if path.name == 'claude-code.jsonl':
                ts = timestamp(row.get('ts'))
                if row.get('event') == 'PreToolUse' and ts is not None and (since is None or ts >= since):
                    tools[iso(ts)[:10]][row.get('payload', {}).get('tool_name', 'unknown')] += 1
            elif row.get('op') and row.get('kind') in ('intent', 'started', 'outcome'):
                joined.setdefault(row['op'], {})[row['kind']] = row
    operations, gpu, build = [], [], []
    for op, events in sorted(joined.items()):
        intent, started, outcome = (events.get(k, {}) for k in ('intent', 'started', 'outcome'))
        merged = {**intent, **started, **outcome}
        start = timestamp(started.get('ts') or intent.get('ts'))
        end = timestamp(outcome.get('ts'))
        elapsed = outcome.get('elapsed_s')
        if start is None and end is not None and elapsed is not None:
            start = end - float(elapsed)
        if since is not None and (end or start or 0) < since:
            continue
        argv = intent.get('argv', [])
        command = ' '.join(map(str, argv)) if isinstance(argv, list) else str(argv)
        tokens = argv if isinstance(argv, list) else shlex.split(str(argv))
        run_dirs = [Path(tokens[i + 1]) for i, token in enumerate(tokens[:-1])
                    if Path(str(token)).name == 'om_run.sh']
        runner_results = []
        for directory in run_dirs:
            directory = directory if directory.is_absolute() else root / directory
            try:
                record = json.loads((directory / 'result.json').read_text())
                if isinstance(record, dict):
                    runner_results.append(record)
            except (OSError, ValueError):
                pass
        runner_failed = any(r.get('status', 'ok') != 'ok' or
                            r.get('exit_code', 0) not in (None, 0) for r in runner_results)
        failed = bool(outcome) and (outcome.get('status', 'ok') != 'ok' or outcome.get('exit_code', 0) not in (None, 0) or runner_failed)
        category = None
        if failed:
            stderr = outcome.get('stderr', {})
            stderr_path = stderr.get('path', '') if isinstance(stderr, dict) else str(stderr or '')
            text = command + '\n' + stderr_path + '\n' + str(outcome.get('status', ''))
            stderr_paths = [stderr_path] if stderr_path else []
            # om_run redirects the engine's stderr into OUTDIR, leaving run_op's
            # stderr with wrapper diagnostics only. Support interpreter prefixes.
            stderr_paths.extend(str(directory / 'stderr.txt') for directory in run_dirs)
            text += '\n' + json.dumps(runner_results)
            for stderr_file in dict.fromkeys(stderr_paths):
                path = Path(stderr_file)
                path = path if path.is_absolute() else root / path
                try:
                    with path.open('rb') as stream:
                        # Bounded tail captures terminal errors without loading large files.
                        stream.seek(0, 2)
                        stream.seek(max(0, stream.tell() - 4096))
                        text += '\n' + stream.read(4096).decode(errors='replace')
                except OSError:
                    pass
            category = classify_failure(text)
        item = dict(op=op, agent=merged.get('agent', 'unknown'), task=merged.get('task', 'unknown'),
                    argv=argv, start=start, end=end, elapsed_s=float(elapsed) if elapsed is not None else (end-start if start is not None and end is not None else None),
                    completed=bool(outcome), failed=failed, failure=category,
                    status=outcome.get('status', 'incomplete'), exit_code=outcome.get('exit_code'),
                    runner_results=runner_results)
        item['class'] = task_class(item['task'])
        operations.append(item)
        if start is not None and end is not None:
            interval = (max(start, since) if since is not None else start, end)
            if GPU_PATTERN.search(command):
                gpu.append(interval)
            if 'build_engine.py' in command:
                build.append(interval)
    jobs, queued = [], []
    for path in sorted((root / 'lab/jobq').glob('*.json')):
        job = read_json(path, warnings)
        if not job or 'class' not in job or 'state' not in job:
            continue
        jobs.append(job)
        start, end, submitted = (timestamp(job.get(k)) for k in ('start', 'end', 'submitted'))
        if start is not None:
            interval = (max(start, since) if since is not None else start, end if end is not None else now)
            if job['class'] == 'gpu':
                gpu.append(interval)
            command = job.get('argv', [])
            command = ' '.join(command) if isinstance(command, list) else str(command)
            if 'build_engine.py' in command:
                build.append(interval)
        if submitted is not None:
            queue_end = start if start is not None else (end if end is not None else now)
            queued.append((max(submitted, since or submitted), queue_end))
    gpu, build, queued = union(gpu), union(build), union(queued)
    overlap = intersection(gpu, build)
    # Only gaps bounded by observed GPU activity; do not invent midnight/now activity.
    gaps = [(gpu[i-1][1], gpu[i][0]) for i in range(1, len(gpu))]
    all_times = [x for seq in (gpu, build, queued) for interval in seq for x in interval]
    all_times += [x for op in operations for x in (op['start'], op['end']) if x is not None]
    days = {}
    if all_times:
        day = int(max(min(all_times), since or min(all_times)) // 86400) * 86400
        while day <= max(all_times):
            window = [(day, day + 86400)]
            idle = intersection(gaps, window)
            queued_idle = intersection(idle, queued)
            longest = max(idle, key=lambda pair: pair[1]-pair[0], default=None)
            days[iso(day)[:10]] = dict(
                gpu_busy_h=duration(intersection(gpu, window))/3600,
                build_h=duration(intersection(build, window))/3600,
                overlap_h=duration(intersection(overlap, window))/3600,
                longest_idle_gap_h=max((b-a for a,b in idle), default=0)/3600,
                longest_idle_gap=[iso(t) for t in longest] if longest else None,
                longest_queued_idle_gap_h=max((b-a for a,b in queued_idle), default=0)/3600 if intersection(queued, window) else None)
            day += 86400
    groups = defaultdict(list)
    failures = {key: dict(count=0, examples=[]) for key in [*FAILURE_PATTERNS, 'other']}
    for op in operations:
        groups[(op['agent'], op['class'])].append(op)
        if op['failed']:
            bucket = failures[op['failure']]
            bucket['count'] += 1
            if len(bucket['examples']) < 3:
                bucket['examples'].append(op['op'])
    summary = []
    for (agent, cls), items in sorted(groups.items()):
        elapsed = [r['elapsed_s'] for r in items if r['elapsed_s'] is not None]
        complete = sum(r['completed'] for r in items)
        failed = sum(r['failed'] for r in items)
        summary.append(dict(agent=agent, task_class=cls, count=len(items), completed=complete,
                            failed=failed, failure_rate=failed/complete if complete else None,
                            total_elapsed_s=sum(elapsed), median_elapsed_s=statistics.median(elapsed) if elapsed else None))
    experiments = dict(result_status=Counter(), abrun_verdict=Counter(), abrun_files=0, records=[])
    paths = [(p, 'result_status', 'status') for p in (root/'experiments/jetson/OM').rglob('result.json')]
    # Abrun stores comparisons.*.verdict, not a top-level verdict; older fixtures may use the latter.
    paths += [(p, 'abrun_verdict', 'verdict') for p in (root/'experiments').rglob('verdict.json')]
    for path, bucket, field in paths:
        record = read_json(path, warnings)
        if not isinstance(record, dict):
            continue
        ts = next((timestamp(record[k]) for k in ('ts', 'ended_at', 'timestamp') if record.get(k)), path.stat().st_mtime)
        if since is not None and ts < since:
            continue
        if bucket == 'abrun_verdict':
            if 'verdict' in record:
                values = [record['verdict']]
            elif isinstance(record.get('comparisons'), dict):
                values = [v.get('verdict', 'unknown') for v in record['comparisons'].values()]
            else:
                continue  # Unrelated handoff verdicts have state/reasons.
            experiments['abrun_files'] += 1
        else:
            values = [record.get(field, 'unknown')]
        for value in values:
            experiments[bucket][str(value)] += 1
        experiments['records'].append(dict(path=str(path.relative_to(root)), values=values, timestamp=ts))
    delegation = aggregate_worklog(list(jsonlines(root/'c2/worklog.jsonl', warnings)), since)
    return dict(since=iso(since) if since is not None else None, generated_at=iso(now),
                ops_summary=summary, operations=operations, failures=failures,
                unmapped_tasks=sorted({r['task'] for r in operations if r['class']=='other'}),
                timeline=days, intervals=dict(gpu=gpu, build=build, overlap=overlap, queued=queued),
                jobs=jobs, tool_calls_per_day={d:dict(c) for d,c in sorted(tools.items())},
                delegation=delegation, experiments=experiments, warnings=warnings,
                totals=dict(ops=len(operations), completed_ops=sum(r['completed'] for r in operations),
                            failed_ops=sum(r['failed'] for r in operations),
                            gpu_busy_h=duration(gpu)/3600, build_h=duration(build)/3600,
                            overlap_h=duration(overlap)/3600,
                            result_count=sum(experiments['result_status'].values()),
                            abrun_count=experiments['abrun_files'],
                            abrun_verdict_count=sum(experiments['abrun_verdict'].values())))


def markdown(report):
    lines = ['# Shape of work', '', f"Generated {report['generated_at']}; since {report['since'] or 'all available evidence'}.",
             'UTC; busy means matching command lifetime, not measured GPU utilization. Inputs are read-only.',
             'Incomplete ops have no inferred end and are excluded from intervals. Running GPU jobs end at this snapshot.',
             'Elapsed totals use whole ops selected by end/start >= since; timeline intervals are clipped to since.',
             'Experiment time uses embedded timestamp, otherwise file mtime. All detail is retained in JSON.', '']

    def table(headers, rows, limit=20):
        lines.append('| ' + ' | '.join(headers) + ' |')
        lines.append('| ' + ' | '.join('---' for _ in headers) + ' |')
        for row in rows[:limit]:
            lines.append('| ' + ' | '.join(str(v).replace('|', '\\|').replace('\n', ' ') if v is not None else '—' for v in row) + ' |')
        if len(rows) > limit:
            lines.append(f'{len(rows)-limit} more rows in JSON.')
        lines.append('')

    lines += ['## 1. Operations', '', 'Failure rate denominator: completed outcomes; pending ops remain in count. Failures include om_run result.json even when the wrapper exits 0.', '']
    table(['Agent', 'Class', 'Ops', 'Completed', 'Elapsed s', 'Median s', 'Failure rate'],
          [[r['agent'], r['task_class'], r['count'], r['completed'], round(r['total_elapsed_s'],3), r['median_elapsed_s'], f"{r['failure_rate']:.1%}" if r['failure_rate'] is not None else None] for r in report['ops_summary']])
    lines += ['Unmapped task names: ' + (', '.join(report['unmapped_tasks']) or 'none') + '.', '']
    table(['Tool calls UTC day', 'Count', 'By tool'], [[d, sum(c.values()), ', '.join(f'{k}: {v}' for k,v in sorted(c.items()))] for d,c in report['tool_calls_per_day'].items()], limit=14)
    lines += ['## 2. Failure taxonomy', '', 'First matching regex wins; stderr reads are bounded to a 4 KiB tail per stderr file, including om_run OUTDIR/stderr.txt.', '']
    table(['Category', 'Count', 'Example ops (up to 3)'], [[k,v['count'], ', '.join(v['examples'])] for k,v in report['failures'].items()])
    lines += ['## 3. Timeline', '', 'Overlap is contamination risk. Idle gaps are bounded by GPU ops and split at UTC midnight.',
              'Queue idle uses observed submitted-to-start intervals for all jobs (including dependency waits); — means no queue coverage that day.', '']
    table(['UTC day', 'GPU busy h', 'Build h', 'Overlap h', 'Longest idle h', 'Queued idle h', 'Idle bounds UTC'],
          [[d, *[round(r[k],4) if r[k] is not None else None for k in ('gpu_busy_h','build_h','overlap_h','longest_idle_gap_h','longest_queued_idle_gap_h')], ' to '.join(r['longest_idle_gap'] or [])] for d,r in report['timeline'].items()])
    lines += ['## 4. Delegation', '', 'Wall/tokens sum end events; latest check-in verdict, cumulative issues, maximum rework rounds, any opus fix.', '']
    table(['Task','Worker','Wall s','Tokens','Verdict','Issues','Rework','Opus fix'], [[r[k] for k in ('task','worker','wall_s','tokens','verdict','issues_count','rework_rounds','opus_fix')] for r in report['delegation']['tasks']], limit=25)
    table(['Worker','Tasks','Wall s','Tokens','Issues','Rework','Opus fixes'], [[w,*[r[k] for k in ('tasks','wall_s','tokens','issues_count','rework_rounds','opus_fixes')]] for w,r in sorted(report['delegation']['workers'].items())], limit=12)
    lines += ['### Lead events', '', 'Missing workers on contamination/result/failure events are inferred as opus.', '']
    table(['UTC time', 'Task', 'Worker', 'Event', 'Detail'],
          [[r.get(k) for k in ('ts', 'task', 'worker', 'event', 'detail')]
           for r in report['delegation']['lead_events']], limit=10)
    lines += ['## 5. Experiments', '', 'Abrun counts comparison verdicts (or a legacy top-level verdict); unrelated handoff verdicts are excluded.', '']
    table(['Kind','Status / verdict','Count'], [[k,v,n] for k in ('result_status','abrun_verdict') for v,n in sorted(report['experiments'][k].items())])
    lines += [f"Totals: {json.dumps(report['totals'], sort_keys=True)}", f"Input warnings: {len(report['warnings'])} (details in JSON).", '']
    assert len(lines) <= 200, 'Markdown line budget exceeded'
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--since', help='ISO timestamp (naive timestamps are UTC)')
    parser.add_argument('--out', default='c2/SHAPE.md')
    parser.add_argument('--json', default='c2/shape.json')
    parser.add_argument('--root', type=Path, default=Path(os.environ.get('SHAPE_ROOT', Path(__file__).resolve().parents[1])), help='Input repository/fixture root; outputs remain relative to cwd')
    args = parser.parse_args()
    try:
        since = timestamp(args.since)
    except ValueError as exc:
        parser.error(str(exc))
    report = generate(args.root.resolve(), since)
    for target, content in ((args.out, markdown(report)), (args.json, json.dumps(report, indent=2, sort_keys=True)+'\n')):
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    print('Unmapped tasks: ' + (', '.join(report['unmapped_tasks']) or 'none'))
    print(json.dumps(report['totals'], sort_keys=True))


if __name__ == '__main__':
    main()
