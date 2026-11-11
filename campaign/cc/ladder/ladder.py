#!/usr/bin/env python3
"""One bounded, resumable GPU-queue step, or an offline frontier report."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import statistics
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cc.abrun import run_summary
from cc.acc import common, kld, server

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / 'experiments/jetson/LADDER'
SPEED_ORDER = ('chat', 'code', 'reason', 'reason', 'code', 'chat')
METRICS = ('eval_ms_per_token', 'prompt_tps', 'nvme_gib_per_token_decode',
           'energy_j_per_token', 'gpu_util_median')
PROTOCOL = 'ladder-v1-chat-greedy-ignore-eos-c4096-n256-ABC-CBA'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    common.write_json(path, value)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def median(values):
    values = [v for v in values if finite(v)]
    return statistics.median(values) if values else None


def configs(model):
    """Spec order is significant; every placement must pin context."""
    result = model['configs']
    if not result or len({c['id'] for c in result}) != len(result):
        raise ValueError('empty or duplicate configuration ladder')
    for config in result:
        argv = config['argv']
        if argv.count('-c') != 1 or argv[argv.index('-c') + 1] != '4096':
            raise ValueError('every configuration must pin -c 4096')
    return result


def successful(summary, tokens=32):
    # common_perf counts the first generated token as prompt evaluation.
    count = summary.get('eval_runs', summary.get('eval_tokens'))
    return (summary.get('status') == 'ok' and not summary.get('watchdog_trips', 0)
            and not summary.get('oom_killed', False) and finite(count) and count >= tokens - 1
            and finite(summary.get('eval_ms_per_token')) and summary['eval_ms_per_token'] > 0
            and summary.get('activation_ok', True))


def select_fit(model, probe, all_configs=False):
    """Probe serially; stop on first usable configuration unless explicitly sweeping."""
    trials, chosen = [], None
    for config in configs(model):
        summary = probe(config)
        fits = successful(summary)
        trials.append(dict(config=config, fits=fits, metrics=summary))
        if fits and chosen is None:
            chosen = config
        if fits and not all_configs:
            break
    return chosen, trials


def completion_argv(model, config, prompt, tokens):
    return [*config['argv'], '-m', '/om/' + Path(model['path']).name,
            '-p', prompt, '-n', str(tokens), '--temp', '0', '-s', '1',
            '-cnv', '-st', '--jinja', '--reasoning', 'off',
            '--chat-template-kwargs', json.dumps(model['template_kwargs'], separators=(',', ':')),
            '--ignore-eos', '--no-display-prompt']


def om_command(out, image, config, argv):
    envflags = shlex.join([word for k, v in sorted(config['env'].items())
                          for word in ('-e', f'{k}={v}')])
    # om_run's prefix branch joins $* into one shell command. Preserve each
    # engine argument there (notably the spaced prompt and JSON kwargs).
    transported = [shlex.quote(a) for a in argv] if os.environ.get('DOCKER_CMD_PREFIX') else argv
    return ['bash', str(ROOT / 'cc/om_run/om_run.sh'), str(out), image,
            envflags, 'llama-completion', *transported]


def show_om(out, image, model, config, prompt, tokens):
    argv = completion_argv(model, config, prompt, tokens)
    print('OM_MODELDIR=' + shlex.quote(str(Path(model['path']).expanduser().parent)) + ' '
          + shlex.join(om_command(out, image, config, argv)))
    # om_run itself provides the runtime; show its expanded container invocation too.
    preview = server.OwnedContainer(image, model['path'], out, 'llama-completion', argv, config['env'])
    print('# container preview (om_run uses --network none --label om=1):')
    print(shlex.join(preview.command()))


def refuse_busy():
    for label in ('om=1', 'acc=1'):
        result = subprocess.run(server.docker_command(['ps', '-q', '--filter', 'label=' + label]),
                                capture_output=True, text=True, check=True, timeout=30)
        if result.stdout.strip():
            raise RuntimeError('another container with label %s is running; submit serially through jobq' % label)


@contextmanager
def exclusive(out):
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.ladder.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('another ladder step holds the output lock') from None
        refuse_busy()
        yield


def bounded(command, deadline, env=None):
    """SIGTERM permits acc's finally/owned-container cleanup before escalation."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('step deadline exhausted; resubmit to resume')
    proc = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True)
    try:
        return proc.wait(timeout=remaining)
    except BaseException:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)
        raise


def clean_owned(out):
    # Only saved IDs/names owned by this step; never kill a label-wide selection.
    for path in Path(out).rglob('container.id'):
        value = path.read_text().strip()
        if re.fullmatch(r'[a-f0-9]{12,64}', value):
            subprocess.run(server.docker_command(['rm', '-f', value]),
                           capture_output=True, timeout=30, check=False)
    for path in Path(out).rglob('invocation.json'):
        name = read(path).get('name', '')
        if re.fullmatch(r'acc-[a-f0-9]{32}', name):
            subprocess.run(server.docker_command(['rm', '-f', name]),
                           capture_output=True, timeout=30, check=False)


def run_checked(command, deadline, env=None):
    rc = bounded(command, deadline, env)
    if rc:
        raise RuntimeError(f'command exited {rc}: {shlex.join(command)}')


def preflight(image, tool, argv, out, deadline):
    env = dict(os.environ, ABRUN_PREFLIGHT_CACHE_DIR=str(out / 'preflight'))
    run_checked([sys.executable, str(ROOT / 'cc/abrun/preflight.py'), image, tool, '--', *argv],
                deadline, env)


def probe_om(out, image, model, config, prompt, tokens, deadline):
    argv = completion_argv(model, config, prompt, tokens)
    preflight(image, 'llama-completion', argv, out, deadline)
    refuse_busy()
    env = dict(os.environ, OM_MODELDIR=str(Path(model['path']).expanduser().parent),
               OM_MAX_S=str(math.ceil(max(1, deadline - time.monotonic()))))
    try:
        rc = bounded(om_command(out, image, config, argv), deadline, env)
    finally:
        clean_owned(out)
    summary = run_summary.summarize(out, config.get('switch_regex', []))
    record = read(out / 'result.json') if (out / 'result.json').exists() else {}
    summary.update(watchdog_trips=record.get('watchdog_trips'), oom_killed=record.get('oom_killed'))
    text = (out / 'stderr.txt').read_text() if (out / 'stderr.txt').exists() else ''
    patterns = config.get('switch_regex', [])
    summary['activation_ok'] = all(re.search(p, text, re.M) for p in patterns)
    if not config['env'] and re.search(r'expert_(?:fill|pregate): active', text):
        summary['activation_ok'] = False
    if rc:
        summary['status'] = 'runner_failed'
    write(out / 'summary.json', summary)
    return summary


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def model_identity(model, required=True):
    path = Path(model['path']).expanduser()
    if not path.is_file():
        if required:
            raise FileNotFoundError(f'model missing: {path}; download it before submitting this step')
        return dict(path=str(path), missing=True)
    stat = path.stat()
    return dict(path=str(path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def step_directory(args, identity):
    parent = args.out / args.model_id
    key = fingerprint(identity)
    if not args.new_run:
        for manifest in sorted(parent.glob(f'{args.step}-*/step.json'), reverse=True):
            if read(manifest).get('identity') == identity:
                return manifest.parent, key
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    path = parent / f'{args.step}-{timestamp}'
    write(path / 'step.json', dict(identity=identity, key=key))
    return path, key


def chosen_config(args, model, image):
    candidates = configs(model)
    if args.config:
        return next(c for c in candidates if c['id'] == args.config)
    identity = model_identity(model, required=not args.dry_run)
    for path in sorted((args.out / args.model_id).glob('fit-*/summary.json'), reverse=True):
        record = read(path)
        if (record.get('complete') and record.get('chosen') and record.get('image') == image
                and record.get('model_identity') == identity):
            chosen = record['chosen']
            if chosen in candidates:
                return chosen
    if args.dry_run:
        print('# no matching fit yet; previewing the first candidate')
        return candidates[0]
    raise ValueError('no successful fit for current model/image; run fit first or select --config explicitly')


def engine_options(config, context=True):
    argv = list(config['argv'])
    if not context:
        i = argv.index('-c')
        del argv[i:i + 2]
    return ['--engine-arg=' + a for a in argv] + [
        item for k, v in sorted(config['env'].items()) for item in ('--env', f'{k}={v}')]


def acc_command(args, image, model, config, out):
    return [sys.executable, str(ROOT / 'cc/acc/mcq.py'), '--image', image,
            '--model', str(Path(model['path']).expanduser()), '--outdir', str(out),
            '--data', str(args.data), '--engine-arg=-np', '--engine-arg=1',
            '--engine-arg=--reasoning', '--engine-arg=off',
            *engine_options(config)]


def tokenizer_identity(path):
    """Read only GGUF metadata; hash tokenizer fields without loading tensor data."""
    path = Path(path).expanduser()
    parts, vocab = {}, None
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    with path.open('rb') as stream:
        end = path.stat().st_size
        def number(fmt):
            data = stream.read(struct.calcsize(fmt))
            if len(data) != struct.calcsize(fmt):
                raise ValueError('truncated GGUF metadata')
            return struct.unpack(fmt, data)[0]
        def skip(n):
            if n < 0 or stream.tell() + n > end:
                raise ValueError('truncated GGUF metadata')
            stream.seek(n, 1)
        def string():
            size = number('<Q')
            skip(size)
        def value(kind):
            if kind in sizes:
                skip(sizes[kind])
            elif kind == 8:
                string()
            elif kind == 9:
                subtype, count = number('<I'), number('<Q')
                if subtype in sizes:
                    skip(sizes[subtype] * count)
                elif subtype == 8:
                    for _ in range(count):
                        string()
                else:
                    raise ValueError('unsupported nested GGUF array')
                return count
            else:
                raise ValueError('unknown GGUF metadata type')
        if stream.read(4) != b'GGUF' or number('<I') not in (2, 3):
            raise ValueError('expected little-endian GGUF v2/v3')
        number('<Q')  # tensor count
        fields = number('<Q')
        for _ in range(fields):
            key_size = number('<Q')
            if key_size > 1024 * 1024:
                raise ValueError('invalid GGUF key length')
            key = stream.read(key_size).decode('utf-8')
            kind = number('<I')
            start = stream.tell()
            count = value(kind)
            stop = stream.tell()
            if key == 'tokenizer.ggml.tokens':
                vocab = count
            if key.startswith('tokenizer.') and not key.startswith('tokenizer.chat_template'):
                stream.seek(start)
                digest = hashlib.sha256(struct.pack('<I', kind))
                for _ in range((stop - start + 1024**2 - 1) // 1024**2):
                    digest.update(stream.read(min(1024**2, stop - stream.tell())))
                parts[key] = digest.hexdigest()
            stream.seek(stop)
    if not vocab or not parts:
        raise ValueError('GGUF tokenizer vocabulary metadata missing')
    return dict(sha256=fingerprint(parts), vocab_size=vocab)


def assert_tokenizers(model, reference):
    left, right = tokenizer_identity(model['path']), tokenizer_identity(reference['path'])
    if left != right:
        raise ValueError('GGUF tokenizer metadata differs; refusing KL comparison')
    for spec, identity in ((model, left), (reference, right)):
        if identity['vocab_size'] > spec['vocab_size_upper_bound']:
            raise ValueError('actual GGUF vocabulary exceeds the declared base-file size bound')
    return left


def capped_chunks(model, reference, requested):
    if (model['tokenizer_group'] != reference['tokenizer_group']
            or model['arch'] != reference['arch']):
        raise ValueError('KLD requires an explicitly declared same-tokenizer/architecture pair')
    # fp16 logits + allowance for engine metadata/header; decimal 3 GB limit.
    vocab = max(model['vocab_size_upper_bound'], reference['vocab_size_upper_bound'])
    cap = (3_000_000_000 - 64 * 1024**2) // (512 * (2 * vocab + 64))
    if requested < 1 or cap < 1:
        raise ValueError('invalid chunks or vocabulary bound too large')
    return min(requested, cap)


def kl_command(args, image, model, config, out, logits, chunks, action):
    return [sys.executable, str(ROOT / 'cc/acc/kld.py'), action, '--image', image,
            '--model', str(Path(model['path']).expanduser()), '--outdir', str(out),
            '--logits', str(logits), '--text', str(args.text), '--chunks', str(chunks),
            '--timeout', str(args.max_seconds), *engine_options(config, context=False)]


def fit_step(args, model, image, out, deadline):
    def probe(config):
        directory = out / config['id']
        if (directory / 'summary.json').exists():
            return read(directory / 'summary.json')
        # Interrupted trial artifacts are retained and a new owned attempt is used.
        directory = fresh_attempt(directory)
        try:
            summary = probe_om(directory, image, model, config,
                               (HERE / 'prompts/chat.txt').read_text().strip(), 32, deadline)
        except RuntimeError as exc:
            summary = dict(status='probe_failed', reason=str(exc))
            write(directory / 'summary.json', summary)
        if directory.name != config['id']:
            write(out / config['id'] / 'summary.json', summary)
        return summary
    chosen, trials = select_fit(model, probe, args.all)
    return dict(chosen=chosen, trials=trials, complete=True)


def fresh_attempt(base):
    if not base.exists():
        return base
    for i in range(1, 10000):
        path = base.parent / f'{base.name}-retry-{i:04}'
        if not path.exists():
            return path
    raise RuntimeError('too many retries')


def speed_step(args, model, image, config, out, deadline):
    runs = []
    for i, prompt in enumerate(SPEED_ORDER):
        canonical = out / f'{i + 1:03}-{prompt}'
        if (canonical / 'summary.json').exists() and successful(read(canonical / 'summary.json'), 256):
            summary = read(canonical / 'summary.json')
        else:
            directory = fresh_attempt(canonical)
            summary = probe_om(directory, image, model, config,
                               (HERE / f'prompts/{prompt}.txt').read_text().strip(), 256, deadline)
            if not successful(summary, 256):
                raise RuntimeError(f'unsuccessful speed run: {directory}; inspect logs then resume')
            write(canonical / 'summary.json', summary)
        runs.append(summary)
    metrics = {k: median([r.get(k) for r in runs]) for k in METRICS}
    ms = metrics['eval_ms_per_token']
    metrics['tok_s'] = 1000 / ms if ms else None
    return dict(config=config, complete=True, order=list(SPEED_ORDER), runs=runs,
                metrics=metrics, coverage={k: sum(finite(r.get(k)) for r in runs) for k in METRICS})


def validate_sessions(out, config):
    """Reject watchdog contamination even if a server survived long enough to score."""
    paths = list(Path(out).rglob('result.json'))
    if not paths:
        raise ValueError('missing owned-container result evidence')
    for path in paths:
        result = read(path)
        if result.get('cleanup_errors'):
            raise ValueError(f'container cleanup errors: {path}')
        state = result.get('container_state')
        if state and json.loads(state).get('OOMKilled'):
            raise ValueError(f'container OOM: {path}')
        watchdog = path.parent / 'watchdog.jsonl'
        if not watchdog.exists():
            raise ValueError(f'missing watchdog evidence: {path.parent}')
        if 'watchdog_trip' in watchdog.read_text():
            raise ValueError(f'watchdog tripped: {path.parent}; retain this as unscored evidence')
        text = (path.parent / 'stderr.txt').read_text()
        # One-token MCQ has no decode-fill activation; validate configured values only.
        for pattern in config.get('switch_regex', []):
            if 'expert_config: effective' in pattern and not re.search(pattern, text, re.M):
                raise ValueError(f'expert effective-config mismatch: {path.parent}')


def acc_step(args, model, image, config, out, deadline):
    command = acc_command(args, image, model, config, out / 'acc')
    preflight(image, 'llama-server', [*config['argv'], '-np', '1', '--jinja', '--reasoning', 'off'], out, deadline)
    refuse_busy()
    run_checked(command, deadline, dict(os.environ, OM_MAX_S=str(args.max_seconds)))
    validate_sessions(out / 'acc/sessions', config)
    result = read(out / 'acc/summary.json')
    template = read(out / 'acc/template.json')
    return dict(config=config, complete=result['complete'], dataset_set=args.set,
                dataset_sha256=result['dataset_sha256'], accuracy=result,
                template_kwargs_used=template['thinking_settings'].get('chat_template_kwargs', {}))


def kld_step(args, model, image, config, out, deadline, reference, refconfig):
    tokenizer = assert_tokenizers(model, reference)
    chunks = capped_chunks(model, reference, args.chunks)
    # Key covers reference identity, placement, corpus bytes and the fixed KL protocol.
    key = fingerprint(dict(reference=model_identity(reference), image=image, config=refconfig,
                           corpus=common.sha256(args.text), chunks=chunks, context=512, ubatch=1))
    logits = args.kl_dir / f'{args.ref}-{key[:20]}-c512-ch{chunks}.bin'
    meta = logits.with_suffix(logits.suffix + '.meta.json')
    if not meta.exists():
        if logits.exists():
            raise ValueError(f'incomplete reference base {logits}; inspect/remove it before retrying')
        preflight(image, 'llama-perplexity', [*refconfig['argv'], '--kl-divergence-base'], out, deadline)
        refuse_busy()
        command = kl_command(args, image, reference, refconfig, fresh_attempt(out / 'reference'), logits, chunks, 'reference')
        run_checked(command, deadline, dict(os.environ, OM_MAX_S=str(args.max_seconds)))
    provenance = read(meta)
    validate_sessions(Path(provenance['run_dir']), refconfig)
    if provenance.get('model_sha256') != common.sha256(Path(reference['path']).expanduser()):
        raise ValueError('reference base model hash mismatch')
    if logits.stat().st_size > 3_000_000_000:
        raise ValueError('reference exceeds 3 GB cap; reduce --chunks')
    preflight(image, 'llama-perplexity', [*config['argv'], '--kl-divergence'], out, deadline)
    refuse_busy()
    candidate = fresh_attempt(out / 'candidate')
    run_checked(kl_command(args, image, model, config, candidate, logits, chunks, 'candidate'), deadline,
                dict(os.environ, OM_MAX_S=str(args.max_seconds)))
    validate_sessions(candidate, config)
    return dict(config=config, complete=True, reference=args.ref, chunks=chunks,
                logits=str(logits), logits_size_bytes=logits.stat().st_size,
                kld=read(candidate / 'summary.json'), context_override=512, tokenizer=tokenizer)


def load_records(out):
    records = []
    for path in sorted(Path(out).glob('*/*/summary.json')):
        record = read(path)
        if record.get('ladder_schema') == 1:
            record['_path'] = str(path)
            records.append(record)
    return records


def pareto(rows):
    """Maximize measured speed and accuracy, only within identical accuracy protocols."""
    def usable(r):
        return (finite(r.get('tok_s')) and r['tok_s'] > 0 and finite(r.get('acc'))
                and r.get('accuracy_complete') and r.get('speed_complete') and not r.get('fixture'))
    result = []
    for i, row in enumerate(rows):
        if not usable(row):
            continue
        group = row.get('accuracy_group')
        dominated = any(usable(other) and other.get('accuracy_group') == group
                        and other['tok_s'] >= row['tok_s'] and other['acc'] >= row['acc']
                        and (other['tok_s'] > row['tok_s'] or other['acc'] > row['acc'])
                        for other in rows)
        if not dominated:
            result.append(i)
    return result


def aggregate(records, spec):
    rows = {}
    def get(record, config):
        model_id = record['model_id']
        model = spec['models'].get(model_id, {})
        # Keep incompatible image/model identities separate rather than merging stale evidence.
        key = (model_id, fingerprint(config), record.get('image'),
               fingerprint(record.get('model_identity')))
        if key not in rows:
            size = record.get('model_identity', {}).get('size')
            rows[key] = dict(model_id=model_id, quant=model.get('quant'), config=config,
                image=record.get('image'), model_identity=record.get('model_identity'),
                size_gib=size / 1024**3 if size else model.get('published_size_gib'),
                size_source='measured file' if size else 'published estimate',
                regime='over-memory' if model.get('over_memory') else 'may fit', fits=None,
                published_scores=model.get('published_scores'), published_quant=model.get('published_quant'),
                evidence=[], fixture=record.get('fixture', False), accuracy_complete=False,
                speed_complete=False)
        row = rows[key]
        row['evidence'].append(record.get('_path'))
        return row
    for record in records:
        step = record['step']
        if step == 'fit':
            for trial in record['trials']:
                row = get(record, trial['config'])
                row.update(fits=trial['fits'], fit_metrics=trial['metrics'])
        elif record.get('config'):
            row = get(record, record['config'])
            if step == 'speed' and record.get('complete'):
                row.update(record['metrics'])
                row['speed_complete'] = True
                row['speed_coverage'] = record.get('coverage')
            elif step == 'acc' and record.get('complete'):
                accuracy = record['accuracy']
                measurement = dict(set=record['dataset_set'], dataset_sha256=record['dataset_sha256'],
                    acc=accuracy['acc'], n=accuracy['n'], correct=accuracy['correct'],
                    wilson95_lo=accuracy['wilson95_lo'], wilson95_hi=accuracy['wilson95_hi'],
                    no_letter=accuracy.get('no_letter'), template_kwargs=record['template_kwargs_used'],
                    protocol=accuracy.get('protocol', {'scoring': 'mcq-letter-max-v1'}))
                row.setdefault('accuracy_sets', {})[record['dataset_set']] = measurement
            elif step == 'kld' and record.get('complete'):
                row.update(kld=record['kld'], kld_reference=record['reference'])
    result = list(rows.values())
    # One row/config, retain every set in JSON; prefer the larger N for the display.
    for row in result:
        sets = row.get('accuracy_sets', {})
        if sets:
            best = max(sets.values(), key=lambda a: a['n'])
            row.update(best)
            row['accuracy_complete'] = True
            row['accuracy_group'] = (best['dataset_sha256'], fingerprint(best['protocol']),
                                     'non-thinking')
        tps = row.get('tok_s')
        row['tiers'] = [tier for tier in (2, 5, 10) if finite(tps) and tps >= tier]
    views = accuracy_views(result)
    frontier = pareto(views)
    for i, row in enumerate(result):
        row['pareto_sets'] = sorted({views[j]['set'] for j in frontier if views[j]['row_index'] == i})
        row['pareto'] = bool(row['pareto_sets'])
    return result


def accuracy_views(rows):
    """Views permit a config measured on both subsets to join both frontiers."""
    views = []
    for i, row in enumerate(rows):
        for measurement in row.get('accuracy_sets', {}).values():
            view = dict(row, **measurement, row_index=i)
            view['accuracy_group'] = (measurement['dataset_sha256'], fingerprint(measurement['protocol']), 'non-thinking')
            views.append(view)
    return views


def format_number(value, digits=2):
    return f'{value:.{digits}f}' if finite(value) else '—'


def render(rows):
    lines = ['# Jetson model ladder', '',
        'Shared DRAM: 7.4 GiB. Published scores use different protocols (often thinking); '
        'only local complete MCQ and speed runs enter a frontier. Missing evidence is —.', '',
        '| Model | Quant | Config | GiB | Regime / fits | ms/token | tok/s | Tiers | MCQ % [Wilson 95%] / n | Set | KLD vs ref | J/token | GPU % | Published MMLU-Pro % | Pareto |',
        '|---|---|---|---:|---|---:|---:|---|---|---|---|---:|---:|---:|---|']
    for row in rows:
        acc = '—'
        if finite(row.get('acc')):
            acc = (f"{100 * row['acc']:.1f} [{100 * row['wilson95_lo']:.1f}, "
                   f"{100 * row['wilson95_hi']:.1f}] / {row['n']}")
        kl = row.get('kld', {})
        kltext = format_number(kl.get('mean_kld'), 6) + (' vs ' + row['kld_reference'] if kl else '')
        lines.append('| ' + ' | '.join(map(str, [row['model_id'], row.get('quant') or '—',
            row['config']['id'], format_number(row.get('size_gib')), f"{row['regime']} / {row['fits']}",
            format_number(row.get('eval_ms_per_token')), format_number(row.get('tok_s')),
            ', '.join('>=' + str(t) for t in row['tiers']) or '—', acc, row.get('set', '—'),
            kltext, format_number(row.get('energy_j_per_token')), format_number(row.get('gpu_util_median')),
            format_number((row.get('published_scores') or {}).get('scores', {}).get('MMLU-Pro')),
            ', '.join(row.get('pareto_sets', [])) if row['pareto'] else '—'])) + ' |')
    if not rows:
        lines += ['', 'No ladder results yet. Run fit, speed and acc through jobq.']
    lines += ['', '## Accuracy vs decode speed (text scatter)', '', '```text']
    for row in sorted(rows, key=lambda r: r.get('tok_s') or 0):
        if finite(row.get('acc')) and finite(row.get('tok_s')):
            lines.append(f"{100 * row['acc']:5.1f}% |" + ' ' * min(60, int(row['tok_s'] * 2))
                         + f"* {row['tok_s']:.2f} tok/s {row['model_id']} / {row['config']['id']} ({row['set']})")
    lines += ['```', '', '## Pareto frontier', '']
    views = accuracy_views(rows)
    selected = [views[i] for i in pareto(views)]
    lines += [f"- {r['model_id']} / {r['config']['id']}: {r['tok_s']:.2f} tok/s, "
              f"{100 * r['acc']:.1f}% ({r['set']})." for r in selected] or ['No comparable complete speed + accuracy pairs.']
    lines += ['', '## Best accuracy at each speed threshold', '']
    groups = {tuple(r['accuracy_group']) for r in views if r.get('accuracy_group') and not r.get('fixture')}
    for group in sorted(groups):
        for tier in (2, 5, 10):
            choices = [r for r in views if tuple(r.get('accuracy_group', ())) == group
                       and r.get('speed_complete') and tier in r['tiers'] and not r.get('fixture')]
            if choices:
                best = max(choices, key=lambda r: (r['acc'], r['tok_s']))
                lines.append(f"- {best['set']} (dataset {group[0][:12]}), >={tier} tok/s: "
                             f"{best['model_id']} / {best['config']['id']}, {100 * best['acc']:.1f}%.")
            else:
                lines.append(f'- Dataset {group[0][:12]}, >={tier} tok/s: no measured qualifying pair.')
    lines += ['', '## Published context and evidence', '']
    for row in rows:
        published = row.get('published_scores')
        scores = ', '.join(f'{k} {v}' for k, v in published['scores'].items()) if published else 'none'
        urls = ' '.join(published.get('urls', [])) if published else ''
        mode = published.get('mode', '') if published else ''
        lines.append(f"- {row['model_id']} / {row['config']['id']}: published {scores}; {mode}. {urls}")
        lines.append(f"  Size: {row['size_source']}. Prompt tok/s {format_number(row.get('prompt_tps'))}; "
                     f"decode NVMe GiB/token {format_number(row.get('nvme_gib_per_token_decode'), 5)}. "
                     f"Evidence: {', '.join(str(p) for p in row['evidence'])}.")
    lines += ['', 'Decode IO/energy windows are approximate elapsed windows from run_summary; '
              'counters include the whole system. Inspect phase_windows, reasons, coverage and raw logs. '
              'No fixed clocks or cold-cache control; medians are descriptive, not a significance test. '
              'Fixture protocol captures are never accuracy benchmarks or frontier entries.', '']
    return '\n'.join(lines)


def fixture_records(spec):
    """Replay actual legacy summaries; never assign Granite protocol samples an accuracy."""
    records = []
    model_id = 'qwen36-35b-a3b'
    arms = {}
    for path in sorted((HERE / 'fixtures/om').glob('*/summary.json')):
        arm = path.parent.name.split('-', 1)[1]
        arms.setdefault(arm, []).append(path)
    for arm, paths in arms.items():
        summaries = [read(path) for path in paths]
        summary = {k: median([r.get(k) for r in summaries]) for k in (*METRICS, 'eval_tps')}
        path = paths[0]
        config = dict(id='legacy-' + arm, argv=['-c', '4096'], env={})
        metrics = {k: summary.get(k) for k in METRICS}
        ms = metrics['eval_ms_per_token']
        metrics['tok_s'] = 1000 / ms if ms else summary.get('eval_tps')
        records.append(dict(step='speed', model_id=model_id, image='legacy OM10c',
            model_identity={}, config=config, complete=True, fixture=True, metrics=metrics, _path=str(path)))
    captured = kld.parse((HERE / 'fixtures/perplexity-kld.granite-Q8_0.txt').read_text())
    records.append(dict(step='kld', model_id='captured-granite', image='captured protocol fixture',
        model_identity={}, config=dict(id='protocol-c128', argv=['-c', '128'], env={}),
        complete=True, fixture=True, reference='captured-base (see cc/acc README)', kld=captured,
        _path=str(HERE / 'fixtures/perplexity-kld.granite-Q8_0.txt')))
    return records


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='step', required=True)
    for name in ('fit', 'speed', 'acc', 'kld', 'report'):
        cmd = commands.add_parser(name)
        cmd.add_argument('--spec', type=Path, default=HERE / 'models.json')
        cmd.add_argument('--out', type=Path, default=DEFAULT_OUT)
        cmd.add_argument('--dry-run', action='store_true')
        if name == 'report':
            cmd.add_argument('--fixtures', action='store_true', help='Only captured protocol fixtures')
            continue
        cmd.add_argument('model_id')
        cmd.add_argument('--image', help='Immutable engine image; default from spec')
        cmd.add_argument('--config', help='Explicit config ID, otherwise the successful fit')
        cmd.add_argument('--max-seconds', type=int, default=14400 if name == 'acc' else 7200)
        cmd.add_argument('--new-run', action='store_true', help='Start fresh instead of resuming identical settings')
        if name == 'fit':
            cmd.add_argument('--all', action='store_true')
        if name == 'acc':
            cmd.add_argument('--set', choices=['mmlu_pro_140', 'mmlu_pro_70'], default=None)
            cmd.add_argument('--data', type=Path, help='Frozen JSONL override')
        if name == 'kld':
            cmd.add_argument('--ref', help='Same-tokenizer reference model ID (or spec kld_ref)')
            cmd.add_argument('--ref-config', help='Reference config override, otherwise its measured fit')
            cmd.add_argument('--chunks', type=int, default=8)
            cmd.add_argument('--text', type=Path, default=Path('~/models/campaign/data/wikitext/wikitext-2-raw/wiki.test.raw').expanduser())
            cmd.add_argument('--kl-dir', type=Path, default=Path('~/models/campaign/kl').expanduser())
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        spec = read(args.spec)
        args.out = args.out.expanduser().resolve()
        if args.step == 'report':
            records = fixture_records(spec) if args.fixtures else load_records(args.out)
            if args.dry_run and not records:
                print('Captured fixture preview (no ladder results; no files written):')
                records = fixture_records(spec)
            rows = aggregate(records, spec)
            markdown = render(rows)
            if not args.dry_run:
                write(args.out / 'ladder.json', dict(schema_version=1, rows=rows))
                (args.out / 'REPORT.md').write_text(markdown)
            print(markdown)
            return 0
        if args.max_seconds < 1:
            raise ValueError('--max-seconds must be positive')
        if args.model_id not in spec['models']:
            raise ValueError(f'unknown model: {args.model_id}')
        model = spec['models'][args.model_id]
        image = args.image or spec['image']
        configs(model)
        if args.config and args.config not in {c['id'] for c in configs(model)}:
            raise ValueError('unknown config ID')
        if args.step == 'fit' and args.config:
            model = dict(model, configs=[c for c in configs(model) if c['id'] == args.config])
        config = configs(model)[0] if args.step == 'fit' else chosen_config(args, model, image)
        if args.step == 'acc':
            args.set = args.set or ('mmlu_pro_70' if model['over_memory'] else 'mmlu_pro_140')
            args.data = (args.data or Path('~/models/campaign/data/acc') / (args.set + '.jsonl')).expanduser().resolve()
        if args.step == 'kld':
            args.text = args.text.expanduser().resolve()
            args.kl_dir = args.kl_dir.expanduser().resolve()
            args.ref = args.ref or model.get('kld_ref')
            if not args.ref or args.ref not in spec['models']:
                raise ValueError('supply --ref with a declared same-tokenizer reference model ID')
            reference = spec['models'][args.ref]
            chunks = capped_chunks(model, reference, args.chunks)
            refargs = argparse.Namespace(**vars(args))
            refargs.model_id, refargs.config = args.ref, args.ref_config
            refconfig = chosen_config(refargs, reference, image)
        if args.dry_run:
            out = args.out / args.model_id / (args.step + '-DRY-RUN')
            if args.step in ('fit', 'speed'):
                selected = configs(model) if args.step == 'fit' and args.all else [config]
                for trial in selected:
                    for prompt in (SPEED_ORDER if args.step == 'speed' else ['chat']):
                        show_om(out / trial['id'], image, model, trial,
                                (HERE / f'prompts/{prompt}.txt').read_text().strip(),
                                32 if args.step == 'fit' else 256)
            elif args.step == 'acc':
                print(shlex.join(acc_command(args, image, model, config, out / 'acc') + ['--dry-run']))
                owned = server.OwnedContainer(image, model['path'], out, 'llama-server',
                    [*config['argv'], '-np', '1', '--jinja', '--reasoning', 'off', '-m', '/om/' + Path(model['path']).name], config['env'])
                print(owned.dry_run())
            else:
                logits = args.kl_dir / f'{args.ref}-DRY-c512-ch{chunks}.bin'
                print(f'# cap: {chunks} chunks, <= 3 GB; KL overrides context to 512, batch 512, ubatch 1')
                for action, subject, placement in [('reference', reference, refconfig), ('candidate', model, config)]:
                    print(shlex.join(kl_command(args, image, subject, placement, out / action, logits, chunks, action) + ['--dry-run']))
            return 0
        model_id = model_identity(model)
        identity = dict(protocol=PROTOCOL, model=model_id, image=image, step=args.step,
                        config=config if args.step != 'fit' else configs(model),
                        prompts={n: fingerprint((HERE / f'prompts/{n}.txt').read_text()) for n in set(SPEED_ORDER)})
        if args.step == 'fit':
            identity['all'] = args.all
        elif args.step == 'acc':
            identity.update(dataset_set=args.set, dataset_sha256=common.sha256(args.data))
        elif args.step == 'kld':
            identity.update(reference=model_identity(reference), ref_config=refconfig,
                            chunks=chunks, corpus_sha256=common.sha256(args.text))
        deadline = time.monotonic() + args.max_seconds
        # Convert queue cancellation into the same owned-container cleanup as a deadline.
        def interrupted(signum, frame):
            raise SystemExit(128 + signum)
        previous = signal.signal(signal.SIGTERM, interrupted)
        try:
            with exclusive(args.out):
                out, key = step_directory(args, identity)
                summary_path = out / 'summary.json'
                if summary_path.exists() and read(summary_path).get('complete'):
                    print(f'already complete: {summary_path}')
                    return 0 if args.step != 'fit' or read(summary_path).get('chosen') else 1
                try:
                    if args.step == 'fit':
                        result = fit_step(args, model, image, out, deadline)
                    elif args.step == 'speed':
                        result = speed_step(args, model, image, config, out, deadline)
                    elif args.step == 'acc':
                        result = acc_step(args, model, image, config, out, deadline)
                    else:
                        result = kld_step(args, model, image, config, out, deadline, reference, refconfig)
                    result.update(ladder_schema=1, model_id=args.model_id, model_identity=model_id,
                                  step=args.step, image=image, identity_key=key, protocol=PROTOCOL)
                    write(summary_path, result)
                    print(json.dumps(result, indent=2))
                finally:
                    clean_owned(out)
        finally:
            signal.signal(signal.SIGTERM, previous)
        return 1 if args.step == 'fit' and not result['chosen'] else 0
    except (OSError, ValueError, RuntimeError, StopIteration, subprocess.SubprocessError) as exc:
        p.exit(1, f'ladder: {exc}\n')


if __name__ == '__main__':
    sys.exit(main())
