#!/usr/bin/env python3
"""Artifact-based MoE factory. Predictions select experiments, not accepted gains."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
WRITE_ROOTS = (HERE, ROOT / 'artifacts/codexFAC')
STAGES = ('inspect', 'system', 'plan', 'recipe', 'build', 'measure', 'report')
DEPS = dict(inspect=(), system=(), plan=('inspect', 'system'), recipe=('plan',),
            build=('recipe',), measure=('recipe', 'build'), report=('plan', 'recipe', 'measure'))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def writable(path):
    path = Path(path).resolve()
    if not any(path.is_relative_to(root.resolve()) for root in WRITE_ROOTS):
        raise ValueError(f'write outside assignment roots: {path}')
    return path


def write(path, value):
    path = writable(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = writable(path.with_suffix(path.suffix + '.tmp'))
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def component(name):
    path = ROOT / 'cc' / name
    module_path = {'model_inspect': 'inspect_model.py', 'resource_observe': 'observe.py',
                   'moe_planner': 'plan.py', 'engine_handoff': 'envelopes.py'}[name]
    spec = importlib.util.spec_from_file_location('factory_' + name, path / module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def allowed_components(registry, include_pending=False):
    ids = [c['id'] for c in registry['components']]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate component IDs')
    for c in registry['components']:
        if c['status'] not in ('accepted', 'pending', 'rejected'):
            raise ValueError(f"invalid status: {c['id']}")
    return [c for c in registry['components'] if c['status'] == 'accepted' or
            (include_pending and c['status'] == 'pending')]


def test_prefix(plan):
    return 'pl_' + hashlib.sha256(plan['id'].encode()).hexdigest()[:12]


def lab_definition(definition, environment):
    """Normalize planner Docker -e flags into bin/lab's native env mapping."""
    td = dict(definition)
    args, env = [], dict(td.get('env', {}))
    source = iter(td.get('docker_args', []))
    for arg in source:
        if arg in ('-e', '--env'):
            key, value = next(source).split('=', 1)
            env[key] = value
        else:
            args.append(arg)
    for key, value in environment.items():
        if key in env and env[key] != value:
            raise ValueError(f'component environment conflict: {key}')
        env[key] = value
    td.update(docker_args=args, env=env)
    return td


def claims(plan, measured=False, failed=False):
    if failed:
        return dict(claim_category='failed candidate', transfer_class='discrete-only')
    if not measured:
        return dict(claim_category='prediction', transfer_class='prediction')
    # D012: timing/placement on a discrete GPU cannot establish unified-memory gains.
    kv = plan.get('flags', [])
    quant_kv = '-ctk' in kv and kv[kv.index('-ctk') + 1] != 'f16'
    category = 'runtime-placement/offload trade-off'
    if plan.get('placement') == 'static' and '--n-cpu-moe' in kv:
        if kv[kv.index('--n-cpu-moe') + 1] == '0':
            category = 'representation trade-off'
    if quant_kv or plan.get('pap'):
        category = 'representation trade-off'
    return dict(claim_category=category, transfer_class='discrete-only',
                metric_classes={'memory_byte_accounting': 'transferable', 'kld': 'transferable',
                                'placement_memory_and_timing': 'discrete-only'},
                scope='Observed workload/host only; no causal engine-code gain or accuracy acceptance inferred')


def result_sources(paths):
    """All files consulted by the shared result envelope adapter, including absences."""
    sources = [ROOT/'cc/engine_handoff/envelopes.py']
    for value in paths:
        path = Path(value).resolve()
        job_dir = path.parent.parent
        sources += [path, job_dir/'job.json', path.parent/'owner.json', path.parent/'state.jsonl',
                    job_dir.parent.parent/'export'/job_dir.name/path.parent.name/'MANIFEST.json']
    return sources


def report_join(predictions, results, error_pct=20.0):
    """Join explicit test names or planner names; retain every occurrence and attempt.

    Legacy fixtures can supply plan.test_names. No heuristic join by similar flags,
    model basename, or coordinate; no quality measurement is moved between tests.
    """
    if 'data' in predictions:
        predictions = predictions['data']
    plans = predictions['plans']
    owners = {}
    for p in plans:
        names = p.get('test_names', [test_prefix(p) + '_' + s for s in
                                    ('engagement', 'timing', 'quality512')])
        for name in names:
            if name in owners:
                raise ValueError(f'ambiguous test mapping: {name}')
            owners[name] = p
    rows, seen, unmatched, envelopes = [], set(), [], []
    for result_path in results:
        result_path = Path(result_path).resolve()
        result = read(result_path)
        finished = result.get('execution', {}).get('status') == 'completed'
        job_path = result_path.parent.parent / 'job.json'
        job = read(job_path) if job_path.is_file() else {}
        envelope_error = None
        if job and re.fullmatch(r'att-[1-9][0-9]*', result_path.parent.name):
            try:
                envelope = component('engine_handoff').make_result(job_path.parent, result_path.parent.name)
                envelopes.append(envelope)
                if envelope['evidence_completeness']['journal_errors']:
                    envelope_error = 'invalid durable journal'
            except (ValueError, KeyError, OSError) as exc:
                envelope_error = str(exc)
        for occurrence, observation in result.get('quality', {}).get('tests', {}).items():
            name = occurrence.rsplit('#', 1)[0]
            if name not in owners:
                unmatched.append(dict(result=str(result_path), test=occurrence))
                continue
            p = owners[name]
            seen.add(p['id'])
            clean = (finished and not envelope_error and observation.get('exit_code') == 0 and
                     observation.get('partial') is False and observation.get('ok') is True)
            measured = {}
            if clean:
                mem = observation.get('mem', {})
                measured = {k: observation.get(k) for k in
                            ('decode_tps', 'prefill_tps', 'ttft_ms', 'kld', 'ppl_ratio')}
                measured.update({k: mem.get(k) for k in
                                 ('gpu_delta_mib', 'gpu_peak_mib', 'gpu_baseline_mib',
                                  'host_cgroup_peak_mib', 'shim_peak_mib')})
                rate, tokens, dec = (observation.get('prefill_tps'),
                                     observation.get('prefill_tokens'), observation.get('decode_tps'))
                prompt_ms = 1000 * tokens / rate if rate and tokens is not None else None
                measured['prefill_ms'] = prompt_ms
                measured['warm_ttft_proxy_ms'] = prompt_ms + 1000 / dec if prompt_ms is not None and dec else None
            errors = {}
            for predicted_key, measured_key in (('gpu_mib', 'gpu_delta_mib'),
                                                ('decode_tps', 'decode_tps'),
                                                ('prefill_tps', 'prefill_tps'),
                                                ('prefill_ms', 'prefill_ms'),
                                                ('ttft_ms', 'warm_ttft_proxy_ms'), ('kld', 'kld')):
                predicted, actual = p.get(predicted_key), measured.get(measured_key)
                if (isinstance(predicted, (int, float)) and isinstance(actual, (int, float))
                        and math.isfinite(predicted) and math.isfinite(actual)):
                    delta = actual - predicted
                    pct = 100 * delta / abs(predicted) if predicted else None
                    tolerance = (max(p.get('gpu_guard_mib', 0), 2.0) if predicted_key == 'gpu_mib'
                                 else .01 if predicted_key == 'kld' else abs(predicted) * error_pct / 100)
                    errors[predicted_key] = dict(predicted=predicted, measured=actual,
                                                 delta=delta, percent=pct,
                                                 flagged=abs(delta) > tolerance,
                                                 comparison=measured_key)
            rows.append(dict(plan_id=p['id'], test=occurrence, job=result.get('job'),
                             attempt=result.get('attempt'), result_path=str(result_path),
                             result_sha256=sha(result_path), clean=clean,
                             inputs_bound=bool(job.get('spec', {}).get('inputs')),
                             envelope_error=envelope_error,
                             measured=measured, prediction_errors=errors,
                             prediction_error=any(e['flagged'] for e in errors.values()),
                             quality_accepted=None,
                             **claims(p, measured=clean, failed=not clean)))
    for p in plans:
        if p['id'] not in seen:
            rows.append(dict(plan_id=p['id'], measured=None, prediction=p,
                             quality_accepted=None, **claims(p)))
    return dict(schema='factory-report/0', rows=rows, unmatched=unmatched, result_envelopes=envelopes,
                limitations=['TTFT proxy excludes load, queueing, tokenization and sampling; actual TTFT remains null unless measured.',
                             'Host working-set prediction is not comparable to cgroup peak; device delta, absolute peak and shim cap stay separate.',
                             'Quality512 is scoped to ctx512; it does not certify the requested context.',
                             'Legacy/unbound results are diagnostic joins, not qualified input identity evidence.',
                             'Component acceptance does not qualify an arbitrary patch combination or prove an engine gain.'])


class Factory:
    def __init__(self, args):
        self.a = args
        self.run = writable(args.run_dir)
        self.model = Path(args.model).resolve()
        self.host = (Path(args.host).resolve() if Path(args.host).is_file() else
                     ROOT / 'cc/engine_handoff/hosts' / (args.host + '.json'))
        self.registry_path = Path(args.components).resolve()
        self.modules = {}
        self.memo = {}

    def mod(self, name):
        if name not in self.modules:
            self.modules[name] = component(name)
        return self.modules[name]

    def identity(self, path):
        path = Path(path).resolve()
        # Hash model payload once per invocation; never replace content identity by stat.
        if path not in self.memo:
            self.memo[path] = sha(path) if path.is_file() else None
        return self.memo[path]

    def artifact(self, stage):
        return self.run / (stage + '.json')

    def data(self, stage):
        return read(self.artifact(stage))['data']

    def inputs(self, stage):
        files = [Path(__file__)] + [self.artifact(d) for d in DEPS[stage]]
        params = {}
        if stage == 'inspect':
            files += [self.model, ROOT/'cc/model_inspect/inspect_model.py']
        elif stage == 'system':
            files += [self.host, ROOT/'cc/resource_observe/observe.py']
            params = {'backends': self.a.backends}
        elif stage == 'plan':
            files += [ROOT/'cc/moe_planner/plan.py', ROOT/'cc/moe_planner/calibration.json',
                      self.registry_path]
            # Planner may discover sibling representations and hybrid manifests.
            cal = read(ROOT/'cc/moe_planner/calibration.json')
            files += [self.model.parent / r['model_file'] for r in cal['representations'].values()]
            files += [ROOT / s[k] for s in cal.get('hybrid_specs', [])
                      for k in ('artifact', 'model_path') if k in s]
            files += [ROOT/'artifacts/kl/bf16.kl']
            params = {k: getattr(self.a, k) for k in ('budget_gpu_mib', 'budget_host_mib', 'ctx',
                      'max_kld', 'top', 'allow_extrapolation', 'objective', 'include_pending')}
        elif stage == 'recipe':
            files += [self.registry_path, ROOT/'cc/pap/make_pap.sh']
            for c in allowed_components(read(self.registry_path), self.a.include_pending):
                if c.get('patch'):
                    files.append(ROOT/c['patch'])
            files += [Path(p['model_path']) for p in self.data('plan')['plans']]
            params = {'include_pending': self.a.include_pending, 'ctx': self.a.ctx}
        elif stage == 'build':
            files += [ROOT/'cc/build_engine/build_engine.py']
            files += list((ROOT/'cc/build_engine/recipes').glob('*'))
            params = {'dry_run': self.a.dry_run, 'base_images': {k: os.environ.get(k) for k in ('BASE_DEVEL', 'BASE_RUNTIME')}}
            # Read-only source identity; Build CC performs the authoritative clean/commit checks.
            params['source'] = self.command(['git', '-C', str(ROOT/'ref/llama.cpp'),
                                             'rev-parse', 'HEAD', 'HEAD^{tree}']).strip()
            params['dirty'] = self.command(['git', '-C', str(ROOT/'ref/llama.cpp'),
                                            'status', '--porcelain', '--untracked-files=all'])
        elif stage == 'measure':
            files += [ROOT/'bin/lab']
            for record in self.data('build').get('builds', []):
                files.append(Path(record['record']))
            if self.a.data:
                files.append(Path(self.a.data).resolve())
            # Bind exactly the model and mounted input files used by emitted tests.
            for recipe in self.data('recipe')['recipes']:
                files.append(Path(recipe['model']['path']))
                for td in recipe['tests'].values():
                    mounts = self.mod('engine_handoff').readonly_binds(td.get('docker_args', []))
                    # Adapter returns (host, container) pairs.
                    for host, target in mounts:
                        for arg in td['args']:
                            if arg.startswith(target.rstrip('/') + '/'):
                                files.append(Path(host) / arg[len(target.rstrip('/')) + 1:])
            params = {'dry_run': self.a.dry_run, 'data': self.a.data}
        elif stage == 'report':
            results = self.a.result + [str(p) for p in (self.run/'lab/jobs').glob('*/att-*/result.json')]
            files += result_sources(results)
            params = {'error_pct': self.a.error_pct}
        return dict(files={str(p.resolve()): self.identity(p) for p in files if not p.is_dir()},
                    parameters=params)

    def command(self, argv, env=None):
        full_env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', GIT_OPTIONAL_LOCKS='0',
                        OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
        full_env.update(env or {})
        r = subprocess.run(argv, cwd=ROOT, env=full_env, text=True, capture_output=True)
        if r.returncode:
            raise ValueError(f'command failed ({r.returncode}): {shlex.join(argv)}\n{r.stderr}')
        return r.stdout

    def valid(self, record, inputs):
        return (record.get('status', 'completed') == 'completed' and
                record.get('inputs_sha256') == digest(inputs) and
                record.get('outputs', {}).get('data_sha256') == digest(record.get('data')) and
                all(Path(p).is_file() and sha(p) == value for p, value in
                    record.get('outputs', {}).get('files', {}).items()))

    def stage(self, stage):
        inputs = self.inputs(stage)
        path = self.artifact(stage)
        try:
            cached = read(path) if path.is_file() else {}
        except (ValueError, OSError):
            cached = {}
        if cached and self.valid(cached, inputs):
            print(f'{stage}: skipped (inputs and outputs unchanged)')
            return
        try:
            data, files = getattr(self, 'do_' + stage)()
        except (ValueError, OSError, KeyError, TypeError) as exc:
            data = {'error': str(exc)}
            write(path, dict(schema='factory-stage/0', stage=stage, status='failed', inputs=inputs,
                             inputs_sha256=digest(inputs), data=data,
                             outputs={'data_sha256': digest(data), 'files': {}}))
            raise
        write(path, dict(schema='factory-stage/0', stage=stage, status='completed', inputs=inputs,
                         inputs_sha256=digest(inputs), data=data,
                         outputs=dict(data_sha256=digest(data),
                                      files={str(Path(p).resolve()): sha(p) for p in files})))
        print(f'{stage}: wrote {path.relative_to(ROOT)}')

    def do_inspect(self):
        return self.mod('model_inspect').inspect(str(self.model)), []

    def do_system(self):
        return dict(host=read(self.host), sample=self.mod('resource_observe').sample(
                    backends=tuple(self.a.backends.split(','))),
                    note='Availability snapshot, not a reservation; resume reuses the sample.'), []

    def do_plan(self):
        budget = dict(gpu_mib=self.a.budget_gpu_mib, ctx=self.a.ctx,
                      max_quality_loss=self.a.max_kld, objective=self.a.objective)
        if self.a.budget_host_mib is not None:
            budget['host_mib'] = self.a.budget_host_mib
        system = self.data('system')
        registry = read(self.registry_path)
        enabled = {c['id'] for c in allowed_components(registry, self.a.include_pending)}
        # Rank enough candidates to refill top-N after component safety gates.
        planner = self.mod('moe_planner')
        options = {}
        if 'components' in inspect.signature(planner.plan).parameters:
            options['components'] = {'components': [dict(c, enabled=True) for c in
                                     allowed_components(registry, self.a.include_pending)]}
        result = planner.plan(self.data('inspect'), system['host'], budget,
                     allow_extrapolation=self.a.allow_extrapolation, top=100000,
                     sample=system['sample'], **options)
        selected = []
        for p in result['plans']:
            reason = None
            if p['placement'] == 'hybrid':
                reason = 'hybrid_split rejected by component registry (D030/D031)'
            if '-ctk' in p['flags'] and p['flags'][p['flags'].index('-ctk')+1] != 'f16' and 'kv_quant_fix' not in enabled:
                reason = 'quantized KV requires pending kv_quant_fix; use --include-pending'
            if reason:
                result['rejected'].append(dict(id=p['id'], reasons=[reason]))
            else:
                selected.append(p)
        result['factory_feasible_count'] = len(selected)
        result['plans'] = selected[:self.a.top]
        tests_path = self.run / 'plan-tests.json'
        write(tests_path, self.mod('moe_planner').emit_lab_tests(result))
        result['emitted_tests'] = str(tests_path)
        return result, [tests_path]

    def do_recipe(self):
        registry = read(self.registry_path)
        enabled = allowed_components(registry, self.a.include_pending)
        patches = [dict(component=c['id'], path=str(ROOT/c['patch']),
                        sha256=sha(ROOT/c['patch']), decisions=c['decisions'])
                   for c in enabled if c.get('patch')]
        bundle = self.run/'ordered-components.patch'
        writable(bundle).parent.mkdir(parents=True, exist_ok=True)
        # Build CC v2 accepts one patch; Git checks the combined patch in its archive.
        bundle.write_bytes(b''.join(Path(p['path']).read_bytes().rstrip(b'\n') + b'\n' for p in patches))
        env = {k: v for c in enabled for k, v in c.get('env', {}).items()}
        engine = dict(base_commit=registry['base_commit'], ordered_patches=patches,
                      patch_bundle=str(bundle) if patches else None,
                      patch_bundle_sha256=sha(bundle), cmake_defines=[],
                      extra_targets=[], linkage=True, env=env)
        recipes = []
        tests = read(self.data('plan')['emitted_tests'])
        for p in self.data('plan')['plans']:
            prefix = test_prefix(p)
            model_path = Path(p['model_path'])
            model_sha = self.identity(model_path)
            transforms = [dict(action='reuse-existing-representation', input=str(model_path),
                               output=str(model_path), input_sha256=model_sha, output_sha256=model_sha)]
            lab_tests = {k: lab_definition(v, env) for k, v in tests.items() if k.startswith(prefix + '_')}
            recipes.append(dict(plan_id=p['id'], model=dict(path=str(model_path), sha256=model_sha),
                                model_transforms=transforms, engine=engine,
                                runtime_argv=['/app/llama-completion', '-m', '/m/' + model_path.name] + p['flags'][2:],
                                runtime_mounts=[dict(source=str(model_path.parent), target='/m', access='ro')],
                                runtime_env=env, tests=lab_tests,
                                prerequisites=p['prerequisites'],
                                component_scope='Accepted components only' if not self.a.include_pending else 'Pending components explicitly opted in; combinations unqualified'))
        # Reuse the PAP recipe as a reviewable alternative, never requantize a low-bit input.
        pap = next((c for c in enabled if c['id'] == 'pap'), None)
        alternatives = []
        if pap:
            bf16 = self.model.parent/'granite-3.1-1b-a400m-instruct-bf16.gguf'
            imatrix = ROOT/'artifacts/codexPAP/models/granite-wtrain.imatrix'
            for p in self.data('plan')['plans']:
                flags = p['flags']
                k = int(flags[flags.index('--n-cpu-moe')+1]) if '--n-cpu-moe' in flags else 0
                if p['representation'] != 'mixE4K' or k not in (12, 18, 24):
                    continue
                for bits in (2, 3):
                    output = self.run/'models'/f'pap{k}_q{bits}.gguf'
                    regex = '^blk\\.(' + '|'.join(map(str, range(k))) + ')\\.ffn_.*_exps\\.weight$=q' + str(bits) + '_K'
                    alternatives.append(dict(plan_id=p['id'], component='pap', status='pending',
                        selected=False, source_recipe=str(ROOT/pap['recipe']),
                        input=str(bf16), input_sha256=None, output=str(output), output_sha256=None,
                        hash_state='Fill full SHA256 after materializing; never inherit baseline predictions or KLD',
                        argv=['llama-quantize', '--imatrix', str(imatrix), '--tensor-type', regex,
                              '--tensor-type', 'ffn_.*_exps=q4_K', str(bf16), str(output), 'Q8_0', '4'],
                        requires=['BF16 source and TRAIN imatrix identities', 'Reinspect/replan transformed output',
                                  'Placement-matched GPU engagement, timing and paired quality']))
        return dict(schema='factory-recipe/0', recipes=recipes, components=enabled,
                    excluded=[c for c in registry['components'] if c not in enabled],
                    transform_alternatives=alternatives,
                    notes=['gpu_budget is infrastructure; no hard VRAM cap is inferred or enabled.',
                           'Planner quantized KV flags include flash attention; long-context f16 flash attention remains a recommendation.',
                           'Ordered patch bundle must pass Build CC git apply checks; component acceptance does not qualify combinations.']), [bundle]

    def do_build(self):
        builds, files = [], []
        recipes = self.data('recipe')['recipes']
        if not recipes:
            return dict(status='no-feasible-plans', builds=[]), []
        engine = recipes[0]['engine']
        key = digest(dict(engine=engine, source=self.inputs('build')['parameters']))[:16]
        out = writable(self.run/'builds'/key)
        argv = [sys.executable, '-B', str(ROOT/'cc/build_engine/build_engine.py'),
                '--source', str(ROOT/'ref/llama.cpp'), '--commit', engine['base_commit'],
                '--tag', 'campaign/llamacpp:factory-' + key, '--out', str(out)]
        if engine['patch_bundle']:
            argv += ['--patch', engine['patch_bundle']]
        for define in engine['cmake_defines']:
            argv += ['--cmake-define', define]
        for target in engine['extra_targets']:
            argv += ['--extra-target', target]
        if engine['linkage']:
            argv += ['--linkage']
        record = out/'build-record.json'
        if self.a.dry_run:
            print('DRY-RUN ' + shlex.join(argv))
        else:
            if out.exists():
                raise ValueError(f'Build evidence exists but stage is invalid; select a fresh --run-dir: {out}')
            self.command(['nice', '-n', '10'] + argv)
            built = read(record)
            if built.get('status') != 'success':
                raise ValueError(f'Build CC did not succeed: {record}')
            files += [p for p in out.rglob('*') if p.is_file()]
        builds.append(dict(plan_ids=[r['plan_id'] for r in recipes], argv=argv,
                           status='dry-run' if self.a.dry_run else 'built', record=str(record)))
        return dict(builds=builds, dry_run=self.a.dry_run), files

    def do_measure(self):
        jobs, files = [], []
        lab = writable(self.run/'lab')
        build = self.data('build').get('builds', [])
        record = read(build[0]['record']) if build and Path(build[0]['record']).is_file() else None
        image_id = record.get('image', {}).get('config_digest') if record else None
        image_name = 'factory-' + (image_id.removeprefix('sha256:') if image_id else 'unbuilt')
        if not self.a.dry_run and not image_id and self.data('recipe')['recipes']:
            raise ValueError('measure requires a successful build with immutable image config_digest')
        for recipe in self.data('recipe')['recipes']:
            # Separate engagement, quality and timing jobs so scheduling cannot imply gate success.
            for phase in ('engagement', 'quality512', 'timing'):
                tests = {k: v for k, v in recipe['tests'].items() if k.endswith('_' + phase)}
                if not tests:
                    continue
                key = 'factory-' + digest(dict(recipe=recipe, phase=phase, image=image_id,
                                               data=self.a.data))[:24]
                job_id = 'job-' + hashlib.sha256(key.encode()).hexdigest()[:10]
                definitions = lab/'tests'/(key + '.json')
                write(definitions, tests)
                files.append(definitions)
                argv = [sys.executable, '-B', str(ROOT/'bin/lab'), 'submit', '--image', image_name,
                        '--model', recipe['model']['path'], '--tests', ','.join(list(tests) * (3 if phase == 'timing' else 1)), '--key', key]
                if phase == 'quality512':
                    argv += ['--data', str(Path(self.a.data).resolve()) if self.a.data else '<WIKITEXT_TEST_PATH_REQUIRED>']
                if self.a.dry_run:
                    print('DRY-RUN ' + shlex.join(['env', 'LAB=' + str(lab)] + argv))
                    status = 'dry-run'
                elif phase == 'quality512' and not self.a.data:
                    status = 'blocked-missing-quality-data'
                else:
                    writable(lab/'jobs'/job_id)
                    write(lab/'images'/(image_name + '.json'), dict(name=image_name, ref=image_id, image_id=image_id,
                                                        repo_digests=[], created=None,
                                                        provenance=build[0]['record']))
                    self.command(argv, env={'LAB': str(lab), 'CAMPAIGN_ID': 'c2-resume'})
                    files += [lab/'images'/(image_name + '.json'), lab/'jobs'/job_id/'job.json']
                    status = 'submitted-not-run'
                jobs.append(dict(plan_id=recipe['plan_id'], phase=phase, job=job_id, argv=argv,
                                 env={'LAB': str(lab)}, status=status,
                                 result_glob=str(lab/'jobs'/job_id/'att-*'/'result.json'),
                                 prerequisites=['Backend evidence of engagement before timing',
                                                'Paired quality at matching context before acceptance',
                                                'Three alternating forward/reverse timing rounds; one GPU job at a time']))
        return dict(jobs=jobs, execution='submission only; no GPU job started',
                    image_id=image_id, dry_run=self.a.dry_run), files

    def do_report(self):
        paths = list(self.a.result)
        paths += [str(p) for p in (self.run/'lab/jobs').glob('*/att-*/result.json')]
        result = report_join(self.data('plan'), sorted(set(paths)), self.a.error_pct)
        result['measurement_jobs'] = self.data('measure')['jobs']
        return result, []

    def execute(self):
        selected = self.a.stages.split(',')
        if any(s not in STAGES for s in selected):
            raise ValueError('--stages must be comma-separated names from ' + ','.join(STAGES))
        # Resolve/validate dependencies even when omitted from --stages; resume never trusts stale artifacts.
        needed = set(selected)
        def include(stage):
            for dep in DEPS[stage]:
                if dep not in needed:
                    needed.add(dep)
                    include(dep)
        for s in selected:
            include(s)
        for s in STAGES:
            if s in needed:
                self.stage(s)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run')
    run.add_argument('--model', required=True)
    run.add_argument('--host', required=True)
    run.add_argument('--budget-gpu-mib', type=float, required=True)
    run.add_argument('--budget-host-mib', type=float)
    run.add_argument('--ctx', type=int, required=True)
    run.add_argument('--max-kld', type=float)
    run.add_argument('--stages', default=','.join(STAGES))
    run.add_argument('--dry-run', action='store_true')
    run.add_argument('--include-pending', action='store_true')
    run.add_argument('--components', default=str(HERE/'components.json'))
    run.add_argument('--run-dir', help='Run directory under cc/factory or artifacts/codexFAC; defaults to a stable run ID')
    run.add_argument('--top', type=int, default=5)
    run.add_argument('--objective', choices=('decode', 'ttft', 'balanced'), default='decode')
    run.add_argument('--allow-extrapolation', action='store_true')
    run.add_argument('--backends', default='nvidia-smi,linux')
    run.add_argument('--data')
    run.add_argument('--result', action='append', default=[])
    run.add_argument('--error-pct', type=float, default=20)
    report = commands.add_parser('report', help='Read-only join of predictions and completed lab attempts')
    report.add_argument('--predictions', required=True)
    report.add_argument('--result', required=True, action='append')
    report.add_argument('--out', required=True)
    report.add_argument('--error-pct', type=float, default=20)
    args = parser.parse_args(argv)
    try:
        if not math.isfinite(args.error_pct) or args.error_pct < 0:
            raise ValueError('--error-pct must be finite and nonnegative')
        if args.command == 'report':
            sources = [Path(args.predictions).resolve()] + result_sources(args.result)
            inputs = dict(files={str(p): sha(p) if p.is_file() else None for p in sources},
                          error_pct=args.error_pct, implementation=sha(__file__))
            if Path(args.out).is_file():
                cached = read(args.out)
                if (cached.get('inputs_sha256') == digest(inputs) and
                        cached.get('outputs', {}).get('data_sha256') == digest(cached.get('data'))):
                    print('report: skipped (inputs and outputs unchanged)')
                    return 0
            value = report_join(read(args.predictions), args.result, args.error_pct)
            write(args.out, dict(schema='factory-stage/0', stage='report', inputs=inputs,
                                 inputs_sha256=digest(inputs), data=value,
                                 outputs={'data_sha256': digest(value), 'files': {}}))
        else:
            if args.top < 1:
                raise ValueError('--top must be positive')
            if args.run_dir is None:
                identity = {k: v for k, v in vars(args).items()
                            if k not in ('run_dir', 'stages', 'dry_run', 'result', 'error_pct')}
                args.run_dir = str(ROOT/'artifacts/codexFAC/factory'/digest(identity)[:16])
            Factory(args).execute()
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f'factory: {exc}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
