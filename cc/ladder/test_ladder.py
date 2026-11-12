"""Offline protocol/config tests. All performance observations are captured data.

Mock process failures and dataset SHAPE_ONLY rows test control flow; they are
never emitted as benchmark evidence. The Granite response is a protocol sample,
not a measured MMLU score.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
from unittest import mock

import pytest

from cc.acc import common, mcq
from cc.ladder import fetch_sets, ladder

SPEC = ladder.read(ladder.HERE / 'models.json')
MODEL = SPEC['models']['qwen35-9b']
MOE = SPEC['models']['qwen36-35b-a3b']
FIXTURES = ladder.HERE / 'fixtures'


def captured_summary():
    # Upgrade a copied legacy summary with precise fields read from the same
    # real stderr, never from rounded throughput. No raw-source file is written.
    path = ladder.ROOT / 'experiments/jetson/OM/J-OM10c/003-pg8i8'
    return ladder.run_summary.summarize(path, write=False)


def fit_ok():
    # Original capture has >=31 decode runs, status ok and precise ms/token.
    return captured_summary()


def protocol_accuracy():
    raw = ladder.read(FIXTURES / 'server-chat-logprobs.granite.json')
    # Same captured-response protocol exercise as cc/acc/test_acc.py; no claim
    # that this constructed item was in a benchmark dataset.
    item = dict(choices=list('ABCD'), answer='B')
    scored = mcq.score(item, raw)
    return dict(common.summary([dict(**scored, elapsed_ms=raw['timings']['prompt_ms'],
                                    prompt_tokens=raw['usage']['prompt_tokens'], decode_tps=None)]),
                complete=True)


def record(step, config=None, **kwargs):
    result = dict(ladder_schema=1, step=step, model_id='qwen36-35b-a3b',
                  image='captured-test-only', model_identity={},
                  _path='captured protocol test', **kwargs)
    if config:
        result['config'] = config
    return result


def test_copies_are_exact():
    for copied in sorted((FIXTURES / 'om').glob('*/summary.json')):
        source = ladder.ROOT / 'experiments/jetson/OM/J-OM10c' / copied.parent.name / copied.name
        assert copied.read_bytes() == source.read_bytes()
    for copied in FIXTURES.glob('*.json'):
        assert copied.read_bytes() == (ladder.ROOT / 'cc/acc/fixtures' / copied.name).read_bytes()
    assert (FIXTURES / 'perplexity-kld.granite-Q8_0.txt').read_bytes() == (
        ladder.ROOT / 'cc/acc/fixtures/perplexity-kld.granite-Q8_0.txt').read_bytes()


def test_dense_ladder_order_and_pinned_context():
    candidates = ladder.configs(MODEL)
    assert candidates[0]['argv'] == ['-ngl', '99', '-fa', 'on', '-c', '4096']
    assert candidates[1]['argv'][-4:] == ['-ctk', 'q8_0', '-ctv', 'q8_0']
    assert '-lm' in candidates[2]['argv']
    assert [c['argv'][1] for c in candidates if c['id'].startswith('partial')] == ['24', '12', '0']
    assert all('-c' in c['argv'] for c in candidates)
    for model in SPEC['models'].values():
        ladder.configs(model)


def test_moe_strict_env_matches_om11_spec():
    reference = ladder.read(ladder.ROOT / 'cc/abrun/specs/J-OM11.json')['arms']['pg8']
    first, second = ladder.configs(MOE)
    assert first['env'] == reference['env']
    assert first['switch_regex'] == reference['switch_regex']
    assert first['env']['LLAMA_EXPERT_PREGATE_K'] == '8'
    assert first['env']['LLAMA_EXPERT_PREGATE_ORDER'] == 'after'
    assert first['env']['LLAMA_EXPERT_PREGATE_MAX_INFLIGHT'] == '4'
    assert second['env'] == {}


@pytest.mark.parametrize('failure', ['watchdog_tripped_survived', 'watchdog_tripped_killed', 'oom_killed', 'failed'])
def test_fit_advances_after_failure_and_stops(failure):
    good = fit_ok()
    probe = mock.Mock(side_effect=[dict(good, status=failure), good])
    chosen, trials = ladder.select_fit(MODEL, probe)
    assert chosen == MODEL['configs'][1]
    assert [t['fits'] for t in trials] == [False, True]
    assert probe.call_count == 2


def test_all_continues_after_success():
    chosen, trials = ladder.select_fit(MODEL, mock.Mock(return_value=fit_ok()), all_configs=True)
    assert chosen == MODEL['configs'][0]
    assert len(trials) == len(MODEL['configs'])


def test_all_fail_cleanly():
    chosen, trials = ladder.select_fit(MOE, mock.Mock(return_value=dict(fit_ok(), status='failed')))
    assert chosen is None and not any(t['fits'] for t in trials)


@pytest.mark.parametrize('field,value', [('watchdog_trips', 1), ('activation_ok', False),
                                        ('eval_runs', 0), ('eval_ms_per_token', None), ('oom_killed', True)])
def test_fit_needs_clean_complete_decode(field, value):
    assert not ladder.successful(dict(fit_ok(), **{field: value}))


def test_config_context_is_mandatory():
    model = dict(MODEL, configs=[dict(MODEL['configs'][0], argv=['-ngl', '99'])])
    with pytest.raises(ValueError, match='pin'):
        ladder.configs(model)


@pytest.mark.parametrize('model_id', ['qwen35-9b', 'qwen36-35b-a3b', 'gemma4-e4b'])
def test_fit_dry_run_is_read_only(model_id, tmp_path, capsys):
    with mock.patch.object(subprocess, 'run', side_effect=AssertionError('Docker forbidden')), \
         mock.patch.object(subprocess, 'Popen', side_effect=AssertionError('process forbidden')):
        assert ladder.main(['fit', model_id, '--out', str(tmp_path / 'absent'), '--dry-run']) == 0
    out = capsys.readouterr().out
    assert '-c 4096' in out and '--runtime nvidia' in out and 'om_run.sh' in out
    assert not (tmp_path / 'absent').exists()


def test_speed_plan_counterbalanced(capsys, tmp_path):
    ladder.main(['speed', 'qwen35-9b', '--dry-run', '--out', str(tmp_path)])
    output = capsys.readouterr().out
    assert output.count('om_run.sh') == 6
    assert output.count('-n 256 --temp 0 -s 1') == 12  # om_run + container preview
    assert ladder.SPEED_ORDER == tuple(reversed(ladder.SPEED_ORDER))
    assert '--reasoning off' in output and '--chat-template-kwargs' not in output  # unsupported by llama-completion at cf34d311
    assert list(tmp_path.iterdir()) == []


def test_missing_model_refuses_before_docker(tmp_path):
    spec = copy.deepcopy(SPEC)
    spec['models']['qwen35-9b']['path'] = str(tmp_path / 'missing.gguf')
    path = tmp_path / 'spec.json'
    ladder.write(path, spec)
    with mock.patch.object(subprocess, 'run', side_effect=AssertionError('Docker forbidden')):
        with pytest.raises(SystemExit):
            ladder.main(['fit', 'qwen35-9b', '--spec', str(path), '--out', str(tmp_path / 'out')])
    assert not (tmp_path / 'out').exists()


def test_refuse_other_om_container():
    with mock.patch.object(subprocess, 'run', return_value=mock.Mock(stdout='owned-elsewhere\n')) as run:
        with pytest.raises(RuntimeError, match='om=1'):
            ladder.refuse_busy()
    assert '--filter' in run.call_args.args[0] and 'label=om=1' in run.call_args.args[0]


def test_step_resume_identity(tmp_path):
    args = argparse.Namespace(out=tmp_path, model_id='qwen35-9b', step='speed', new_run=False)
    identity = dict(model='SHAPE_ONLY', config=MODEL['configs'][0])
    first, key = ladder.step_directory(args, identity)
    assert ladder.step_directory(args, identity) == (first, key)
    different, _ = ladder.step_directory(args, dict(identity, image='other'))
    assert different != first
    args.new_run = True
    assert ladder.step_directory(args, identity)[0] != first


def test_chosen_config_does_not_reuse_other_image_or_model(tmp_path):
    args = argparse.Namespace(out=tmp_path, model_id='qwen35-9b', config=None, dry_run=False)
    identity = dict(path='SHAPE_ONLY', size=1)
    with mock.patch.object(ladder, 'model_identity', return_value=identity):
        inputs = dict(ladder.identity_inputs(MODEL, 'same'), step='fit')
        ladder.write(tmp_path / args.model_id / 'fit-captured/summary.json', dict(
            complete=True, chosen=MODEL['configs'][1], image='same', model_identity=identity,
            identity_inputs=inputs, identity_key=ladder.fingerprint(inputs)))
        assert ladder.chosen_config(args, MODEL, 'same') == MODEL['configs'][1]
        with pytest.raises(ValueError, match='fit'):
            ladder.chosen_config(args, MODEL, 'changed')
        with mock.patch.object(ladder, 'model_identity', return_value=dict(identity, size=2)):
            with pytest.raises(ValueError, match='fit'):
                ladder.chosen_config(args, MODEL, 'same')


def test_report_from_captured_summaries_and_accuracy_protocol(tmp_path):
    config = MOE['configs'][0]
    captured = fit_ok()
    accuracy = protocol_accuracy()
    records = [record('fit', complete=True, trials=[dict(config=config, fits=True, metrics=captured)]),
               record('speed', config, complete=True, metrics=dict(
                   **{k: captured[k] for k in ladder.METRICS}, tok_s=1000 / captured['eval_ms_per_token'])),
               record('acc', config, complete=True, accuracy=accuracy, dataset_set='protocol-only',
                      dataset_sha256=common.sha256(FIXTURES / 'server-chat-logprobs.granite.json'),
                      template_kwargs_used={})]
    for i, entry in enumerate(records):
        ladder.write(tmp_path / entry['model_id'] / f'{entry["step"]}-{i}/summary.json', entry)
    loaded = ladder.load_records(tmp_path)
    rows = ladder.aggregate(loaded, SPEC)
    assert len(rows) == 1
    row = rows[0]
    assert row['fits'] and row['acc'] == accuracy['acc'] and row['n'] == accuracy['n']
    assert row['wilson95_lo'] == accuracy['wilson95_lo']
    assert row['eval_ms_per_token'] == captured['eval_ms_per_token']
    assert row['nvme_gib_per_token_decode'] == captured['nvme_gib_per_token_decode']
    assert row['energy_j_per_token'] == captured['energy_j_per_token']
    assert row['pareto']
    markdown = ladder.render(rows)
    assert '| Pareto |' in markdown and 'Wilson' in markdown and 'Published' in markdown


def test_fixture_report_never_claims_mmlu_or_frontier(tmp_path, capsys):
    records = ladder.fixture_records(SPEC)
    rows = ladder.aggregate(records, SPEC)
    assert rows and not any(r['pareto'] for r in rows)
    assert all('acc' not in r for r in rows)
    granite = next(r for r in rows if r['model_id'] == 'captured-granite')
    assert granite['kld']['mean_kld'] == ladder.kld.parse(
        (FIXTURES / 'perplexity-kld.granite-Q8_0.txt').read_text())['mean_kld']
    with mock.patch.object(subprocess, 'run', side_effect=AssertionError('Docker forbidden')):
        ladder.main(['report', '--dry-run', '--out', str(tmp_path / 'absent')])
    assert '| Pareto |' in capsys.readouterr().out
    assert not (tmp_path / 'absent').exists()


def test_report_writes_only_requested_output(tmp_path, capsys):
    ladder.main(['report', '--fixtures', '--out', str(tmp_path)])
    assert (tmp_path / 'REPORT.md').exists() and (tmp_path / 'ladder.json').exists()
    assert ladder.read(tmp_path / 'ladder.json')['rows']


def test_incomplete_results_and_mismatched_protocols_are_not_frontier():
    config = MOE['configs'][0]
    data = fit_ok()
    accuracy = protocol_accuracy()
    speed = record('speed', config, complete=True, metrics=dict(tok_s=data['eval_tps']))
    acc = record('acc', config, complete=False, accuracy=accuracy,
                 dataset_set='protocol-only', dataset_sha256='captured', template_kwargs_used={})
    rows = ladder.aggregate([speed, acc], SPEC)
    assert not rows[0]['pareto']
    # Geometry uses only captured response-derived accuracy and real throughput.
    point = dict(tok_s=data['eval_tps'], acc=accuracy['acc'], accuracy_complete=True,
                 speed_complete=True, accuracy_group='same')
    equal = dict(point)
    assert ladder.pareto([point, equal]) == [0, 1]
    faster = dict(point, tok_s=1000 / data['eval_ms_per_token'])
    assert ladder.pareto([point, faster]) == [0]  # printed 1.40 vs precise 1.39559...
    separate = dict(faster, accuracy_group='different-subset')
    assert ladder.pareto([point, separate]) == [0, 1]
    assert ladder.pareto([dict(point, fixture=True), dict(point, acc=None)]) == []


def test_same_tokenizer_and_3gb_bound():
    candidate = SPEC['models']['qwen35-4b-q4']
    reference = SPEC['models']['qwen35-4b-q8']
    count = ladder.capped_chunks(candidate, reference, requested=999)
    worst_size = count * 512 * (2 * candidate['vocab_size_upper_bound'] + 64) + 64 * 1024**2
    assert worst_size <= 3_000_000_000
    assert count >= 1
    with pytest.raises(ValueError, match='same-tokenizer'):
        ladder.capped_chunks(candidate, MODEL, 8)
    with pytest.raises(ValueError):
        ladder.capped_chunks(candidate, reference, 0)


def gguf_metadata(path, token):
    def text(value):
        raw = value.encode()
        return struct.pack('<Q', len(raw)) + raw
    # SHAPE_ONLY synthetic metadata, not model tensors or benchmark numbers.
    path.write_bytes(b'GGUF' + struct.pack('<IQQ', 3, 0, 2)
                     + text('tokenizer.ggml.model') + struct.pack('<I', 8) + text('test')
                     + text('tokenizer.ggml.tokens') + struct.pack('<IIQ', 9, 8, 1) + text(token))


def test_kld_checks_actual_gguf_tokenizer(tmp_path):
    left, right = tmp_path / 'left.gguf', tmp_path / 'right.gguf'
    gguf_metadata(left, 'A')
    gguf_metadata(right, 'A')
    assert ladder.tokenizer_identity(left) == ladder.tokenizer_identity(right)
    a = dict(MODEL, path=str(left))
    b = dict(MODEL, path=str(right))
    assert ladder.assert_tokenizers(a, b)['vocab_size'] == 1
    gguf_metadata(right, 'B')
    with pytest.raises(ValueError, match='tokenizer'):
        ladder.assert_tokenizers(a, b)
    left.write_bytes(b'not a model')
    with pytest.raises(ValueError):
        ladder.tokenizer_identity(left)


def test_kld_context_override_avoids_duplicate_context(tmp_path):
    args = argparse.Namespace(text=tmp_path / 'corpus', max_seconds=7200)
    command = ladder.kl_command(args, SPEC['image'], MODEL, MODEL['configs'][0],
                                tmp_path / 'out', tmp_path / 'base.bin', 8, 'candidate')
    assert '--engine-arg=-c' not in command
    assert '--engine-arg=-ngl' in command
    assert command[2] == 'candidate'


def test_equal_category_fetch_is_seeded_and_nested():
    # SHAPE_ONLY rows, not dataset observations. Sampling is not proportional.
    rows = [dict(id=f'{category}:{i}', meta=dict(category=str(category)))
            for category in range(14) for i in range(10 + category)]
    full = fetch_sets.balanced_subset(rows, 10)
    half = fetch_sets.balanced_subset(rows, 5)
    assert len(full) == 140 and len(half) == 70
    assert {r['id'] for r in half} <= {r['id'] for r in full}
    assert full == fetch_sets.balanced_subset(list(reversed(rows)), 10)
    assert len({r['id'] for r in full}) == 140
    assert all(sum(r['meta']['category'] == str(c) for r in full) == 10 for c in range(14))
    with pytest.raises(ValueError, match='14'):
        fetch_sets.balanced_subset([], 5)


def test_deadline_terminates_process_group_for_cleanup():
    process = mock.Mock(pid=99999)
    process.wait.side_effect = [subprocess.TimeoutExpired('mock', 1), 0]
    with mock.patch.object(subprocess, 'Popen', return_value=process), \
         mock.patch.object(ladder.os, 'killpg') as killpg:
        with pytest.raises(subprocess.TimeoutExpired):
            ladder.bounded(['mock'], ladder.time.monotonic() + 1)
        killpg.assert_called_once_with(process.pid, ladder.signal.SIGTERM)


def test_acc_dry_run_uses_server_harness_and_records_policy(tmp_path, capsys):
    ladder.main(['acc', 'qwen35-9b', '--dry-run', '--out', str(tmp_path)])
    out = capsys.readouterr().out
    assert 'cc/acc/mcq.py' in out and '--runtime nvidia' in out
    assert 'mmlu_pro_140.jsonl' in out and '--engine-arg=--reasoning' in out
    ladder.main(['acc', 'qwen36-35b-a3b', '--dry-run', '--out', str(tmp_path)])
    assert 'mmlu_pro_70.jsonl' in capsys.readouterr().out


def test_config_can_join_both_subset_frontiers():
    data, accuracy = fit_ok(), protocol_accuracy()
    config = MOE['configs'][0]
    records = [record('speed', config, complete=True, metrics=dict(tok_s=data['eval_tps']))]
    for name, source in [('protocol-full', 'captured-full-hash'), ('protocol-half', 'captured-half-hash')]:
        records.append(record('acc', config, complete=True, accuracy=accuracy,
            dataset_set=name, dataset_sha256=source, template_kwargs_used={}))
    rows = ladder.aggregate(records, SPEC)
    assert len(rows) == 1
    assert rows[0]['pareto_sets'] == ['protocol-full', 'protocol-half']
    assert {v['set'] for v in ladder.accuracy_views(rows)} == set(rows[0]['pareto_sets'])
    assert 'protocol-full' in ladder.render(rows) and 'protocol-half' in ladder.render(rows)


def test_speed_resume_skips_successful_slots(tmp_path):
    args = argparse.Namespace()
    captured = fit_ok()
    # The fixture is a 64-token legacy capture; lower the *test-only* completion
    # check to its real length rather than invent 256-token performance data.
    real_check = ladder.successful
    def check_fixture(summary, tokens):
        return real_check(summary, 32)
    with mock.patch.object(ladder, 'probe_om', return_value=captured) as probe, \
         mock.patch.object(ladder, 'successful', side_effect=check_fixture):
        first = ladder.speed_step(args, MOE, SPEC['image'], MOE['configs'][0], tmp_path,
                                  ladder.time.monotonic() + 7200)
        assert probe.call_count == 6
        assert first['metrics']['eval_ms_per_token'] == captured['eval_ms_per_token']
        probe.reset_mock()
        second = ladder.speed_step(args, MOE, SPEC['image'], MOE['configs'][0], tmp_path,
                                   ladder.time.monotonic() + 7200)
        probe.assert_not_called()
        assert second['metrics'] == first['metrics']
        assert second['coverage']['energy_j_per_token'] == len(ladder.SPEED_ORDER)


def test_watchdog_trip_survived_keeps_accuracy_unscored(tmp_path):
    ladder.write(tmp_path / 'session/result.json', dict(container_state='{}', cleanup_errors=[]))
    (tmp_path / 'session/watchdog.jsonl').write_text('{"event":"watchdog_trip"}\n')
    (tmp_path / 'session/stderr.txt').write_text('')
    with pytest.raises(ValueError, match='watchdog tripped'):
        ladder.validate_sessions(tmp_path, MODEL['configs'][0])


def test_report_separates_changed_image_identity():
    data = fit_ok()
    config = MOE['configs'][0]
    first = record('speed', config, complete=True, metrics=dict(tok_s=data['eval_tps']))
    second = dict(first, image='different captured image')
    assert len(ladder.aggregate([first, second], SPEC)) == 2


def test_om_prefix_preserves_prompt_and_json_as_single_arguments(tmp_path):
    import shlex
    argv = ladder.completion_argv(MODEL, MODEL['configs'][0],
                                  (ladder.HERE / 'prompts/chat.txt').read_text().strip(), 32)
    with mock.patch.dict(ladder.os.environ, {'DOCKER_CMD_PREFIX': 'sg docker -c'}):
        command = ladder.om_command(tmp_path, SPEC['image'], MODEL['configs'][0], argv)
    # Emulate only om_run's shell-string transport; do not run sg or Docker.
    assert shlex.split(' '.join(command[6:])) == argv
    with mock.patch.dict(ladder.os.environ, {'DOCKER_CMD_PREFIX': ''}):
        assert ladder.om_command(tmp_path, SPEC['image'], MODEL['configs'][0], argv)[6:] == argv
