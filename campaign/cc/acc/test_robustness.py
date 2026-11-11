"""Offline regression tests for fetching, selection and container ownership."""
from collections import Counter
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
import importlib.machinery
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import urllib.error
from unittest.mock import Mock, patch

import pytest

from cc.acc import datasets, mcq, server


def envelope(offset=0, count=1, total=1):
    # SHAPE_ONLY: documented HF rows envelope and MMLU-Pro columns, no benchmark data.
    return dict(rows=[dict(row_idx=i, row=dict(question=f"question {i}",
                options=["one", "two"], answer="A", category="small" if i < 3 else "large"))
                for i in range(offset, offset + count)], num_rows_total=total)


@pytest.mark.parametrize('mode_flags,mode,n,counts', [
    (['--per-category', '2'], 'per-category', 2, {'small': 2, 'large': 2}),
    (['-n', '2'], 'per-category', 2, {'small': 2, 'large': 2}),
    (['--proportional', '-n', '5'], 'proportional', 5, {'large': 5}),
    (['--proportional', '-n', '101'], 'proportional', 101, {'small': 3, 'large': 98}),
])
def test_selection_cli_and_page_provenance(tmp_path, mode_flags, mode, n, counts):
    calls = []
    def opener(url, timeout):
        calls.append(url)
        offset = 100 if len(calls) == 2 else 0
        return io.BytesIO(json.dumps(envelope(offset, 100 if offset == 0 else 1, 101)).encode())
    raw, pages = datasets.fetch_rows('test', 'default', 'test', opener=opener, page_delay=0)
    with patch.object(sys, 'argv', ['datasets', 'mmlu-pro', *mode_flags, '--outdir', str(tmp_path)]), \
         patch.object(datasets, 'fetch_rows', return_value=(raw, pages)):
        datasets.main()
    output = next(tmp_path.glob('*.jsonl'))
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert Counter(r['meta']['category'] for r in rows) == counts
    meta = json.loads(output.with_suffix('.meta.json').read_text())
    assert meta['mode'] == mode and meta['seed'] == 42
    assert meta['categories'] == counts and meta['n'] == len(rows) and meta['requested_n'] == n
    for row in rows:
        assert row['source'] == pages[row['meta']['row_offset'] // 100]['url']
    # Every row, including page two, must retain its own page URL.
    all_rows = [datasets.normalize('mmlu-pro', i, r, pages[i // 100]['url']) for i, r in raw]
    assert all_rows[-1]['source'] == calls[1]
    assert datasets.subset(all_rows, 2, 42, 'per-category') == datasets.subset(all_rows, 2, 42, 'per-category')
    assert datasets.subset(all_rows, 2, 42, 'per-category') != datasets.subset(all_rows, 2, 43, 'per-category')
    with pytest.raises(ValueError, match='category'):
        datasets.subset(all_rows, 4, 42, 'per-category')


@pytest.mark.parametrize('error', [
    urllib.error.HTTPError('url', 429, 'limited', {'Retry-After': '9'}, None),
    urllib.error.HTTPError('url', 503, 'unavailable', {}, None),
    urllib.error.URLError('offline'),
])
def test_fetch_retry_backoff(error):
    with patch.object(datasets.time, 'sleep') as sleep, \
         patch.object(datasets.random, 'uniform', return_value=.25):
        opener = Mock(
            side_effect=[error, urllib.error.URLError('again'), io.BytesIO(b'ok')])
        assert datasets.fetch_page('url', opener) == b'ok'
    assert opener.call_count == 3
    assert [c.args[0] for c in sleep.call_args_list] == [9 if getattr(error, 'code', None) == 429 else 1.25, 2.25]


def test_retry_limits_permanent_errors_and_retry_after_date():
    with patch.object(datasets.time, 'sleep') as sleep:
        def fail(*args, **kwargs):
            raise urllib.error.URLError('offline')
        with pytest.raises(urllib.error.URLError):
            datasets.fetch_page('url', fail)
        assert sleep.call_count == 7
        sleep.reset_mock()
        def forbidden(*args, **kwargs):
            raise urllib.error.HTTPError('url', 403, 'forbidden', {}, None)
        with pytest.raises(urllib.error.HTTPError):
            datasets.fetch_page('url', forbidden)
        sleep.assert_not_called()
    with patch.object(datasets.time, 'time', return_value=100):
        assert datasets.retry_after({'Retry-After': format_datetime(datetime.fromtimestamp(120, timezone.utc))}) == 20
    assert datasets.retry_after({'Retry-After': 'invalid'}) == 0


def test_polite_page_delay():
    def opener(url, timeout):
        offset = 100 if 'offset=100' in url else 0
        return io.BytesIO(json.dumps(envelope(offset, 100 if offset == 0 else 1, 101)).encode())
    with patch.object(datasets.time, 'sleep') as sleep:
        datasets.fetch_rows('test', 'default', 'test', opener=opener)
        sleep.assert_called_once_with(.5)


def container(cid, job=None, age=7, state='running'):
    return dict(Id=cid, State=dict(Status=state), Config=dict(Labels={'jobq_job': job} if job else {}),
                Created=(datetime.now(timezone.utc) - timedelta(hours=age)).isoformat())


def docker_mock(containers, jobs):
    def run(command, **kwargs):
        if command[0] == str(server.ROOT / 'bin/jobq'):
            state = jobs[command[-1]]
            return subprocess.CompletedProcess(command, 1 if state is None else 0,
                stdout=json.dumps({'state': state}), stderr='jobq: unknown job: missing' if state is None else '')
        assert command[0] == 'docker'
        if command[1] == 'ps':
            stdout = '\n'.join(containers)
        elif command[1] == 'inspect':
            stdout = json.dumps([containers[command[2]]])
        else:
            assert command[1:3] == ['rm', '-f']
            stdout = ''
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr='')
    return run


def test_reap_dry_run_commands_only_and_actual_removal():
    containers = {c['Id']: c for c in [container('active', 'a'), container('done', 'd'),
        container('unknown', 'u', state='exited'), container('old'), container('new', age=1),
        container('created', 'd', state='created')]}
    containers['old']['Created'] = '2000-01-01T00:00:00.123456789Z'
    fake = docker_mock(containers, dict(a='running', d='succeeded', u=None))
    output = io.StringIO()
    with patch.dict('os.environ', {}, clear=True), patch.object(subprocess, 'run', side_effect=fake) as run, redirect_stdout(output):
        commands = server.reap_containers()
        assert output.getvalue().splitlines() == ['docker rm -f done', 'docker rm -f old', 'docker rm -f unknown']
        assert not any(c.args[0][1:3] == ['rm', '-f'] for c in run.call_args_list)
        assert len(commands) == 3
        server.reap_containers(dry_run=False)
        assert sum(c.args[0][1:3] == ['rm', '-f'] for c in run.call_args_list) == 3


def test_reap_cli_and_jobq_reap_defaults():
    containers = {c['Id']: c for c in [container('done', 'd'), container('old')]}
    fake = docker_mock(containers, dict(d='cancelled'))
    loader = importlib.machinery.SourceFileLoader('jobq_reap_test', str(server.ROOT / 'bin/jobq'))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    jobq = importlib.util.module_from_spec(spec)
    loader.exec_module(jobq)
    for argv, main, expected in [(['server', 'reap', '--dry-run'], server.main, ['done', 'old']),
                                 (['jobq', 'reap-containers'], jobq.main, ['done'])]:
        output = io.StringIO()
        with patch.dict('os.environ', {}, clear=True), patch.object(sys, 'argv', argv), \
             patch.object(subprocess, 'run', side_effect=fake) as run, redirect_stdout(output):
            main()
        assert output.getvalue().splitlines() == ['docker rm -f ' + cid for cid in expected]
        assert not any(c.args[0][1:3] == ['rm', '-f'] for c in run.call_args_list)
        if main == jobq.main:
            assert {c.args[0][-1] for c in run.call_args_list if c.args[0][1:2] == ['ps']} == {'label=om=1', 'label=acc=1'}
    with patch.dict('os.environ', {}, clear=True), patch.object(sys, 'argv', ['jobq', 'reap-containers', '--yes']), \
         patch.object(subprocess, 'run', side_effect=fake) as run:
        jobq.main()
    assert [c.args[0] for c in run.call_args_list if c.args[0][1:3] == ['rm', '-f']] == [['docker', 'rm', '-f', 'done']]


def test_container_job_label_and_reap_before_start(tmp_path):
    with patch.dict('os.environ', {'JOBQ_JOB_ID': 'job-abc'}):
        owned = server.OwnedContainer('image', '/tmp/model', tmp_path, 'test', [])
        assert 'jobq_job=job-abc' in owned.command() and 'acc=1' in owned.command()
    with patch.object(server, 'reap_containers', side_effect=RuntimeError('reap first')), \
         patch.object(subprocess, 'run') as run:
        with pytest.raises(RuntimeError, match='reap first'):
            owned.__enter__()
        run.assert_not_called()


def test_mcq_retains_only_no_letter():
    result = mcq.score(dict(id='test', answer='A', choices=['a', 'b']),
                       {'choices': [{'logprobs': {'content': [{'top_logprobs': []}]}}]})
    assert result['no_letter'] is True
    assert 'argmax_letter_found' not in result


def test_reap_preserves_container_on_job_lookup_failure():
    owned = container('owned', 'job-abc')
    fake = docker_mock({'owned': owned}, {})
    def run(command, **kwargs):
        if command[0] == str(server.ROOT / 'bin/jobq'):
            return subprocess.CompletedProcess(command, 1, stdout='', stderr='state unreadable')
        return fake(command, **kwargs)
    with patch.dict('os.environ', {}, clear=True), patch.object(subprocess, 'run', side_effect=run) as calls:
        with pytest.raises(RuntimeError, match='jobq lookup failed'):
            server.reap_containers(dry_run=False)
    assert not any(c.args[0][1:3] == ['rm', '-f'] for c in calls.call_args_list)
