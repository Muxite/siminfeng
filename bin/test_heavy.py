"""Exercise heavy with a temporary queue and fake commands, without builds."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

BIN = Path(__file__).resolve().parent


@pytest.mark.parametrize('rc', [0, 7])
def test_heavy_output_and_exit_code(tmp_path, rc):
    wrapper = tmp_path / 'run_op'
    wrapper.write_text('#!/usr/bin/env python3\nimport os,sys\n'
                       'argv=sys.argv[sys.argv.index("--")+1:]\n'
                       'os.execv(argv[0], argv)\n')
    wrapper.chmod(0o755)
    state = tmp_path / 'queue'
    env = dict(os.environ, JOBQ_DIR=str(state), JOBQ_RUN_OP=str(wrapper))
    command = ('import sys; print("command stdout", sys.argv[1]); '
               'print("command stderr", file=sys.stderr); sys.exit(int(sys.argv[2]))')
    heavy = subprocess.Popen([str(BIN / 'heavy'), '--name', 'fake-build', '--timeout', '5',
                              '--', sys.executable, '-c', command, 'arg with spaces', str(rc)],
                             cwd=tmp_path, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 5
        while not list(state.glob('job-*.json')):
            assert heavy.poll() is None
            assert time.monotonic() < deadline, 'heavy did not submit a job'
            time.sleep(.02)
        subprocess.run([str(BIN / 'jobq'), 'worker', '--once'], env=env,
                       capture_output=True, text=True, check=True, timeout=10)
        stdout, stderr = heavy.communicate(timeout=5)
        assert heavy.returncode == rc, stderr
        record_path, = state.glob('job-*.json')
        record = json.loads(record_path.read_text())
        assert record['class'] == 'build'
        assert record['name'] == 'fake-build' and record['timeout'] == 5
        assert record['cwd'] == str(tmp_path)
        assert record['rc'] == rc
        assert str(record_path.with_suffix('.log')) in stdout
        assert 'command stdout arg with spaces' in stdout
        assert 'command stderr' in stdout
    finally:
        if heavy.poll() is None:
            heavy.terminate()
        heavy.communicate(timeout=5)
