#!/usr/bin/env bash
# Run inside campaign/cuda-tools:c1 with /src read-only and /build writable.
# No downloads, source edits, Docker invocations, or GPU builds.
set -eu
export PYTHONDONTWRITEBYTECODE=1
exec python3 - <<'PY'
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

SOURCE = Path('/src')
BUILD = Path('/build')
PIN = 'ffa80d73133ea5c548aefe57889a16e8085a938f'
SELECTED = [
    ('exchange_storage_test', 'standalone'),
    ('routed_stream_test', 'cmake'),
    ('expert_multi_test', 'standalone'),
    ('gguf_geometry_test', 'cmake'),
    ('model_family_test', 'cmake'),
]
BUILD.mkdir(parents=True, exist_ok=True)
LOGS = BUILD / 'logs'
LOGS.mkdir(exist_ok=True)
TMP = BUILD / 'tmp'
TMP.mkdir(exist_ok=True)
STANDALONE = BUILD / 'standalone'
STANDALONE.mkdir(exist_ok=True)
os.chdir(BUILD)
# Avoid inherited Strata switches changing the selected default tests. All scratch
# files, including the synthetic GGUF fixtures, belong on the writable mount.
env = dict(os.environ)
for key in list(env):
    if key.startswith('STRATA_') or key in ('CXXFLAGS', 'CPPFLAGS', 'LDFLAGS',
                                            'CMAKE_TOOLCHAIN_FILE', 'CMAKE_GENERATOR',
                                            'CMAKE_PROJECT_TOP_LEVEL_INCLUDES'):
        del env[key]
env.update(TMPDIR=str(TMP), TMP=str(TMP), TEMP=str(TMP),
           CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='void',
           PYTHONDONTWRITEBYTECODE='1', LC_ALL='C')
report = {
    'source': str(SOURCE), 'expected_revision': PIN,
    'validation': 'Executed by this runner; see individual statuses.',
    'toolchain': {}, 'configure': None, 'tests': [],
}

def save():
    temporary = BUILD / 'results.json.tmp'
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(BUILD / 'results.json')


def run(label, command, timeout):
    """Log both streams; kill the entire command group on timeout, then continue."""
    path = LOGS / (label + '.log')
    start = time.monotonic()
    print('+ ' + shlex.join(command), flush=True)
    timed_out = False
    code = 127
    with path.open('wb') as output:
        try:
            process = subprocess.Popen(command, cwd=BUILD, env=env, stdout=output,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                code = 124
                output.write(('\nRunner timeout after %.3f seconds\n' % timeout).encode())
        except OSError as error:
            output.write((str(error) + '\n').encode())
    with path.open('rb') as output:
        output.seek(0, os.SEEK_END)
        output.seek(max(0, output.tell() - 8192))
        tail = output.read().decode('utf-8', errors='replace')
    result = dict(command=command, exit_code=code,
                  seconds=round(time.monotonic() - start, 6),
                  stdout_tail=tail, log=str(path), timed_out=timed_out)
    print('%s: exit %d (%.3fs)' % (label, code, result['seconds']), flush=True)
    return result


save()
for name, command in [
    ('cmake', ['cmake', '--version']), ('g++', ['g++', '--version']),
    ('ninja', ['ninja', '--version']), ('git', ['git', '--version']),
    ('python3', ['python3', '--version']), ('nvcc', ['nvcc', '--version']),
    ('machine', ['uname', '-a']),
]:
    report['toolchain'][name] = run('version-' + name.replace('+', 'p'), command, 15)
report['revision_check'] = run('revision',
    ['git', '-c', 'safe.directory=/src', '-C', '/src', 'rev-parse', 'HEAD'], 15)
revision = report['revision_check']['stdout_tail'].strip()
report['actual_revision'] = revision
if report['revision_check']['exit_code'] != 0 or revision != PIN:
    report['preflight_error'] = 'Cannot verify the required source revision; no builds attempted.'
    for name, method in SELECTED:
        report['tests'].append(dict(test=name, method=method, built=False,
            exit_code=None, seconds=0.0, stdout_tail=report['preflight_error'],
            status='blocked', build_exit_code=None))
    save()
    sys.exit(2)

# Native experts default ON and would fetch llama.cpp. OFF removes that branch.
# FULLY_DISCONNECTED is extra defense; the Docker recipe also has --network none.
configure = [
    'cmake', '-S', '/src', '-B', '/build', '-G', 'Ninja',
    '-DCMAKE_CXX_COMPILER=g++', '-DCMAKE_BUILD_TYPE=Release',
    '-DCMAKE_CXX_FLAGS=', '-DCMAKE_CXX_FLAGS_RELEASE=-O2 -UNDEBUG',
    '-DSTRATA_BUILD_TESTS=ON', '-DSTRATA_ENABLE_CUDA=OFF',
    '-DSTRATA_ENABLE_HIP=OFF', '-DSTRATA_HIP_GFX906=OFF',
    '-DSTRATA_ENABLE_SYCL=OFF', '-DSTRATA_NATIVE_EXPERTS=OFF',
    '-DSTRATA_PREFILL_MMQ=OFF', '-DSTRATA_ORCA_Q4KS_MMQ=OFF',
    '-DSTRATA_BUILD_CONVERSATION_TESTS=OFF', '-DSTRATA_BUILD_DSV41=OFF',
    '-DSTRATA_BUILD_SPLIT=OFF', '-DSTRATA_WERROR=OFF',
    '-DFETCHCONTENT_FULLY_DISCONNECTED=ON',
    '-DFETCHCONTENT_UPDATES_DISCONNECTED=ON',
]
report['configure'] = run('configure', configure, 120)
save()

for name, method in SELECTED:
    started = time.monotonic()
    entry = dict(test=name, method=method, built=False, exit_code=None,
                 seconds=0.0, stdout_tail='', status='build_failed', build_exit_code=None)
    if method == 'cmake' and report['configure']['exit_code'] != 0:
        entry.update(status='configure_failed',
                     stdout_tail=report['configure']['stdout_tail'])
    else:
        common = ['g++', '-std=c++20', '-O2', '-UNDEBUG', '-pthread', '-I/src/include']
        if method == 'standalone':
            executable = STANDALONE / name
            if name == 'exchange_storage_test':
                # Only the header-defined ExchangeStorage implementation is needed.
                commands = [common + ['/src/tests/core/exchange_storage_test.cpp',
                                      '-o', str(executable)]]
            else:
                # The native-OFF CMake library omits q2_avx1.cpp even though
                # expert_layout.cpp references it. Compile the unchanged CPU
                # sources directly, retaining each kernel's own ISA flags.
                sources = [
                    ('expert_multi_test.cpp', []),
                    ('expert.cpp', ['-mavx512f', '-mavx512bw', '-mavx512vl',
                                    '-mavx512dq', '-mavx512vnni', '-mavx512vbmi',
                                    '-mf16c', '-mfma']),
                    ('expert_layout.cpp', []),
                    ('q2_avx2.cpp', ['-mavx2', '-mfma', '-mf16c', '-DSTRATA_AVXVNNI=0']),
                    ('q2_avx1.cpp', ['-msse4.2', '-mavx', '-mno-avx2',
                                    '-mno-fma', '-mno-f16c']),
                    ('hadamard.cpp', []),
                ]
                objects = []
                commands = []
                for source, flags in sources:
                    obj = STANDALONE / (source + '.o')
                    objects.append(str(obj))
                    commands.append(common + flags + ['-c',
                        '/src/src/kernels/cpu/' + source, '-o', str(obj)])
                commands.append(['g++', '-pthread'] + objects + ['-o', str(executable)])
        else:
            executable = BUILD / name
            commands = [['cmake', '--build', '/build', '--target', name, '--parallel', '8']]
        build_started = time.monotonic()
        steps = []
        for index, command in enumerate(commands):
            remaining = max(0.001, 180 - (time.monotonic() - build_started))
            step = run('build-' + name + '-' + str(index), command, remaining)
            steps.append(step)
            if step['exit_code'] != 0:
                break
        build_result = dict(steps=steps, exit_code=steps[-1]['exit_code'],
                            seconds=round(time.monotonic() - build_started, 6),
                            stdout_tail=steps[-1]['stdout_tail'])
        entry['build'] = build_result
        entry['build_exit_code'] = build_result['exit_code']
        entry['built'] = build_result['exit_code'] == 0 and executable.is_file()
        if entry['built']:
            # No --bench, --real, --rotation-gpu, or CTest-wide invocation.
            result = run('run-' + name, [str(executable)], 60)
            entry.update(exit_code=result['exit_code'], stdout_tail=result['stdout_tail'],
                         run_seconds=result['seconds'], run=result)
            if result['exit_code'] == 0:
                entry['status'] = 'passed'
            elif result['exit_code'] == 77 and name == 'expert_multi_test':
                entry['status'] = 'skipped_isa'
            else:
                entry['status'] = 'failed'
        else:
            entry['stdout_tail'] = build_result['stdout_tail']
            if build_result['exit_code'] == 0:
                entry['stdout_tail'] += '\nExpected executable is missing: ' + str(executable)
    entry['seconds'] = round(time.monotonic() - started, 6)
    report['tests'].append(entry)
    save()

statuses = [item['status'] for item in report['tests']]
report['summary'] = {status: statuses.count(status) for status in sorted(set(statuses))}
save()
print(json.dumps(report['summary']), flush=True)
print('Results: /build/results.json; complete logs: /build/logs/', flush=True)
# A skip is explicitly distinguishable from a complete pass.
if any(status not in ('passed', 'skipped_isa') for status in statuses):
    sys.exit(1)
sys.exit(3 if 'skipped_isa' in statuses else 0)
PY
