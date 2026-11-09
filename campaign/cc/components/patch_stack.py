#!/usr/bin/env python3
"""Inspect patch layers and exhaustively test application order on a git archive.

Only the caller-owned scratch/output paths are written; source is read-only.
Tree IDs are Git tree hashes (paths, contents, executable bits and symlinks).
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def layer(path):
    if Path(path).name == 'CMakeLists.txt' or path.endswith('.cmake') or path.startswith('cmake/'):
        return 'build'
    for prefix, name in [('ggml/src/ggml-cuda/', 'backend-cuda'),
                         ('ggml/src/ggml-cpu/', 'backend-cpu'),
                         ('src/', 'libllama'), ('common/', 'libllama-common'),
                         ('tools/', 'tool')]:
        if path.startswith(prefix):
            return name
    if re.match(r'^ggml/src/(ggml-backend[^/]*\.cpp|ggml\.c|ggml-alloc[^/]*)$', path):
        return 'ggml-core'
    return 'unmapped'


def inspect_patch(path):
    files, current, old = [], None, None
    for line in Path(path).read_text().splitlines():
        if line.startswith('--- '):
            old = line[4:].split('\t')[0]
        elif line.startswith('+++ '):
            new = line[4:].split('\t')[0]
            name = old if new == '/dev/null' else new
            if name.startswith(('a/', 'b/')):
                name = name[2:]
            current = {'path': name, 'old_path': old, 'new_path': new,
                       'layer': layer(name), 'hunks': []}
            files.append(current)
        elif line.startswith('@@ '):
            match = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)', line)
            if current is None or match is None:
                raise ValueError('invalid patch hunk: ' + line)
            a, n, b, m, context = match.groups()
            current['hunks'].append({'old_start': int(a), 'old_count': int(n or 1),
                                     'new_start': int(b), 'new_count': int(m or 1),
                                     'context': context.strip()})
    if not files:
        raise ValueError('no unified file headers: ' + str(path))
    return {'patch': str(path), 'sha256': sha256(path), 'files': files,
            'layers': sorted({f['layer'] for f in files})}


def overlaps(patches):
    result = []
    for a, b in itertools.combinations(patches, 2):
        common = []
        for left in a['files']:
            for right in b['files']:
                if left['path'] != right['path']:
                    continue
                pairs = []
                for i, x in enumerate(left['hunks']):
                    for j, y in enumerate(right['hunks']):
                        # Include insertion points; overlap is advisory, apply is authoritative.
                        if max(x['old_start'], y['old_start']) < min(
                                x['old_start'] + max(1, x['old_count']),
                                y['old_start'] + max(1, y['old_count'])):
                            pairs.append([i, j])
                common.append({'path': left['path'], 'overlapping_hunks': pairs})
        result.append({'patches': [a['patch'], b['patch']], 'shared_files': common,
                       'shared_layers': sorted(set(a['layers']) & set(b['layers']))})
    return result


def git(cwd, *args, check=True):
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull,
               GIT_CEILING_DIRECTORIES=str(Path(cwd).resolve().parent))
    result = subprocess.run(['git', '-C', str(cwd), *args], env=env, capture_output=True)
    if check and result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace'))
    return result


def test_stack(source, commit, patches, scratch, release=None):
    patches = [Path(p).resolve() for p in patches]
    if not patches or len(set(patches)) != len(patches):
        raise ValueError('stack must contain distinct patches')
    metadata = [inspect_patch(p) for p in patches]
    resolved = git(source, 'rev-parse', commit + '^{commit}').stdout.decode().strip()
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    permutations = []
    combined = None
    with tempfile.TemporaryDirectory(prefix='stack-', dir=scratch) as temporary:
        root = Path(temporary)
        archive = root / 'source.tar'
        # Stream archive to disk, without materializing the source in Python memory.
        with archive.open('wb') as out:
            env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
            subprocess.run(['git', '-C', str(Path(source).resolve()), 'archive', resolved],
                           stdout=out, stderr=subprocess.PIPE, env=env, check=True)
        tree = root / 'tree'
        tree.mkdir()
        with tarfile.open(archive) as tar:
            tar.extractall(tree, filter='data')
        git(tree, 'init', '-q')
        git(tree, 'add', '-f', '-A')
        baseline = git(tree, 'write-tree').stdout.decode().strip()
        for order in itertools.permutations(range(len(patches))):
            git(tree, 'read-tree', '--reset', '-u', baseline)
            git(tree, 'clean', '-fdx')
            attempt = {'order': [str(patches[i]) for i in order], 'applies': True,
                       'tree_hash': None, 'conflict': None}
            for i in order:
                applied = git(tree, 'apply', '--index', '--whitespace=nowarn', str(patches[i]), check=False)
                if applied.returncode:
                    attempt.update(applies=False, conflict={
                        'patch': str(patches[i]), 'returncode': applied.returncode,
                        'stderr': applied.stderr.decode(errors='replace')})
                    break
            if attempt['applies']:
                attempt['tree_hash'] = git(tree, 'write-tree').stdout.decode().strip()
                if combined is None:
                    combined = git(tree, 'diff', '--cached', '--binary', '--no-ext-diff', baseline).stdout
            permutations.append(attempt)
    hashes = sorted({p['tree_hash'] for p in permutations if p['applies']})
    independent = all(p['applies'] for p in permutations) and len(hashes) == 1
    report = {'base_commit': resolved, 'base_tree_hash': baseline, 'patches': metadata,
              'pairwise_overlap': overlaps(metadata), 'permutations': permutations,
              'tree_hashes': hashes, 'order_independent': independent, 'release': None}
    if release is not None:
        if not independent:
            raise ValueError('refusing release patch: stack is not order-independent')
        Path(release).write_bytes(combined)
        report['release'] = {'path': str(release), 'sha256': sha256(release)}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('patches', nargs='+')
    parser.add_argument('--source', default='ref/llama.cpp')
    parser.add_argument('--commit', default='31288c9c6')
    parser.add_argument('--scratch', default='artifacts/codexCOMP')
    parser.add_argument('--out', required=True)
    parser.add_argument('--release')
    args = parser.parse_args()
    report = test_stack(args.source, args.commit, args.patches, args.scratch, args.release)
    Path(args.out).write_text(json.dumps(report, indent=2) + '\n')
    return 0 if report['order_independent'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
