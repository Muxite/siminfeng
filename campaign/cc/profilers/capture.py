"""Emit Docker argv as JSON; never execute Docker or profiling tools."""
import argparse
import json
from pathlib import Path
import re

try:
    from .ncu_summary import METRICS
except ImportError:
    from ncu_summary import METRICS

TOOLS_IMAGE = 'campaign/cuda-tools:c1'


def host_path(value):
    path = str(Path(value).absolute())
    if ',' in path or '\n' in path:
        raise ValueError('mount paths cannot contain commas or newlines')
    return path


def extract(image, directory, name='profw-extract'):
    if not image or image.startswith('-') or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]*', name):
        raise ValueError('valid image and temporary container name required')
    return [['docker', 'create', '--name', name, '--entrypoint', '/bin/true', image],
            ['docker', 'cp', f'{name}:/app/.', host_path(directory)],
            ['docker', 'rm', name]]


def capture(tool, app, models, output, engine_args, launch_skip=2000, launch_count=600,
            graph_trace='graph'):
    if tool not in ('nsys', 'ncu'):
        raise ValueError('tool must be nsys or ncu')
    if launch_skip < 0 or launch_count <= 0:
        raise ValueError('launch-skip must be nonnegative and launch-count positive')
    if graph_trace not in ('graph', 'node'):
        raise ValueError('graph_trace must be graph or node')
    argv = ['docker', 'run', '--rm', '--label', 'campaign=c1', '--gpus', 'all',
            '--cpus', '4', '--memory', '8g', '--pids-limit', '256']
    if tool == 'ncu':
        argv += ['--cap-add', 'SYS_ADMIN']
    for source, target, readonly in ((app, '/app', True), (models, '/m', True), (output, '/out', False)):
        argv += ['--mount', f'type=bind,src={host_path(source)},dst={target}' + (',readonly' if readonly else '')]
    argv += ['--workdir', '/out', '--env', 'LD_LIBRARY_PATH=/app', TOOLS_IMAGE]
    if tool == 'nsys':
        argv += ['nsys', 'profile', '--trace=cuda,nvtx,osrt', '--sample=none', '--cpuctxsw=none',
                 f'--cuda-graph-trace={graph_trace}', '-o', '/out/capture', '-f', 'true']
    else:
        argv += ['ncu', '--launch-skip', str(launch_skip), '--launch-count', str(launch_count),
                 '--metrics', ','.join(METRICS), '--csv', '--log-file', '/out/ncu.csv']
    return argv + ['/app/llama-bench'] + list(engine_args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    child = sub.add_parser('extract')
    child.add_argument('--image', required=True)
    child.add_argument('--directory', required=True)
    child.add_argument('--name', default='profw-extract')
    for tool in ('nsys', 'ncu'):
        child = sub.add_parser(tool)
        for option in ('app', 'models', 'output'):
            child.add_argument('--' + option, required=True)
        child.add_argument('--launch-skip', type=int, default=2000)
        child.add_argument('--launch-count', type=int, default=600)
        child.add_argument('--graph-trace', choices=('graph', 'node'), default='graph')
        child.add_argument('engine_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.mode == 'extract':
        result = extract(args.image, args.directory, args.name)
    else:
        engine_args = args.engine_args
        if engine_args[:1] == ['--']:
            engine_args = engine_args[1:]
        result = capture(args.mode, args.app, args.models, args.output, engine_args,
                         args.launch_skip, args.launch_count, args.graph_trace)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
