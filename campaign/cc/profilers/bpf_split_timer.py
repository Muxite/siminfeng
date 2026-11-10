"""Generate CPU graph uprobes and parse bpftrace's aggregate output."""
import argparse
import json
import math
from pathlib import Path
import re
import shlex


def positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('duration and tokens/s must be positive and finite')
    return value


def script(pid, library, seconds=5, per_thread=False):
    seconds = positive(seconds)
    if int(pid) <= 0 or not re.fullmatch(r'/[\w/+.\-]+', library):
        raise ValueError('positive host PID and an absolute, literal library path required')
    thread = '\n  @thread_graphs[tid] = count();\n  @thread_tot_us[tid] = sum($us);' if per_thread else ''
    return f'''// PROFW v1: elapsed graph-call time, not summed worker CPU time.
uprobe:{library}:ggml_graph_compute /pid == {int(pid)}/ {{
  @s[tid] = nsecs;
}}
uretprobe:{library}:ggml_graph_compute /pid == {int(pid)} && @s[tid]/ {{
  $us = (nsecs - @s[tid]) / 1000;
  @graphs = count();
  @graph_tot_us = sum($us);
  @graph_us = hist($us);{thread}
  delete(@s[tid]);
}}
interval:ms:{max(1, round(seconds * 1000))} {{ exit(); }}
END {{
  printf("PROFW window_s={max(1, round(seconds * 1000)) / 1000}\\n");
  clear(@s);
}}
'''


def container_command(container, seconds=5, per_thread=False):
    """Emit a shell recipe; do not inspect Docker or invoke sudo here."""
    command = ['python3', str(Path(__file__).resolve()), 'script', '--pid', '"$pid"',
               '--library', '"$library"', '--seconds', str(positive(seconds))]
    if per_thread:
        command.append('--per-thread')
    invocation = ' '.join(x if x in ('"$pid"', '"$library"') else shlex.quote(x) for x in command)
    return f'''set -eu
pid=$(docker inspect -f '{{{{.State.Pid}}}}' {shlex.quote(container)})
[ "$pid" -gt 0 ]
library=
for candidate in /proc/"$pid"/root/app/libggml-cpu.so.*; do
  [ -f "$candidate" ] && [ ! -L "$candidate" ] || continue
  [ -z "$library" ] || {{ echo 'Multiple CPU backend files; select --library explicitly' >&2; exit 1; }}
  library="$candidate"
done
[ -n "$library" ]
{invocation} | sudo bpftrace -
'''


def parse(source, seconds=None, tokens_per_second=None):
    if seconds is None:
        window = re.search(r'^PROFW window_s=([\d.]+)$', source, re.M)
        if not window:
            raise ValueError('legacy captures require explicit --seconds')
        seconds = float(window[1])
    seconds = positive(seconds)
    def scalar(name):
        matches = re.findall(rf'^@{name}:\s*(\d+)\s*$', source, re.M)
        if len(matches) != 1:
            raise ValueError(f'expected one @{name} aggregate')
        return int(matches[0])
    count, total_us = scalar('graphs'), scalar('graph_tot_us')
    hist = []
    section = re.search(r'^@graph_us:\s*\n((?:\[[^\n]*\n?)+)', source, re.M)
    def bound(value):
        match = re.fullmatch(r'(\d+)([KM]?)', value.strip())
        if not match:
            raise ValueError(f'invalid histogram bound: {value}')
        return int(match[1]) * {'': 1, 'K': 1024, 'M': 1024**2}[match[2]]
    if section:
        for label, n in re.findall(r'^\[([^\n]+?)[\)\]]\s+(\d+)', section[1], re.M):
            edges = label.split(',')
            low = bound(edges[0])
            high = bound(edges[1]) if len(edges) == 2 else low + 1
            hist.append(dict(low_us=low, high_us=high, count=int(n)))
    if sum(b['count'] for b in hist) != count:
        raise ValueError('histogram does not account for every completed split')
    threads = {}
    for name, tid, value in re.findall(r'^@thread_(graphs|tot_us)\[(\d+)\]:\s*(\d+)$', source, re.M):
        threads.setdefault(tid, {})[name] = int(value)
    return dict(splits=count, splits_per_token=count / (seconds * positive(tokens_per_second))
                if tokens_per_second is not None else None,
                cpu_time_frac=total_us / (seconds * 1e6), hist=hist,
                window_s=seconds, total_us=total_us, mean_us=total_us / count if count else None,
                threads=threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    for mode in ('script', 'container'):
        child = sub.add_parser(mode)
        child.add_argument('--seconds', type=float, default=5)
        child.add_argument('--per-thread', action='store_true')
        if mode == 'script':
            child.add_argument('--pid', type=int, required=True)
            child.add_argument('--library', required=True)
        else:
            child.add_argument('--container', required=True)
    child = sub.add_parser('parse')
    child.add_argument('input', type=Path)
    child.add_argument('--seconds', type=float)
    child.add_argument('--tokens-per-second', type=float)
    args = parser.parse_args()
    if args.mode == 'parse':
        print(json.dumps(parse(args.input.read_text(), args.seconds, args.tokens_per_second), indent=2))
    elif args.mode == 'script':
        print(script(args.pid, args.library, args.seconds, args.per_thread), end='')
    else:
        print(container_command(args.container, args.seconds, args.per_thread), end='')


if __name__ == '__main__':
    main()
