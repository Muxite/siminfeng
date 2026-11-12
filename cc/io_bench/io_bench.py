#!/usr/bin/env python3
"""Linux read-only, cold-cache expert I/O microbenchmark (stdlib only)."""
import argparse
import concurrent.futures
import ctypes
import json
import math
import mmap
import os
from pathlib import Path
import random
import struct
import subprocess
import threading
import time

DEFAULT_MODEL = '/home/muk/models/campaign/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf'
# GGML type -> (elements per block, bytes per block); no tensor payload is loaded.
QUANTS = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22),
    7: (32, 24), 8: (32, 34), 9: (32, 36), 10: (256, 84), 11: (256, 110),
    12: (256, 144), 13: (256, 176), 14: (256, 210), 15: (256, 292),
    16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50),
    20: (32, 18), 21: (256, 110), 22: (256, 82), 23: (256, 136),
    24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8),
    29: (256, 56), 30: (1, 2), 34: (256, 54), 35: (256, 66),
    39: (32, 17), 40: (64, 36), 41: (128, 18), 42: (64, 18), 43: (1, 1),
}


def size_bytes(value):
    value = value.upper().removesuffix('IB').removesuffix('B')
    factor = {'K': 1024, 'M': 1024**2, 'G': 1024**3}
    n = float(value[:-1]) * factor[value[-1]] if value[-1:] in factor else float(value)
    if not math.isfinite(n) or n < 1:
        raise argparse.ArgumentTypeError('size must be positive')
    return int(n)


class GGUFReader:
    def __init__(self, file):
        self.file = file
        self.end = os.fstat(file.fileno()).st_size

    def read(self, n):
        if n < 0 or n > self.end - self.file.tell():
            raise ValueError('truncated GGUF')
        data = self.file.read(n)
        if len(data) != n:
            raise ValueError('truncated GGUF')
        return data

    def number(self, fmt):
        return struct.unpack('<' + fmt, self.read(struct.calcsize('<' + fmt)))[0]

    def string(self):
        n = self.number('Q')
        if n > 16 * 1024 * 1024:
            raise ValueError('GGUF string too large')
        return self.read(n).decode('utf-8')

    def value(self, kind, depth=0):
        formats = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i',
                   6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
        if kind in formats:
            return self.number(formats[kind])
        if kind == 8:
            return self.string()
        if kind == 9 and depth < 8:
            subtype, count = self.number('I'), self.number('Q')
            if count > self.end - self.file.tell():
                raise ValueError('invalid GGUF array length')
            for _ in range(count):
                self.value(subtype, depth + 1)
            return None
        raise ValueError('unsupported GGUF metadata type/depth')


def extract_ranges(path):
    """Return absolute file offsets, exact tensor sizes and nb[2] expert slices."""
    with open(path, 'rb') as file:
        r = GGUFReader(file)
        if r.read(4) != b'GGUF' or r.number('I') not in (2, 3):
            raise ValueError('expected little-endian GGUF v2/v3')
        tensors, metadata = r.number('Q'), r.number('Q')
        if tensors > r.end // 24 or metadata > r.end // 12:
            raise ValueError('invalid GGUF counts')
        alignment = 32
        for _ in range(metadata):
            key = r.string()
            value = r.value(r.number('I'))
            if key == 'general.alignment':
                alignment = value
        if not isinstance(alignment, int) or alignment < 1 or alignment > 1024 * 1024:
            raise ValueError('invalid GGUF alignment')
        found = []
        for _ in range(tensors):
            name, ndim = r.string(), r.number('I')
            if not 1 <= ndim <= 4:
                raise ValueError('invalid GGUF dimensions')
            dims = [r.number('Q') for _ in range(ndim)]
            kind, offset = r.number('I'), r.number('Q')
            if '_exps' not in name:
                continue
            if ndim != 3 or any(d < 1 for d in dims) or kind not in QUANTS:
                raise ValueError('unsupported expert tensor: ' + name)
            block, width = QUANTS[kind]
            if dims[0] % block:
                raise ValueError('unaligned quantized row: ' + name)
            stride = dims[0] // block * width * dims[1]
            found.append(dict(name=name, offset=offset, size=stride * dims[2],
                              nb2=stride, experts=dims[2], type=kind, shape=dims))
        base = (file.tell() + alignment - 1) // alignment * alignment
        slices = []
        for tensor in found:
            tensor['offset'] += base
            if tensor['offset'] + tensor['size'] > r.end:
                raise ValueError('expert tensor exceeds file')
            slices.extend([tensor['offset'] + i * tensor['nb2'], tensor['nb2']]
                          for i in range(tensor['experts']))
        if not slices:
            raise ValueError('no *_exps tensors; use --raw-file for a non-GGUF fixture')
        return dict(schema=1, model=str(Path(path).resolve()), tensors=found, slices=slices)


def load_ranges(path, file_size):
    data = json.loads(Path(path).read_text())
    ranges = data['slices'] if isinstance(data, dict) else data
    if not ranges:
        raise ValueError('empty ranges')
    for entry in ranges:
        if (not isinstance(entry, list) or len(entry) != 2 or
                any(type(n) is not int for n in entry) or entry[0] < 0 or entry[1] <= 0 or
                entry[0] + entry[1] > file_size):
            raise ValueError('ranges must contain in-file [offset, size] pairs')
    return ranges


def make_plan(ranges, request_size, total_bytes, pattern, seed=1, direct=False):
    rng = random.Random(seed)
    spans = list(ranges)
    requests = []
    left = total_bytes
    while left > 0:
        if pattern == 'random-expert':
            rng.shuffle(spans)
        progressed = False
        for offset, length in spans:
            if direct:
                end = (offset + length) // 4096 * 4096
                offset = (offset + 4095) // 4096 * 4096
                length = end - offset
            chunk = request_size // 4096 * 4096 if direct else request_size
            if chunk < 1:
                raise ValueError('direct I/O requires requests >= 4096')
            while length > 0 and left > 0:
                n = min(chunk, length, left)
                if direct:
                    n = n // 4096 * 4096
                if not n:
                    break
                requests.append((offset, n))
                offset, length, left = offset + n, length - n, left - n
                progressed = True
            if left <= 0:
                break
        if not progressed:
            break
    if not requests:
        raise ValueError('no readable ranges after alignment')
    return requests


def evict(fd, requests):
    page = mmap.PAGESIZE
    spans = sorted((off // page * page, (off + n + page - 1) // page * page)
                   for off, n in requests)
    start, end = spans[0]
    for lo, hi in spans[1:]:
        if lo <= end:
            end = max(end, hi)
        else:
            os.posix_fadvise(fd, start, end - start, os.POSIX_FADV_DONTNEED)
            start, end = lo, hi
    os.posix_fadvise(fd, start, end - start, os.POSIX_FADV_DONTNEED)


def residency(fd, requests, limit=256):
    page = mmap.PAGESIZE
    pages = sorted({off // page * page for off, _ in requests} |
                   {(off + n - 1) // page * page for off, n in requests})
    pages = pages[::max(1, math.ceil(len(pages) / limit))]
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    resident = 0
    for offset in pages:
        length = min(page, os.fstat(fd).st_size - offset)
        with mmap.mmap(fd, length, flags=mmap.MAP_PRIVATE,
                       prot=mmap.PROT_READ | mmap.PROT_WRITE, offset=offset) as mapping:
            address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
            vec = ctypes.c_ubyte()
            if libc.mincore(address, length, ctypes.byref(vec)):
                raise OSError(ctypes.get_errno(), 'mincore')
            resident += bool(vec.value & 1)
    return dict(sample_pages=len(pages), resident_pages=resident,
                resident_fraction=resident / len(pages))


def percentile(values, percent):
    return sorted(values)[max(0, math.ceil(len(values) * percent / 100) - 1)] if values else None


def python_cell(fd, requests, concurrency):
    groups = [requests[i::concurrency] for i in range(min(concurrency, len(requests)))]
    barrier = threading.Barrier(len(groups) + 1)

    def worker(group):
        latencies = []
        barrier.wait()
        for offset, length in group:
            start = time.monotonic_ns()
            data = os.pread(fd, length, offset)
            latencies.append(time.monotonic_ns() - start)
            if len(data) != length:
                raise OSError('short pread')
        return latencies

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(groups)) as pool:
        futures = [pool.submit(worker, group) for group in groups]
        start = time.monotonic_ns()
        barrier.wait()
        latencies = [n for future in futures for n in future.result()]
        elapsed = time.monotonic_ns() - start
    return dict(bytes_read=sum(n for _, n in requests), elapsed_ns=elapsed,
                mb_s=sum(n for _, n in requests) / elapsed * 1000,
                p50_ns=percentile(latencies, 50), p95_ns=percentile(latencies, 95))


def generate_fio_jobs(model, requests, concurrency, engine, direct, directory):
    """Replay identical explicit offsets, one synchronous-depth stream per worker."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    for old_log in directory.glob('lat*.*.log'):
        old_log.unlink()
    model = str(Path(model).resolve())
    if any(c in model for c in '\n\r\\') or any(c in str(directory) for c in '\n\r'):
        raise ValueError('unsupported path characters for fio replay')
    lines = ['[global]', 'rw=read', 'allow_file_create=0',
             'ioengine=' + engine, 'direct=' + str(int(direct)), 'iodepth=1',
             'thread=1', 'replay_no_stall=1', 'log_avg_msec=0',
             'disable_lat=0', 'disable_clat=0', 'per_job_logs=1']
    for i in range(min(concurrency, len(requests))):
        log = directory / ('worker%d.iolog' % i)
        content = ['fio version 2 iolog', model + ' add', model + ' open']
        content.extend('%s read %d %d' % (model, off, n) for off, n in requests[i::concurrency])
        content.append(model + ' close')
        log.write_text('\n'.join(content) + '\n')
        lines += ['', '[worker%d]' % i, 'read_iolog=' + str(log),
                  'write_lat_log=' + str(directory / ('lat%d' % i))]
    job = directory / 'job.fio'
    job.write_text('\n'.join(lines) + '\n')
    return job


def fio_cell(model, requests, concurrency, engine, direct, directory, timeout):
    job = generate_fio_jobs(model, requests, concurrency, engine, direct, directory)
    output = job.parent / 'fio.json'
    result = subprocess.run(['/usr/bin/fio', str(job), '--readonly', '--output-format=json',
                             '--output=' + str(output)], capture_output=True, text=True, timeout=timeout)
    raw = output.read_text() if output.exists() else ''
    try:
        data = json.loads(raw) if raw else {}
    except ValueError as error:
        raise RuntimeError((result.stderr + raw)[:1500]) from error
    jobs = data.get('jobs', [])
    if result.returncode or not jobs or any(j.get('error') for j in jobs):
        raise RuntimeError((result.stderr or str(data))[:1500])
    latencies = []
    for path in job.parent.glob('lat*_lat.*.log'):
        for line in path.read_text().splitlines():
            latencies.append(int(line.split(',')[1]))
    nbytes = sum(j['read']['io_bytes'] for j in jobs)
    elapsed = max(j['read']['runtime'] for j in jobs) * 1000000
    if nbytes != sum(n for _, n in requests) or len(latencies) != len(requests) or not elapsed:
        raise RuntimeError('incomplete fio replay or missing latency logs')
    return dict(bytes_read=nbytes, elapsed_ns=elapsed, mb_s=nbytes / elapsed * 1000,
                p50_ns=percentile(latencies, 50), p95_ns=percentile(latencies, 95))


def table(cells):
    lines = ['| pattern | engine | direct | request cap | workers | MB/s | p50 ms | p95 ms | resident before/after | status |',
             '|---|---|---:|---:|---:|---:|---:|---:|---|---|']
    for c in cells:
        before, after = c.get('residency_before', {}), c.get('residency_after', {})
        resident = '%s/%s' % (before.get('resident_fraction', '?'), after.get('resident_fraction', '?'))
        lines.append('| %s | %s | %d | %d | %d | %s | %s | %s | %s | %s |' % (
            c['pattern'], c['engine'], c['direct'], c['request_bytes'], c['concurrency'],
            '%.1f' % c['mb_s'] if 'mb_s' in c else '-',
            '%.3f' % (c['p50_ns'] / 1e6) if c.get('p50_ns') is not None else '-',
            '%.3f' % (c['p95_ns'] / 1e6) if c.get('p95_ns') is not None else '-', resident, c['status']))
    candidates = [c['mb_s'] for c in cells if c.get('status') == 'ok' and
                  c['pattern'] == 'random-expert' and not c['direct'] and
                  c['request_bytes'] == 2 * 1024**2 and c['concurrency'] == 16]
    lines += ['', 'ceiling at engine pattern (2 MiB/16 workers buffered): ' +
              ('%.1f MB/s' % max(candidates) if candidates else 'unavailable')]
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', nargs='?', default=DEFAULT_MODEL)
    parser.add_argument('--model', dest='model_option', help='alternative to positional model path')
    parser.add_argument('--ranges', help='JSON slices: [[absolute offset, size], ...], or extracted object')
    parser.add_argument('--extract-ranges', metavar='JSON', help='extract GGUF ranges and exit')
    parser.add_argument('--raw-file', action='store_true', help='use whole file as synthetic expert spans')
    parser.add_argument('--sizes', default=None, help='comma-separated sizes, default 256K,640K,1M,1.83M,2M,4M')
    parser.add_argument('--concurrency', default=None, help='comma-separated worker counts, default 1,2,4,8,16,32')
    parser.add_argument('--pattern', choices=['random-expert', 'sequential', 'both'], default='both')
    parser.add_argument('--engines', default='python,psync,io_uring', help='comma-separated python,psync,io_uring')
    parser.add_argument('--bytes-per-cell', type=size_bytes, default=None, help='default 64M, quick 32M; max 512M')
    parser.add_argument('--output-dir', default='io-bench-results')
    parser.add_argument('--quick', action='store_true', help='2 sizes x 2 worker counts; 75 second budget')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args(argv)
    args.model = args.model_option or args.model
    try:
        output = Path(args.output_dir).resolve()
        file_size = os.stat(args.model).st_size
        if args.extract_ranges:
            Path(args.extract_ranges).write_text(json.dumps(extract_ranges(args.model), indent=2) + '\n')
            return 0
        if args.ranges:
            ranges = load_ranges(args.ranges, file_size)
        elif args.raw_file:
            # Synthetic expert boundaries preserve the request-size sweep on raw fixtures.
            ranges = [[off, min(8 * 1024**2, file_size - off)] for off in range(0, file_size, 8 * 1024**2)]
        else:
            extracted = extract_ranges(args.model)
            output.mkdir(parents=True, exist_ok=True)
            (output / 'ranges.json').write_text(json.dumps(extracted, indent=2) + '\n')
            ranges = extracted['slices']
        if not ranges:
            raise ValueError('empty input file')
        sizes = [size_bytes(s) for s in (args.sizes or ('256K,2M' if args.quick else '256K,640K,1M,1.83M,2M,4M')).split(',')]
        workers = [int(s) for s in (args.concurrency or ('1,16' if args.quick else '1,2,4,8,16,32')).split(',')]
        if any(n < 1 or n > 256 for n in workers):
            raise ValueError('worker counts must be 1..256')
        engines = args.engines.split(',')
        if set(engines) - {'python', 'psync', 'io_uring'}:
            raise ValueError('unknown engine')
        total = args.bytes_per_cell or (32 if args.quick else 64) * 1024**2
        if total > 512 * 1024**2:
            raise ValueError('bytes per cell exceeds 512 MiB')
        patterns = ['random-expert', 'sequential'] if args.pattern == 'both' else [args.pattern]
        output.mkdir(parents=True, exist_ok=True)
        cells, unsupported = [], set()
        started = time.monotonic()
        deadline = started + (75 if args.quick else 570)
        with open(args.model, 'rb') as file:
            for pattern in patterns:
                spans = ranges if pattern == 'random-expert' else [[0, file_size]]
                for size in sizes:
                    for concurrency in workers:
                        for engine in engines:
                            for direct in ([False] if engine == 'python' else [False, True]):
                                cell = dict(pattern=pattern, engine=engine, direct=direct,
                                            request_bytes=size, concurrency=concurrency)
                                cells.append(cell)
                                if (engine, direct) in unsupported:
                                    cell['status'] = 'unsupported (see earlier error)'
                                    continue
                                if time.monotonic() > deadline - 3:
                                    cell['status'] = 'skipped (sweep time budget)'
                                    continue
                                try:
                                    requests = make_plan(spans, size, total, pattern, args.seed, direct)
                                    cell.update(requests=len(requests), actual_workers=min(concurrency, len(requests)), planned_bytes=sum(n for _, n in requests),
                                                actual_request_min=min(n for _, n in requests),
                                                actual_request_max=max(n for _, n in requests))
                                    evict(file.fileno(), requests)
                                    cell['residency_before'] = residency(file.fileno(), requests)
                                    if engine == 'python':
                                        cell.update(python_cell(file.fileno(), requests, concurrency))
                                    else:
                                        cell.update(fio_cell(args.model, requests, concurrency, engine, direct,
                                                             output / ('cell-%03d' % len(cells)),
                                                             min(15, max(1, deadline - time.monotonic()))))
                                    cell['residency_after'] = residency(file.fileno(), requests)
                                    cell['status'] = 'ok'
                                except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
                                    cell.update(status='error', error=str(error))
                                    if engine != 'python':
                                        unsupported.add((engine, direct))
                                # Preserve partial evidence if the operator interrupts the sweep.
                                (output / 'results.json').write_text(json.dumps(dict(model=str(Path(args.model).resolve()),
                                    args=vars(args), elapsed_s=time.monotonic() - started, cells=cells), indent=2) + '\n')
        report = table(cells)
        (output / 'results.json').write_text(json.dumps(dict(model=str(Path(args.model).resolve()),
            args=vars(args), elapsed_s=time.monotonic() - started, cells=cells), indent=2) + '\n')
        (output / 'results.md').write_text(report + '\n')
        print(report)
        print('Results: ' + str(output / 'results.json'))
        return 0 if any(c['status'] == 'ok' for c in cells) else 1
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
