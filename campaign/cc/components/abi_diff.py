#!/usr/bin/env python3
"""Conservative ELF surface compatibility check; never proves C/C++ layout ABI."""
import argparse
import hashlib
import json
import mmap
from pathlib import Path
import struct
import sys

# Reuse T-N3's ELF validation/dependency reader, extending its section traversal
# for SONAME, exports and GNU version definitions without changing T-N3 files.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'build_engine'))
from host_linkage import inspect_elf  # noqa: E402


def inspect_library(path):
    result = inspect_elf(path)
    with Path(path).open('rb') as src:
        digest = hashlib.file_digest(src, 'sha256').hexdigest()
        with mmap.mmap(src.fileno(), 0, access=mmap.ACCESS_READ) as data:
            endian = '<' if data[5] == 1 else '>'
            wide = data[4] == 2

            def unpack(fmt, offset):
                fmt = endian + fmt
                if offset < 0 or offset + struct.calcsize(fmt) > len(data):
                    raise ValueError('ELF structure outside file')
                return struct.unpack_from(fmt, data, offset)

            header = unpack('HHIQQQIHHHHHH' if wide else 'HHIIIIIHHHHHH', 16)
            shoff, shsize, shnum = header[5], header[10], header[11]
            sfmt = 'IIQQQQIIQQ' if wide else 'IIIIIIIIII'
            if shnum == 0:
                shnum = unpack(sfmt, shoff)[5]
            sections = [unpack(sfmt, shoff + i * shsize) for i in range(shnum)]
            for s in sections:
                if s[1] != 8 and s[4] + s[5] > len(data):
                    raise ValueError('ELF section outside file')

            def string(index, offset):
                if index >= len(sections):
                    raise ValueError('invalid string table link')
                s = sections[index]
                if s[1] != 3 or not 0 <= offset < s[5]:
                    raise ValueError('invalid string table offset')
                end = data.find(b'\0', s[4] + offset, s[4] + s[5])
                if end < 0:
                    raise ValueError('unterminated ELF string')
                return data[s[4] + offset:end].decode('utf8')

            versions, versyms, soname = {}, {}, None
            for s in sections:
                if s[1] == 6:
                    for pos in range(s[4], s[4] + s[5], s[9]):
                        tag, value = unpack('qQ' if wide else 'iI', pos)
                        if tag == 0:
                            break
                        if tag == 14:
                            soname = string(s[6], value)
                elif s[1] == 0x6fffffff:  # SHT_GNU_versym, linked to DYNSYM
                    if s[5] % 2:
                        raise ValueError('invalid GNU versym size')
                    versyms[s[6]] = [unpack('H', p)[0] for p in range(s[4], s[4] + s[5], 2)]
                elif s[1] == 0x6ffffffd:  # SHT_GNU_verdef
                    pos, end = s[4], s[4] + s[5]
                    while pos < end:
                        if pos + 20 > end:
                            raise ValueError('truncated GNU verdef')
                        ver, flags, index, count, _, aux, nxt = unpack('HHHHIII', pos)
                        if ver != 1 or not count or aux < 20 or pos + aux + 8 > end:
                            raise ValueError('invalid GNU verdef')
                        name, _ = unpack('II', pos + aux)
                        versions[index] = string(s[6], name)
                        if not nxt:
                            break
                        if nxt < 20:
                            raise ValueError('invalid GNU verdef chain')
                        pos += nxt
            exports = []
            for index, s in enumerate(sections):
                if s[1] != 11:
                    continue
                values = versyms.get(index)
                if values is not None and len(values) != s[5] // s[9]:
                    raise ValueError('GNU versym/dynsym count mismatch')
                for number, pos in enumerate(range(s[4], s[4] + s[5], s[9])):
                    row = unpack('IBBHQQ' if wide else 'IIIBBH', pos)
                    name, info, other, defined, size = ((row[0], row[1], row[2], row[3], row[5]) if wide
                                                     else (row[0], row[3], row[4], row[5], row[2]))
                    if not defined or info >> 4 not in (1, 2, 10) or other & 3 not in (0, 3):
                        continue  # global/weak/GNU unique, default/protected, defined
                    symbol_name = string(s[6], name)
                    if not symbol_name:
                        continue
                    version = values[number] if values is not None else 1
                    vid = version & 0x7fff
                    if vid == 0:
                        continue  # local version index
                    if vid > 1 and vid not in versions:
                        raise ValueError('export has unknown GNU version definition')
                    exports.append({'name': symbol_name, 'version': versions.get(vid),
                                    'default_version': vid > 1 and not bool(version & 0x8000),
                                    'binding': info >> 4, 'type': info & 15,
                                    'visibility': other & 3,
                                    'size': size if info & 15 in (1, 6) else None})
    return {'sha256': digest, 'soname': soname, 'dt_needed': sorted(result['dt_needed']),
            'elf_class': 64 if wide else 32, 'byte_order': endian, 'machine': header[1],
            'exports': sorted(exports, key=lambda s: (s['name'], s['version'] or '', s['default_version']))}


def inventory(directory):
    root = Path(directory).resolve()
    files, aliases = {}, {}
    for path in sorted(root.rglob('*')):
        if '.so' not in path.name or not (path.is_file() or path.is_symlink()):
            continue
        target = path.resolve(strict=True)
        if not target.is_relative_to(root):
            raise ValueError('library symlink escapes directory: ' + str(path))
        name = str(target.relative_to(root))
        aliases[str(path.relative_to(root))] = name
        if name not in files:
            try:
                files[name] = inspect_library(target)
            except (ValueError, OSError, struct.error) as exc:
                files[name] = {'error': str(exc)}
    return files, aliases


def compare_directories(before, after):
    left, la = inventory(before)
    right, ra = inventory(after)
    libraries = []
    for name in sorted(left.keys() | right.keys()):
        a, b = left.get(name), right.get(name)
        item = {'library': name, 'before': a, 'after': b, 'compatible': False,
                'exports_added': [], 'exports_removed': [], 'reasons': []}
        if a is None or b is None:
            item['reasons'].append('library added or removed')
        elif 'error' in a or 'error' in b:
            item['reasons'].append('ELF inspection unavailable')
        else:
            encode = lambda s: json.dumps(s, sort_keys=True)
            sa, sb = {encode(s) for s in a['exports']}, {encode(s) for s in b['exports']}
            item['exports_added'] = [json.loads(s) for s in sorted(sb - sa)]
            item['exports_removed'] = [json.loads(s) for s in sorted(sa - sb)]
            for key in ('soname', 'dt_needed', 'elf_class', 'byte_order', 'machine'):
                if a[key] != b[key]:
                    item['reasons'].append(key + ' changed')
            if item['exports_removed']:
                item['reasons'].append('exports removed or changed')
            item['compatible'] = not item['reasons']
        item['bytes_changed'] = a is None or b is None or a.get('sha256') != b.get('sha256')
        libraries.append(item)
    aliases_equal = la == ra
    return {'before': str(before), 'after': str(after), 'libraries': libraries,
            'aliases_before': la, 'aliases_after': ra, 'aliases_equal': aliases_equal,
            'compatible': bool(libraries) and aliases_equal and all(i['compatible'] for i in libraries),
            'policy': 'Conservative before->after exported surface check. Additions allowed; '
                      'SONAME, dependencies, architecture and existing symbol versions/types/bindings/'
                      'visibility/object sizes must match. Does not prove layouts, semantics, '
                      'toolchain ABI, dependency availability or GPU correctness.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before')
    parser.add_argument('after')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = compare_directories(args.before, args.after)
    Path(args.out).write_text(json.dumps(result, indent=2) + '\n')
    return 0 if result['compatible'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
