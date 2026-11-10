#!/usr/bin/env python3
"""Explicit GPU operator helper; calibrate.py itself never executes probes."""
import argparse
import json
import subprocess
import time
from pathlib import Path
import sys
sys.dont_write_bytecode = True
from calibrate import output_path, read, emit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('capture')
    a=p.parse_args()
    out=output_path(a.capture)
    manifest=read(out/'manifest.json')
    started=time.monotonic()
    for j in manifest['jobs']:
        name=j['name']; cmd=j['argv']
        if cmd[:3]!=['docker','run','--rm']:
            p.error('only docker run plans are accepted')
        container='host-cal-'+str(time.time_ns())
        cmd=cmd[:3]+['--name',container]+cmd[3:]
        try:
            result=subprocess.run(cmd,capture_output=True,timeout=29)
            stdout,stderr,code=result.stdout,result.stderr,result.returncode
        except subprocess.TimeoutExpired as exc:
            stdout,stderr,code=exc.stdout or b'',exc.stderr or b'',124
        finally:
            # Remove container even if Docker CLI was killed; daemon may still own it.
            subprocess.run(['docker','rm','-f',container],capture_output=True,timeout=1)
        (out/f'{name}.stdout').write_bytes(stdout)
        (out/f'{name}.stderr').write_bytes(stderr)
        emit(out/f'{name}.status.json',dict(returncode=code,argv=cmd))
        print(f'{name}: {code}',flush=True)
        if code:
            raise SystemExit(f'probe failed; inspect {out/name}')
    emit(out/'elapsed.json',dict(seconds=time.monotonic()-started))

if __name__=='__main__': main()
