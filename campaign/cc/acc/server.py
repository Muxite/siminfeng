"""Owned NVIDIA containers, watchdog/telemetry and localhost llama-server."""
import argparse
import atexit
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
import uuid

try:
    from .common import request_json, write_json
except ImportError:
    from common import request_json, write_json

ROOT = Path(__file__).resolve().parents[2]
OBSERVER = ROOT / "cc/resource_observe/resource_observe.py"


def docker_command(argv, prefix=None):
    prefix = os.environ.get("DOCKER_CMD_PREFIX", "") if prefix is None else prefix
    command = ["docker", *map(str, argv)]
    return shlex.split(prefix) + [shlex.join(command)] if prefix else command


def reap_containers(dry_run=True, labels=("acc=1",), unowned_max_age=6 * 3600):
    """Inspect only labelled containers; unknown jobs are stale, lookup errors fail closed.

    Dry runs perform read-only discovery and print only proposed Docker removals.
    Set unowned_max_age=None to retain containers without a job label.
    """
    ids = set()
    for label in labels:
        result = subprocess.run(docker_command(["ps", "-aq", "--filter", "label=" + label]),
                                check=True, capture_output=True, text=True, timeout=30)
        ids.update(result.stdout.split())
    commands = []
    states = {}
    for cid in sorted(ids):
        result = subprocess.run(docker_command(["inspect", cid]), check=True,
                                capture_output=True, text=True, timeout=30)
        container = json.loads(result.stdout)[0]
        if container["State"]["Status"] not in ("running", "exited"):
            continue
        job = (container["Config"].get("Labels") or {}).get("jobq_job")
        if job:
            if job not in states:
                status = subprocess.run([str(ROOT / "bin/jobq"), "show", job],
                                        capture_output=True, text=True, timeout=30)
                if status.returncode:
                    if "unknown job:" not in status.stderr and "invalid job id" not in status.stderr:
                        raise RuntimeError("jobq lookup failed: " + status.stderr)
                    states[job] = "unknown"
                else:
                    states[job] = json.loads(status.stdout)["state"]
            stale = states[job] != "running"
        elif unowned_max_age is not None:
            # Docker emits nanoseconds; Python 3.10 accepts microseconds only.
            timestamp = re.sub(r"\.(\d+)", lambda m: "." + (m[1] + "000000")[:6],
                               container["Created"])
            created = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            stale = (datetime.now(timezone.utc) - created).total_seconds() > unowned_max_age
        else:
            stale = False
        if stale:
            command = docker_command(["rm", "-f", cid])
            commands.append(command)
            if dry_run:
                print(shlex.join(command))
            else:
                subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    return commands


def parse_env(values):
    result = {}
    for value in values:
        key, sep, val = value.partition("=")
        if not sep or not key or not all(c.isalnum() or c == "_" for c in key):
            raise ValueError("--env must be NAME=VALUE")
        result[key] = val
    return result


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class OwnedContainer:
    def __init__(self, image, model, outdir, tool, argv, env=None, mounts=()):
        self.image = image
        self.model = Path(model).expanduser().resolve()
        self.outdir = Path(outdir)
        self.tool = tool
        self.argv = list(argv)
        self.env = dict(env or {})
        self.mounts = list(mounts)
        self.name = "acc-" + uuid.uuid4().hex
        self.children = []
        self.started = False
        self.closed = False
        self.signals = {}
        self.start_time = None

    def command(self):
        args = ["run", "-d", "--name", self.name, "--runtime", "nvidia",
                "--network", "host", "--label", "campaign=c2", "--label", "acc=1"]
        if os.environ.get("JOBQ_JOB_ID"):
            args += ["--label", "jobq_job=" + os.environ["JOBQ_JOB_ID"]]
        env = {**self.env, "NVIDIA_VISIBLE_DEVICES": "all",
               "NVIDIA_DRIVER_CAPABILITIES": "compute,utility"}
        for key, value in sorted(env.items()):
            args += ["-e", f"{key}={value}"]
        args += ["-v", f"{self.model.parent}:/om:ro"]
        for host, target, readonly in self.mounts:
            args += ["-v", f"{Path(host).expanduser().resolve()}:{target}" + (":ro" if readonly else ":rw")]
        return docker_command(args + [self.image, "/app/" + self.tool, *self.argv])

    def kill_command(self):
        return docker_command(["kill", self.name])

    def dry_run(self):
        return "\n".join([shlex.join(self.command()),
                          "# finally / atexit / SIGTERM cleanup:",
                          shlex.join(self.kill_command()),
                          shlex.join(docker_command(["rm", "-f", self.name]))])

    def _signal(self, signum, frame):
        raise SystemExit(128 + signum)

    def __enter__(self):
        reap_containers(dry_run=False)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.start_time = time.monotonic()
        atexit.register(self.close)
        try:
            for sig in (signal.SIGTERM, signal.SIGINT):
                self.signals[sig] = signal.signal(sig, self._signal)
            (self.outdir / "cmd.txt").write_text(self.dry_run() + "\n")
            write_json(self.outdir / "invocation.json", dict(image=self.image, model=str(self.model),
                       argv=self.argv, env=self.env, name=self.name))
            # Mark before launch: cleanup also covers a launch that succeeds but loses its reply.
            self.started = True
            result = subprocess.run(self.command(), check=True, capture_output=True, text=True, timeout=120)
            (self.outdir / "container.id").write_text(result.stdout)
            watchdog = [sys.executable, str(OBSERVER), "watchdog",
                "--min-available-mib", os.getenv("OM_MIN_AVAIL_MIB", "600"),
                "--max-swap-used-mib", os.getenv("OM_MAX_SWAP_MIB", "2560"),
                "--max-swap-rate-mibs", os.getenv("OM_MAX_SWAP_RATE", "1000"),
                "--interval-ms", "250", "--kill-cmd", shlex.join(self.kill_command()),
                "--out", str(self.outdir / "watchdog.jsonl")]
            sampler = [sys.executable, str(OBSERVER), "sample", "--backends", "io,proc,tegrastats",
                "--path", str(self.model.parent), "--interval-ms", "500", "--duration-s",
                os.getenv("OM_MAX_S", "1800"), "--out", str(self.outdir / "telemetry.jsonl")]
            for cmd in (watchdog, sampler):
                self.children.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
            return self
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.closed:
            return
        self.closed = True
        errors = []
        state = image = None
        def save(path, value):
            try:
                path.write_text(value)
            except OSError as exc:
                errors.append(str(exc))
        # Kill the owned container first: an observer shutdown failure must never
        # prevent the GPU workload from being stopped.
        if self.started:
            def run(args):
                try:
                    return subprocess.run(args, capture_output=True, text=True, timeout=30)
                except (OSError, subprocess.SubprocessError) as exc:
                    errors.append(str(exc))
                    return None
            run(self.kill_command())
            logs = run(docker_command(["logs", self.name]))
            if logs is not None:
                save(self.outdir / "stdout.txt", logs.stdout)
                save(self.outdir / "stderr.txt", logs.stderr)
            state = run(docker_command(["inspect", self.name, "--format", "{{json .State}}",]))
            image = run(docker_command(["inspect", self.name, "--format", "{{.Image}}",]))
            run(docker_command(["rm", "-f", self.name]))
        for child in self.children:
            try:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
            except (OSError, subprocess.SubprocessError) as exc:
                errors.append(str(exc))
        for sig, previous in self.signals.items():
            signal.signal(sig, previous)
        atexit.unregister(self.close)
        if self.started:
            try:
                write_json(self.outdir / "result.json", dict(wall_s=time.monotonic() - self.start_time,
                           container_state=state.stdout.strip() if state else None,
                           image_id=image.stdout.strip() if image else None, cleanup_errors=errors))
            except OSError:
                pass  # Cleanup must still finish when the output filesystem is full.

    def __exit__(self, exc_type, exc, traceback):
        self.close()


class Server(OwnedContainer):
    def __init__(self, image, model, outdir, engine_argv=(), env=None, port=None, health_timeout=600):
        self.port = port or free_port()
        self.health_timeout = health_timeout
        self.url = f"http://127.0.0.1:{self.port}"
        argv = [*engine_argv, "-m", "/om/" + Path(model).name, "--host", "127.0.0.1",
                "--port", str(self.port), "--jinja"]
        super().__init__(image, model, outdir, "llama-server", argv, env)

    def __enter__(self):
        super().__enter__()
        try:
            deadline = time.monotonic() + self.health_timeout
            while time.monotonic() < deadline:
                try:
                    if request_json(self.url + "/health", timeout=2).get("status") == "ok":
                        self.props = request_json(self.url + "/props", timeout=10)
                        write_json(self.outdir / "props.json", self.props)
                        return self
                except (OSError, ValueError):
                    pass
                state = subprocess.run(docker_command(["inspect", self.name, "--format", "{{.State.Running}}"]),
                                       capture_output=True, text=True, timeout=15)
                if state.returncode or state.stdout.strip() != "true":
                    raise RuntimeError("llama-server exited during startup; inspect session logs")
                time.sleep(.5)
            raise TimeoutError("llama-server health timeout")
        except BaseException:
            self.close()
            raise


def add_arguments(parser):
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", required=True, help="Host GGUF; parent mounted read-only as /om")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--env", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--engine-arg", action="append", default=[], help="Repeat; use --engine-arg=-ngl for flags")
    parser.add_argument("--health-timeout", type=float, default=600)
    parser.add_argument("--dry-run", action="store_true", help="Print Docker start and cleanup, execute nothing")


def from_args(args, outdir=None):
    return Server(args.image, args.model, outdir or args.outdir, args.engine_arg,
                  parse_env(args.env), health_timeout=args.health_timeout,
                  port=18080 if args.dry_run else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    if sys.argv[1:2] == ["reap"]:
        parser.add_argument("action", choices=["reap"])
        parser.add_argument("--dry-run", action="store_true")
        args = parser.parse_args()
        reap_containers(dry_run=args.dry_run)
        return
    add_arguments(parser)
    parser.add_argument("--duration", type=float, default=1800, help="Seconds to serve after health succeeds")
    args = parser.parse_args()
    server = from_args(args)
    if args.dry_run:
        print(server.dry_run())
        return
    with server:
        print(server.url, flush=True)
        time.sleep(args.duration)


if __name__ == "__main__":
    main()
