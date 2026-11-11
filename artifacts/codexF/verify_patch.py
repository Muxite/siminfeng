"""Offline verification; all generated files stay under artifacts/codexF."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/codexF"
SOURCE = ROOT / "ref/llama.cpp"
PATCH = ROOT / "cc/routing_trace/routing-trace.patch"
COPY = OUT / "verification-tree"
env = dict(os.environ, GIT_OPTIONAL_LOCKS="0", GIT_CEILING_DIRECTORIES=str(OUT),
           TMPDIR=str(OUT))
steps = []


def run(command, cwd=ROOT):
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
    steps.append({"argv": command, "cwd": str(cwd), "exit_code": result.returncode,
                  "stdout": result.stdout, "stderr": result.stderr})
    if result.returncode:
        (OUT / "verification.json").write_text(json.dumps(steps, indent=2) + "\n")
        raise RuntimeError(result.stderr)
    return result.stdout.strip()


commit = run(["git", "-C", str(SOURCE), "rev-parse", "HEAD"])
assert commit == "31288c9c66e59377f7788d920678a565bb89dd37"
status_before = run(["git", "-C", str(SOURCE), "status", "--porcelain", "--untracked-files=no"])
COPY.mkdir(exist_ok=False)
archive = OUT / "source.tar"
with archive.open("wb") as stream:
    subprocess.run(["git", "-C", str(SOURCE), "archive", commit], stdout=stream,
                   env=env, check=True)
with tarfile.open(archive) as stream:
    stream.extractall(COPY, filter="data")
archive.unlink()
run(["git", "apply", "--check", str(PATCH)], cwd=COPY)
run(["git", "apply", str(PATCH)], cwd=COPY)
cpp = COPY / "examples/routing-trace/routing-trace.cpp"
run(["g++", "--version"])
run(["g++", "-fsyntax-only", "-std=c++17", "-Wall", "-Wextra", "-Wpedantic",
     "-I" + str(COPY / "ggml/include"), "-I" + str(COPY / "include"),
     "-I" + str(COPY / "common"), "-I" + str(COPY / "tools"), str(cpp)])
assert status_before == run(["git", "-C", str(SOURCE), "status", "--porcelain", "--untracked-files=no"])
summary = {"source_commit": commit, "source_status_before": status_before,
           "source_status_unchanged": True,
           "patch_sha256": hashlib.sha256(PATCH.read_bytes()).hexdigest(),
           "steps": steps,
           "not_verified": ["full CMake build/link", "CUDA compilation", "model execution",
                            "real device transfers", "numerical equivalence", "Docker/Build CC image"]}
(OUT / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")
print("PASS: git apply --check, apply, C++17 syntax-only; source tracked status unchanged")
