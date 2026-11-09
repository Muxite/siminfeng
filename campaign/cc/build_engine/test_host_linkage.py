"""Small offline ELF and layered archive fixtures; no CUDA toolkit required."""
from functools import lru_cache
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tarfile
import tempfile

import pytest

from host_linkage import collect_host_linkage, extract_backend, inspect_elf


def crafted_elf(import_malloc=True, wide=True, endian="<"):
    """Fallback ELF with dynamic sections, including defined/lookalike symbols."""
    strings = b"\0libcudart.so.12\0cudaMalloc\0cudaFreeAsync\0cudaMallocAsync\0cudaFree\0cudaMemGetInfo\0cuMemCreate\0cudaMallocExtra\0"
    name = lambda value: strings.index(value.encode() + b"\0")
    symfmt = endian + ("IBBHQQ" if wide else "IIIBBH")
    def symbol(value, section=0):
        args = (name(value), 0x12, 0, section, 0, 0) if wide else (name(value), 0, 0, 0x12, 0, section)
        return struct.pack(symfmt, *args)
    symbols = bytes(struct.calcsize(symfmt))
    symbols += symbol("cudaMalloc", 0 if import_malloc else 1)
    symbols += symbol("cudaFreeAsync") + symbol("cudaMallocAsync")
    symbols += symbol("cudaFree", 1) + symbol("cudaMemGetInfo", 1) + symbol("cuMemCreate", 1)
    symbols += symbol("cudaMallocExtra")
    dynfmt = endian + ("qQ" if wide else "iI")
    dynamic = struct.pack(dynfmt, 1, 1) + struct.pack(dynfmt, 0, 0)
    shfmt = endian + ("IIQQQQIIQQ" if wide else "IIIIIIIIII")
    shsize = struct.calcsize(shfmt)
    shoff = 64
    start = shoff + shsize * 4
    section = lambda kind, offset, size, link=0, stride=0: struct.pack(shfmt, 0, kind, 0, 0, offset, size, link, 0, 1, stride)
    table = bytes(shsize) + section(3, start, len(strings))
    table += section(11, start + len(strings), len(symbols), 1, struct.calcsize(symfmt))
    table += section(6, start + len(strings) + len(symbols), len(dynamic), 1, struct.calcsize(dynfmt))
    ident = b"\x7fELF" + bytes([2 if wide else 1, 1 if endian == '<' else 2, 1]) + bytes(9)
    fmt = endian + ("HHIQQQIHHHHHH" if wide else "HHIIIIIHHHHHH")
    header = struct.pack(fmt, 3, 62 if wide else 3, 1, 0, 0, shoff, 0, 64 if wide else 52, 0, 0, shsize, 4, 0)
    return (ident + header).ljust(shoff, b"\0") + table + strings + symbols + dynamic


@lru_cache(None)
def fixture_elf(import_malloc=True):
    cc = shutil.which("cc")
    if cc is None:
        return crafted_elf(import_malloc)
    with tempfile.TemporaryDirectory(prefix="cuda-elf-fixture-") as directory:
        root = Path(directory)
        (root / "cudart.c").write_text("void cudaMalloc(void) {}\nvoid cudaFreeAsync(void) {}\nvoid cudaMallocAsync(void) {}\n")
        (root / "backend.c").write_text(
            "extern void cudaMalloc(void), cudaFreeAsync(void), cudaMallocAsync(void);\n"
            "void cudaFree(void) {}\nvoid cudaMemGetInfo(void) {}\nvoid cuMemCreate(void) {}\n"
            "void backend(void) { " + ("cudaMalloc(); " if import_malloc else "") + "cudaFreeAsync(); cudaMallocAsync(); }\n")
        subprocess.run([cc, "-shared", "-fPIC", "-Wl,-soname,libcudart.so.12", "-o", str(root / "libcudart.so.12"), str(root / "cudart.c")], check=True, capture_output=True)
        subprocess.run([cc, "-shared", "-fPIC", "-o", str(root / "backend.so"), str(root / "backend.c"), "-L" + str(root), "-l:libcudart.so.12"], check=True, capture_output=True)
        return (root / "backend.so").read_bytes()


def layer_bytes(entries):
    """Entries: (path, bytes) or (path, ('symlink'/'hardlink'/'dir', target))."""
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w") as archive:
        for path, payload in entries:
            entry = tarfile.TarInfo(path)
            if isinstance(payload, tuple):
                kind, entry.linkname = payload
                entry.type = {"symlink": tarfile.SYMTYPE, "hardlink": tarfile.LNKTYPE, "dir": tarfile.DIRTYPE}[kind]
                archive.addfile(entry)
            else:
                entry.size = len(payload)
                archive.addfile(entry, io.BytesIO(payload))
    return out.getvalue()


def oci_fixture(path, layers, compressed=True, corrupt=False):
    objects = {}
    def blob(raw):
        digest = "sha256:" + hashlib.sha256(raw).hexdigest()
        objects["blobs/sha256/" + digest[7:]] = raw
        return {"digest": digest, "size": len(raw)}
    config = {"os": "linux", "architecture": "amd64", "rootfs": {"diff_ids": ["sha256:" + hashlib.sha256(raw).hexdigest() for raw in layers]}}
    config_desc = blob(json.dumps(config).encode())
    descs = [blob(gzip.compress(raw) if compressed else raw) for raw in layers]
    manifest = blob(json.dumps({"schemaVersion": 2, "config": config_desc, "layers": descs}).encode())
    # Duplicate references in nested OCI indexes, plus ignored foreign platform/attestation.
    nested = blob(json.dumps({"manifests": [manifest]}).encode())
    objects["index.json"] = json.dumps({"manifests": [nested, manifest,
        {"digest": "sha256:" + "0" * 64, "platform": {"os": "linux", "architecture": "arm64"}},
        {"digest": "sha256:" + "1" * 64, "annotations": {"vnd.docker.reference.type": "attestation-manifest"}}]}).encode()
    if corrupt:
        objects["blobs/sha256/" + descs[0]["digest"][7:]] += b"tampered"
    with tarfile.open(path, "w") as archive:
        for name, raw in objects.items():
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
    return config_desc["digest"]


@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("endian", ["<", ">"])
def test_python_elf_classes_and_byte_orders(tmp_path, wide, endian):
    path = tmp_path / "backend.so"
    path.write_bytes(crafted_elf(wide=wide, endian=endian))
    result = inspect_elf(path)
    assert result["dt_needed"] == ["libcudart.so.12"]
    assert result["undefined_imports"] == {"cudaMalloc": True, "cudaFree": False,
        "cudaMallocAsync": True, "cudaFreeAsync": True, "cudaMemGetInfo": False, "cuMemCreate": False}


def test_host_cc_elf(tmp_path):
    path = tmp_path / "backend.so"
    path.write_bytes(fixture_elf())
    result = inspect_elf(path)
    assert result["libcudart_dynamic"] is True
    assert result["undefined_imports"]["cudaMalloc"] is True
    assert result["undefined_imports"]["cudaFree"] is False


@pytest.mark.parametrize("compressed", [False, True])
def test_oci_whiteouts_and_link_chain(tmp_path, compressed):
    path = tmp_path / "image.tar"
    layers = [layer_bytes([("app/libggml-cuda.so", b"old"), ("app/hidden/old", b"hidden"), ("unrelated/large", b"unused" * 10000)]),
              layer_bytes([("app/libggml-cuda.so", ("symlink", "libggml-cuda.so.1")),
                           ("app/libggml-cuda.so.1", ("symlink", "/app/current/backend")),
                           ("app/current", ("symlink", "hidden")),
                           ("app/hidden/backend", fixture_elf()),
                           ("app/hidden/.wh..wh..opq", b""), ("app/.wh.libggml-cuda.so", b"")])]
    identity = oci_fixture(path, layers, compressed)
    output = tmp_path / "extracted"
    output.mkdir()
    library, evidence = extract_backend(path, output, identity)
    assert evidence["resolved_library"] == "/app/hidden/backend"
    assert library.read_bytes() == fixture_elf()
    assert list(output.iterdir()) == [library]
    result = collect_host_linkage(path, identity, tmp_path)
    assert result["status"] == "passed"  # only cudaMalloc is required
    assert result["symbols"]["tool"] == "python-elf"
    assert not list(tmp_path.glob("linkage-*"))


@pytest.mark.parametrize("entries", [
    [("app/.wh.libggml-cuda.so", b"")],
    [("app/.wh..wh..opq", b"")],
    [(".wh.app", b"")],
    [("app", b"replacement")],
    [("app/libggml-cuda.so", ("symlink", "missing"))],
    [("app/libggml-cuda.so", ("symlink", "libggml-cuda.so"))],
])
def test_merged_library_unavailable(tmp_path, entries):
    path = tmp_path / "image.tar"
    oci_fixture(path, [layer_bytes([("app/libggml-cuda.so", fixture_elf())]), layer_bytes(entries)])
    result = collect_host_linkage(path, temp_parent=tmp_path)
    assert result["status"] == "unavailable"
    assert result["undefined_imports"]["cudaMalloc"] is None


def test_hardlink_and_parent_relative_symlink(tmp_path):
    path = tmp_path / "image.tar"
    oci_fixture(path, [layer_bytes([("app/libggml-cuda.so", ("symlink", "../opt/link")),
        ("opt/link", ("hardlink", "opt/backend")), ("opt/backend", fixture_elf())])])
    assert collect_host_linkage(path, temp_parent=tmp_path)["status"] == "passed"


def test_layer_digest_corruption(tmp_path):
    path = tmp_path / "image.tar"
    oci_fixture(path, [layer_bytes([("app/libggml-cuda.so", fixture_elf())])], compressed=False, corrupt=True)
    result = collect_host_linkage(path, temp_parent=tmp_path)
    assert result["status"] == "unavailable"
    assert "SHA-256 mismatch" in result["reason"]


@pytest.mark.parametrize("raw", [b"", b"not an ELF" * 10, crafted_elf()[:200]])
def test_bad_elf_unavailable(tmp_path, raw):
    path = tmp_path / "image.tar"
    oci_fixture(path, [layer_bytes([("app/libggml-cuda.so", raw)])])
    result = collect_host_linkage(path, temp_parent=tmp_path)
    assert result["status"] == "unavailable"
    assert all(value is None for value in result["undefined_imports"].values())


def test_missing_malloc_fails(tmp_path):
    path = tmp_path / "image.tar"
    oci_fixture(path, [layer_bytes([("app/libggml-cuda.so", fixture_elf(False))])])
    result = collect_host_linkage(path, temp_parent=tmp_path)
    assert result["status"] == "failed"
    assert result["libcudart_dynamic"] is True
    assert result["undefined_imports"]["cudaMalloc"] is False


def test_missing_cudart_dependency_fails(tmp_path):
    path = tmp_path / "image.tar"
    elf = crafted_elf().replace(b"libcudart.so.12", b"libnotrt.so.12x")
    oci_fixture(path, [layer_bytes([("app/libggml-cuda.so", elf)])])
    result = collect_host_linkage(path, temp_parent=tmp_path)
    assert result["status"] == "failed"
    assert result["undefined_imports"]["cudaMalloc"] is True
    assert result["libcudart_dynamic"] is False


def test_upper_layer_replaces_symlink_target(tmp_path):
    path = tmp_path / "image.tar"
    oci_fixture(path, [layer_bytes([("app/libggml-cuda.so", ("symlink", "backend")),
                                   ("app/backend", fixture_elf(False))]),
                       layer_bytes([("app/backend", fixture_elf())])])
    assert collect_host_linkage(path, temp_parent=tmp_path)["status"] == "passed"


def test_config_mismatch_unavailable(tmp_path):
    path = tmp_path / "image.tar"
    oci_fixture(path, [layer_bytes([("app/libggml-cuda.so", fixture_elf())])])
    result = collect_host_linkage(path, "sha256:" + "0" * 64, tmp_path)
    assert result["status"] == "unavailable"
    assert "matching" in result["reason"]
