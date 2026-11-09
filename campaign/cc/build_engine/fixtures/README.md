These are captured host `ldd`, `readelf -Ws`, and `nm -D` outputs for a CPU-only
ELF fixture, not a CUDA engine. A mock `libcudart.so.12` defines three dummy
functions; a shared `libggml-cuda.so` calls them and leaves them as imports.
The sources and compiled fixture are under `artifacts/codexN/linkage-fixture/`.
The fixture has an `$ORIGIN` runpath. Addresses and absolute paths are incidental;
tests parse library and symbol names only. No Docker, CUDA toolkit, or GPU was used.
