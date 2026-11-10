set -eu
pid=$(docker inspect -f '{{.State.Pid}}' ENGINE_CONTAINER)
[ "$pid" -gt 0 ]
library=
for candidate in /proc/"$pid"/root/app/libggml-cpu.so.*; do
  [ -f "$candidate" ] && [ ! -L "$candidate" ] || continue
  [ -z "$library" ] || { echo 'Multiple CPU backend files; select --library explicitly' >&2; exit 1; }
  library="$candidate"
done
[ -n "$library" ]
python3 /home/muk/projects/siminfeng/campaign/cc/profilers/bpf_split_timer.py script --pid "$pid" --library "$library" --seconds 5.0 --per-thread | sudo bpftrace -
