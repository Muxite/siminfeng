#!/usr/bin/env bash
# chain_factory.sh F10 F11 ... — run factory Codex assignments sequentially; stop if tests fail after one.
set -u; cd "$(dirname "$0")/.."
for T in "$@"; do
  A=$(ls assignments/$T-*.md | head -1)
  CODEX_CWD=$HOME/projects/jiuwenswarm c2/launch_codex.sh "$T" "$A"
  if ! (cd ~/projects/jiuwenswarm && timeout 900 .venv/bin/python -m pytest tests/factory --no-cov -q -p no:cacheprovider -x > /tmp/claude-1000/chain-$T-pytest.txt 2>&1); then
    echo "STOP after $T: tests failing (see /tmp/claude-1000/chain-$T-pytest.txt)"; tail -5 /tmp/claude-1000/chain-$T-pytest.txt; exit 1
  fi
  echo "$T ok: $(tail -1 /tmp/claude-1000/chain-$T-pytest.txt)"
done
