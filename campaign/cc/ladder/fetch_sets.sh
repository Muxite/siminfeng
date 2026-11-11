#!/usr/bin/env bash
# LEAD ONLY: network fetch; no GPU or Docker. Never overwrites frozen sets.
set -euo pipefail
exec python3 "$(dirname "$0")/fetch_sets.py" "$@"
