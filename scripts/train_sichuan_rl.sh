#!/usr/bin/env bash
# RJob/container entrypoint; the caller supplies resources and persistent paths.
set -euo pipefail
mahjong_project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$mahjong_project_root"
exec "${MAHJONG_PYTHON:-python}" -m mahjong_sandbox rl-train "$@"
