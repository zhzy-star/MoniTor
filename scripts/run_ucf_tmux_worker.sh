#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

env_file="$1"
start_stage="$2"
end_stage="$3"
if [[ ! -r "$env_file" ]]; then
    echo "Missing tmux environment file: $env_file" >&2
    exit 2
fi
source "$env_file"
rm -f "$env_file"
if [[ -f /etc/network_turbo ]]; then
    source /etc/network_turbo
fi
exec bash scripts/run_ucf_all.sh "$start_stage" "$end_stage"
