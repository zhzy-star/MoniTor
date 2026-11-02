#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-/root/miniconda3/envs/Moni/bin/python}"
START_STAGE="${1:-summarize}"
END_STAGE="${2:-evaluate}"
SESSION_4="${TMUX_SESSION_GLM4:-ucf_glm4}"
SESSION_53="${TMUX_SESSION_GLM53:-ucf_glm53}"

: "${GLM4_API_KEY:?Set GLM4_API_KEY before starting the GLM-4 session}"
: "${GLM53_API_KEY:?Set GLM53_API_KEY before starting the GLM-5.3 session}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 2
fi
if tmux has-session -t "$SESSION_4" 2>/dev/null; then
    echo "tmux session already exists: $SESSION_4" >&2
    exit 2
fi
if tmux has-session -t "$SESSION_53" 2>/dev/null; then
    echo "tmux session already exists: $SESSION_53" >&2
    exit 2
fi

start_session() {
    local session="$1"
    local config="$2"
    local api_key="$3"
    local env_file
    env_file="$(mktemp "${TMPDIR:-/tmp}/monitor_tmux_env.XXXXXX")"
    chmod 600 "$env_file"
    {
        printf 'export PYTHON=%q\n' "$PYTHON_BIN"
        printf 'export UCF_CONFIG=%q\n' "$config"
        printf 'export ZHIPUAI_API_KEY=%q\n' "$api_key"
        if [[ -n "${UCF_MEMORY_LOCK:-}" ]]; then
            printf 'export UCF_MEMORY_LOCK=%q\n' "$UCF_MEMORY_LOCK"
        fi
    } > "$env_file"
    if ! tmux new-session -d -s "$session" -c "$ROOT_DIR" \
        "bash scripts/run_ucf_tmux_worker.sh '$env_file' '$START_STAGE' '$END_STAGE'"; then
        rm -f "$env_file"
        return 1
    fi
}

start_session "$SESSION_4" "configs/ucf_glm4.json" "$GLM4_API_KEY"
start_session "$SESSION_53" "configs/ucf_glm53.json" "$GLM53_API_KEY"

echo "Started sessions: $SESSION_4, $SESSION_53"
echo "Attach with: tmux attach -t $SESSION_4"
echo "Attach with: tmux attach -t $SESSION_53"
