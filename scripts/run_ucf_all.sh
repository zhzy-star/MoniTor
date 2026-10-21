#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Accept start/end stages so completed stages need not be repeated.
start_stage="${1:-extract}"
end_stage="${2:-evaluate}"
test_config="${UCF_CONFIG:-configs/ucf.json}"
test_python="${PYTHON:-/root/miniconda3/envs/Moni/bin/python}"
stages=(extract caption caption-index clean summarize score summary-index refine evaluate)
# Paper mode evaluates the fused score directly; ImageBind summary vectors
# are prepared before score workers start. No extra retrieval refinement.
paper_mode=$("$test_python" -c 'import json,sys; print(int(json.load(open(sys.argv[1])).get("paper_alignment", False)))' "$test_config")
if [[ "$paper_mode" == 1 ]]; then
    stages=(extract caption caption-index clean summarize memory-embeddings score evaluate)
fi
start_index=-1
end_index=-1
for index in "${!stages[@]}"; do
    [[ "${stages[$index]}" != "$start_stage" ]] || start_index="$index"
    [[ "${stages[$index]}" != "$end_stage" ]] || end_index="$index"
done
if (( start_index < 0 || end_index < start_index )); then
    echo "Usage: bash scripts/run_ucf_all.sh [start_stage] [end_stage]" >&2
    echo "Stages: ${stages[*]}" >&2
    exit 2
fi
for (( index=start_index; index<=end_index; index++ )); do
    if [[ "${stages[$index]}" == summarize || "${stages[$index]}" == score ]]; then
        : "${ZHIPUAI_API_KEY:?Export ZHIPUAI_API_KEY before running summary/scoring stages}"
    fi
done
if [[ ! -x "$test_python" ]]; then
    echo "Python executable not found: $test_python" >&2
    exit 2
fi
test_logs="logs/ucf_$(date -u +%Y%m%d_%H%M%S)_$$"
mkdir -p "$test_logs"
for (( index=start_index; index<=end_index; index++ )); do
    stage="${stages[$index]}"
    echo "Running $stage; logs: $test_logs"
    if [[ "$stage" == evaluate ]]; then
        score_sources=(raw refined)
        [[ "$paper_mode" != 1 ]] || score_sources=(raw)
        for source in "${score_sources[@]}"; do
            "$test_python" -u -m src.reproduce --config "$test_config" \
                --stage evaluate --score-source "$source" 2>&1 | tee "$test_logs/evaluate_$source.log"
        done
    else
        "$test_python" -u -m src.reproduce --config "$test_config" \
            --stage "$stage" --resume 2>&1 | tee "$test_logs/$stage.log"
    fi
done
echo "Completed $start_stage through $end_stage; logs: $test_logs"
