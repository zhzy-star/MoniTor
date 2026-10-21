#!/usr/bin/env bash
# Reuse extracted UCF frames; caption ten videos with five separate models.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export PYTHON="${PYTHON:-/root/miniconda3/envs/Moni/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

if [[ -z "${ZHIPUAI_API_KEY:-}" ]]; then
    read -r -s -p "输入 GLM API Key: " monitor_glm_key
    printf '\n'
    export ZHIPUAI_API_KEY="$monitor_glm_key"
    unset monitor_glm_key
fi
: "${ZHIPUAI_API_KEY:?GLM API Key must not be empty}"

# Only read the existing extraction list; never decode or extract videos here.
export UCF_CONFIG
UCF_CONFIG=$("$PYTHON" -u - <<'PY'
import json
import sys
import tempfile
from pathlib import Path
from src.data.video_record import VideoRecord
from src.reproduce import ROOT, read_config

config = read_config(ROOT / "configs/ucf.json")
rows = [line.strip() for line in Path(config["annotationfile_path"]).read_text().splitlines()
        if line.strip()][:10]
if len(rows) != 10:
    raise SystemExit("已有抽帧清单不足 10 个视频")
videos = [VideoRecord(row.split(), config["root_path"]) for row in rows]
if len({video.path.name for video in videos}) != 10:
    raise SystemExit("所选视频清单有重复名称")
models = ["blip2-flan-t5-xl", "blip2-flan-t5-xl-coco", "blip2-flan-t5-xxl",
          "blip2-opt-6.7b", "blip2-opt-6.7b-coco"]
config["caption_models"] = models
run_root = ROOT / "runs"
run_root.mkdir(parents=True, exist_ok=True)
run_dir = Path(tempfile.mkdtemp(prefix="ucf_10videos_5models_", dir=str(run_root)))
(run_dir / "configs").mkdir()
(run_dir / "logs").mkdir()
list_path = run_dir / "test.txt"
list_path.write_text("\n".join(rows) + "\n")
config["annotationfile_path"] = str(list_path)
split_path = run_dir / "videos.txt"
split_path.write_text("\n".join(video.path.name + ".mp4" for video in videos) + "\n")
config["video_split_file"] = str(split_path)
outputs = {
    "raw_caption_root": "caption/raw", "caption_index_dir": "index/captions",
    "clean_captions_dir": "caption/clean", "summaries_dir": "caption/summary",
    "summary_index_dir": "index/summaries", "raw_scores_dir": "score/raw",
    "refined_scores_dir": "score/refined", "retrieval_dir": "retrieval",
    "output_dir": "output",
}
for key, suffix in outputs.items():
    config[key] = str(run_dir / suffix)
config_path = run_dir / "config.json"
config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
for model in models:
    one = dict(config, caption_models=[model])
    (run_dir / "configs" / f"{model}.json").write_text(
        json.dumps(one, ensure_ascii=False, indent=2) + "\n")
print("本次视频：" + ", ".join(video.path.name for video in videos), file=sys.stderr)
print(config_path)
PY
)
monitor_run_dir="$(dirname "$UCF_CONFIG")"
printf '实验目录：%s\n' "$monitor_run_dir"

# Each model captions all ten videos in its own process, sequentially.
for monitor_model in \
    blip2-flan-t5-xl \
    blip2-flan-t5-xl-coco \
    blip2-flan-t5-xxl \
    blip2-opt-6.7b \
    blip2-opt-6.7b-coco
do
    printf '运行字幕模型：%s\n' "$monitor_model"
    "$PYTHON" -u -m src.reproduce \
        --config "$monitor_run_dir/configs/$monitor_model.json" \
        --stage caption --resume \
        2>&1 | tee "$monitor_run_dir/logs/caption_$monitor_model.log"
done

# Save a per-frame, per-model view. The index reads the original five folders.
"$PYTHON" -u - "$UCF_CONFIG" <<'PY'
import json
import sys
from pathlib import Path
from src.data.video_record import VideoRecord
from src.reproduce import read_config

config = read_config(sys.argv[1])
merged_dir = Path(config["raw_caption_root"]).parent / "merged"
merged_dir.mkdir(parents=True, exist_ok=True)
for row in Path(config["annotationfile_path"]).read_text().splitlines():
    video = VideoRecord(row.split(), config["root_path"])
    expected = [str(idx) for idx in range(0, video.num_frames, config["frame_interval"])]
    merged = {key: {} for key in expected}
    for model in config["caption_models"]:
        path = Path(config["raw_caption_root"]) / model / f"{video.path.name}.json"
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict) or set(payload) != set(expected):
            raise SystemExit(f"字幕帧编号不完整：{path}")
        if not all(isinstance(text, str) and text.strip() for text in payload.values()):
            raise SystemExit(f"字幕为空或格式不正确：{path}")
        for key in expected:
            merged[key][model] = payload[key]
    (merged_dir / f"{video.path.name}.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n")
print(f"五模型字幕已按帧合并：{merged_dir}", flush=True)
PY

# One joint analysis after every model has finished, then both evaluations.
bash scripts/run_ucf_all.sh caption-index evaluate \
    2>&1 | tee "$monitor_run_dir/logs/analysis.log"
printf '完成。原始指标：%s/output/raw/metrics.json\n' "$monitor_run_dir"
printf '细化指标：%s/output/refined/metrics.json\n' "$monitor_run_dir"
