"""Prepare an isolated reference-implementation run from ten-video captions.

Preparation is the default. --run explicitly starts inference/API stages.
The anomaly prior is a reconstruction; its exact text was not disclosed in the paper.
"""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.video_record import VideoRecord
from src.reproduce import read_config


def prepare(source):
    source = Path(source).resolve()
    config = read_config(source / "config.json")
    rows = [row for row in Path(config["annotationfile_path"]).read_text().splitlines() if row.strip()]
    videos = [VideoRecord(row.split(), config["root_path"]) for row in rows]
    if len(videos) != 10 or len({v.path.name for v in videos}) != 10 or len(config["caption_models"]) != 5:
        raise ValueError("Expected the previous ten unique videos and five models")
    caption_files = []
    for video in videos:
        expected = {str(i) for i in range(0, video.num_frames, config["frame_interval"])}
        for model in config["caption_models"]:
            path = Path(config["raw_caption_root"]) / model / f"{video.path.name}.json"
            data = json.loads(path.read_text())
            if (not isinstance(data, dict) or set(data) != expected
                    or not all(isinstance(text, str) and text.strip() for text in data.values())):
                raise ValueError(f"Incomplete original captions: {path}")
            caption_files.append((model, path))
    clipped = ROOT / "ucf_crime/annotations/temporal_test_clipped.txt"
    if not clipped.is_file():
        raise ValueError(f"Missing isolated evaluation annotations: {clipped}")
    (ROOT / "runs").mkdir(exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="ucf_paper_10videos_", dir=ROOT / "runs"))
    for key, suffix in {
        "raw_caption_root": "caption/raw", "caption_index_dir": "index/captions",
        "clean_captions_dir": "caption/clean", "summaries_dir": "caption/summary",
        "memory_embeddings_dir": "index/memory", "raw_scores_dir": "score/raw",
        "output_dir": "output",
    }.items():
        config[key] = str(run_dir / suffix)
    # Copy small JSON files; preserve original BLIP outputs and existing experiments.
    for model, source_path in caption_files:
        target = Path(config["raw_caption_root"]) / model / source_path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target)
    (run_dir / "test.txt").write_text("\n".join(rows) + "\n")
    (run_dir / "videos.txt").write_text("\n".join(v.path.name + ".mp4" for v in videos) + "\n")
    config.update(annotationfile_path=str(run_dir / "test.txt"),
                  video_split_file=str(run_dir / "videos.txt"),
                  temporal_annotation_file=str(clipped),
                  score_alpha=0.7, score_previous="raw", score_queue_source="raw")
    (run_dir / "prompts").mkdir()
    for key, source_path in {
        "summary_prompt_file": ROOT / "prompts/summary_paper.txt",
        "context_prompt_file": ROOT / "prompts/ucf_context_paper.txt",
        "format_prompt_file": Path(config["format_prompt_file"]),
    }.items():
        target = run_dir / "prompts" / source_path.name
        shutil.copyfile(source_path, target)
        config[key] = str(target)
    config["reference_notes"] = {
        "anomaly_prior": "Reconstructed definitions/examples, not verbatim paper prompts",
        "fusion": "Literal formula: current raw and previous raw; queue uses raw scores",
        "target": "Main empty-initialization setting; ten-video diagnostic only",
        "source_run": str(source),
    }
    path = run_dir / "config.json"
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "runs/ucf_10videos_5models_0plr3lhp")
    parser.add_argument("--run", action="store_true", help="Start model/API stages after preparation")
    args = parser.parse_args()
    config_path = prepare(args.source)
    print(f"Prepared: {config_path}", flush=True)
    command = [sys.executable, "-u", "-m", "src.reproduce"]
    if args.run:
        log_dir = config_path.parent / "logs"
        log_dir.mkdir()
        for stage in ("caption-index", "clean", "summarize", "memory-embeddings", "score", "evaluate"):
            print(f"Running {stage}; logs: {log_dir}", flush=True)
            with (log_dir / f"{stage}.log").open("w") as log:
                subprocess.run(command + ["--config", str(config_path), "--stage", stage],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"Metrics: {config_path.parent / 'output/raw/metrics.json'}")
    else:
        print("Preparation only; no model or API was started.")
        print(f"UCF_CONFIG={shlex.quote(str(config_path))} bash scripts/run_ucf_all.sh caption-index evaluate")
