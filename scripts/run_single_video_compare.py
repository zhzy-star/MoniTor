"""Caption one video in separate processes, then compare one and five models."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from src.data.video_record import VideoRecord
from src.reproduce import ROOT, build_commands, check_inputs, read_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-config", default="configs/ucf_single_video_1model.json")
    parser.add_argument("--five-config", default="configs/ucf_single_video_5models.json")
    args = parser.parse_args()
    # The container exposes more CPUs than its quota; limit BLAS thread pools.
    child_env = dict(os.environ)
    child_env.setdefault("OMP_NUM_THREADS", "8")
    child_env.setdefault("MKL_NUM_THREADS", "8")
    single = read_config(args.single_config)
    five = read_config(args.five_config)
    for key in ("root_path", "annotationfile_path", "video_split_file", "frame_interval"):
        if single[key] != five[key]:
            raise ValueError(f"Both runs must share {key}")
    if len(single["caption_models"]) != 1 or len(five["caption_models"]) != 5:
        raise ValueError("Expected one and five caption models")
    if single["caption_models"][0] not in five["caption_models"]:
        raise ValueError("The baseline model must be one of the five models")
    split = Path(five["video_split_file"]).read_text().split()
    if len(split) != 1:
        raise ValueError("This runner accepts exactly one video")

    run_dir = Path(five["annotationfile_path"]).parent
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    info_path = run_dir / "comparison_run.json"
    info = {"video": split[0], "caption_models": five["caption_models"], "stages": []}

    def run_commands(label, commands):
        start = time.monotonic()
        log_path = log_dir / f"{label}.log"
        print(f"START {label}; log: {log_path}", flush=True)
        record = {"stage": label, "log": str(log_path), "status": "running"}
        info["stages"].append(record)
        info_path.write_text(json.dumps(info, indent=2) + "\n")
        with log_path.open("w") as log:
            for command in commands:
                result = subprocess.run(command, cwd=ROOT, env=child_env,
                                        stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    record.update(status="failed", exit_code=result.returncode,
                                  seconds=round(time.monotonic() - start, 2))
                    info_path.write_text(json.dumps(info, indent=2) + "\n")
                    print(f"FAILED {label}; see {log_path}", flush=True)
                    raise subprocess.CalledProcessError(result.returncode, command)
        record.update(status="passed", seconds=round(time.monotonic() - start, 2))
        info_path.write_text(json.dumps(info, indent=2) + "\n")
        print(f"DONE {label}: {record['seconds']} s", flush=True)

    def stage(config, name, label, source="raw"):
        errors = check_inputs(config, name, source)
        if errors:
            raise ValueError("\n".join(errors[:20]))
        run_commands(label, build_commands(config, name, source, resume=True))

    if not Path(five["annotationfile_path"]).exists():
        stage(five, "extract", "extract")
    lines = Path(five["annotationfile_path"]).read_text().splitlines()
    if len(lines) != 1 or lines[0].split()[0] != Path(split[0]).stem:
        raise ValueError("Existing extracted video list differs from the selected video")
    video = VideoRecord(lines[0].split(), five["root_path"])
    expected = {str(i) for i in range(0, video.num_frames, five["frame_interval"])}

    merged = {key: {} for key in sorted(expected, key=int)}
    for model in five["caption_models"]:
        config = dict(five, caption_models=[model])
        if model == "blip2-flan-t5-xxl":
            # Amortize CPU/GPU weight transfers without changing sampled frames.
            config["batch_size"] = 4
        output = Path(five["raw_caption_root"]) / model / f"{video.path.name}.json"
        if output.exists():
            payload = json.loads(output.read_text())
            if set(payload) != expected:
                raise ValueError(f"Incomplete caption file: {output}")
            print(f"SKIP completed caption/{model}", flush=True)
        else:
            stage(config, "caption", f"caption_{model}")
            payload = json.loads(output.read_text())
        if set(payload) != expected or not all(isinstance(text, str) and text.strip()
                                              for text in payload.values()):
            raise ValueError(f"Invalid caption file: {output}")
        for key, text in payload.items():
            merged[key][model] = text

    merged_path = Path(five["raw_caption_root"]).parent / "merged" / f"{video.path.name}.json"
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    merged_path.write_text(json.dumps(merged, indent=2) + "\n")

    baseline = single["caption_models"][0]
    src = Path(five["raw_caption_root"]) / baseline / f"{video.path.name}.json"
    dst = Path(single["raw_caption_root"]) / baseline / src.name
    if dst.exists() and dst.read_bytes() != src.read_bytes():
        raise ValueError(f"Baseline captions differ from the five-model run: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

    for label, config in (("1model", single), ("5models", five)):
        for name in ("caption-index", "clean", "summarize", "score", "summary-index", "refine"):
            stage(config, name, f"{label}_{name}")
        for source in ("raw", "refined"):
            stage(config, "evaluate", f"{label}_evaluate_{source}", source)
            metrics = json.loads((Path(config["output_dir"]) / source / "metrics.json").read_text())
            print(f"METRICS {label}/{source}: {json.dumps(metrics)}", flush=True)
    print(f"COMPLETE; report: {info_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
