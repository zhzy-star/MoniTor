"""Finish the existing single-video GPU run and preserve logs and metrics."""
import json
from pathlib import Path
import subprocess
import sys
import time

from src.reproduce import ROOT, build_commands, read_config


def main():
    run_dir = Path(sys.argv[1]).resolve()
    config_path = run_dir / "config.json"
    config = read_config(config_path)
    info = json.loads((run_dir / "run_info.json").read_text())
    info["scope"] = "single-video pipeline test with one BLIP model; not paper-level evaluation"
    info["caption_models"] = config["caption_models"]
    info["stages"] = []
    for stage in ("caption-index", "clean", "summarize", "memory-embeddings", "score"):
        start = time.monotonic()
        print(f"START {stage}", flush=True)
        record = {"stage": stage}
        with (run_dir / f"{stage}.log").open("w") as log:
            for command in build_commands(config, stage):
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    record.update(status="failed", exit_code=result.returncode,
                                  seconds=round(time.monotonic()-start, 2))
                    info["stages"].append(record)
                    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2)+"\n")
                    print(f"FAILED {stage}; see {run_dir / f'{stage}.log'}", flush=True)
                    return result.returncode
        record.update(status="passed", seconds=round(time.monotonic()-start, 2))
        info["stages"].append(record)
        (run_dir / "run_info.json").write_text(json.dumps(info, indent=2)+"\n")
        print(f"DONE {stage}: {record['seconds']} s", flush=True)
    print("START evaluate/raw", flush=True)
    with (run_dir / "evaluate_raw.log").open("w") as log:
        result = subprocess.run(build_commands(config, "evaluate", "raw")[0], cwd=ROOT,
                                stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        print("FAILED evaluate/raw", flush=True)
        return result.returncode
    metrics_path = Path(config["output_dir"]) / "raw" / "metrics.json"
    print(f"METRICS raw {metrics_path.read_text().strip()}", flush=True)
    print(f"COMPLETE {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
