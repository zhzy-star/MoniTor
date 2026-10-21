"""Save metrics, score diagnostics and a frame-aligned comparison plot."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.reproduce import read_config


def main():
    configs = [("1model", read_config("configs/ucf_single_video_1model.json")),
               ("5models", read_config("configs/ucf_single_video_5models.json"))]
    run_dir = Path(configs[0][1]["annotationfile_path"]).parent
    results = {}
    labels = None
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for model_label, config in configs:
        results[model_label] = {}
        for axis, source in zip(axes, ("raw", "refined")):
            output = Path(config["output_dir"]) / source
            metrics = json.loads((output / "metrics.json").read_text())
            current_labels = np.load(output / "frame_labels.npy")
            scores = np.load(output / "frame_scores.npy")
            if labels is None:
                labels = current_labels
            if not np.array_equal(labels, current_labels):
                raise ValueError("Ground truth differs between runs")
            if len(scores) != len(labels) or not np.isfinite(scores).all():
                raise ValueError("Scores must be finite and frame aligned")
            metrics.update(
                normal_frames=int((labels == 0).sum()),
                anomaly_frames=int((labels == 1).sum()),
                mean_normal_score=float(scores[labels == 0].mean()),
                mean_anomaly_score=float(scores[labels == 1].mean()),
            )
            if source == "raw":
                video = Path(config["video_split_file"]).read_text().strip()
                payload = json.loads((Path(config["raw_scores_dir"]) / f"{Path(video).stem}.json").read_text())
                metrics["unparsed_sample_scores"] = sum(value["new_score"] == -1 for value in payload.values())
            results[model_label][source] = metrics
            axis.plot(np.arange(len(scores)), scores, label=model_label, linewidth=1.3)
    for axis, source in zip(axes, ("raw", "refined")):
        axis.fill_between(np.arange(len(labels)), 0, 1, where=labels == 1,
                          color="red", alpha=0.12, label="ground-truth anomaly")
        axis.set(title=source, ylabel="Anomaly score", ylim=(-0.02, 1.02))
        axis.legend(loc="upper right")
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Source frame ID")
    fig.suptitle(Path(configs[0][1]["video_split_file"]).read_text().strip())
    fig.tight_layout()
    fig.savefig(run_dir / "comparison.png", dpi=160)
    plt.close(fig)
    (run_dir / "comparison_metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    lines = ["# Single-video comparison", "", "Video: Abuse028_x264.mp4. Sampling interval: 16 frames.",
             "", "| Models | Scores | ROC-AUC | PR-AUC | AP | Mean normal score | Mean anomaly score |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for model_label, sources in results.items():
        for source, metrics in sources.items():
            values = [metrics[key] for key in ("roc_auc", "pr_auc", "average_precision",
                                              "mean_normal_score", "mean_anomaly_score")]
            lines.append(f"| {model_label} | {source} | " + " | ".join(f"{value:.6f}" for value in values) + " |")
    lines.extend(["", "![Frame scores](comparison.png)", "",
                  "The two runs share the video, sampled frames, prompts and downstream parameters.",
                  "The one-model baseline reuses the FLAN-T5-XL captions from the five-model run.",
                  "XXL captions used batch size 4 to reduce CPU/GPU transfer overhead; other captioners used batch size 1.",
                  "GLM calls use temperature 0.6; one run cannot isolate model-count effects from API randomness.",
                  "This experiment checks one video and does not reproduce dataset-level paper metrics.", ""])
    (run_dir / "RESULTS.md").write_text("\n".join(lines))
    print(json.dumps(results, indent=2))
    print(f"Report: {run_dir / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
