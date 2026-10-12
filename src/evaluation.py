"""CPU evaluation with explicit frame alignment and temporal ground truth."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score

from src.data.video_record import VideoRecord


def temporal_testing_annotations(temporal_annotation_file, dataset="ucf", temporal_index_base=0):
    annotations = {}
    with open(temporal_annotation_file) as handle:
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            name = Path(parts[0]).stem
            # UCF: filename category start end ...; XD: filename start end ...
            values = parts[2:] if dataset == "ucf" else parts[1:]
            if len(values) % 2:
                raise ValueError(f"Unpaired temporal boundaries for {name}")
            intervals = []
            for start, end in zip(values[::2], values[1::2]):
                start, end = int(start), int(end)
                if (start, end) == (-1, -1):
                    continue
                if start < 0 or end < start:
                    raise ValueError(f"Invalid temporal interval for {name}: {start}, {end}")
                start, end = start - temporal_index_base, end - temporal_index_base
                if start < 0:
                    raise ValueError(f"Temporal index base is incompatible with {name}")
                intervals.append((start, end))
            if name in annotations:
                raise ValueError(f"Duplicate temporal annotation: {name}")
            annotations[name] = intervals
    return annotations


def get_video_labels(video_record, annotations, normal_label=0):
    name = video_record.path.name
    if name not in annotations:
        raise ValueError(f"Missing temporal ground truth for {name}; include normal videos explicitly")
    labels = np.full(video_record.num_frames, normal_label, dtype=int)
    anomaly_label = 1 if normal_label != 1 else 0
    for start, end in annotations[name]:
        if end >= video_record.num_frames:
            raise ValueError(f"Temporal boundary exceeds frame count for {name}: {end}")
        labels[start:end + 1] = anomaly_label
    return labels


def score_value(value, num_neighbors):
    if isinstance(value, dict):
        if "new_score" in value:
            return score_value(value["new_score"], num_neighbors)
        # Refiner output: {neighbor_id: scalar_or_score_record}.
        keys = sorted(value, key=int)[:num_neighbors]
        if not keys:
            raise ValueError("Empty neighbor scores")
        scores = [score_value(value[key], num_neighbors) for key in keys]
        valid = [score for score in scores if score != -1]
        return float(np.mean(valid)) if valid else -1.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Expected numeric anomaly score, got {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value) or (value != -1 and not 0 <= value <= 1):
        raise ValueError(f"Anomaly score must be -1 or in [0, 1], got {value}")
    return value


def replace_negative_scores_with_previous(scores_dict, num_neighbors=10):
    result, previous = {}, 0.0
    for key in sorted(scores_dict, key=int):
        value = score_value(scores_dict[key], num_neighbors)
        if value != -1:
            previous = value
        result[key] = previous
    return result


def calculate_weighted_scores(scores_dict, num_neighbors, frame_interval, num_frames=None):
    # Keys are source frame IDs (0, 16, 32, ...), not consecutive sample IDs.
    if frame_interval < 1 or num_neighbors < 1 or not scores_dict:
        raise ValueError("Scores, frame_interval and num_neighbors must be nonempty/positive")
    ordered = sorted(scores_dict, key=int)
    expected_count = (num_frames + frame_interval - 1) // frame_interval if num_frames else len(ordered)
    expected = list(range(0, expected_count * frame_interval, frame_interval))
    if [int(key) for key in ordered] != expected:
        raise ValueError("Score keys must cover every sampled source frame; missing or extra scores")
    numeric = replace_negative_scores_with_previous(scores_dict, num_neighbors)
    scores = np.repeat([numeric[key] for key in ordered], frame_interval)
    return scores[:num_frames] if num_frames is not None else scores


def load_json_file(file_path):
    with open(file_path) as handle:
        return json.load(handle)


def save_metric(output_dir, metric_name, num_neighbors, metric_value):
    (output_dir / f"{metric_name}_nn_{num_neighbors}.txt").write_text(f"{metric_value}\n")


def main(root_path, annotationfile_path, temporal_annotation_file, raw_scores_dir,
         captions_dir, output_dir, frame_interval=16, normal_label=0,
         num_neighbors=10, without_labels=False, visualize=False, video_fps=None,
         dataset="ucf", frame_labels_file=None, temporal_index_base=0):
    if visualize:
        raise ValueError("Video rendering is unavailable in this checkout; use the saved frame arrays")
    if normal_label is None:
        raise ValueError("normal_label must be explicit")
    with open(annotationfile_path) as handle:
        videos = [VideoRecord(line.split(), root_path) for line in handle if line.strip()]
    if not videos or len({video.path.name for video in videos}) != len(videos):
        raise ValueError("Video list must be nonempty and contain unique names")
    annotations = None
    ground_truth = None
    if not without_labels:
        if frame_labels_file:
            ground_truth = np.load(frame_labels_file, allow_pickle=False).reshape(-1)
            if len(ground_truth) != sum(video.num_frames for video in videos):
                raise ValueError("Frame ground truth length does not match the ordered video list")
            if not np.isin(ground_truth, [0, 1]).all():
                raise ValueError("Frame ground truth must contain binary labels")
        else:
            annotations = temporal_testing_annotations(temporal_annotation_file, dataset, temporal_index_base)
    flat_scores, flat_labels = [], []
    for video in videos:
        scores_dict = load_json_file(Path(raw_scores_dir) / f"{video.path.name}.json")
        scores = calculate_weighted_scores(scores_dict, num_neighbors, frame_interval, video.num_frames)
        flat_scores.extend(scores)
        if annotations is not None:
            flat_labels.extend(get_video_labels(video, annotations, normal_label) != normal_label)
    scores = np.asarray(flat_scores)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"dataset": dataset, "videos": len(videos), "frames": len(scores),
               "frame_interval": frame_interval, "num_neighbors": num_neighbors}
    if not without_labels:
        labels = ground_truth if ground_truth is not None else np.asarray(flat_labels)
        if len(np.unique(labels)) != 2:
            raise ValueError("Evaluation requires both normal and anomalous frames")
        precision, recall, _ = precision_recall_curve(labels, scores)
        metrics.update(roc_auc=float(roc_auc_score(labels, scores)),
                       pr_auc=float(auc(recall, precision)),
                       average_precision=float(average_precision_score(labels, scores)))
        for name in ("roc_auc", "pr_auc", "average_precision"):
            save_metric(output_dir, name, num_neighbors, metrics[name])
            print(f"{name}: {metrics[name]}")
        np.save(output_dir / "frame_labels.npy", labels)
    np.save(output_dir / "frame_scores.npy", scores)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def parse_args(dataset="ucf"):
    parser = argparse.ArgumentParser(description=f"CPU evaluation for {dataset}")
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--annotationfile_path", required=True)
    parser.add_argument("--temporal_annotation_file")
    parser.add_argument("--frame_labels_file", help="Binary .npy in the exact video-list order")
    parser.add_argument("--temporal_index_base", type=int, choices=(0, 1), default=0)
    parser.add_argument("--raw_scores_dir", required=True)
    parser.add_argument("--captions_dir", help="Accepted for compatibility; metrics need only scores")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--frame_interval", type=int, default=16)
    parser.add_argument("--normal_label", type=int, default=0)
    parser.add_argument("--num_neighbors", type=int, default=10)
    parser.add_argument("--without_labels", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--video_fps", type=float)
    args = parser.parse_args()
    if not args.without_labels and not (args.temporal_annotation_file or args.frame_labels_file):
        parser.error("Provide temporal_annotation_file or frame_labels_file for labeled evaluation")
    if args.temporal_annotation_file and args.frame_labels_file:
        parser.error("Choose one ground truth format")
    return args
