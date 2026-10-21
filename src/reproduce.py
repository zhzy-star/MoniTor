"""Configure, inspect, and run one explicit stage of the reproduction pipeline."""
import argparse
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

from src.data.video_record import VideoRecord

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("extract", "caption", "caption-index", "clean", "summarize", "memory-embeddings", "score",
          "summary-index", "refine", "evaluate")


def read_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    if config.get("dataset") not in {"ucf", "xd"}:
        raise ValueError("dataset must be ucf or xd")
    for key, value in list(config.items()):
        if key.endswith(("_dir", "_root", "_path", "_file")):
            # Make paths absolute while preserving the canonical symlink view.
            config[key] = os.path.abspath(ROOT / Path(value).expanduser())
    for key in ("frame_interval", "batch_size", "num_samples", "num_neighbors", "fps", "clip_duration"):
        if config.get(key, 0) <= 0:
            raise ValueError(f"{key} must be positive")
    paper = config.get("paper_alignment", False)
    if not isinstance(paper, bool) or (paper and config["dataset"] != "ucf"):
        raise ValueError("paper_alignment must be boolean and is supported only for ucf")
    alpha = config.get("score_alpha", 0.7 if paper else 0.6)
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("score_alpha must be a number between 0 and 1")
    for key in ("score_previous", "score_queue_source"):
        if config.get(key, "raw") not in ("raw", "fused"):
            raise ValueError(f"{key} must be raw or fused")
    if paper and not config.get("memory_embeddings_dir"):
        raise ValueError("paper_alignment requires memory_embeddings_dir")
    models = config.get("caption_models", [])
    for key in ("summary_workers", "score_workers", "score_request_workers"):
        workers = config.get(key, 1)
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError(f"{key} must be a positive integer")
    prefetch = config.get("score_prefetch_frames", 1)
    if isinstance(prefetch, bool) or not isinstance(prefetch, int) or prefetch not in (0, 1):
        raise ValueError("score_prefetch_frames must be 0 or 1")
    allowed_models = {"blip2-flan-t5-xl", "blip2-flan-t5-xl-coco", "blip2-flan-t5-xxl",
                      "blip2-opt-6.7b", "blip2-opt-6.7b-coco"}
    if not models or len(set(models)) != len(models) or not set(models) <= allowed_models:
        raise ValueError("caption_models must list unique supported BLIP2 model names")
    return config


def command(module, **arguments):
    result = [sys.executable, "-m", module]
    for key, value in arguments.items():
        if value is None or value is False:
            continue
        result.append(f"--{key}")
        if value is not True:
            result.extend(str(item) for item in (value if isinstance(value, list) else [value]))
    return result


def build_commands(config, stage, score_source="raw", resume=False):
    c = config
    shared = dict(root_path=c["root_path"], annotationfile_path=c["annotationfile_path"],
                  batch_size=c["batch_size"], frame_interval=c["frame_interval"])
    clip = dict(fps=c["fps"], clip_duration=c["clip_duration"], num_samples=c["num_samples"],
                num_neighbors=c["num_neighbors"])
    if stage == "extract":
        return [command("src.preprocessing.extract_frames", videos_dir=c["videos_dir"],
                        frames_dir=c["root_path"], annotations_file=c["annotationfile_path"],
                        video_split_file=c.get("video_split_file"))]
    if stage == "caption":
        return [command("src.model.image_captioner", **shared,
                        pretrained_model_name=str(Path(c["model_root"]) / model) if c.get("model_root") else f"Salesforce/{model}",
                        device_map=c.get("caption_device_map"),
                        max_gpu_memory=c.get("caption_max_gpu_memory"),
                        output_dir=str(Path(c["raw_caption_root"]) / model), resume=resume)
                for model in c["caption_models"]]
    if stage == "caption-index":
        return [command("src.model.create_index", **shared,
                        captions_dirs=[str(Path(c["raw_caption_root"]) / model) for model in c["caption_models"]],
                        output_dir=c["caption_index_dir"],
                        paper_alignment=c.get("paper_alignment", False))]
    if stage == "clean":
        return [command("src.model.image_text_caption_cleaner", **shared, **clip,
                        captions_dir_template=str(Path(c["raw_caption_root"]) / "{}"),
                        index_dir=c["caption_index_dir"], output_dir=c["clean_captions_dir"], resume=resume,
                        paper_alignment=c.get("paper_alignment", False))]
    if stage == "summarize":
        return [command("src.model.summarize_captions", root_path=c["root_path"],
                        annotationfile_path=c["annotationfile_path"], frame_interval=c["frame_interval"],
                        captions_dir=c["clean_captions_dir"], output_dir=c["summaries_dir"],
                        prompt_file=c["summary_prompt_file"], resume=resume,
                        workers=c.get("summary_workers", 1))]
    if stage == "memory-embeddings":
        return [command("src.model.memory_embeddings", **shared,
                        captions_dir=c["summaries_dir"], output_dir=c["memory_embeddings_dir"], resume=resume)]
    if stage == "score":
        paper_args = dict(
            paper_alignment=c.get("paper_alignment", False),
            memory_embeddings_dir=c.get("memory_embeddings_dir") if c.get("paper_alignment") else None,
            score_alpha=c.get("score_alpha"),
            score_previous=c.get("score_previous", "raw"),
            score_queue_source=c.get("score_queue_source", "raw"),
        ) if c["dataset"] == "ucf" else {}
        return [command("src.model.04new" if c["dataset"] == "ucf" else "src.model.xd_04",
                        **dict(shared, batch_size=1), captions_dir=c["summaries_dir"],
                        output_scores_dir=c["raw_scores_dir"],
                        num_jobs=c.get("score_workers", 1) if c["dataset"] == "ucf" else None,
                        request_workers=c.get("score_request_workers", 3) if c["dataset"] == "ucf" else None,
                        prefetch_frames=c.get("score_prefetch_frames", 1) if c["dataset"] == "ucf" else None,
                        resume=resume, **paper_args,
                        context_prompt=Path(c["context_prompt_file"]).read_text().strip(),
                        format_prompt=Path(c["format_prompt_file"]).read_text().strip())]
    if stage == "summary-index":
        return [command("src.model.create_summary_index", **shared,
                        captions_dir=c["summaries_dir"], output_dir=c["summary_index_dir"])]
    if stage == "refine":
        ret = Path(c["retrieval_dir"])
        return [command("src.model.video_text_score_refiner", **shared, **clip,
                        captions_dir=c["summaries_dir"], index_dir=c["summary_index_dir"],
                        scores_dir=c["raw_scores_dir"], output_scores_dir=c["refined_scores_dir"],
                        output_summary_dir=str(ret / "captions"), output_similarity_dir=str(ret / "similarity"),
                        output_indices_dir=str(ret / "indices"), output_filenames_dir=str(ret / "filenames"),
                        resume=resume)]
    return [command(f"src.{c['dataset']}_eval", root_path=c["root_path"],
                    annotationfile_path=c["annotationfile_path"],
                    temporal_annotation_file=c.get("temporal_annotation_file"),
                    frame_labels_file=c.get("frame_labels_file"),
                    temporal_index_base=c.get("temporal_index_base", 0),
                    raw_scores_dir=c["raw_scores_dir" if score_source == "raw" else "refined_scores_dir"],
                    output_dir=str(Path(c["output_dir"]) / score_source), frame_interval=c["frame_interval"],
                    num_neighbors=c["num_neighbors"], normal_label=c["normal_label"])]


def check_inputs(c, stage, score_source="raw"):
    errors = []
    def require(path):
        if not Path(path).exists():
            errors.append(f"Missing input: {path}")
    require(ROOT / "libs/ImageBind/imagebind/models/imagebind_model.py")
    if stage == "extract":
        require(c["videos_dir"])
        if c.get("video_split_file"):
            require(c["video_split_file"])
        if Path(c["annotationfile_path"]).exists():
            errors.append("Extraction would overwrite the video list; choose a new annotationfile_path")
        return errors
    if stage == "caption" and c.get("model_root"):
        for model in c["caption_models"]:
            model_dir = Path(c["model_root"]) / model
            require(model_dir / "config.json")
            require(model_dir / "preprocessor_config.json")
    require(c["annotationfile_path"])
    if errors:
        return errors
    with open(c["annotationfile_path"]) as handle:
        videos = [VideoRecord(line.split(), c["root_path"]) for line in handle if line.strip()]
    if not videos or len({v.path.name for v in videos}) != len(videos):
        errors.append("Video list must be nonempty and contain unique names")
    directory_keys = {
        "summarize": ["clean_captions_dir"], "memory-embeddings": ["summaries_dir"], "score": ["summaries_dir"],
        "summary-index": ["summaries_dir"], "refine": ["summaries_dir", "raw_scores_dir"],
        "evaluate": ["raw_scores_dir" if score_source == "raw" else "refined_scores_dir"],
    }.get(stage, [])
    directories = [Path(c[key]) for key in directory_keys]
    if stage in {"caption-index", "clean"}:
        directories.extend(Path(c["raw_caption_root"]) / model for model in c["caption_models"])
    for video in videos:
        for directory in directories:
            path = directory / f"{video.path.name}.json"
            require(path)
            if path.exists():
                try:
                    payload = json.loads(path.read_text())
                    if not isinstance(payload, dict):
                        raise ValueError("expected a frame-keyed JSON object")
                    expected = {str(idx) for idx in range(0, video.num_frames, c["frame_interval"])}
                    if set(payload) != expected:
                        raise ValueError("missing or extra sampled source frame IDs")
                except (ValueError, TypeError) as error:
                    errors.append(f"Invalid {path}: {error}")
        if stage in {"caption", "clean", "refine"}:
            require(video.path)
            if video.path.exists():
                for idx in range(video.num_frames):
                    frame_path = video.path / f"{idx:06d}.jpg"
                    if not frame_path.is_file():
                        errors.append(f"Missing frame: {frame_path}")
                        break
        if stage in {"clean", "refine"}:
            key = "caption_index_dir" if stage == "clean" else "summary_index_dir"
            for suffix in ("json", "bin"):
                require(Path(c[key]) / f"{video.path.name}.{suffix}")
    if c.get("paper_alignment") and stage == "clean":
        for video in videos:
            require(Path(c["caption_index_dir"]) / f"{video.path.name}.paper.json")
    if c.get("paper_alignment") and stage == "score":
        from src.model.memory_embeddings import load_vectors
        for video in videos:
            vector_path = Path(c["memory_embeddings_dir"]) / f"{video.path.name}.npz"
            require(vector_path)
            summary_path = Path(c["summaries_dir"]) / f"{video.path.name}.json"
            if vector_path.exists() and summary_path.exists():
                try:
                    load_vectors(vector_path, json.loads(summary_path.read_text()))
                except (ValueError, KeyError, OSError) as error:
                    errors.append(str(error))
    if stage == "evaluate":
        if not (c.get("temporal_annotation_file") or c.get("frame_labels_file")):
            errors.append("Configure temporal_annotation_file or frame_labels_file")
        for key in ("temporal_annotation_file", "frame_labels_file"):
            if c.get(key):
                require(c[key])
        if not errors:
            from src.evaluation import calculate_weighted_scores, get_video_labels, temporal_testing_annotations
            if c.get("frame_labels_file"):
                import numpy as np
                labels = np.load(c["frame_labels_file"], allow_pickle=False).reshape(-1)
                if len(labels) != sum(v.num_frames for v in videos) or not np.isin(labels, [0, 1]).all():
                    errors.append("Binary frame ground truth must match the exact ordered video list")
                elif len(np.unique(labels)) != 2:
                    errors.append("Ground truth must include normal and anomalous frames")
            else:
                annotations = temporal_testing_annotations(c["temporal_annotation_file"], c["dataset"], c.get("temporal_index_base", 0))
                for video in videos:
                    get_video_labels(video, annotations, c["normal_label"])
            for video in videos:
                payload = json.loads((directories[0] / f"{video.path.name}.json").read_text())
                calculate_weighted_scores(payload, c["num_neighbors"], c["frame_interval"], video.num_frames)
    if stage in {"summarize", "score"}:
        for key in (("summary_prompt_file",) if stage == "summarize" else ("context_prompt_file", "format_prompt_file")):
            require(c[key])
        if not os.environ.get("ZHIPUAI_API_KEY"):
            errors.append("Set ZHIPUAI_API_KEY before running this API stage")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--score-source", choices=("raw", "refined"), default="raw")
    parser.add_argument("--check", action="store_true", help="Validate inputs without loading models or sending API calls")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without checking data or executing")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        config = read_config(args.config)
        if not args.dry_run:
            errors = check_inputs(config, args.stage, args.score_source)
            if errors:
                for error in errors[:20]:
                    print(error, file=sys.stderr)
                if len(errors) > 20:
                    print(f"... {len(errors) - 20} additional input errors", file=sys.stderr)
                return 2
        if args.check and not args.dry_run:
            print(f"Input checks passed: {config['dataset']}/{args.stage}")
            return 0
        for cmd in build_commands(config, args.stage, args.score_source, args.resume):
            print(shlex.join(cmd), flush=True)
            if not args.dry_run:
                subprocess.run(cmd, cwd=ROOT, check=True)
        return 0
    except (ValueError, OSError, KeyError, ImportError) as error:
        print(f"Reproduction input error: {error}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as error:
        return error.returncode


if __name__ == "__main__":
    raise SystemExit(main())
