"""Precompute ImageBind summary vectors; scoring reads them without loading a model."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.data.video_record import VideoRecord


def captions_digest(captions):
    payload = json.dumps(captions, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_vectors(path, captions):
    frames = sorted(map(int, captions))
    with np.load(path, allow_pickle=False) as saved:
        if (str(saved["encoder"].item()) != "imagebind_huge_text"
                or str(saved["captions_sha256"].item()) != captions_digest(captions)
                or not np.array_equal(saved["frames"], frames)):
            raise ValueError(f"Stale or mismatched memory vectors: {path}")
        vectors = np.asarray(saved["vectors"], dtype=np.float32)
    if (vectors.ndim != 2 or len(vectors) != len(frames) or vectors.shape[1] == 0
            or not np.isfinite(vectors).all()):
        raise ValueError(f"Invalid memory vectors: {path}")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise ValueError(f"Zero memory vector: {path}")
    vectors = vectors / norms
    return {str(frame): vector for frame, vector in zip(frames, vectors)}


def run(root_path, annotationfile_path, captions_dir, output_dir,
        frame_interval=16, batch_size=4, resume=False):
    if batch_size < 1 or frame_interval < 1:
        raise ValueError("batch_size and frame_interval must be positive")
    # Only this explicit stage imports/loads ImageBind. Score workers never do.
    import torch
    from imagebind import data
    from imagebind.models.imagebind_model import ModalityType
    from src.utils.torch_utils import initialize_vlm_model_and_device

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, device = initialize_vlm_model_and_device()
    for row in Path(annotationfile_path).read_text().splitlines():
        if not row.strip():
            continue
        video = VideoRecord(row.split(), root_path)
        captions = json.loads((Path(captions_dir) / f"{video.path.name}.json").read_text())
        frames = list(range(0, video.num_frames, frame_interval))
        if (set(captions) != {str(frame) for frame in frames}
                or not all(isinstance(text, str) and text.strip() for text in captions.values())):
            raise ValueError(f"Incomplete summaries: {video.path.name}")
        output_path = output_dir / f"{video.path.name}.npz"
        if resume and output_path.exists():
            load_vectors(output_path, captions)
            continue
        unique = list(dict.fromkeys(captions[str(frame)] for frame in frames))
        batches = []
        for start in range(0, len(unique), batch_size):
            inputs = {ModalityType.TEXT: data.load_and_transform_text(unique[start:start + batch_size], device)}
            with torch.no_grad():
                batches.append(model(inputs)[ModalityType.TEXT].cpu().numpy())
        encoded = np.concatenate(batches)
        encoded /= np.linalg.norm(encoded, axis=1, keepdims=True)
        lookup = dict(zip(unique, encoded))
        vectors = np.stack([lookup[captions[str(frame)]] for frame in frames])
        temporary = output_path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, frames=frames, vectors=vectors,
                            captions_sha256=captions_digest(captions), encoder="imagebind_huge_text")
        temporary.replace(output_path)
        print(f"Memory vectors saved: {output_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root_path", "annotationfile_path", "captions_dir", "output_dir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--frame_interval", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    run(**vars(parser.parse_args()))
