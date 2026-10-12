"""Search only captions whose source frames have already been observed."""
import faiss
import numpy as np


def search_past_captions(index, file_names, vectors, frame_indices, num_neighbors):
    if num_neighbors < 1 or index.ntotal != len(file_names):
        raise ValueError("Invalid neighbor count or index/metadata length mismatch")
    if len(vectors) != len(frame_indices):
        raise ValueError("Search vectors must match the query frame count")
    source_frames = [int(name.rsplit("/", 1)[-1]) for name in file_names]
    distances, indices = [], []
    for vector, frame_idx in zip(vectors, frame_indices):
        available = [i for i, source in enumerate(source_frames) if source <= frame_idx]
        if not available:
            raise ValueError(f"No causal captions available at frame {frame_idx}")
        causal_index = faiss.IndexFlatIP(index.d)
        causal_index.add(np.stack([index.reconstruct(i) for i in available]))
        scores, positions = causal_index.search(
            np.asarray([vector], dtype=np.float32), min(num_neighbors, len(available))
        )
        distances.append(scores[0])
        indices.append([available[int(position)] for position in positions[0]])
    return distances, indices
