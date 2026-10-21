"""Offline regression checks for the paper-alignment path.

These tests use synthetic vectors and mocked API replies.  They must not load
ImageBind, contact the network, decode video, or run the ten-video experiment.
"""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import socket
import unittest
from unittest.mock import patch

import numpy as np

from src.model.image_text_caption_cleaner import ImageTextCaptionCleaner
from src.model.summarize_captions import caption_text
from src.model.memory_embeddings import captions_digest, load_vectors
from src.reproduce import build_commands, read_config
import importlib

scoring = importlib.import_module("src.model.04new")


class _FakeClient:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        self.calls.append(request)
        content = next(self.replies)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    def close(self):
        pass


class _FakeIndex:
    def __init__(self, vectors):
        self.vectors = vectors

    def reconstruct(self, index):
        return self.vectors[index]


class PaperAlignmentTests(unittest.TestCase):
    def setUp(self):
        # Fail locally if a missing mock ever attempts a connection or weight load.
        for target in ("socket.socket.connect", "socket.create_connection", "torch.load",
                       "src.utils.torch_utils.initialize_vlm_model_and_device",
                       "src.model.image_text_caption_cleaner.initialize_vlm_model_and_device"):
            guard = patch(target, side_effect=AssertionError("Network/model loading forbidden"))
            guard.start()
            self.addCleanup(guard.stop)

    def test_top10_keeps_sources_and_causal_window(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            names = []
            vectors = []
            for model in ("m1", "m2", "m3", "m4", "m5"):
                for frame in (0, 16):
                    names.append(f"{model}/demo/{frame}")
                    vectors.append(np.array([1.0, 0.0], dtype=np.float32))
                (root / model).mkdir()
                (root / model / "demo.json").write_text(json.dumps({"0": f"{model}-old", "16": f"{model}-current"}))
            cleaner = ImageTextCaptionCleaner.__new__(ImageTextCaptionCleaner)
            cleaner.frame_interval = 16
            cleaner.captions_dir_template = str(root / "{}")
            output = {}
            cleaner._retrieve_top_captions(_FakeIndex(vectors), names,
                                           np.array([[1.0, 0.0]], dtype=np.float32),
                                           [16], "demo", output)
            self.assertEqual(len(output["16"]), 10)
            self.assertTrue(all(item["source_frame"] <= 16 for item in output["16"]))
            self.assertEqual({item["model"] for item in output["16"]}, {"m1", "m2", "m3", "m4", "m5"})
            self.assertEqual({item["text"] for item in output["16"]},
                             {f"{model}-{suffix}" for model in ("m1", "m2", "m3", "m4", "m5")
                              for suffix in ("old", "current")})

    def test_summary_accepts_top10_and_rejects_future_source(self):
        entries = [{"text": "old event", "source_frame": 0, "rank": 1},
                   {"text": "current event", "source_frame": 16, "rank": 2}]
        self.assertEqual(caption_text(entries, "demo", 16), "old event\ncurrent event")
        with self.assertRaisesRegex(ValueError, "future"):
            caption_text(entries + [{"text": "future", "source_frame": 32, "rank": 1}], "demo", 16)

    def test_paper_memory_uses_ten_steps_and_strict_similarity_gate(self):
        client = _FakeClient([])
        with patch.object(scoring, "ZhipuAI", return_value=client):
            scorer = scoring.GLMAnomalyScorer("offline", ".", 1, 16, "context", "format", ".", ".",
                                              request_workers=1, prefetch_frames=0,
                                              paper_alignment=True, memory_embeddings_dir=".")
        captions = {str(i * 16): f"caption-{i}" for i in range(11)}
        vectors = {str(i * 16): np.array([1.0, 0.0], dtype=np.float32) for i in range(11)}
        vectors["0"] = np.array([0.0, 1.0], dtype=np.float32)
        vectors["16"] = np.array([0.5, np.sqrt(0.75)], dtype=np.float32)
        scorer._memory_vectors = vectors
        long_memory, short_memory = scorer._select_memories(captions, 160)
        self.assertEqual(long_memory, [f"caption-{i}" for i in range(2, 10)])
        self.assertEqual(short_memory, ["caption-8", "caption-9"])
        trace = scorer._memory_trace["160"]["long_candidates"]
        self.assertEqual(len(trace), 10)
        self.assertFalse(trace[0]["retained"])
        self.assertFalse(trace[1]["retained"])

    def test_paper_fusion_uses_previous_raw_and_raw_queue(self):
        client = _FakeClient(["[0.2]", "long", "short", "prediction", "[0.8]",
                              "long", "short", "prediction", "[0.4]"])
        with patch.object(scoring, "ZhipuAI", return_value=client), \
             patch.object(scoring, "load_vectors", return_value={str(i): np.array([1.0]) for i in (0, 16, 32)}):
            scorer = scoring.GLMAnomalyScorer("offline", ".", 1, 16, "context", "format", ".", ".",
                                              request_workers=1, prefetch_frames=0,
                                              paper_alignment=True, memory_embeddings_dir=".")
            video = SimpleNamespace(num_frames=33, path=Path("demo"))
            result = scorer._score_temporal_summaries(video, {"0": "first", "16": "second", "32": "third"})
        self.assertEqual(result["0"]["new_score"], 0.2)
        self.assertAlmostEqual(result["16"]["new_score"], 0.62)
        self.assertEqual(result["16"]["parsed_score"], 0.8)
        self.assertEqual(result["16"]["previous_score"], 0.2)
        self.assertEqual(result["16"]["queue_source"], "raw")
        self.assertEqual(scorer.score_queue[0.2], [0])
        self.assertEqual(scorer.score_queue[0.8], [16])
        self.assertNotIn("smooth_score", result["16"])
        self.assertAlmostEqual(result["32"]["new_score"], 0.52)
        self.assertEqual(result["32"]["previous_score"], 0.8)
        self.assertIn("Event with score 0.8: 'second'.", result["32"]["prompt"])

    def test_top10_excludes_future_and_older_than_five_steps(self):
        cleaner = ImageTextCaptionCleaner.__new__(ImageTextCaptionCleaner)
        cleaner.frame_interval = 16
        # Excluded vectors would rank highest if they entered the candidate set.
        names = ["m/demo/0", "m/demo/112"] + [f"m/demo/{i}" for i in (16, 32, 48, 64, 80, 96)]
        vectors = [np.array([100.0]), np.array([100.0])] + [np.array([float(i)]) for i in range(6)]
        with tempfile.TemporaryDirectory() as temp:
            cleaner.captions_dir_template = str(Path(temp) / "{}")
            (Path(temp) / "m").mkdir()
            (Path(temp) / "m/demo.json").write_text(json.dumps({str(i): str(i) for i in range(0, 113, 16)}))
            output = {}
            cleaner._retrieve_top_captions(_FakeIndex(vectors), names, [np.array([1.0])], [96], "demo", output)
        self.assertEqual([item["source_frame"] for item in output["96"]], [96, 80, 64, 48, 32, 16])
        self.assertEqual([item["rank"] for item in output["96"]], list(range(1, 7)))

    def test_memory_vector_validation_rejects_stale_or_invalid_data(self):
        captions = {"0": "first", "16": "second"}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "vectors.npz"
            def save(vectors):
                np.savez(path, frames=[0, 16], vectors=vectors,
                         encoder="imagebind_huge_text", captions_sha256=captions_digest(captions))
            save([[2., 0.], [0., 3.]])
            actual = load_vectors(path, captions)
            np.testing.assert_allclose(actual["0"], [1., 0.])
            with self.assertRaisesRegex(ValueError, "Stale"):
                load_vectors(path, dict(captions, **{"16": "changed"}))
            for invalid in ([[0., 0.], [1., 0.]], [[float("nan"), 0.], [1., 0.]]):
                save(invalid)
                with self.assertRaises(ValueError):
                    load_vectors(path, captions)

    def test_paper_entry_flags_and_legacy_default(self):
        root = Path(__file__).resolve().parents[1]
        legacy = read_config(root / "configs/ucf.json")
        self.assertNotIn("--paper_alignment", build_commands(legacy, "score")[0])
        paper = dict(legacy, paper_alignment=True, memory_embeddings_dir="/tmp/offline-vectors",
                     score_alpha=0.7, score_previous="raw", score_queue_source="raw")
        for stage in ("caption-index", "clean", "score"):
            self.assertIn("--paper_alignment", build_commands(paper, stage)[0])
        command = build_commands(paper, "memory-embeddings")[0]
        self.assertIn("src.model.memory_embeddings", command)
        self.assertEqual(command[command.index("--output_dir") + 1], paper["memory_embeddings_dir"])


if __name__ == "__main__":
    unittest.main()
