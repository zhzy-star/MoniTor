"""Regression checks use CPU arrays, synthetic videos, and mocked API replies."""
import contextlib
import importlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import faiss
import numpy as np
import torch

from src.data.video_record import VideoRecord
from src.evaluation import calculate_weighted_scores, main as evaluate, temporal_testing_annotations
from src.model.image_text_caption_cleaner import ImageTextCaptionCleaner
from src.model.image_captioner import ImageCaptioner
from src.model.video_text_score_refiner import VideoTextScoreRefiner
from src.preprocessing.extract_frames import main as extract_frames
from src.reproduce import STAGES, build_commands, check_inputs, read_config
from src.utils.path_utils import find_unprocessed_videos
from src.utils.retrieval_utils import search_past_captions


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write_json(self, path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
        return path

    def test_extraction_preserves_frames_and_nested_video_paths(self):
        videos = self.root / "videos" / "category"
        videos.mkdir(parents=True)
        writer = cv2.VideoWriter(str(videos / "demo.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32))
        self.assertTrue(writer.isOpened())
        for index in range(19):
            writer.write(np.full((32, 32, 3), index * 10, dtype=np.uint8))
        writer.release()
        with contextlib.redirect_stdout(io.StringIO()):
            extract_frames(self.root / "videos", self.root / "frames", str(self.root / "list.txt"))
        self.assertEqual((self.root / "list.txt").read_text(), "demo 0 18 0\n")
        frames = sorted((self.root / "frames" / "demo").glob("*.jpg"))
        self.assertEqual(len(frames), 19)
        means = [cv2.imread(str(frame)).mean() for frame in frames]
        self.assertTrue(all(right > left for left, right in zip(means, means[1:])))

    def test_frame_alignment_and_refined_score_schema(self):
        unordered = {"32": {"new_score": 0.9}, "0": 0.1, "16": {"new_score": -1}}
        scores = calculate_weighted_scores(unordered, 10, 16, 35)
        np.testing.assert_allclose(scores, [0.1] * 32 + [0.9] * 3)
        refined = {"0": {"0": {"new_score": 0.2}},
                   "16": {"0": {"new_score": 0.6}, "1": 0.2}}
        np.testing.assert_allclose(calculate_weighted_scores(refined, 2, 16, 18), [0.2] * 16 + [0.4] * 2)
        with self.assertRaisesRegex(ValueError, "missing or extra"):
            calculate_weighted_scores({"0": 0.1, "32": 0.9}, 10, 16, 35)

    def test_extraction_uses_only_official_split_in_its_order(self):
        split = self.root / "official_test.txt"
        split.write_text("b.mp4\na.mp4\n")
        annotations = self.root / "list.txt"
        with patch("src.preprocessing.extract_frames.extract_frames_with_sliding_window",
                   side_effect=[("b", 18), ("a", 19)]) as extract:
            extract_frames(self.root / "videos", self.root / "frames", annotations,
                           video_split_file=split)
        self.assertEqual([Path(call.args[0]).name for call in extract.call_args_list], ["b.mp4", "a.mp4"])
        self.assertEqual(annotations.read_text(), "b 0 17 0\na 0 18 0\n")

    def test_blip_local_model_loading_and_missing_path_failure(self):
        model_dir = self.root / "models" / "blip2-flan-t5-xl"
        model_dir.mkdir(parents=True)
        with patch("src.model.image_captioner.torch.cuda.is_available", return_value=False), \
             patch("src.model.image_captioner.Blip2Processor.from_pretrained") as processor, \
             patch("src.model.image_captioner.Blip2ForConditionalGeneration.from_pretrained") as model:
            ImageCaptioner(1, 16, "{:06d}.jpg", model_dir, "float32", self.root / "captions")
            processor.assert_called_once_with(str(model_dir), local_files_only=True, use_fast=False)
            model.assert_called_once_with(str(model_dir), torch_dtype=torch.float32, local_files_only=True)
            model.return_value.to.assert_called_once_with("cpu")
            processor.reset_mock()
            model.reset_mock()
            with self.assertRaisesRegex(FileNotFoundError, "BLIP model directory"):
                ImageCaptioner(1, 16, "{:06d}.jpg", model_dir / "missing", "float32", self.root / "captions")
            processor.assert_not_called()
            model.assert_not_called()
        config = read_config(Path(__file__).resolve().parents[1] / "configs/ucf.json")
        commands = build_commands(config, "caption")
        for command, name in zip(commands, config["caption_models"]):
            self.assertEqual(command[command.index("--pretrained_model_name") + 1],
                             f"/root/autodl-tmp/models/{name}")
        self.assertEqual(config["videos_dir"], "/root/autodl-tmp/video_add/ucf_crime/videos/test")
        self.assertEqual(config["temporal_annotation_file"],
                         "/root/autodl-tmp/video_add/ucf_crime/annotations/temporal_test.txt")

    def test_cpu_evaluation_uses_temporal_labels_and_all_frames(self):
        records = self.root / "list.txt"
        records.write_text("normal 0 16 0\nanomaly 0 17 0\n")
        truth = self.root / "temporal.txt"
        truth.write_text("normal.mp4 Normal -1 -1\nanomaly.mp4 Abuse 16 17\n")
        score_dir = self.root / "scores"
        self.write_json(score_dir / "normal.json", {"0": 0.0, "16": 0.0})
        self.write_json(score_dir / "anomaly.json", {"16": {"new_score": 1.0}, "0": {"new_score": 0.0}})
        with contextlib.redirect_stdout(io.StringIO()):
            metrics = evaluate(self.root, records, truth, score_dir, None, self.root / "out")
        self.assertEqual(metrics["frames"], 35)
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["average_precision"], 1.0)
        labels = np.load(self.root / "out" / "frame_labels.npy")
        np.testing.assert_array_equal(labels, [0] * 33 + [1] * 2)
        truth.write_text("anomaly.mp4 Abuse 16 17\n")
        with self.assertRaisesRegex(ValueError, "Missing temporal ground truth"):
            evaluate(self.root, records, truth, score_dir, None, self.root / "bad")

    def test_automatic_model_placement_preserves_accelerate_dispatch(self):
        model_dir = self.root / "large_model"
        model_dir.mkdir()
        with patch("src.model.image_captioner.torch.cuda.is_available", return_value=True), \
             patch("src.model.image_captioner.torch.cuda.device_count", return_value=1), \
             patch("src.model.image_captioner.Blip2Processor.from_pretrained"), \
             patch("src.model.image_captioner.Blip2ForConditionalGeneration.from_pretrained") as model:
            ImageCaptioner(1, 16, "{:06d}.jpg", model_dir, "float16", self.root / "out",
                           device_map="auto", max_gpu_memory="20GiB")
            model.assert_called_once_with(str(model_dir), torch_dtype=torch.float16,
                                          local_files_only=True, device_map="auto",
                                          low_cpu_mem_usage=True,
                                          max_memory={0: "20GiB", "cpu": "120GiB"})
            model.return_value.to.assert_not_called()
            model.return_value.eval.assert_called_once()
        config = read_config(Path(__file__).resolve().parents[1] / "configs/ucf.json")
        for cmd in build_commands(config, "caption", resume=True):
            self.assertEqual(cmd[cmd.index("--device_map") + 1], "auto")
            self.assertIn("--resume", cmd)
        self.assertIn("--resume", build_commands(config, "score", resume=True)[0])

    def test_xd_ground_truth_array_and_index_base(self):
        truth = self.root / "temporal.txt"
        truth.write_text("demo.mp4 1 2\n")
        self.assertEqual(temporal_testing_annotations(truth, "xd", 1), {"demo": [(0, 1)]})
        records = self.root / "list.txt"
        records.write_text("demo 0 17 0\n")
        scores = self.root / "scores"
        self.write_json(scores / "demo.json", {"0": 0.0, "16": 1.0})
        labels_file = self.root / "gt.npy"
        np.save(labels_file, [0] * 16 + [1] * 2)
        with contextlib.redirect_stdout(io.StringIO()):
            metrics = evaluate(self.root, records, None, scores, None, self.root / "out",
                               dataset="xd", frame_labels_file=labels_file)
        self.assertEqual(metrics["average_precision"], 1.0)
        np.save(labels_file, [0, 1])
        with self.assertRaisesRegex(ValueError, "length"):
            evaluate(self.root, records, None, scores, None, self.root / "bad",
                     dataset="xd", frame_labels_file=labels_file)

    def test_online_retrieval_never_uses_future_or_negative_neighbors(self):
        index = faiss.IndexFlatIP(2)
        index.add(np.array([[0, 1], [1, 0], [1, 0]], dtype=np.float32))
        names = ["model/demo/0", "model/demo/16", "model/demo/32"]
        _, indices = search_past_captions(index, names, np.array([[1, 0]] * 3, dtype=np.float32), [0, 16, 32], 10)
        self.assertEqual(indices[0], [0])
        self.assertEqual(set(indices[1]), {0, 1})
        self.assertEqual(set(indices[2]), {0, 1, 2})

    def test_cleaner_maps_variable_clip_lengths_to_correct_centers(self):
        cleaner = ImageTextCaptionCleaner(None, "cpu", self.root / "clean", 10, 1,
                                          self.root / "index", str(self.root / "{}"),
                                          10, 30, "{:06d}.jpg", 2, 16)
        video = VideoRecord(["demo", "0", "32", "0"], self.root)
        indices, paths = cleaner._prepare_frame_data(video, [0, 16], 300)
        self.assertEqual(cleaner._sample_centers, [0] + [16] * 10)
        self.write_json(self.root / "model" / "demo.json", {"0": "scene"})
        result = {}
        from collections import defaultdict
        result = defaultdict(dict)
        cleaner._retrieve_captions([[0]] * len(paths), indices, ["model/demo/0"],
                                  "demo", [0, 16], result)
        self.assertEqual(len(result["0"]), 1)
        self.assertEqual(len(result["16"]), 10)
        self.assertTrue(all(int(key) <= 16 for key in result["16"]))

    def test_refiner_reads_jpeg_clips_and_preserves_score_records(self):
        class FakeModel:
            def __call__(self, inputs):
                return {"vision": inputs["vision"]}
        refiner = VideoTextScoreRefiner(FakeModel(), "cpu", self.root / "refined",
                                       self.root / "summary", self.root / "similarity",
                                       self.root / "indices", self.root / "names", 10, 10,
                                       self.root / "index", self.root / "captions", self.root / "scores",
                                       10, 30, "{:06d}.jpg", 1, 16)
        with patch("src.model.video_text_score_refiner.data.load_and_transform_vision_data",
                   side_effect=[torch.tensor([[1., 0.]]), torch.tensor([[0., 1.], [1., 0.]])]):
            inputs = refiner._load_and_transform_data([["0.jpg"], ["0.jpg", "16.jpg"]])
            vectors = refiner._calculate_search_vectors(inputs)
        np.testing.assert_allclose(vectors, [[1, 0], [0.5, 0.5]])
        video = VideoRecord(["demo", "0", "17", "0"], self.root)
        self.write_json(self.root / "scores" / "demo.json", {"0": {"new_score": 0.1}, "16": {"new_score": 0.9}})
        self.write_json(self.root / "names" / "demo.json", {"0": {"0": "demo/0"}, "16": {"0": "demo/16", "1": "demo/0"}})
        refiner.refine_scores(video)
        result = json.loads((self.root / "refined" / "demo.json").read_text())
        self.assertEqual(result["16"]["0"]["new_score"], 0.9)
        self.assertEqual(len(calculate_weighted_scores(result, 10, 16, 18)), 18)

    def test_mocked_api_summary_to_score_chain_and_failure_propagation(self):
        summary_module = importlib.import_module("src.model.summarize_captions")
        records = self.root / "list.txt"
        records.write_text("demo 0 17 0\n")
        prompt = self.root / "prompt.txt"
        prompt.write_text("Summarize past frames.")
        self.write_json(self.root / "clean" / "demo.json", {"0": {"0": "normal scene"}, "16": {"0": "normal scene", "16": "fighting scene"}})
        calls = []
        def reply(**kwargs):
            calls.append(kwargs)
            content = kwargs["messages"][-1]["content"]
            payload = {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
                                   to_dict=lambda **options: payload)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=reply)), close=lambda: None)
        with patch.object(summary_module, "ZhipuAI", return_value=client):
            summary_module.run(self.root, records, self.root / "clean", self.root / "summaries", prompt, "test")
        self.assertEqual(len(calls), 2)
        for name in ("04new", "xd_04"):
            scorer_module = importlib.import_module(f"src.model.{name}")
            with patch.object(scorer_module, "ZhipuAI", return_value=client):
                scorer = scorer_module.GLMAnomalyScorer("test", self.root, 1, 16,
                                                       "Assess", "Return [0.0]", self.root / "scores", self.root / "summaries")
            def score_reply(**kwargs):
                content = "[0.8]" if "current frame" in kwargs["messages"][-1]["content"] else "past summary"
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
            with patch.object(scorer_module, "ZhipuAI", return_value=client), \
                 patch.object(client.chat.completions, "create", side_effect=score_reply):
                scorer.process_video(VideoRecord(["demo", "0", "17", "0"], self.root))
            scores = json.loads((self.root / "scores" / "demo.json").read_text())
            self.assertEqual(set(scores), {"0", "16"})
            self.assertEqual(len(calculate_weighted_scores(scores, 10, 16, 18)), 18)
            with patch.object(scorer_module, "ZhipuAI", return_value=client), \
                 patch.object(client.chat.completions, "create", side_effect=RuntimeError("offline")):
                with self.assertRaisesRegex(RuntimeError, "Scoring failed"):
                    scorer.process_video(VideoRecord(["demo", "0", "17", "0"], self.root))

    def test_resume_handles_holes_and_configured_stages(self):
        videos = [VideoRecord([name, "0", "17", "0"], self.root) for name in ("a", "b", "c")]
        self.write_json(self.root / "output" / "b.json", {})
        self.assertEqual([v.path.name for v in find_unprocessed_videos(videos, self.root / "output", "*.json")], ["a", "c"])
        for dataset in ("ucf", "xd"):
            config = read_config(Path(__file__).resolve().parents[1] / f"configs/{dataset}.json")
            for stage in STAGES:
                commands = build_commands(config, stage)
                self.assertTrue(commands)
                self.assertTrue(all(command[1:3] == ["-m", command[2]] for command in commands))
            self.assertTrue(check_inputs(config, "evaluate"))


if __name__ == "__main__":
    unittest.main()
