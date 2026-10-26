"""Offline checks for prompt equivalence, causal scoring, and bounded prefetch."""
from collections import Counter
import contextlib
import importlib
import io
import json
from pathlib import Path
import re
import socket
import tempfile
from threading import Barrier, Event, Lock, get_ident
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.reproduce import build_commands, read_config

scoring = importlib.import_module("src.model.04new")


class FakeAPI:
    def __init__(self, overlap=False, fail=None):
        self.calls = []
        self.clients = []
        self.lock = Lock()
        self.overlap = overlap
        self.fail = fail
        self.started = Event()
        self.prepared = Event()
        self.barrier = Barrier(3)
        self.active = 0
        self.peak = 0

    def client(self, **kwargs):
        owner = get_ident()
        client = SimpleNamespace(closed=False)

        def create(**request):
            if get_ident() != owner:
                raise AssertionError("API client shared across threads")
            return self.reply(request)

        def close():
            client.closed = True

        client.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
        client.close = close
        with self.lock:
            self.clients.append(client)
        return client

    def reply(self, request):
        user = request["messages"][-1]["content"]
        system = request["messages"][0]["content"]
        kind = "score" if "current frame" in user else "summary" if system.startswith("Please summarize") else "prediction"
        with self.lock:
            self.calls.append(request)
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            if self.overlap:
                if kind == "score" and "scene0" in user:
                    self.started.set()
                    if not self.prepared.wait(5):
                        raise AssertionError("next-frame preparation did not overlap scoring")
                    with self.lock:
                        if len(self.calls) != 4:
                            raise AssertionError("prefetched more than one frame")
                elif kind != "score" and "scene0" in user and "scene1" not in user:
                    if not self.started.wait(5):
                        raise AssertionError("score request did not start")
                    if self.barrier.wait(5) == 0:
                        self.prepared.set()
            if self.fail == kind:
                raise RuntimeError("mock request failed")
            if kind == "score":
                index = int(re.search(r"scene(\d+)", user).group(1))
                content = ["[0.2]", "score is 0.9", "unparseable", "[0.7]", "[0.4]", "[0.1]", "[0.8]"][index % 7]
            else:
                content = f"{kind}: {user}"
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
        finally:
            with self.lock:
                self.active -= 1


class ScorePrefetchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.captions = {str(i * 16): f"scene{i} distinctive action{i}" for i in range(7)}
        self.video = SimpleNamespace(num_frames=97, path=self.root / "demo")
        # A missed mock must fail locally, never contact the real API.
        self.network = patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden"))
        self.network.start()
        self.addCleanup(self.network.stop)

    def run_scorer(self, api, workers=3, prefetch=1, captions=None, save=False):
        frame_captions = self.captions if captions is None else captions
        memory_vectors = {frame: np.array([1.0]) for frame in frame_captions}
        with patch.object(scoring, "ZhipuAI", side_effect=api.client), \
             patch.object(scoring, "load_vectors", return_value=memory_vectors), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            scorer = scoring.GLMAnomalyScorer("offline", self.root, 1, 16,
                                             "Assess", "Return [0.0]", self.root / "scores", self.root,
                                             request_workers=workers, prefetch_frames=prefetch,
                                             memory_embeddings_dir=self.root / "memory")
            try:
                if save:
                    (self.root / "demo.json").write_text(json.dumps(self.captions))
                    scorer.process_video(self.video)
                    result = json.loads((self.root / "scores" / "demo.json").read_text())
                else:
                    result = scorer._score_temporal_summaries(self.video, self.captions if captions is None else captions)
                return scorer, result
            finally:
                scorer.client.close()

    def test_parallel_prompts_requests_scores_and_state_match_serial(self):
        serial_api = FakeAPI()
        serial, expected = self.run_scorer(serial_api, workers=1, prefetch=0)
        for workers, prefetch in ((3, 0), (1, 1), (3, 1)):
            with self.subTest(workers=workers, prefetch=prefetch):
                api = FakeAPI()
                scorer, actual = self.run_scorer(api, workers, prefetch)
                self.assertEqual(actual, expected)
                self.assertEqual(scorer.score_queue, serial.score_queue)
                self.assertEqual(scorer.long_term_memory, serial.long_term_memory)
                self.assertEqual(scorer.short_term_memory, serial.short_term_memory)
                self.assertEqual(scorer.prediction1, serial.prediction1)
                self.assertEqual(Counter(json.dumps(c, sort_keys=True) for c in api.calls),
                                 Counter(json.dumps(c, sort_keys=True) for c in serial_api.calls))
                self.assertEqual(len(api.calls), 25)  # 1 + 6 * 4, including duplicate-input summaries.
                self.assertTrue(all(client.closed for client in api.clients))
        for actual_score, expected_score in zip(
            [row["new_score"] for row in expected.values()],
            [0.2, 0.69, -1, 0.7, 0.49, 0.19, 0.59],
        ):
            self.assertAlmostEqual(actual_score, expected_score)
        self.assertIn("Event with score 0.9: 'scene1 distinctive action1'.", expected["32"]["prompt"])
        self.assertIn("Short-term memory summarizes the previous two sampling steps in chronological order: summary: scene0 distinctive action0 scene1 distinctive action1.", expected["32"]["prompt"])
        self.assertNotIn("Event with score", expected["0"]["prompt"])
        self.assertEqual(serial_api.peak, 1)

    def test_three_preparations_overlap_score_with_only_one_frame_ahead(self):
        api = FakeAPI(overlap=True)
        _, result = self.run_scorer(api)
        self.assertEqual(api.peak, 4)
        self.assertEqual(list(result), list(self.captions))
        scored = [c["messages"][-1]["content"] for c in api.calls
                  if "current frame" in c["messages"][-1]["content"]]
        self.assertEqual([int(re.search(r"scene(\d+)", c).group(1)) for c in scored], list(range(7)))

    def test_preparation_and_score_errors_propagate_and_close_clients(self):
        for kind, message in (("summary", "Preparation failed at frame 16"),
                              ("prediction", "Preparation failed at frame 16"),
                              ("score", "Scoring failed at frame 0")):
            with self.subTest(kind=kind):
                api = FakeAPI(fail=kind)
                with self.assertRaisesRegex(RuntimeError, message) as raised:
                    self.run_scorer(api, save=True)
                self.assertEqual(str(raised.exception.__cause__), "mock request failed")
                self.assertTrue(all(client.closed for client in api.clients))
                self.assertLessEqual(len(api.calls), 4)
                self.assertFalse((self.root / "scores" / "demo.json").exists())

    def test_saved_json_matches_serial_output(self):
        _, expected = self.run_scorer(FakeAPI(), workers=1, prefetch=0)
        _, actual = self.run_scorer(FakeAPI(), save=True)
        self.assertEqual(actual, expected)

    def test_empty_and_single_frame_do_not_prefetch(self):
        for frames in (0, 1):
            self.video.num_frames = frames
            api = FakeAPI()
            _, result = self.run_scorer(api, captions={"0": self.captions["0"]} if frames else {})
            self.assertEqual(len(api.calls), frames)
            self.assertEqual(len(result), frames)

    def test_config_and_command_flags_validate_and_allow_serial_fallback(self):
        root = Path(__file__).resolve().parents[1]
        config = read_config(root / "configs/ucf.json")
        command = build_commands(config, "score")[0]
        self.assertEqual(command[command.index("--request_workers") + 1], "3")
        self.assertEqual(command[command.index("--prefetch_frames") + 1], "1")
        config.update(score_request_workers=1, score_prefetch_frames=0)
        command = build_commands(config, "score")[0]
        self.assertEqual(command[command.index("--prefetch_frames") + 1], "0")
        for key, invalid in (("score_request_workers", 0), ("score_request_workers", True),
                             ("score_prefetch_frames", 2), ("score_prefetch_frames", False),
                             ("score_prefetch_frames", 1.0)):
            path = self.root / "bad.json"
            path.write_text(json.dumps(dict(config, **{key: invalid})))
            with self.assertRaisesRegex(ValueError, key):
                read_config(path)


if __name__ == "__main__":
    unittest.main()
