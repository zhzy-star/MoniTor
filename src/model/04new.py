#You can replace any prompt with a Chinese version for better performance.
import argparse
import json
import os
import re
from contextlib import contextmanager
from threading import Lock, local
from pathlib import Path
import numpy as np
from tqdm import tqdm
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.data.video_record import VideoRecord
from zhipuai import ZhipuAI
from src.utils.path_utils import find_unprocessed_videos
from src.model.memory_embeddings import load_vectors

class GLMAnomalyScorer:
    def __init__(
        self,
        api_key,
        root_path,
        batch_size,
        frame_interval,
        context_prompt,
        format_prompt,
        output_scores_dir,
        captions_dir,
        request_workers=3,
        prefetch_frames=1,
        paper_alignment=False,
        memory_embeddings_dir=None,
        score_alpha=None,
        score_previous="raw",
        score_queue_source="raw",
    ):
        if batch_size != 1:
            raise ValueError("Online queue updates require batch_size=1")
        if isinstance(request_workers, bool) or not isinstance(request_workers, int) or request_workers < 1:
            raise ValueError("request_workers must be a positive integer")
        if isinstance(prefetch_frames, bool) or not isinstance(prefetch_frames, int) or prefetch_frames not in (0, 1):
            raise ValueError("prefetch_frames must be 0 or 1")
        if paper_alignment and not memory_embeddings_dir:
            raise ValueError("paper_alignment requires memory_embeddings_dir")
        self.paper_alignment = paper_alignment
        self.memory_embeddings_dir = memory_embeddings_dir
        self.score_alpha = (0.7 if paper_alignment else 0.6) if score_alpha is None else score_alpha
        if not 0 <= self.score_alpha <= 1:
            raise ValueError("score_alpha must be between 0 and 1")
        if score_previous not in ("raw", "fused") or score_queue_source not in ("raw", "fused"):
            raise ValueError("Score sources must be raw or fused")
        self.score_previous = score_previous
        self.score_queue_source = score_queue_source
        self._memory_vectors = {}
        self._memory_trace = {}
        self.api_key = api_key
        self.root_path = root_path
        self.batch_size = batch_size
        self.frame_interval = frame_interval
        self.context_prompt = context_prompt
        self.format_prompt = format_prompt
        self.output_scores_dir = output_scores_dir
        self.captions_dir = captions_dir
        self.client = ZhipuAI(api_key=self.api_key)
        self.request_workers = request_workers
        self.prefetch_frames = prefetch_frames
        self._request_local = local()
        self._worker_clients = []
        self._clients_lock = Lock()

        # Initialize the score queue
        self.score_queue = {score: [] for score in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
        # Initialize prediction memory
        self.prediction1 = "No prediction available" 
        self.prediction2 = "No prediction available"  
        # Initialize short-term and long-term memory
        self.long_term_memory = []  
        self.short_term_memory = []  

    def _parse_score(self, response):
        pattern = r"\[(\d+(?:\.\d+)?)\]"
        match = re.search(pattern, response)
        if match:
            score = float(match.group(1))
        else:
            alternative_pattern = r"score is (\d+(?:\.\d+)?)"
            match = re.search(alternative_pattern, response)
            score = float(match.group(1)) if match else -1

        if score != -1:
            allowed_scores = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            score = min(allowed_scores, key=lambda x: abs(x - score))
        return score

    def _update_score_queue(self, frame_idx, score):
        # Update the queue with the latest frame index for the given score
        if score in self.score_queue:
            self.score_queue[score].append(frame_idx)

    @contextmanager
    def _preparation_executor(self):
        # The main thread alone scores and updates state; workers only read captions.
        executor = None
        if self.request_workers > 1 or self.prefetch_frames:
            executor = ThreadPoolExecutor(max_workers=self.request_workers,
                                          thread_name_prefix="score-prepare")
        try:
            yield executor
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
            for client in self._worker_clients:
                close = getattr(client, "close", None)
                if close is not None:
                    close()
            self._worker_clients.clear()

    def _preparation_request(self, operation, *args):
        # Each preparation thread owns a client; the scoring client stays separate.
        if not hasattr(self._request_local, "client"):
            client = ZhipuAI(api_key=self.api_key)
            self._request_local.client = client
            with self._clients_lock:
                self._worker_clients.append(client)
        return operation(*args, client=self._request_local.client)

    def _lstm_summarize(self, texts, client=None):
        # Simulate LSTM-based summarization by combining texts into a summary
        combined_text = " ".join(texts)
        summary = (client if client is not None else self.client).chat.completions.create(
            model="glm-4-flash",
            messages=[
                {"role": "system", "content": "Please summarize the following descriptions into a concise sentence."}, # modify to your summary prompt
                {"role": "user", "content": combined_text}
            ],
            stream=False,
            timeout=30,
            temperature=0.6
        ).choices[0].message.content.strip()
        return summary

    def _calculate_similarity(self, text1, text2):
        # Use TF-IDF Vectorizer to convert text to vectors
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
        except ValueError:
            return 1.0 if text1 == text2 else 0.0

        # Calculate cosine similarity between the vectors
        similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        similarity_score = similarity_matrix[0][0]

        return similarity_score

    def _select_memories(self, captions, current_idx):
        if self.paper_alignment:
            current = self._memory_vectors[str(current_idx)]
            candidates = []
            long_memory = []
            for idx in range(max(0, current_idx - 10 * self.frame_interval),
                             current_idx, self.frame_interval):
                similarity = float(np.dot(current, self._memory_vectors[str(idx)]))
                keep = similarity > 0.5
                candidates.append(dict(frame=idx, similarity=similarity, retained=keep))
                if keep:
                    long_memory.append(captions[str(idx)])
            short_frames = list(range(max(0, current_idx - 2 * self.frame_interval),
                                      current_idx, self.frame_interval))
            self._memory_trace[str(current_idx)] = dict(long_candidates=candidates,
                                                        short_frames=short_frames)
            return long_memory, [captions[str(idx)] for idx in short_frames]
        # Prepare long-term memory (summarizing previous 5 seconds of descriptions)
        long_term_captions = [
            captions[str(i)] for i in range(max(0, current_idx - 5 * self.frame_interval), current_idx, self.frame_interval)
            if str(i) in captions
        ]
        # Apply a forgetting mechanism based on similarity
        filtered_long_term_captions = []
        for caption in long_term_captions:
            if not filtered_long_term_captions:
                filtered_long_term_captions.append(caption)
            else:
                similarity = self._calculate_similarity(filtered_long_term_captions[-1], caption)
                forgetting_threshold = 0.5  # Set threshold for forgetting based on similarity
                if similarity < forgetting_threshold:
                    filtered_long_term_captions.append(caption)

        long_term_memory = filtered_long_term_captions[-5:]  # Limit long-term memory to 5 descriptions

        # Prepare short-term memory (summarizing the last 2 frames)
        short_term_captions = [
            captions[str(current_idx - i * self.frame_interval)] for i in range(1, 3)
            if str(current_idx - i * self.frame_interval) in captions
        ]
        return long_term_memory, short_term_captions[-2:]

    def _predict(self, previous_caption, client=None):
        prediction_prompt = ("If you are law enforcement, please predict what might happen next in the described scene, considering potential suspicious activities or behaviors, "
                             "such as abuse, arrest, arson, assault, burglary, disturbing the peace, explosion, fighting, robbery, shooting, stealing, shoplifting, or vandalism. "
                             "Please provide a concise prediction based on the current context.\n"
                             "Please provide the prediction in a concise sentence, focusing on describing the behavior or event that might occur next in the scene, avoiding any additional explanations.")
        response = (client if client is not None else self.client).chat.completions.create(
            model="glm-4-flash",
            messages=[
                {"role": "system", "content": prediction_prompt},
                {"role": "user", "content": f"The description of the previous frame is: '{previous_caption}', please provide the description of the next frame."}
            ],
            stream=False,
            timeout=60,
            temperature=0.6
        )
        return response.choices[0].message.content.strip()

    def _submit_preparation(self, captions, frame_idx, executor):
        long_memory, short_memory = self._select_memories(captions, frame_idx)

        def submit(operation, value, fallback):
            if value is not None and executor is not None:
                return executor.submit(self._preparation_request, operation, value)
            future = Future()
            future.set_result(operation(value) if value is not None else fallback)
            return future

        return long_memory, short_memory, (
            submit(self._lstm_summarize, long_memory or None, "No long-term memory"),
            submit(self._lstm_summarize, short_memory or None, "No short-term memory"),
            submit(self._predict,
                   captions.get(str(frame_idx - self.frame_interval), "No previous caption available")
                   if frame_idx - self.frame_interval >= 0 else None,
                   "No previous frame available for prediction"),
        )

    def _prepare_score_prompts(self, captions):
        # Prepare score queue information to be used as a prompt for scoring
        score_prompts = []
        for score, frames in self.score_queue.items():
            if frames:
                frame_idx = frames[-1]
                caption = captions[str(frame_idx)]
                score_prompts.append(f"Event with score {score}: '{caption}'.")
        return "\n".join(score_prompts)

    def _prepare_dialog(self, captions, idx, prepared):
        # Get score queue information as part of the prompt
        score_prompts = self._prepare_score_prompts(captions)
        prompt = self.context_prompt + " " + self.format_prompt + "\n" + score_prompts

        # Resolve in frame order. Prefetch must never mutate the current memories.
        long_memory, short_memory, futures = prepared
        long_term_summary, short_term_summary, prediction_result = [future.result() for future in futures]
        self.long_term_memory = long_memory
        self.short_term_memory = short_memory
        self.prediction2 = prediction_result
        self.prediction1 = self.prediction2
        if idx - self.frame_interval >= 0:
            print(f"Prediction result: {prediction_result}")

        current_caption = f"This is the description of the current frame: '{captions[str(idx)]}'."
        memory_info = f"Long-term memory contains a summary of events from the previous five seconds: {long_term_summary}. Short-term memory contains descriptions of the previous two frames: {short_term_summary}."
        if self.paper_alignment:
            memory_info = (f"Long-term memory summarizes retained events from the previous ten sampling steps: {long_term_summary}. "
                           f"Short-term memory summarizes the previous two sampling steps in chronological order: {short_term_summary}.")
        scoring_prompt = (prompt + f"\n{memory_info}\nPrediction from the previous frame: '{prediction_result}'")
        return [
            {"role": "system", "content": scoring_prompt},
            {"role": "user", "content": f"{current_caption} "}
        ]

    def _score_temporal_summaries(self, video, temporal_captions):
        video_scores = {}
        fy = {}
        frame_idxs = list(range(0, video.num_frames, self.frame_interval))
        if self.paper_alignment:
            self._memory_vectors = load_vectors(
                Path(self.memory_embeddings_dir) / f"{Path(video.path).name}.npz", temporal_captions)
            self._memory_trace = {}
        previous_raw = None

        with self._preparation_executor() as executor:
            def prepare(frame_idx):
                try:
                    return self._submit_preparation(temporal_captions, frame_idx, executor)
                except Exception as error:
                    raise RuntimeError(f"Preparation failed at frame {frame_idx} of {video.path}") from error

            prepared = prepare(frame_idxs[0]) if frame_idxs else None
            for position, frame_idx in enumerate(tqdm(frame_idxs, desc=f"Processing {video.path}", unit="batch")):
                try:
                    dialog = self._prepare_dialog(temporal_captions, frame_idx, prepared)
                except Exception as error:
                    raise RuntimeError(f"Preparation failed at frame {frame_idx} of {video.path}") from error

                next_idx = frame_idxs[position + 1] if position + 1 < len(frame_idxs) else None
                # At most one future frame; no score prompt is built before its turn.
                prepared = None
                if self.prefetch_frames and next_idx is not None:
                    prepared = prepare(next_idx)
                try:
                    response = self.client.chat.completions.create(
                        model="glm-4-flash",
                        messages=dialog,
                        stream=False,
                        timeout=60,  # Increase timeout
                        temperature=0.6  # Lower temperature to reduce randomness
                    )
                except Exception as e:
                    raise RuntimeError(f"Scoring failed at frame {frame_idx} of {video.path}") from e

                result_content = response.choices[0].message.content
                current_score = self._parse_score(result_content)
                previous_score = video_scores.get(str(frame_idx - self.frame_interval), None)
                if self.paper_alignment:
                    reference = previous_raw if self.score_previous == "raw" else previous_score
                    score = (self.score_alpha * current_score + (1 - self.score_alpha) * reference
                             if current_score != -1 and reference is not None and reference != -1
                             else float(current_score))
                    queue_score = current_score if self.score_queue_source == "raw" else score
                    if queue_score != -1:
                        self._update_score_queue(frame_idx, round(queue_score, 1))
                else:
                    if current_score != -1 and previous_score is not None and previous_score != -1:
                        score = round(self.score_alpha * current_score + (1 - self.score_alpha) * previous_score, 1)
                    else:
                        score = round(current_score, 1)
                    self._update_score_queue(frame_idx, score)
                previous_raw = current_score

                fy[str(frame_idx)] = {'new_score': score, 'raw_score': result_content, 'caption': dialog[1]["content"], 'prompt': dialog[0]["content"]}
                if self.paper_alignment:
                    fy[str(frame_idx)].update(
                        parsed_score=current_score, alpha=self.score_alpha,
                        previous_source=self.score_previous, previous_score=reference,
                        queue_source=self.score_queue_source,
                        memory=self._memory_trace[str(frame_idx)])
                video_scores[str(frame_idx)] = score
                if not self.prefetch_frames and next_idx is not None:
                    prepared = prepare(next_idx)

        if self.paper_alignment:
            return fy  # No forward-looking smooth_score in the paper path.

        # Smoothing scores
        sorted_indices = sorted(video_scores.keys(), key=lambda x: int(x))
        scores = [video_scores[idx] for idx in sorted_indices]
        smoothed_scores = np.convolve(scores, np.ones(3) / 3, mode='valid') if len(scores) >= 3 else scores
        smoothed_scores = [round(score, 1) for score in smoothed_scores]
        for i, idx in enumerate(sorted_indices[:len(smoothed_scores)]):
            video_scores[idx] = smoothed_scores[i]

        # Update smoothed scores
        for i, idx in enumerate(sorted_indices[:len(smoothed_scores)]):
            fy[idx]['smooth_score'] = smoothed_scores[i]

        return fy

    def process_video(self, video):
        video_name = Path(video.path).name
        temporal_captions_path = Path(self.captions_dir) / f"{video_name}.json"
        try:
            with open(temporal_captions_path) as f:
                temporal_captions = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot load captions for {video_name}") from error
        expected = {str(idx) for idx in range(0, video.num_frames, self.frame_interval)}
        if set(temporal_captions) != expected or not all(isinstance(value, str) and value.strip() for value in temporal_captions.values()):
            raise ValueError(f"Expected one nonempty string caption per sampled frame for {video_name}")

        video_scores = self._score_temporal_summaries(video, temporal_captions)
        output_path = Path(self.output_scores_dir) / f"{video_name}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(video_scores, f, indent=4)

        print(f"Scores successfully saved to {output_path}")

def process_video_parallel(video, scorer_params):
    scorer = GLMAnomalyScorer(**scorer_params)
    try:
        scorer.process_video(video)
    finally:
        close = getattr(scorer.client, "close", None)
        if close is not None:
            close()

def run(
    root_path,
    annotationfile_path,
    batch_size,
    frame_interval,
    context_prompt,
    format_prompt,
    output_scores_dir,
    captions_dir,
    api_key,
    num_jobs,
    resume=False,
    request_workers=3,
    prefetch_frames=1,
    paper_alignment=False,
    memory_embeddings_dir=None,
    score_alpha=None,
    score_previous="raw",
    score_queue_source="raw",
):
    video_list = [VideoRecord(x.strip().split(), root_path) for x in open(annotationfile_path)]
    if resume:
        video_list = find_unprocessed_videos(video_list, output_scores_dir, "*.json")

    scorer_params = {
        "api_key": api_key,
        "root_path": root_path,
        "batch_size": batch_size,
        "frame_interval": frame_interval,
        "context_prompt": context_prompt,
        "format_prompt": format_prompt,
        "output_scores_dir": output_scores_dir,
        "captions_dir": captions_dir,
        "request_workers": request_workers,
        "prefetch_frames": prefetch_frames,
        "paper_alignment": paper_alignment,
        "memory_embeddings_dir": memory_embeddings_dir,
        "score_alpha": score_alpha,
        "score_previous": score_previous,
        "score_queue_source": score_queue_source,
    }

    with ProcessPoolExecutor(max_workers=num_jobs) as executor:
        list(executor.map(process_video_parallel, video_list, [scorer_params] * len(video_list)))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--annotationfile_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--frame_interval", type=int, default=16)
    parser.add_argument("--context_prompt", type=str, required=True)
    parser.add_argument("--format_prompt", type=str, required=True)
    parser.add_argument("--output_scores_dir", type=str, required=True)
    parser.add_argument("--captions_dir", type=str, required=True)
    parser.add_argument("--api_key", type=str, default=os.environ.get("ZHIPUAI_API_KEY"))
    parser.add_argument("--num_jobs", type=int, default=1)
    parser.add_argument("--request_workers", type=int, default=3,
                        help="Preparation requests per video; scoring uses one additional client")
    parser.add_argument("--prefetch_frames", type=int, choices=(0, 1), default=1,
                        help="Prepare the next frame while scoring the current frame")
    parser.add_argument("--paper_alignment", action="store_true")
    parser.add_argument("--memory_embeddings_dir")
    parser.add_argument("--score_alpha", type=float)
    parser.add_argument("--score_previous", choices=("raw", "fused"), default="raw")
    parser.add_argument("--score_queue_source", choices=("raw", "fused"), default="raw")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.api_key:
        parser.error("Set ZHIPUAI_API_KEY or provide --api_key")
    if args.batch_size != 1:
        parser.error("Online queue updates require --batch_size 1")
    if args.frame_interval < 1 or args.num_jobs < 1 or args.request_workers < 1:
        parser.error("frame_interval, num_jobs and request_workers must be positive")
    return args

if __name__ == "__main__":
    args = parse_args()
    run(
        root_path=args.root_path,
        annotationfile_path=args.annotationfile_path,
        batch_size=args.batch_size,
        frame_interval=args.frame_interval,
        context_prompt=args.context_prompt,
        format_prompt=args.format_prompt,
        output_scores_dir=args.output_scores_dir,
        captions_dir=args.captions_dir,
        api_key=args.api_key,
        num_jobs=args.num_jobs,
        resume=args.resume,
        request_workers=args.request_workers,
        prefetch_frames=args.prefetch_frames,
        paper_alignment=args.paper_alignment,
        memory_embeddings_dir=args.memory_embeddings_dir,
        score_alpha=args.score_alpha,
        score_previous=args.score_previous,
        score_queue_source=args.score_queue_source,
    )
