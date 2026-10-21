"""Convert the cleaner's nested captions into scorer-compatible summaries."""
import argparse
import hashlib
import json
import os
import random
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from tqdm import tqdm
from zhipuai import APIConnectionError, APIStatusError, ZhipuAI

from src.data.video_record import VideoRecord


class EmptySummaryError(ValueError):
    """A successful API response without usable summary text."""


class BlockedSummaryError(ValueError):
    """An explicit refusal, recorded without automatic retries."""


def write_json(path, payload):
    """Replace a checkpoint only after its new contents reach disk."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def validate_summaries(result, frame_ids, complete=False):
    expected = {str(idx) for idx in frame_ids}
    if (not isinstance(result, dict) or not set(result).issubset(expected)
            or (complete and set(result) != expected)
            or any(not isinstance(text, str) or not text.strip() for text in result.values())):
        raise ValueError("Invalid summary checkpoint/output: frame IDs or nonempty text required")


def caption_text(frame_captions, video_name, frame_idx):
    if not isinstance(frame_captions, dict) or not frame_captions:
        raise ValueError(f"Expected nested cleaned captions at {video_name}/{frame_idx}")
    keys = sorted(frame_captions, key=int)
    if any(int(key) > frame_idx for key in keys):
        raise ValueError("Future frame found in online caption summary")
    descriptions = [frame_captions[key] for key in keys]
    if not all(isinstance(text, str) and text.strip() for text in descriptions):
        raise ValueError("Cleaned captions must be nonempty strings")
    return "\n".join(dict.fromkeys(descriptions))


def run(root_path, annotationfile_path, captions_dir, output_dir, prompt_file,
        api_key, frame_interval=16, model="glm-4-flash", resume=False,
        workers=1, max_retries=4, recovery_rounds=3, recovery_delay=60):
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if max_retries < 0 or frame_interval < 1 or recovery_rounds < 0 or recovery_delay < 0:
        raise ValueError("Retry counts/delay must be nonnegative and frame_interval positive")
    prompt = Path(prompt_file).read_text().strip()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(annotationfile_path) as handle:
        videos = [VideoRecord(line.split(), root_path) for line in handle if line.strip()]
    local = threading.local()
    clients = []
    usage_lock = threading.Lock()

    def summarize(text, video_name, frame_idx):
        if not hasattr(local, "client"):
            local.client = ZhipuAI(api_key=api_key, max_retries=0)
            clients.append(local.client)
        for attempt in range(max_retries + 1):
            try:
                response = local.client.chat.completions.create(
                    model=model, messages=[{"role": "system", "content": prompt},
                                           {"role": "user", "content": text}],
                    stream=False, timeout=60, temperature=0.6,
                )
                choices = getattr(response, "choices", None) or []
                choice = choices[0] if choices else None
                finish_reason = getattr(choice, "finish_reason", None)
                message = getattr(choice, "message", None)
                summary = getattr(message, "content", None)
                usage = getattr(response, "usage", None)
                response_model = getattr(response, "model", None)
                request_id = getattr(response, "request_id", None) or getattr(response, "id", None)
                response_payload = response.to_dict(mode="json", exclude_none=False)
                counts = {key: getattr(usage, key, None) for key in
                          ("prompt_tokens", "completion_tokens", "total_tokens")}
                # Record every returned response, including empty attempts, before
                # retrying. Only the caller's frame ID determines output alignment.
                with usage_lock, (usage_dir / f"{video_name}.jsonl").open("a") as usage_log:
                    usage_log.write(json.dumps(dict(
                        frame_id=frame_idx, attempt=attempt + 1,
                        requested_model=model, response_model=response_model,
                        request_id=request_id,
                        finish_reason=finish_reason,
                        content=summary,
                        reasoning_content=getattr(message, "reasoning_content", None),
                        tool_calls=getattr(message, "tool_calls", None),
                        response=response_payload,
                        empty=not isinstance(summary, str) or not summary.strip(),
                        **counts,
                    ), ensure_ascii=False, default=str) + "\n")
                if (finish_reason in ("content_filter", "sensitive", "refusal")
                        or getattr(message, "refusal", None)):
                    raise BlockedSummaryError(f"Summary blocked by API for {video_name}/{frame_idx}; "
                                            f"finish_reason={finish_reason}")
                if not isinstance(summary, str) or not summary.strip():
                    raise EmptySummaryError(f"Empty summary for {video_name}/{frame_idx}; "
                                            f"finish_reason={finish_reason}")
                return summary.strip()
            except (APIConnectionError, APIStatusError, EmptySummaryError) as error:
                status = getattr(error, "status_code", None)
                retryable = status is None or status in (408, 409, 429) or status >= 500
                if not retryable or attempt == max_retries:
                    raise
                delay = min(60, 2 ** attempt + random.random())
                response = getattr(error, "response", None)
                if response is not None:
                    try:
                        delay = min(60, max(delay, float(response.headers.get("retry-after", 0))))
                    except ValueError:
                        pass
                tqdm.write(f"Summary request retry {attempt + 1}/{max_retries} "
                           f"for {video_name}/{frame_idx}: "
                           f"{type(error).__name__}; waiting {delay:.1f}s")
                time.sleep(delay)

    usage_dir = output_dir / "_usage"
    usage_dir.mkdir(exist_ok=True)
    checkpoint_dir = output_dir / "_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    print(f"Summary workers: {workers}; max retries per request: {max_retries}", flush=True)

    def process_video(executor, video, use_saved):
        video_name = video.path.name
        output_path = output_dir / f"{video_name}.json"
        checkpoint_path = checkpoint_dir / output_path.name
        frame_ids = list(range(0, video.num_frames, frame_interval))
        if use_saved and output_path.exists():
            validate_summaries(json.loads(output_path.read_text()), frame_ids, complete=True)
            return 0, 0
        captions = json.loads((Path(captions_dir) / output_path.name).read_text())
        # Validate all inputs before requests; bind partial results to these exact inputs.
        texts = {idx: caption_text(captions[str(idx)], video_name, idx) for idx in frame_ids}
        fingerprint = hashlib.sha256(json.dumps(
            [model, prompt, 0.6, frame_interval, texts], sort_keys=True,
        ).encode()).hexdigest()
        state = dict(version=1, fingerprint=fingerprint, results={}, failures={})
        if use_saved and checkpoint_path.exists():
            state = json.loads(checkpoint_path.read_text())
            if state.get("version") != 1 or state.get("fingerprint") != fingerprint:
                raise ValueError(f"Checkpoint inputs/settings changed for {video_name}; use a new output directory")
            validate_summaries(state["results"], frame_ids)
            if not isinstance(state["failures"], dict):
                raise ValueError(f"Invalid failure queue for {video_name}")
        result, failures = state["results"], state["failures"]
        for key in list(failures):
            if key in result:
                failures.pop(key)
        # Without resume, stale final output must not masquerade as the new run.
        if not use_saved:
            output_path.unlink(missing_ok=True)
        write_json(checkpoint_path, state)
        remaining = iter(idx for idx in frame_ids if str(idx) not in result
                         and failures.get(str(idx), {}).get("retryable", True))
        pending = {}

        def submit_next():
            idx = next(remaining, None)
            if idx is not None:
                pending[executor.submit(summarize, texts[idx], video_name, idx)] = idx

        def save_future(future, idx):
            key = str(idx)
            try:
                summary = future.result()
            except (APIConnectionError, APIStatusError, EmptySummaryError, BlockedSummaryError) as error:
                status = getattr(error, "status_code", None)
                retryable = (not isinstance(error, BlockedSummaryError)
                             and (status is None or status in (408, 409, 429) or status >= 500))
                # Authentication and invalid API parameters are global failures.
                if not retryable and not isinstance(error, BlockedSummaryError):
                    raise
                failures[key] = dict(error_type=type(error).__name__, status_code=status,
                                     retryable=retryable,
                                     failed_cycles=failures.get(key, {}).get("failed_cycles", 0) + 1)
                write_json(checkpoint_path, state)
                tqdm.write(f"Queued summary failure: {video_name}/{idx}; "
                           f"{type(error).__name__}; retryable={retryable}")
            else:
                result[key] = summary
                failures.pop(key, None)
                write_json(checkpoint_path, state)

        for _ in range(workers):
            submit_next()
        with tqdm(total=len(frame_ids), initial=len(result),
                  desc=f"Summarizing {video_name}", unit="frame") as progress:
            try:
                while pending:
                    finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in finished:
                        idx = pending.pop(future)
                        save_future(future, idx)
                        if str(idx) in result:
                            progress.update(1)
                    for _ in finished:
                        submit_next()
            except BaseException:
                # Keep successes returned by already running requests, even on interruption.
                for future in pending:
                    future.cancel()
                for future, idx in pending.items():
                    if not future.cancelled():
                        try:
                            save_future(future, idx)
                        except Exception:
                            pass
                raise
        missing = [idx for idx in frame_ids if str(idx) not in result]
        if not missing:
            ordered = {str(idx): result[str(idx)] for idx in frame_ids}
            write_json(output_path, ordered)
            checkpoint_path.unlink(missing_ok=True)
            return 0, 0
        retryable_count = sum(failures.get(str(idx), {}).get("retryable", True) for idx in missing)
        return len(missing), retryable_count

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for round_idx in range(recovery_rounds + 1):
                missing = retryable_count = 0
                for video in videos:
                    count, retry_count = process_video(executor, video, resume or round_idx > 0)
                    missing += count
                    retryable_count += retry_count
                print(f"Summary sweep {round_idx + 1}: {missing} missing frames; "
                      f"{retryable_count} eligible for another sweep", flush=True)
                if not missing:
                    return
                if round_idx == recovery_rounds or not retryable_count:
                    raise RuntimeError(f"Summary incomplete: {missing} frames remain; "
                                       f"saved results/failure queue: {checkpoint_dir}. "
                                       "Resume to retry eligible missing frames. Scoring must wait.")
                delay = min(300, recovery_delay * 2 ** round_idx)
                print(f"Waiting {delay}s before summary recovery sweep {round_idx + 1}", flush=True)
                time.sleep(delay)
    finally:
        for client in clients:
            client.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root_path", "annotationfile_path", "captions_dir", "output_dir", "prompt_file"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--api_key", default=os.environ.get("ZHIPUAI_API_KEY"))
    parser.add_argument("--frame_interval", type=int, default=16)
    parser.add_argument("--model", default="glm-4-flash")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent independent summary requests")
    parser.add_argument("--max_retries", type=int, default=4,
                        help="Retries after a transient API failure or empty summary")
    parser.add_argument("--recovery_rounds", type=int, default=3,
                        help="Additional full sweeps over eligible missing frames")
    parser.add_argument("--recovery_delay", type=float, default=60,
                        help="Initial seconds between recovery sweeps (doubles, capped at 300)")
    args = parser.parse_args()
    if not args.api_key:
        parser.error("Set ZHIPUAI_API_KEY or provide --api_key")
    if args.frame_interval < 1:
        parser.error("frame_interval must be positive")
    if args.workers < 1 or args.max_retries < 0 or args.recovery_rounds < 0 or args.recovery_delay < 0:
        parser.error("workers must be positive and retry counts/delay nonnegative")
    return args


if __name__ == "__main__":
    run(**vars(parse_args()))
