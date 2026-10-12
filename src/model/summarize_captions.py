"""Convert the cleaner's nested captions into scorer-compatible summaries."""
import argparse
import json
import os
from pathlib import Path

from zhipuai import ZhipuAI

from src.data.video_record import VideoRecord


def run(root_path, annotationfile_path, captions_dir, output_dir, prompt_file,
        api_key, frame_interval=16, model="glm-4-flash", resume=False):
    client = ZhipuAI(api_key=api_key)
    prompt = Path(prompt_file).read_text().strip()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(annotationfile_path) as handle:
        videos = [VideoRecord(line.split(), root_path) for line in handle if line.strip()]
    for video in videos:
        output_path = output_dir / f"{video.path.name}.json"
        if resume and output_path.exists():
            continue
        captions = json.loads((Path(captions_dir) / output_path.name).read_text())
        result = {}
        for frame_idx in range(0, video.num_frames, frame_interval):
            frame_captions = captions[str(frame_idx)]
            if not isinstance(frame_captions, dict) or not frame_captions:
                raise ValueError(f"Expected nested cleaned captions at {video.path.name}/{frame_idx}")
            keys = sorted(frame_captions, key=int)
            if any(int(key) > frame_idx for key in keys):
                raise ValueError("Future frame found in online caption summary")
            descriptions = [frame_captions[key] for key in keys]
            if not all(isinstance(text, str) and text.strip() for text in descriptions):
                raise ValueError("Cleaned captions must be nonempty strings")
            text = "\n".join(dict.fromkeys(descriptions))
            response = client.chat.completions.create(
                model=model, messages=[{"role": "system", "content": prompt},
                                       {"role": "user", "content": text}],
                stream=False, timeout=60, temperature=0.6,
            )
            summary = response.choices[0].message.content
            if not summary or not summary.strip():
                raise ValueError(f"Empty summary for {video.path.name}/{frame_idx}")
            result[str(frame_idx)] = summary.strip()
        output_path.write_text(json.dumps(result, indent=2) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root_path", "annotationfile_path", "captions_dir", "output_dir", "prompt_file"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--api_key", default=os.environ.get("ZHIPUAI_API_KEY"))
    parser.add_argument("--frame_interval", type=int, default=16)
    parser.add_argument("--model", default="glm-4-flash")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.api_key:
        parser.error("Set ZHIPUAI_API_KEY or provide --api_key")
    if args.frame_interval < 1:
        parser.error("frame_interval must be positive")
    return args


if __name__ == "__main__":
    run(**vars(parse_args()))
