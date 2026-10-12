import argparse
import os
from pathlib import Path

import cv2

def extract_frames_with_sliding_window(video_path, frames_dir, window_size, step_size):
    video_name = Path(video_path).stem
    video_frames_dir = os.path.join(frames_dir, video_name)
    os.makedirs(video_frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    frame_count = 0

    # Decode each source frame once; overlapping windows duplicate frames and
    # shift the temporal ground truth. Windowing belongs to downstream models.
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(video_frames_dir, f"{frame_count:06d}.jpg")
            if not cv2.imwrite(frame_path, frame):
                raise OSError(f"Cannot write frame: {frame_path}")
            frame_count += 1
    finally:
        cap.release()
    if not frame_count:
        raise ValueError(f"Video has no readable frames: {video_path}")
    print(f"Extracted {frame_count} frames from {video_path} to {video_frames_dir}")
    return video_name, frame_count


def main(videos_dir, frames_dir, annotations_file, window_size=16, step_size=8, video_split_file=None):
    os.makedirs(frames_dir, exist_ok=True)
    Path(annotations_file).parent.mkdir(parents=True, exist_ok=True)
    if video_split_file:
        filenames = [line.strip() for line in Path(video_split_file).read_text().splitlines() if line.strip()]
        if len(filenames) != len(set(filenames)) or any(Path(name).name != name for name in filenames):
            raise ValueError("Video split must contain unique flat filenames")
        video_paths = [Path(videos_dir) / name for name in filenames]
    else:
        video_paths = sorted(
            path for path in Path(videos_dir).rglob("*")
            if path.is_file() and path.suffix.lower() in {".avi", ".mp4", ".mkv"}
        )
    if not video_paths:
        raise ValueError(f"No videos found in {videos_dir}")
    names = [path.stem for path in video_paths]
    if len(names) != len(set(names)):
        raise ValueError("Video basenames must be unique across dataset folders")

    with open(annotations_file, "w") as f:
        # Loop through all the video files in the video directory
        for video_path in video_paths:
            video_name, num_frames = extract_frames_with_sliding_window(
                str(video_path), frames_dir, window_size, step_size
            )
            # Placeholder category; evaluation uses temporal ground truth.
            f.write(f"{video_name} 0 {num_frames - 1} 0\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--videos_dir",
        type=str,
        required=True,
        help="Directory path to the videos.",
    )

    parser.add_argument(
        "--frames_dir",
        type=str,
        required=True,
        help="Directory path to the frames.",
    )
    parser.add_argument(
        "--annotations_file",
        type=str,
        required=True,
        help="Path to the annotations file.",
    )
    parser.add_argument("--video_split_file", help="Official flat video filenames in split order")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args.videos_dir, args.frames_dir, args.annotations_file, video_split_file=args.video_split_file)
