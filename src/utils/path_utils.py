from pathlib import Path


def find_unprocessed_videos(video_list, output_dir, pathname):
    processed_names = {path.stem for path in Path(output_dir).glob(pathname)}
    return [video for video in video_list if Path(video.path).name not in processed_names]


def find_last_processed_video_index(video_list, last_processed_video_path):
    last_processed_video_name = last_processed_video_path.stem
    return [video.path.name for video in video_list].index(last_processed_video_name)
