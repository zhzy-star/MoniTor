"""Dataset evaluation entry point; shared frame alignment lives in evaluation.py."""
from src.evaluation import (
    calculate_weighted_scores, get_video_labels, load_json_file,
    main as evaluate, parse_args as parse_shared_args,
    replace_negative_scores_with_previous, save_metric,
    temporal_testing_annotations as parse_temporal_annotations,
)


def temporal_testing_annotations(path):
    return parse_temporal_annotations(path, 'ucf')


def main(*args, **kwargs):
    return evaluate(*args, dataset='ucf', **kwargs)


def parse_args():
    return parse_shared_args('ucf')


if __name__ == "__main__":
    main(**vars(parse_args()))
