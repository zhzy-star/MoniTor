from pathlib import Path
from typing import List, Union


class VideoRecord:
    """This class represents a video sample's metadata.

    Args:
        root_datapath: the system path to the root folder
                       of the videos.
        row: A list with four or more elements where 1) The first
             element is the path to the video sample's frames excluding
             the root_datapath prefix 2) The  second element is the starting frame id of the video
             3) The third element is the inclusive ending frame id of the video
             4) The fourth element is the label index. If the video has multiple labels,
                the fourth element is a comma-separated string of label indices.
    """

    def __init__(self, row, root_datapath):
        if len(row) != 4:
            raise ValueError("Video records require: video_name start_frame end_frame label")
        self._data = row
        self._path = Path(root_datapath) / row[0]
        if self.start_frame != 0 or self.end_frame < self.start_frame:
            raise ValueError("The reproduction pipeline requires zero-based, inclusive frame ranges")
        self.label  # Validate the label column at the input boundary.

    @property
    def path(self) -> Path:
        return self._path

    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame + 1  # +1 because end frame is inclusive

    @property
    def start_frame(self) -> int:
        return int(self._data[1])

    @property
    def end_frame(self) -> int:
        return int(self._data[2])

    @property
    def label(self) -> Union[int, List[int]]:
        return [int(label_id) for label_id in self._data[3].split(",")]

