from pathlib import Path

from source.read_image import ImageSource
from source.read_video import VideoSource


class Source:

    def __init__(self, source_name: str, source_folder: Path) -> None:
        is_image = source_name.endswith('jpg')
        if is_image:
            self.source = ImageSource.create_from_source(
                image_path=source_name,
                image_folder=source_folder
            )
        else:
            self.source = VideoSource.create_from_source(
                video_path=source_name,
                video_folder=source_folder
            )
