from pathlib import Path

import cv2


class VideoSource:

    def __init__(self, source_path: Path, source_folder: str) -> None:
        self.source_folder = source_folder
        self.source_path = source_path

    @classmethod
    def create_from_source(cls, video_path: str, video_folder=''):
        # Get videos from the product assignment module
        VIDEO_MAIN_FOLDER = Path.cwd() / 'data' / 'videos'

        if video_path == '0':
            source_folder = ''
            source_path = 0
        else:
            if video_folder == VIDEO_MAIN_FOLDER or not video_folder:
                source_path = VIDEO_MAIN_FOLDER / video_path
                source_folder = VIDEO_MAIN_FOLDER
            else:
                source_path = Path(video_folder) / video_path
                source_folder = video_folder

        return cls(source_path, source_folder)

    def get_ocv_video_capture(self) -> cv2.VideoCapture:
        if self.source_path == 0:
            video_source = 0
        else:
            video_source = str(self.source_path)
        return cv2.VideoCapture(video_source)
