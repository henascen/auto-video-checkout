from pathlib import Path
from typing import Tuple

import cv2
import numpy


class ImageSource:

    def __init__(self, image_path: Path, image_folder: str) -> None:
        self.image_path = image_path
        self.image_folder = image_folder

        self.image_array = self.read_image(image_path)

        image_height, image_width = self.get_height_width(self.image_array)
        self.height = image_height
        self.width = image_width

    @staticmethod
    def read_image(image_path: Path) -> numpy.array:
        image_array = cv2.imread(str(image_path))
        return image_array

    @classmethod
    def create_from_source(cls, image_path: str, image_folder=''):

        VIDEO_MAIN_FOLDER = Path.cwd() / 'data' / 'images'

        if image_folder == VIDEO_MAIN_FOLDER or not image_folder:
            source_path = VIDEO_MAIN_FOLDER / image_path
            source_folder = VIDEO_MAIN_FOLDER
        else:
            source_path = Path(image_folder) / image_folder
            source_folder = image_folder

        return cls(source_path, source_folder)

    @staticmethod
    def get_height_width(image_array) -> Tuple[int, int]:
        image_height, image_width, _ = image_array.shape
        return image_height, image_width
