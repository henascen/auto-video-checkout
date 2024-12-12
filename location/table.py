import cv2
import numpy

from pathlib import Path

class TableLocation:
    
    TRANSFORMS_FOLDER = Path.cwd() / 'location' / 'transforms'
    LOCATION_RESOURCES_FOLDER = Path.cwd() / 'location' / 'resources'
    
    def __init__(self, prsp_matrix_path: str, table_top_image_path: str) -> None:
        self.perspective_matrix = self.load_perspective_matrix(
            prsp_matrix_path=prsp_matrix_path
        )

        table_top_image = cv2.imread(table_top_image_path)
        self.table_top_img_narray = table_top_image
    
    @staticmethod
    def load_perspective_matrix(prsp_matrix_path: str):
        try:
            with open(prsp_matrix_path, 'rb') as f:
                prsp_matrix = numpy.load(f)
        except Exception as e:
            raise e
        return prsp_matrix
    
    @classmethod
    def create_from_prsp_matrix_name(
        cls,
        prsp_matrix_name: str,
        table_top_image_name: str,
        format='npy',
    ):
        prsp_matrix_path = (
            cls.TRANSFORMS_FOLDER / (prsp_matrix_name + '.' + format)
        )
        table_top_image_path = (
            cls.LOCATION_RESOURCES_FOLDER / table_top_image_name
        )

        return cls(
            prsp_matrix_path=str(prsp_matrix_path),
            table_top_image_path=str(table_top_image_path)
        )

