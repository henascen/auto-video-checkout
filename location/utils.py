import numpy
import cv2

import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# create handler
c_handler = logging.StreamHandler()
# create formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# add formatter to ch
c_handler.setFormatter(formatter)
# Add handler to logger
logger.addHandler(c_handler)


class LocationUtils:
    def __init__(self) -> None:
        pass

    @staticmethod
    def transform_pnt_with_perspective(
        point: numpy.ndarray,
        perspective_matrix: numpy.ndarray
    ) -> numpy.ndarray:
        point_original = numpy.array(
            [[[point[0], point[1]]]],
            dtype=numpy.float32
        )
        point_transformed = cv2.perspectiveTransform(
            point_original,
            perspective_matrix
        )
        transformed_x = point_transformed[0, 0, 0]
        transformed_y = point_transformed[0, 0, 1]

        return numpy.array([transformed_x, transformed_y])

    @staticmethod
    def transform_pnts_with_perspective(
        points_narray: numpy.ndarray,
        perspective_matrix: numpy.ndarray
    ) -> numpy.ndarray:
        points_transformed = cv2.perspectiveTransform(
            numpy.array([points_narray], dtype=numpy.float32),
            perspective_matrix
        )
        points = points_transformed.shape[1]
        points_dims = points_transformed.shape[2]
        n_points_transformed = numpy.reshape(
            points_transformed,
            (points, points_dims)
        )

        return n_points_transformed