import numpy

from typing import Tuple
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


class DetectionUtils:
    def __init__(self) -> None:
        pass

    @staticmethod
    def resize_preds_w_scale_and_padds(
        predictions: numpy.array,
        scale_ratio: float,
        diff_padds: Tuple[int, int]
    ):
        bboxes_resized = (
            predictions[:, 0:4] - numpy.array(diff_padds * 2)
        ) / scale_ratio
        predictions[:, 0:4] = bboxes_resized.round().astype(numpy.int32)

        return predictions
