import numpy
from scipy.optimize import linear_sum_assignment as scipy_linsum_assignment
from scipy.spatial import distance as scipy_distance

from typing import Tuple, List
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


class AssignUtils:

    @staticmethod
    def norfair_points_to_standard_bbox(norfair_points: numpy.ndarray):
        standard_bbox_points = numpy.array(
            [
                norfair_points[0][0],
                norfair_points[0][1],
                norfair_points[1][0],
                norfair_points[1][1],
            ]
        )
        return standard_bbox_points
    
    @staticmethod
    def check_bbox1_inside_bbox2(
        bbox1: numpy.ndarray,
        bbox2: numpy.ndarray,
        alpha_x=35,
        alpha_y=35
    ) -> bool:
        """
        Method to check if a bbox 1 is inside bbox 2. Returns True if that is
        the case, else returns False.
        - bbox1: Numpy array containing the bounding box coordinates.
        - bbox2: Numpy array containing the bounding box coordinates.
        - alpha_x: Value to increase bbox2 in the x direction. For example to
        make it larger so that the evaluation with bbox1 can cover more area.
        - alpha_y: Value to increase bbox2 in the y direction. For example to
        make it larger so that the evaluation with bbox1 can cover more area.
        """
        x1a_in_x2a = True if bbox1[0] >= (bbox2[0] - alpha_x) else False
        y1a_in_y2a = True if bbox1[1] >= (bbox2[1] - alpha_y) else False

        x1b_in_x2b = True if bbox1[2] <= (bbox2[2] + alpha_x) else False
        y1b_in_y2b = True if bbox1[3] <= (bbox2[3] + alpha_y) else False

        if x1a_in_x2a and y1a_in_y2a and x1b_in_x2b and y1b_in_y2b:
            return True

        return False
    
    @staticmethod
    def compute_iou(
        bbox1: numpy.ndarray,
        bbox2: numpy.ndarray,
    ) -> float :
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(bbox1[0], bbox2[0])
        yA = max(bbox1[1], bbox2[1])
        xB = min(bbox1[2], bbox2[2])
        yB = min(bbox1[3], bbox2[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        bbox_oneArea = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
        bbox_twoArea = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(bbox_oneArea + bbox_twoArea - interArea)

        # return the intersection over union value
        return iou

    @staticmethod
    def get_linear_sum_assignment_w_matrix(
        assignment_score_matrix: numpy.ndarray,
        duplicate_cols: bool
    ) -> Tuple[List, List] :
        if duplicate_cols:
            assignment_score_matrix = numpy.concatenate(
                (assignment_score_matrix, assignment_score_matrix),
                axis=1
            )
        
        row_indxs, col_indxs = scipy_linsum_assignment(
            assignment_score_matrix,
            maximize=True
        )

        return row_indxs, col_indxs

    @staticmethod
    def compute_euclidean_distance(
        point1: numpy.ndarray,
        point2: numpy.ndarray
    ):
        dist_btwn_point1_point2 = scipy_distance.euclidean(
            u=point1,
            v=point2
        )
        return dist_btwn_point1_point2
