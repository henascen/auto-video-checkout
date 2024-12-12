import norfair
import numpy

from assigning.utils import AssignUtils


class Interaction:
    def __init__(
            self,
            bbox_tracking,
            bbox_last_detection,
            class_name,
            track_id,
    ) -> None:
        self.bbox_tracking = bbox_tracking
        self.bbox_last_detection = bbox_last_detection
        self.class_name = class_name
        self.track_id = track_id
        self.center_location = self.compute_center_location(self.bbox_tracking)

    @classmethod
    def create_from_norfair_tracked(
        cls,
        tracked_object: norfair.tracker.TrackedObject
    ):
        bbox_tracking = tracked_object.get_estimate()
        bbox_tracking = AssignUtils.norfair_points_to_standard_bbox(
            norfair_points=bbox_tracking
        )

        bbox_last_detection = tracked_object.last_detection.points
        bbox_last_detection = AssignUtils.norfair_points_to_standard_bbox(
            norfair_points=bbox_last_detection
        )

        label_last_detection = tracked_object.last_detection.label

        id_tracking = tracked_object.id

        return cls(
            bbox_tracking,
            bbox_last_detection,
            label_last_detection,
            id_tracking
        )
    
    @staticmethod
    def compute_center_location(bbox: numpy.ndarray) -> numpy.ndarray:
        center_x = int( ((bbox[2] - bbox[0]) / 2) + bbox[0])
        center_y = int( ((bbox[3] - bbox[1]) / 2) + bbox[1])
        center_location = numpy.array([center_x, center_y], dtype=numpy.int32)

        return center_location

    def __repr__(self) -> str:
        return repr(f'{self.__class__.__name__}: {self.track_id}')
