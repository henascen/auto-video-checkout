import numpy

import random

from assigning.interaction import Interaction


class Person(Interaction):
    def __init__(
            self,
            bbox_tracking,
            bbox_last_detection,
            class_name,
            track_id
    ) -> None:
        super().__init__(
            bbox_tracking,
            bbox_last_detection,
            class_name,
            track_id
        )

        self.hand_one = None
        self.hand_two = None
        self.hands_bboxes = numpy.array([])
        self.hands_bboxes_prev = numpy.array([])

        self.color = tuple(random.choices(range(256), k=3))
