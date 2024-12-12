import norfair
from norfair.filter import OptimizedKalmanFilterFactory
import numpy

from typing import List, Optional, Dict, Union
import logging
import collections

from assigning.person import Person
from assigning.hand import Hand
from assigning.product import Product
from assigning.assign import DetectionLabel


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


class NorfairTracker:

    PRODUCTS_LABELS = [label for label in DetectionLabel if label < 4]

    def __init__(self,) -> None:
        self.tracker = norfair.Tracker(
            initialization_delay=5,
            distance_function='iou',
            distance_threshold=2,
            hit_counter_max=10,
            filter_factory=OptimizedKalmanFilterFactory(),
            past_detections_length=2,
        )

    @staticmethod
    def convert_predictions_to_norfair(
        predictions: numpy.array
    ) -> List[norfair.Detection]:
        norfair_detections = []
        for prediction in predictions:
            bbox_norfair = numpy.array(
                [
                    [prediction[0], prediction[1]],
                    [prediction[2], prediction[3]]
                ]
            )
            detection = norfair.Detection(
                points=bbox_norfair,
                scores=numpy.array([prediction[5], prediction[5]]),
                label=int(prediction[4])
            )
            norfair_detections.append(detection)

        return norfair_detections

    def update_from_predictions(
            self,
            predictions: Union[Optional[numpy.array], None] = None,
            period: int = 1
    ):
        if predictions is None:
            tracked_objects = self.tracker.update(
                period=period
            )
        else:
            norfair_detections = self.convert_predictions_to_norfair(
                predictions=predictions
            )
            tracked_objects = self.tracker.update(
                detections=norfair_detections,
                period=period
            )

        return tracked_objects

    @staticmethod
    def draw_norfair_tracked_boxes(image, norfair_tracked_objects):
        image_with_tracked = norfair.draw_boxes(
            frame=image,
            drawables=norfair_tracked_objects
        )
        return image_with_tracked

    @staticmethod
    def get_label_from_tracked_obj(
        norfair_tracked_object: norfair.tracker.TrackedObject
    ):
        label = norfair_tracked_object.last_detection.label
        return label

    @staticmethod
    def separate_interactions_from_norfair_tracked_objs(
        norfair_tracked_objects: List[norfair.tracker.TrackedObject],
        products_labels = PRODUCTS_LABELS
    ) -> Dict[int, List[Union[Person, Hand,Product]]]:
        """
        Save the tracked objects as interaction objects, returns a dictionary
        with keys equal to the existing labels of tracked objects, and items
        with lists of either Person, Hand or Product objects.
        """
        interaction_objs = collections.defaultdict(list)

        for tracked_object in norfair_tracked_objects:
            tracked_obj_label = NorfairTracker.get_label_from_tracked_obj(
                tracked_object
            )

            """
            match tracked_obj_label:
                case DetectionLabel.PERSON:
                    interaction_obj = Person.create_from_norfair_tracked(
                        tracked_object=tracked_object
                    )
                    interaction_key = DetectionLabel.PERSON
                case DetectionLabel.HAND:
                    interaction_obj = Hand.create_from_norfair_tracked(
                        tracked_object=tracked_object
                    )
                    interaction_key = DetectionLabel.HAND
                case product if product in products_labels:
                    interaction_obj = Product.create_from_norfair_tracked(
                        tracked_object
                    )
                    interaction_key = DetectionLabel.PRODUCTS
                case _:
                    continue
            """
            
            if tracked_obj_label == DetectionLabel.PERSON:
                interaction_obj = Person.create_from_norfair_tracked(
                    tracked_object=tracked_object
                )
                interaction_key = DetectionLabel.PERSON
            elif tracked_obj_label == DetectionLabel.HAND:
                interaction_obj = Hand.create_from_norfair_tracked(
                    tracked_object=tracked_object
                )
                interaction_key = DetectionLabel.HAND
            elif tracked_obj_label in products_labels:
                interaction_obj = Product.create_from_norfair_tracked(
                    tracked_object
                )
                interaction_key = DetectionLabel.PRODUCTS
            else:
                continue

            if interaction_key not in interaction_objs:
                interaction_objs[interaction_key] = [interaction_obj]
            else:
                interaction_objs[interaction_key].append(interaction_obj)

        return interaction_objs
