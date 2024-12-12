import cv2
import numpy

from typing import List, Tuple
from assigning.interaction import Interaction
from assigning.product import Product
from assigning.person import Person
from assigning.hand import Hand
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


class Utils:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def draw_assignment_frame(
        assignments: List[Tuple[Hand, Person]],
        original_frame: numpy.ndarray
    ):
        if assignments:
            for hand, person in assignments:

                rect_frame = cv2.rectangle(
                    original_frame,
                    tuple(hand.bbox_tracking[:2].astype(int)),
                    tuple(hand.bbox_tracking[2:].astype(int)),
                    person.color,
                    2
                )
                rect_frame = cv2.rectangle(
                    rect_frame,
                    tuple(person.bbox_tracking[:2].astype(int)),
                    tuple(person.bbox_tracking[2:].astype(int)),
                    person.color,
                    2
                )
                # Write the person id with cv2 text in the rect frame
                cv2.putText(
                    rect_frame,
                    str(person.track_id),
                    (
                        (person.bbox_tracking[0] + 5).astype(int),
                        (person.bbox_tracking[1] + 5).astype(int)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3,
                    person.color,
                    2,
                )
                # Write the hand id with cv2 text in the rect frame
                cv2.putText(
                    rect_frame,
                    str(hand.track_id),
                    (
                        (hand.bbox_tracking[0] + 3).astype(int),
                        (hand.bbox_tracking[1] + 3).astype(int)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    person.color,
                    2,
                )

        else:
            rect_frame = original_frame

        return rect_frame

    @staticmethod
    def draw_interaction_bboxes_in_frame(
        interaction_objs: List[Interaction],
        original_frame: numpy.ndarray,
        bbox_color = (255, 0, 0)
    ):
        """
        Draws the bounding boxes of the interaction objects in the frame.
        :param interaction_objs: List of interaction objects
        :param original_frame: Frame to draw the bounding boxes in
        :return: Frame with bounding boxes drawn
        """
        if interaction_objs:
            for interaction in interaction_objs:
                inter_frame = cv2.rectangle(
                    original_frame,
                    tuple(interaction.bbox_tracking[:2].astype(int)),
                    tuple(interaction.bbox_tracking[2:].astype(int)),
                    bbox_color,
                    1,
                )
        else:
            inter_frame = original_frame
        
        return inter_frame

    @staticmethod
    def draw_product_top_narray_xy_pnts_over_image(
        top_narrays: List[numpy.ndarray],
        top_img: numpy.ndarray,
        products: List[Product],
        radius=5
    ) -> numpy.ndarray:
        img_copy = top_img.copy()
        if top_narrays:
            for xy_pnt, product in zip(top_narrays, products):
                new_img = cv2.circle(
                    img_copy,
                    tuple(xy_pnt[:2].astype(int)),
                    radius,
                    product.color,
                    -1,
                )
                new_img = cv2.putText(
                    new_img,
                    str(product.track_id),
                    tuple(xy_pnt[:2].astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    product.color,
                    1,
                    cv2.LINE_AA
                )
        else:
            new_img = img_copy
        
        return new_img

