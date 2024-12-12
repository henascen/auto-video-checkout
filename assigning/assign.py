import numpy

from typing import List, Tuple, Dict, Union, Set
import logging
from enum import IntEnum, IntFlag
from collections import Counter

from assigning.person import Person
from assigning.hand import Hand
from assigning.utils import AssignUtils


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


class AssignPersonHands:

    WEIGHT_HANDS = 0.3
    WEIGHT_PERSON = 0.5
    WEIGHT_SAME_ID = 0.2

    def __init__(self) -> None:
        self.curr_assignment = None
        self.prev_frame_assignment = None

        self.persons_objs = None
        self.hands_objs = None

        self.score_matrix = None

    def assign_from_norfair_tracked_obj(
            self,
            interaction_objects: Dict[int, List[Union[Hand, Person]]]
    ) -> List[Tuple[Hand, Person]]:
        persons = interaction_objects.get(DetectionLabel.PERSON, None)
        hands = interaction_objects.get(DetectionLabel.HAND, None)

        if persons and hands:
            # There are persons and hands in the frame

            # We make the first or pure assignments, which are the hands
            # that are only inside one person in the frame. We also recover
            # the intercepted hands set which includes the hands that are
            # inside two person objects.

            potential_pure_assignments, intercepted_hands = (
                self.check_each_hand_inside_each_person(
                    person_objs=persons,
                    hand_objs=hands
                )
            )
            intercepted_persons = self.get_persons_hands_w_hand_intrcptn(
                intercepted_hands=intercepted_hands
            )
            new_inter_hands, new_pure_assigns = (
                self.remove_intrcptd_persons_pure_assignments(
                    pure_assignments=potential_pure_assignments,
                    intercepted_persons=intercepted_persons
                )
            )
            # Join the sets from intercepted hands and the new hands from the
            # removed pure assignments given the persons interception to get
            # the final hands.
            intercepted_hands.update(new_inter_hands)
            
            if new_pure_assigns:
                # If assignments have been made,
                # then check for persons with 3 hands
                first_assignment = self.remove_third_hand_from_persons(
                    assignments=new_pure_assigns
                )
            else:
                first_assignment = []

            # From the intercepted hands we need to make the second assignments
            if intercepted_hands:
                # If there are intercepted hands
                second_assignment = self.assign_intercepted_hands(
                    intercepted_hands=intercepted_hands,
                    prev_assignment=self.prev_frame_assignment,
                    intercepted_persons=intercepted_persons
                )
            else:
                second_assignment = []

            final_assignment = first_assignment + second_assignment
        else:
            # Persons, hands or both are missing from the frame
            final_assignment = None

        if final_assignment:
            final_assignment = self.delete_assignments_with_none(
                assignments=final_assignment
            )

        self.assignment = final_assignment

        self.save_curr_as_previous_assignment(
            current_assignment=self.assignment
        )

        return self.assignment

    def save_curr_as_previous_assignment(self, current_assignment):
        self.prev_frame_assignment = current_assignment

    def get_assignment(self):
        return self.assignment

    @staticmethod
    def check_each_hand_inside_each_person(
            person_objs: List[Person],
            hand_objs: List[Hand]
    ) -> Tuple[List[Tuple[Hand, Person]], Set[Hand]]:
        intercepted_hands = set()
        assignments = []

        # For each hand that has been tracked and created as interaction
        for hand in hand_objs:
            # We go through each person that has also been tracked and created
            for person in person_objs:

                # We check if the hand is inside a person
                is_hand_inside_person = AssignUtils.check_bbox1_inside_bbox2(
                    bbox1=hand.bbox_tracking,
                    bbox2=person.bbox_tracking
                )
                # If it is then we can add that person as a potential
                # assignment for the hand
                if is_hand_inside_person:
                    hand.add_potential_person_assignment(
                        potential_person=person
                    )

            # Once we have gone through all the persons we check if the hand
            # is inside two people.
            if (hand.n_potential_assignments > 1):
                # If it is then we add the hand to a set of intercepted hands
                intercepted_hands.add(hand)
            elif (hand.n_potential_assignments == 1):
                # If it only has one potential assignment then we can save
                # the assignment
                assignments.append((hand, hand.potential_persons[0]))
            else:
                # Hand without person
                logger.info(f'Hand {hand.track_id} is not inside any person')

        return assignments, intercepted_hands
    
    def assign_intercepted_hands(
        self,
        intercepted_hands: Set[Hand],
        prev_assignment: List[Tuple[Hand, Person]],
        intercepted_persons: Set[Person],
    ) -> List[Tuple[Hand, Person]]:
        
        # We get the intercepted hands and persons from the set
        intercptd_hands_ordered = list(intercepted_hands)
        # We turned in into a set first to avoid duplicates
        intercptd_persons_ordered = list(intercepted_persons)

        intercptd_hands_to_ref, intercptd_persons_to_ref = (
            self.fill_uneven_intercepted_interactions(
                intercepted_hands_ordered=intercptd_hands_ordered,
                intercepted_persons_ordered=intercptd_persons_ordered,
            )
        )

        # We check if there has been a previous assignment, so that
        # we can use that info on the next assignments
        if prev_assignment:
            # If there are prev assignments then use IoU as score
            assignment_score_matrix = (
                self.compute_score_intercepted_hands_persons(
                    intercepted_hands=intercptd_hands_ordered,
                    intercepted_persons=intercptd_persons_ordered,
                    prev_assignments=prev_assignment,
                    weight_hand=self.WEIGHT_HANDS,
                    weight_person=self.WEIGHT_PERSON,
                    weight_same_id=self.WEIGHT_SAME_ID
                )
            )
        else:
            # But if there are not previous assignments, then we need to
            # make the assignments based only on the information we
            # have

            # IDEA: We can use the distance as the cost inside the
            # matrix, the distance can be only the x distance
            assignment_score_matrix = (
                self.compute_score_intercepted_hands_persons(
                    intercepted_hands=intercptd_hands_ordered,
                    intercepted_persons=intercptd_persons_ordered,
                    prev_assignments=None
                )
            )

        # We use the score matrix to perform the assignment
        hand_indxs, person_indxs = (
            AssignUtils.get_linear_sum_assignment_w_matrix(
                assignment_score_matrix=assignment_score_matrix,
                duplicate_cols=True
            )
        )

        intercepted_assignments = (
            self.create_assignments_from_hand_person_indxs(
                hand_indxs=hand_indxs,
                person_indxs=person_indxs,
                intercepted_hands=intercptd_hands_to_ref,
                intercepted_persons=intercptd_persons_to_ref
            )
        )

        return intercepted_assignments


    
    @staticmethod
    def compute_score_intercepted_hands_persons(
        intercepted_hands: List[Hand],
        intercepted_persons: List[Person],
        prev_assignments: List[Tuple[Hand, Person]],
        weight_hand: float,
        weight_person: float,
        weight_same_id: float
    ) -> numpy.ndarray :
        # We create the score matrix to return the cost of each combination
        assignment_score_matrix = numpy.zeros(
            (
                len(intercepted_hands),
                len(intercepted_persons)
            )
        )

        hand_row_score_counter = 0
        person_column_score_counter = 0
        for person in intercepted_persons:
            for hand in intercepted_hands:
                # Check if the person is inside the potential persons
                
                # For each hand and person prepare the list to save the wsum 
                # of the iou for each hand with each person using the prev 
                # assignment if exists or distance if not
                hand_scores_for_a_person = []

                if prev_assignments:

                    for prev_assignment in prev_assignments:
                        prev_assign_hand = prev_assignment[0]
                        prev_assign_person = prev_assignment[1]

                        # Compute the score of sharing tracking ID
                        if (
                            (hand.track_id == prev_assign_hand.track_id) and
                            (person.track_id == prev_assign_person.track_id)
                        ):
                            curr_prev_track_ids = 1
                        else:
                            curr_prev_track_ids = 0

                        # Compute IoU for the current and prev hand
                        curr_prev_hands_iou = AssignUtils.compute_iou(
                            bbox1=hand.bbox_tracking,
                            bbox2=prev_assign_hand.bbox_tracking
                        )
                        # Compute IoU for the current and prev person
                        curr_prev_persons_iou = AssignUtils.compute_iou(
                            bbox1=person.bbox_tracking,
                            bbox2=prev_assign_person.bbox_tracking
                        )
                        wsum_curr_prev_iou = (
                            (weight_same_id * curr_prev_track_ids) +
                            (weight_hand * curr_prev_hands_iou) +
                            (weight_person * curr_prev_persons_iou)
                        )
                        hand_scores_for_a_person.append(wsum_curr_prev_iou)

                else:
                    # If there are not previous assignments then we need to
                    # use a different score computation

                    # Compute the euclidean distance between hand and
                    # person, the result is the potential new score
                    hand_person_dist = (
                        AssignUtils.compute_euclidean_distance(
                            point1=hand.center_location,
                            point2=person.center_location
                        )
                    )
                    hand_scores_for_a_person.append(hand_person_dist)

                # After going through all the previous assignment, use
                # the max iou weighted sum as the score for the hand for
                # the correspondent person
                score_hand_person = max(hand_scores_for_a_person)

                # Fill the score matrix for the hand (row) and person (column)
                assignment_score_matrix[
                    hand_row_score_counter,
                    person_column_score_counter
                ] = score_hand_person
                # We will fill the next row the next iteration of hands
                hand_row_score_counter += 1

            # We finished all the hands for one person, then fill the next col
            person_column_score_counter += 1
            # Restart the rows counter
            hand_row_score_counter = 0
        
        return assignment_score_matrix

    @staticmethod
    def create_assignments_from_hand_person_indxs(
        hand_indxs: numpy.ndarray,
        person_indxs: numpy.ndarray,
        intercepted_hands: List[Hand],
        intercepted_persons: List[Person]
    ) -> List[Tuple[Hand, Person]]:
        assignments = []
        for hand_indx, person_indx in zip(hand_indxs, person_indxs):
            assignments.append(
                (
                    intercepted_hands[hand_indx],
                    intercepted_persons[person_indx]
                )
            )
        
        return assignments

    @staticmethod
    def remove_third_hand_from_persons(assignments: List[Tuple[Hand, Person]]):
        """
        A pure assignment is when a hand only is inside a person.

        When pure assignments are done, we want to make sure that those
        pure assignments only include two hands by person.

        If a person has more than two hands, then we need to remove one of the
        assignments, the removal decision will be choosing the last assignment
        in the list.
        """
        person_counter = Counter()
        new_assignments = []

        for hand, person in assignments:
            if person_counter[person] < 2:
                new_assignments.append((hand, person))
            person_counter.update([person])

        return new_assignments

    @staticmethod
    def fill_uneven_intercepted_interactions(
        intercepted_hands_ordered: List[Hand],
        intercepted_persons_ordered=List[Person],
    ):
        intercepted_persons_to_ref = (
            intercepted_persons_ordered + intercepted_persons_ordered
        )
        n_hands = len(intercepted_hands_ordered)
        n_persons = len(intercepted_persons_to_ref)

        if n_hands > n_persons:
            missing_persons_cols = n_hands - n_persons
            intercepted_persons_to_ref = (
                intercepted_persons_ordered + (missing_persons_cols * [None])
            )
            intercepted_hands_to_ref = intercepted_hands_ordered
        elif n_hands < n_persons:
            missing_hands_rows = n_persons - n_hands
            intercepted_hands_to_ref = (
                intercepted_hands_ordered + (missing_hands_rows * [None])
            )
        else:
            intercepted_hands_to_ref = intercepted_hands_ordered
        
        return intercepted_hands_to_ref, intercepted_persons_to_ref

    @staticmethod
    def delete_assignments_with_none(
        assignments: List[Tuple[Hand, Person]]
    ) -> List[Tuple[Hand, Person]]:
        assignments_wo_none = [
            assignment
            for assignment in assignments
            if None not in assignment
        ]

        return assignments_wo_none

    @staticmethod
    def get_persons_hands_w_hand_intrcptn(
        intercepted_hands: Set[Hand]
    ) -> Set[Person]:
        persons_intercepted_by_hand = {
            potential_person
            for intercepted_hand in intercepted_hands
            for potential_person in intercepted_hand.potential_persons
        }

        return persons_intercepted_by_hand
    
    @staticmethod
    def remove_intrcptd_persons_pure_assignments(
        pure_assignments: List[Tuple[Hand, Person]],
        intercepted_persons: Set[Person]
    ) -> Tuple[Set[Hand], List[Tuple[Hand, Person]]]:
        new_intercepted_hands = set()
        new_pure_assignments = []
        for pure_hand, pure_person in pure_assignments:
            if pure_person in intercepted_persons:
                new_intercepted_hands.add(pure_hand)
            else:
                new_pure_assignments.append((pure_hand, pure_person))
        
        return new_intercepted_hands, new_pure_assignments

class DetectionLabel(IntEnum):
    PERSON = 4
    HAND = 5
    PRODUCTS = 6
    PRODUCT_A = 0
    PRODUCT_B = 1
    PRODUCT_C = 2
    PRODUCT_D = 3
