from typing import List, Tuple, Dict, Optional

from costumer.costumer import Costumer
from assigning.person import Person
from assigning.hand import Hand


class Purchase:
    def __init__(
            self,
            active_costumers : Optional[List[Costumer]] = [],
            on_hold_costumers: Optional[List[Costumer]] = []
    ) -> None:
        """
        This methods handles the costumers in the session.

        - active_costumers = List of Costumers that have been saved from the
        last frame that had an assignment available.
        - on_hold_costumers = List of Costumers that were created but have not
        had an assignment for an x number of frames. They are on hold as they
        could be converted to active if an assignment comes within a window
        of frames.

        """
        self.active_costumers = active_costumers
        self.on_hold_costumers = on_hold_costumers

    def manage_costumers(
            self,
            person_hand_assignments: List[Tuple[Person, Hand]]
    ):
        """
        Handle the person_hand_assignments from the current frame to:
            - Create new costumers if the person does not exist
            - Update hands of active costumers
            - Forget (Delete) costumers that have not been active
        """
        active_costumers_person_ids = {
            costumer.person_id: costumer
            for costumer in self.active_costumers
        }
        on_hold_costumers_person_ids = {
            costumer.person_id: costumer
            for costumer in self.on_hold_costumers
        }

        created_updated_costumers = self.get_managing_costumers_data(
            active_costumers_person_ids=active_costumers_person_ids,
            on_hold_costumers_person_ids=on_hold_costumers_person_ids,
            person_hand_assignments=person_hand_assignments
        )

        updated_costumers_person_ids_set = set(created_updated_costumers)

        new_on_hold_costumers = set(self.active_costumers).difference(
            updated_costumers_person_ids_set
        )

        self.active_costumers = created_updated_costumers
        self.on_hold_costumers = list(new_on_hold_costumers)

    @staticmethod
    def get_managing_costumers_data(
            active_costumers_person_ids: Dict[int, Costumer],
            on_hold_costumers_person_ids: Dict[int, Costumer],
            person_hand_assignments: List[Tuple[Hand, Person]]
    ) -> List[Costumer]:
        # Costumers to be active after this processing
        updated_created_costumers = []

        if person_hand_assignments:
            for person_hand_assignment in person_hand_assignments:
                create_costumer = False
                incoming_person = person_hand_assignment[1]
                incoming_hand = person_hand_assignment[0]

                if incoming_person.track_id in active_costumers_person_ids:
                    # Update the costumer that is already in active costumers
                    costumers_to_update = active_costumers_person_ids
                elif incoming_person.track_id in on_hold_costumers_person_ids:
                    # The costumer was on hold, it should be reactivated and
                    # updated
                    costumers_to_update = on_hold_costumers_person_ids
                else:
                    # Create the costumer that was not in either the active and
                    # on hold costumers
                    create_costumer = True
                
                if create_costumer:
                    # Create a new costumer instance
                    costumer_to_update = (
                        Costumer.create_from_person_hand_assignment(
                            assignment=(incoming_hand, incoming_person)
                        )
                    )
                else:
                    # Update the costumer instance from the appropiate set
                    # of costumers
                    costumer_to_update = costumers_to_update.get(
                        incoming_person.track_id,
                        None
                    )

                if costumer_to_update in updated_created_costumers:
                    # If the costumer was already created or updated in this
                    # frame, then update the second hand
                    costumer_to_update.update_hand(
                        hand=incoming_hand,
                        first_hand=False
                    )
                elif not create_costumer:
                    # Update the hand if the costumer already existed
                    costumer_to_update.update_hand(
                        hand=incoming_hand
                    )
                    updated_created_costumers.append(costumer_to_update)
                else:
                    # If it's a new costumer and first hand, then only
                    # add it to the list of costumers created in this frame
                    updated_created_costumers.append(costumer_to_update)
        
        return updated_created_costumers

    def get_active_costumers(self):
        return self.active_costumers
