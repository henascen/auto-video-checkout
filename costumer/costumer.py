from typing import List, Tuple
import queue

from assigning.person import Person
from assigning.hand import Hand

class Costumer:

    MAX_FRAMES_CLOSE_HISTORY = 10

    def __init__(self, person: Person, hands: List[Hand]) -> None:
        self.person = person
        self.person_id = person.track_id

        self.hands = hands
        self.first_hand = hands[0]
        self.second_hand = None

        self.products = []
        self.close_products_history = queue.LifoQueue(
            maxsize=self.MAX_FRAMES_CLOSE_HISTORY
        )

        self.active = True
        self.n_frames_inactive = 0

    @classmethod
    def create_from_person_hand_assignment(
        cls,
        assignment: Tuple[Hand, Person]
    ):
        return cls(
            person=assignment[1],
            hands=[assignment[0]]
        )
    
    def add_second_hand(self, second_hand: Hand):
        if second_hand.track_id != self.first_hand.track_id:
            self.second_hand = second_hand
            self.hands = self.hands + [self.second_hand]
        elif second_hand is not None:
            raise 'Costumer alread has a second hand'
        else:
            raise 'Hand already exists in the Costumer'
    
    def update_hand(self, hand: Hand, first_hand=True):
        if first_hand:
            self.first_hand = hand
        else:
            self.add_second_hand(second_hand=hand)
    
    def __repr__(self) -> str:
        return (
            f'Costumer with {self.person} and '
            f'hands: {self.first_hand} & {self.second_hand}'
        )

