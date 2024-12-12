import numpy

from assigning.interaction import Interaction
from assigning.person import Person


class Hand(Interaction):
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

        self.person = None
        self.person_bbox = numpy.array([])
        self.person_bbox_prev = numpy.array([])

        self.potential_persons = []
        self.assignment_scores = []

    def add_potential_person_assignment(self, potential_person: Person):
        self.potential_persons.append(potential_person)
    
    @property
    def n_potential_assignments(self):
        return len(self.potential_persons)
