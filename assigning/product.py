from typing import Tuple

from assigning.interaction import Interaction
from assigning.assign import DetectionLabel


class Product(Interaction):
    TYPE_COLOR_LIST = {
        DetectionLabel.PRODUCT_A: {
            'color': (0, 0, 255),
            'name': 'cocas'
        },
        DetectionLabel.PRODUCT_B: {
            'color': (20, 230, 230),
            'name': 'ememes'
        },
        DetectionLabel.PRODUCT_C: {
            'color': (220, 100, 180),
            'name': 'pringles'
        },
        DetectionLabel.PRODUCT_D: {
            'color': (10, 100, 205),
            'name': 'doritos'
        },
    }

    def __init__(
            self,
            bbox_tracking,
            bbox_last_detection,
            class_name,
            track_id,
    ) -> None:
        super().__init__(
            bbox_tracking,
            bbox_last_detection,
            class_name,
            track_id
        )

        self.top_view_location = None

        product_info = self.assign_product_info(class_name=class_name)
        self.color = product_info[0]
        self.name = product_info[1]

        product_code = self.create_product_code(
            product_id=self.track_id,
            product_name=self.name
        )
        self.code = product_code

    def assign_product_info(self, class_name: int) -> Tuple[Tuple[int], str]:
        product_color = self.TYPE_COLOR_LIST[class_name]['color']
        product_name = self.TYPE_COLOR_LIST[class_name]['name']
        return product_color, product_name
    
    @staticmethod
    def create_product_code(product_id: int, product_name: str) -> str:
        product_code = f'{product_name}#{product_id}'
        return product_code

    def __repr__(self) -> str:
        return repr(f'Product ID: {self.track_id} and Name: {self.name}')
