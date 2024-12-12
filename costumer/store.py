import numpy

from typing import List, Tuple
import logging

from costumer.purchase import Purchase
from costumer.costumer import Costumer
from location.products import Products
from assigning.product import Product
from location.utils import LocationUtils
from costumer.utils import CostumerUtils


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


class Store:
    """
    A Store aggregates information from costumers purchase and products. Using 
    their location and attritbutes estimates the product assignment to a given
    costumer.
    """

    CLOSE_DISTANCE_THRESHOLD = 50

    def __init__(self, products: Products, costumers: Purchase) -> None:
        self.products = products
        self.costumers = costumers

    def transform_filter_costumers_location(self):
        active_hands_location_ids = self.extract_costumers_hands_location(
            active_costumers=self.costumers.active_costumers
        )
        active_hands_location = active_hands_location_ids[0]
        active_hands_ids = active_hands_location_ids[1]

        active_hands_location_top = (
            self.transform_costumers_location_table_top(
                active_hands_location=active_hands_location,
                table_prsp_matrix=self.products.table_prsp_matrix
            )
        )
        logger.info(
            'Array with active hands location from '
            f'top {active_hands_location_top.shape}'
        )

        positive_active_hands_location_top_ids = (
            self.filter_positive_hands_location(
                active_hands_location_top=active_hands_location_top,
                active_hands_ids=active_hands_ids
            )
        )
        positive_active_hands_location_top = (
            positive_active_hands_location_top_ids[0]
        )
        positive_active_hands_location_ids = (
            positive_active_hands_location_top_ids[1]
        )

        logger.info(
            'Array with POSITIVE active hands location from '
            f'top {positive_active_hands_location_top.shape}'
        )

        return (
            positive_active_hands_location_top,
            positive_active_hands_location_ids
        )

    @staticmethod
    def extract_costumers_hands_location(
        active_costumers: List[Costumer]
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        # Save the location of each active hand
        active_costumers_hands_location = []
        # Save the id corresponding to the hand in the same position in the
        # active_costumers_hands_location list
        active_costumers_from_location = []
        
        for costumer in active_costumers:
            costumer_hands = costumer.hands
            n_costumer_hands = len(costumer_hands)
            if n_costumer_hands > 1:
                costumer_hands_location = [
                    costumer.first_hand.center_location,
                    costumer.second_hand.center_location
                ]
                costumers_from_location = [
                    costumer,
                    costumer
                ]
            elif n_costumer_hands == 1:
                costumer_hands_location = [
                    costumer.first_hand.center_location
                ]
                costumers_from_location = [
                    costumer
                ]
            else:
                continue

            active_costumers_hands_location.extend(costumer_hands_location)
            active_costumers_from_location.extend(costumers_from_location)

        return (
            numpy.array(active_costumers_hands_location),
            numpy.array(active_costumers_from_location)
        )
    
    @staticmethod
    def transform_costumers_location_table_top(
        active_hands_location: numpy.ndarray,
        table_prsp_matrix: numpy.ndarray
    ) -> numpy.ndarray:
        transformed_active_hands = (
            LocationUtils.transform_pnts_with_perspective(
                points_narray=active_hands_location,
                perspective_matrix=table_prsp_matrix
            )
        )

        return transformed_active_hands
    
    @staticmethod
    def filter_positive_hands_location(
        active_hands_location_top: numpy.ndarray,
        active_hands_ids: numpy.ndarray
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        mask_only_positive_coords = (
            (active_hands_location_top[:, 0] >= 0) &
            (active_hands_location_top[:, 1] >= 0)
        )
        only_positive_hands_locations_top = (
            active_hands_location_top[mask_only_positive_coords]
        )
        only_positive_hands_ids_top = (
            active_hands_ids[mask_only_positive_coords]
        )

        return only_positive_hands_locations_top, only_positive_hands_ids_top

    def determine_close_hands_products(
        self,
        products_location_top: List[numpy.ndarray]
    ) -> Tuple[numpy.ndarray, List[Tuple[Costumer, Product]]]:
        active_hands_location_top, active_costumers_location = (
            self.transform_filter_costumers_location()
        )
        products_location_top = products_location_top
        products_location_top = numpy.array(products_location_top)

        if active_hands_location_top.size > 0:
            hands_products_distances = (
                self.compute_distance_active_hands_products(
                    pos_active_hands_location_top=active_hands_location_top,
                    products_location_top=products_location_top
                )
            )

            close_hands_products = self.filter_close_hands_products(
                close_distance_threshold=self.CLOSE_DISTANCE_THRESHOLD,
                hands_products_distances=hands_products_distances
            )
            
            products_active = numpy.array(self.products.current_products)
            costumer_product_close = self.match_distance_idx_with_names_ids(
                active_costumers_location=active_costumers_location,
                products_active=products_active,
                close_hands_products=close_hands_products
            )

        else:
            hands_products_distances = None
            costumer_product_close = []

        return hands_products_distances, costumer_product_close
    
    @staticmethod
    def compute_distance_active_hands_products(
        pos_active_hands_location_top: numpy.ndarray,
        products_location_top: numpy.ndarray
    ) -> numpy.ndarray:
        hands_products_distances = (
            CostumerUtils.compute_distance_btwn_two_coords_vectors(
                coords_vec_one=pos_active_hands_location_top,
                coords_vec_two=products_location_top
            )
        )

        return hands_products_distances
    
    @staticmethod
    def filter_close_hands_products(
        close_distance_threshold: int,
        hands_products_distances: numpy.ndarray
    ) -> numpy.ndarray:
        close_hands_products = numpy.argwhere(
            hands_products_distances < close_distance_threshold
        )
        return close_hands_products

    @staticmethod
    def match_distance_idx_with_names_ids(
        active_costumers_location: numpy.ndarray,
        products_active: numpy.ndarray,
        close_hands_products: numpy.ndarray
    ) -> List[Tuple[Costumer, Product]]:
        """
        This functions takes the pair of indices that are close in the distance
        matrix. Relates that indices to the ids of the products and costumers.

        Returns a list of tuples of the costumers associated to which product
        they are close to.
        """
        if close_hands_products.size > 0:
            costumers_products_close = []
            for close_hand_product in close_hands_products:
                hand_idx = close_hand_product[0]
                product_idx = close_hand_product[1]

                costumer_from_hand = active_costumers_location[hand_idx]
                product_from_product_idx = products_active[product_idx]

                costumer_product_close = (
                    costumer_from_hand,
                    product_from_product_idx
                )

                costumers_products_close.append(costumer_product_close)
        else:
            costumers_products_close = []
        
        return costumers_products_close
    
    def store_close_products_into_costumer(
            self
    ):
        # USE THE TUPLE COSTUMER PRODUCTS CLOSE TO GO THROUGH EACH ONE
        # AND SAVE THE PRODUCT IN THE COSTUMER HISTORY

        # SHOULD WE SAVE COSTUMER PRODUCTS CLOSENESS WITH THE DISTANCE THAT THEY
        # HAD TO FURTHER FILTER THE ASSIGNMENTS?
        pass
