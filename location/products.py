import numpy

from typing import List, Set, Tuple, Union
import logging

from assigning.product import Product
from location.utils import LocationUtils as location_utils

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



class Products:
    def __init__(self, table_prsp_matrix: numpy.ndarray) -> None:
        
        self.table_prsp_matrix = table_prsp_matrix
        self.current_products = None
        self.prev_frame_products = None

        self.curr_product_top_loc = []

        self.current_products_codes = []

        self.current_products_codes_set = {}
        self.prev_frame_products_codes_set = {}

    
    def update_current_products(self, products: List[Product]):
        self.curr_product_top_loc = []
        self.current_products = products
        return self.current_products
    
    def assign_top_location_current_products(
            self,
            current_products: List[Product]
    ) -> List[numpy.ndarray]:
        if current_products:
            for product in current_products:
                try:
                    # FIX -- INSTEAD OF DOING THIS ONCE FOR EACH PRODUCT
                    # GATHER ALL THE DATA AND DO ONE OPENCV TRANSFORMATION

                    # MISSING FILTERING ONLY POSITIVES COORDS FOR PRODUCTS
                    product_top_location = (
                        location_utils.transform_pnt_with_perspective(
                            point=product.center_location,
                            perspective_matrix=self.table_prsp_matrix
                        )
                    )
                    product.top_view_location = product_top_location
                    self.curr_product_top_loc.append(product_top_location)
                except Exception as e:
                    logger.error('Perspective Transformation Failed')
                    raise e
        else:
            logger.info("No current products to assign")

    def get_products_top_location(self) -> List[numpy.ndarray]:
        return self.curr_product_top_loc
    
    def compute_products_top_location_narrays(
            self,
            products: List[Product]
    ) -> List[numpy.ndarray]:

        self.prev_frame_products = self.current_products
        
        self.update_current_products(products=products)

        self.assign_top_location_current_products(
            current_products=self.current_products
        )

        if self.current_products:
            self.products_codes = self.extract_products_codes(
                products=self.current_products
            )
            self.current_products_codes_set = set(
                self.products_codes
            )

            prev_frame_products_data = self.manage_prev_frame_products()
            self.prev_frame_products_codes = prev_frame_products_data[0]
            self.prev_frame_products_codes_set = prev_frame_products_data[1]

        products_top_location = self.get_products_top_location()

        return products_top_location
    
    @staticmethod
    def extract_products_ids(products: List[Product]) -> List[int]:
        products_ids = [product.track_id for product in products]
        return products_ids
    
    @staticmethod
    def extract_products_names(products: List[Product]) -> List[str]:
        products_names = [product.name for product in products]
        return products_names
    
    @staticmethod
    def extract_products_codes(products: List[Product]) -> List[str]:
        products_codes = [product.code for product in products]
        return products_codes

    def manage_prev_frame_products(
        self,
    ):
        if self.prev_frame_products is not None:
            prev_frame_products_codes = self.extract_products_codes(
                products=self.prev_frame_products
            )
            prev_frame_products_codes_set = set(prev_frame_products_codes)
        else:
            prev_frame_products_codes = []
            prev_frame_products_codes_set = set()
        
        return (prev_frame_products_codes, prev_frame_products_codes_set)

    @staticmethod
    def compute_products_diff_btwn_curr_prev_frame(
        current_products_set: Set[Union[str, int]],
        prev_frame_products_set: Set[Union[str, int]],
    ) -> Tuple[Set[Union[str, int]], Set[Union[str, int]]]:
        try:
            products_added_curr_prev = current_products_set.difference(
                prev_frame_products_set
            )
            products_gone_curr_prev = prev_frame_products_set.difference(
                current_products_set
            )
        except Exception as e:
            logger.error(f'Product set difference failed by: {e}')

        return products_added_curr_prev, products_gone_curr_prev
    
    def get_products_codes_difference_prev_curr(
        self
    ) -> Tuple[Set[Union[str, int]], Set[Union[str, int]]]:
        diff_products_added_codes, diff_products_gone_codes = (
            self.compute_products_diff_btwn_curr_prev_frame(
                current_products_set=self.current_products_codes_set,
                prev_frame_products_set=self.prev_frame_products_codes_set
            )
        )

        return diff_products_added_codes, diff_products_gone_codes

