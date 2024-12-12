import queue
from typing import Set, List, Tuple

from location.products import Products
from costumer.purchase import Purchase
from costumer.costumer import Costumer
from assigning.product import Product



class Transaction:
    """
    This class takes the products gone and added, as well as the
    active costumers to compute the transaction that corresponds between them.
    For example, if a costumer takes away a product this class should assign
    that product to that costumer as a purchase.
    """
    def __init__(self, products: Products, costumers: Purchase):
        self.products = products
        self.costumers = costumers
    
    def manage_products_transactions(
            self,
    ):
        products_added_codes, products_gone_codes = (
            self.products.get_products_codes_difference_prev_curr()
        )
    
    def manage_products_gone_transaction(
        self,
        products_gone_codes: Set[str],
    ):
        pass

