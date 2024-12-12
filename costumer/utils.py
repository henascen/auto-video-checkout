import numpy
from sklearn.metrics.pairwise import (
    euclidean_distances as sklearn_pairwise_distances
)


class CostumerUtils:
    @staticmethod
    def compute_distance_btwn_two_coords_vectors(
        coords_vec_one: numpy.ndarray,
        coords_vec_two: numpy.ndarray
    ) -> numpy.ndarray:
        """
        Compute the distance matrix between each pair from a vector
        array X and Y.

        X, Y are coordinates ndarrays with shape (N, 2). Where N is the number
        of instances, and 2 the dimensions of the coordinates

        Returns:
            - A distances numpy array with shape (XN, YN). Where the rows are
            the elements coresponding to vector one, and the columns are the
            elements conrresponding to vector two. For example: The distance
            between element X0 and Y0, will be in position (0, 0) in this
            array of distances. 
        """
        coords_distances_narray = sklearn_pairwise_distances(
            coords_vec_one,
            coords_vec_two
        )
        return coords_distances_narray
