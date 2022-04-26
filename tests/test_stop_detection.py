"""Test stop detection algorithm and helper functions.
"""

import unittest

from geodata.geodata import route as rt
from geodata.geodata import point as pt
from geodata.geodata import point_t as ptt
import pandas as pd
import numpy as np

from stop_detection.stop_detection import union, intersection, calculate_centroid, extract_pois


def move_(coordinates, x_delta, y_delta):
    """
    Moves the coordinates tuple in x and y direction. The coordinates object is changed in place and returned.

    Parameters
    ----------
    coordinates : list
        A list of two float coordinates x and y.
    x_delta : float
        The distance which should be added onto the x-coordinate.
    y_delta : float
        The distance which should be added onto the y-coordinate.

    Returns
    -------
    coordinates : list
        The coordinates increased by x_delta and y_delta.
    """
    coordinates[0] += x_delta
    coordinates[1] += y_delta
    return coordinates


def set_(coordinates, new_coordinates):
    """
    Set the coordinates tuple to new coordinate values. The coordinates object is changed in place and returned.

    Parameters
    ----------
    coordinates : list
        A list of two float coordinates x and y.
    new_coordinates : list
        A list of the two new float coordinates x and y.

    Returns
    -------
    coordinates : list
        The coordinates object whose values are changed to the provided new values.
    """
    coordinates[0] = new_coordinates[0]
    coordinates[1] = new_coordinates[1]
    return coordinates


class TestStopDetection(unittest.TestCase):
    """Test the stop detection and helper functions."""

    def setUp(self) -> None:
        self.route_a = rt.Route([[0, 0], [1, 1], [2, 2], [3, 3]])
        self.route_b = rt.Route([[1, 1], [3, 3]])
        self.route_c = rt.Route([[4, 4]])
        self.route_d = rt.Route()

    def test_union(self):
        """Test union function.
        """
        self.assertEqual(rt.Route([[0, 0], [1, 1], [2, 2], [3, 3]]), union(self.route_a, self.route_b))
        self.assertEqual(rt.Route([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]), union(self.route_a, self.route_c))
        self.assertEqual(rt.Route([[0, 0], [1, 1], [2, 2], [3, 3]]), union(self.route_a, self.route_d))

    def test_intersection(self):
        """Test intersection function.
        """
        self.assertEqual(rt.Route([[1, 1], [3, 3]]), intersection(self.route_a, self.route_b))
        self.assertEqual(rt.Route(), intersection(self.route_a, self.route_c))
        self.assertEqual(rt.Route(), intersection(self.route_a, self.route_d))

    def test_calculate_centroid(self):
        """Test centroid calculation function.
        """
        route = rt.Route([pt.Point([0, 0], 'cartesian'),
                         pt.Point([0, 1], 'cartesian'),
                         pt.Point([1, 0], 'cartesian'),
                         pt.Point([1, 1], 'cartesian')])
        for point in route:
            point.to_latlon_()

        expected_centroid = pt.Point([0.5, 0.5], 'cartesian')
        expected_centroid.to_latlon_()

        self.assertEqual(expected_centroid, calculate_centroid(route))

    def test_extract_pois(self):
        """Test stop-detection algorithm.
        """
        start_time = pd.Timestamp(0)
        poi_home = [0, 0]
        poi_shop = [30, 10]
        poi_pizza = [90, 15]
        # start and stay at home for an hour
        # transit to a shop in 20 minutes
        # stay in shop for 20 minutes
        # transit to pick up a pizza in 30 minutes
        # pick up pizza very quickly
        # transit back home in 40 minutes
        # stay home for two hours
        #
        # route points:
        # [[0, 0], [1, 1], [10, 5], [20, 10], [30, 10], [31, 11], [31, 12], [32, 12], [31, 11], [30, 12], [29, 13],
        # [70, 15], [90, 15], [55, 20], [10, 20], [1, 1], [1, 2]]

        location = [0, 0]
        route = rt.Route([ptt.PointT(set_(location, poi_home), start_time, 'cartesian'),
                          ptt.PointT(move_(location, 1, 1), start_time + pd.Timedelta(minutes=50), 'cartesian'),
                          ptt.PointT(set_(location, [10, 5]), start_time + pd.Timedelta(minutes=60), 'cartesian'),
                          ptt.PointT(set_(location, [20, 10]), start_time + pd.Timedelta(minutes=70), 'cartesian'),
                          ptt.PointT(set_(location, poi_shop), start_time + pd.Timedelta(minutes=80), 'cartesian'),
                          ptt.PointT(move_(location, 1, 1), start_time + pd.Timedelta(minutes=83), 'cartesian'),
                          ptt.PointT(move_(location, 0, 1), start_time + pd.Timedelta(minutes=90), 'cartesian'),
                          ptt.PointT(move_(location, 1, 0), start_time + pd.Timedelta(minutes=92), 'cartesian'),
                          ptt.PointT(move_(location, -1, -1), start_time + pd.Timedelta(minutes=95), 'cartesian'),
                          ptt.PointT(move_(location, -1, 1), start_time + pd.Timedelta(minutes=96), 'cartesian'),
                          ptt.PointT(move_(location, -1, 1), start_time + pd.Timedelta(minutes=100), 'cartesian'),
                          ptt.PointT(set_(location, [70, 15]), start_time + pd.Timedelta(minutes=115), 'cartesian'),
                          ptt.PointT(set_(location, poi_pizza), start_time + pd.Timedelta(minutes=130), 'cartesian'),
                          ptt.PointT(set_(location, [55, 20]), start_time + pd.Timedelta(minutes=145), 'cartesian'),
                          ptt.PointT(set_(location, [10, 20]), start_time + pd.Timedelta(minutes=160), 'cartesian'),
                          ptt.PointT(set_(location, poi_home), start_time + pd.Timedelta(minutes=170), 'cartesian'),
                          ptt.PointT(move_(location, 0, 1), start_time + pd.Timedelta(minutes=290), 'cartesian'),
                          ptt.PointT(move_(location, 0, 1), start_time + pd.Timedelta(minutes=291), 'cartesian')])
        for point in route:
            point.to_latlon_()
        max_distance = 0
        for i in range(len(route)-1):
            distance = pt.get_distance(route[i], route[i+1])
            if distance > max_distance:
                max_distance = distance

        for distance_threshold_meters, time_threshold_minutes, expected_pois, min_points in [
            [2_000, 5, [poi_home, poi_shop], 1],
            [2_000, 5, [poi_home], 2]
        ]:
            extracted_pois = extract_pois(route, pd.Timedelta(minutes=time_threshold_minutes),
                                          distance_threshold_meters, min_points, print_comments=False)
            for extracted_poi in extracted_pois:
                extracted_poi.to_cartesian_()

            # number of found pois should be correct
            self.assertEqual(len(expected_pois), len(extracted_pois))

            # all pois should be found
            accuracy = 2  # meters?
            for poi in expected_pois:
                poi_valid = False
                for extracted_poi in extracted_pois:
                    x_diff = np.absolute(poi[0] - extracted_poi.x_lon)
                    y_diff = np.absolute(poi[1] - extracted_poi.y_lat)
                    if x_diff < accuracy and y_diff < accuracy:
                        poi_valid = True
                        continue
                self.assertTrue(poi_valid)
