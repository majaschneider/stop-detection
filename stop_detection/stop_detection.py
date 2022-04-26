"""
An algorithm to detect stops based on time- and space-based clustering. The implementation is according to the
approach of Primault, V. (2018) Practically Preserving and Evaluating Location Privacy.
"""

from geodata.geodata import point as pt
from geodata.geodata import route as rt
import numpy as np


def calculate_centroid(route):
    """
    Calculates the euclidian centroid of a route.

    Parameters
    ----------
    route : rt.Route
        The route representing a collection of geographical points in 'latlon' format to calculate the centroid for.

    Returns
    -------
    centroid : pt.Point
        The centroid of route's points in 'latlon' formate, calculated by averaging the points' coordinates in the
        euclidian domain.
    """
    route = route.deep_copy()
    for point in route:
        point.to_cartesian_()
    x_mean = np.mean([point.x_lon for point in route])
    y_mean = np.mean([point.y_lat for point in route])
    centroid = pt.Point([x_mean, y_mean], 'cartesian')
    centroid.to_latlon_()
    return centroid


def intersection(route_a, route_b):
    """
    Returns the points of route_a, that are also present in route_b.

    Parameters
    ----------
    route_a : rt.Route
        The first route.
    route_b : rt.Route
        The second route.

    Returns
    -------
    common_points : rt.Route
        The points of route_a, that are also present in route_b.
    """
    common_points = rt.Route()
    for point in route_a:
        if point in route_b:
            common_points.append(point)
    return common_points


def union(route_a, route_b):
    """
    Returns a union of points of route_a and route_b.

    Parameters
    ----------
    route_a : rt.Route
        The first route.
    route_b : rt.Route
        The second route.

    Returns
    -------
    union : rt.Route
        The points of route_a and route_b without duplicates.
    """
    union_a_b = route_a.deep_copy()
    for point in route_b:
        if point not in route_a:
            union_a_b.append(point)
    return union_a_b


def extract_pois(route, time_threshold, distance_threshold, min_points=1, merge_threshold=0.5, print_comments=False):
    """
    Extracts places of interest from a route of geographical points with timestamps. Implementation according to
    Primault, V. (2018) Practically Preserving and Evaluating Location Privacy, p. 44.

    The algorithm has two parts:
        1. Extracting stays:
        Stays are defined according to Hariharan, R. and Toyama, K. (2004) ‘Project lachesis: Parsing and modeling
        location histories’. The authors describe stays as 'spending some time in one place', characterized by the
        roaming distance and the stay duration. Depending on the scale of these parameters, the stay has different
        interpretability, e.g. when considering a short stay in a shop or a longer stay in a holiday resort.

        2. Aggregating stays into POIs:
        Frequent and nearby stays are merged such that clusters have a minimum number of stays and a minimum distance
        from each other.

    Parameters
    ----------
    route : rt.Route
        A route containing geographical points with timestamps in 'latlon' format, indicating a trajectory of a moving
        object.
    time_threshold : pandas.Timedelta
        The minimum time duration that has to be spent in every stay.
    distance_threshold : float
        The maximal diameter of the stay area in meters.
    min_points : int
        A minimum number of stays necessary to create a POI.
    merge_threshold : float
        Defines the maximum distance in percent, under which two distinct clusters are merged into a single one. The
        default is 50 % of the distance threshold, which means, such two clusters are merged, that have half of their
        area in common.
    print_comments : bool
        Indicates whether comments should be printed to help with debugging.

    Returns
    -------
    pois : list
        A list of geodata.point.Point objects each representing a place of interest found in the route.
    """
    # 1. Extract stays
    # Route objects are used as point collections
    stays = rt.Route()      # Stays extracted so far
    candidate_stay = rt.Route()     # collection of points that form a stay region
    idx = 0
    route_len = len(route)
    route = route.deep_copy()
    while idx < route_len:
        # get max distance between current route point and all points in stay
        candidate_stay_diameter = 0
        for event in candidate_stay:
            distance_event_to_route_point = pt.get_distance(route[idx], event)
            if distance_event_to_route_point > candidate_stay_diameter:
                candidate_stay_diameter = distance_event_to_route_point
        if print_comments:
            print("max distance in stay", candidate_stay_diameter)

        # check if adding this route point to the candidate stay will surpass its allowed diameter
        # if the candidate stay is still empty, the current route point will be added to it by default
        if candidate_stay_diameter <= distance_threshold:
            candidate_stay.append(route[idx])
            if print_comments:
                print("appending point to candidate stay", route[idx].to_cartesian())
            idx += 1
        # if the diameter is surpassed, check if the elapsed time inside the candidate stay is above the time_threshold
        else:
            if print_comments:
                print("max allowed distance in stay of", distance_threshold, "surpassed")
            # since Route objects are kept sorted by timestamp, min/max timestamp correlate to first/last route point
            max_events_time = candidate_stay[len(candidate_stay) - 1].timestamp
            min_events_time = candidate_stay[0].timestamp
            # if time_threshold is surpassed, the candidate stay is valid and appended to the valid stays
            if print_comments:
                print("max time in candidate stay is", (max_events_time - min_events_time))
            if max_events_time - min_events_time >= time_threshold:
                if print_comments:
                    print("passed time in candidate stay and is bigger than allowed threshold of", time_threshold)
                centroid = calculate_centroid(candidate_stay)
                stays.append(centroid)
                if print_comments:
                    print("appending centroid to stay", centroid.to_cartesian())
                # reset candidate stay
                candidate_stay = rt.Route()
                if print_comments:
                    print("")
            else:
                if print_comments:
                    print("passed time in candidate stay and is below the allowed threshold of", time_threshold)
                    print("removing candidate stay", candidate_stay[0].to_cartesian())
                candidate_stay.remove(candidate_stay[0])

    # 2. Aggregate POIs
    clusters = []   # list of Route objects
    for stay in stays:
        # get neighbourhood around stay consisting of other stays
        neighbourhood = rt.Route()
        for neighbour in stays:
            if pt.get_distance(neighbour, stay) <= merge_threshold * distance_threshold:
                neighbourhood.append(neighbour)
        # if neighbourhood is big enough (number of stays in a region surpasses a threshold)
        if len(neighbourhood) >= min_points:
            # if clusters are empty, the first neighbourhood will be appended by default
            for cluster in clusters:
                # if at least one stay of the neighbourhood is already in one of the clusters, extend that cluster by
                # the neighbourhood
                if len(intersection(neighbourhood, cluster)) > 0:
                    # extend the cluster by all stays of the neighbourhood
                    neighbourhood = union(neighbourhood, cluster)
                    clusters.remove(cluster)
            clusters.append(neighbourhood)
    pois = [calculate_centroid(cluster) for cluster in clusters]
    return pois
