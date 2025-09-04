"""
coordinate_converter.py

This module provides utility functions for converting geographic coordinates
(latitude, longitude, altitude) into metric coordinates (meters) relative to
a reference point. The primary function supports converting latitude and
longitude to Cartesian x/y coordinates, preserving altitude as-is.

Functions:
-----------
latlon_to_meters(lat, lon, ref_lat=0.0, ref_lon=0.0)
    Convert arrays of latitude and longitude (in degrees) into x and y
    coordinates in meters relative to a reference point (ref_lat, ref_lon).

Usage Example:
--------------
from coordinate_converter import latlon_to_meters

latitudes = [1.3521, 1.3530]
longitudes = [103.8198, 103.8205]
x, y = latlon_to_meters(latitudes, longitudes, ref_lat=1.3521, ref_lon=103.8198)
"""

import numpy as np


def latlon_to_meters(lat, lon, ref_lat=0.0, ref_lon=0.0):
    """
    Convert latitude and longitude to meters relative to a reference point.

    Args:
        lat, lon: Arrays of latitude/longitude in degrees
        ref_lat, ref_lon: Reference point (degrees)

    Returns:
        x, y: Coordinates in meters
    """
    # Earth radius
    R = 6378137  # meters

    # Convert degrees to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    ref_lat_rad = np.radians(ref_lat)
    ref_lon_rad = np.radians(ref_lon)

    x = R * (lon_rad - ref_lon_rad) * np.cos(ref_lat_rad)
    y = R * (lat_rad - ref_lat_rad)
    return x, y
