"""
utils.py — small shared helpers (no heavy dependencies).
"""

from __future__ import annotations

import math

from config import EARTH_RADIUS_KM


def haversine(coord_a: tuple[float, float], coord_b: tuple[float, float]) -> float:
    """Great-circle distance in km between two (lat, lon) points.

    Earth radius 6371 km (config.EARTH_RADIUS_KM).
    """
    lat1, lon1 = coord_a
    lat2, lon2 = coord_b
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))
