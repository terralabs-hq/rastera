from unittest.mock import MagicMock

import numpy as np
from affine import Affine

from rastera.geo import BBox
from rastera.meta import Profile


def make_profile(width=100, height=100, scale=10.0, count=1, dtype=np.uint8):
    """North-up profile with origin at (0, height*scale)."""
    transform = Affine(scale, 0, 0, 0, -scale, height * scale)
    return Profile(
        width=width,
        height=height,
        count=count,
        dtype=np.dtype(dtype),
        transform=transform,
        res=(scale, scale),
        crs_epsg=32632,
        bounds=BBox(0, 0, width * scale, height * scale),
    )


def make_mock_geotiff(
    width=100, height=100, scale=10.0, count=3,
    tile_width=256, tile_height=256, dtype=np.dtype("u2"),
    nodata=None, crs_epsg=32632,
):
    """Build a mock async_geotiff.GeoTIFF suitable for Profile.from_geotiff()."""
    gt = MagicMock()
    gt.width = width
    gt.height = height
    gt.count = count
    gt.dtype = dtype
    gt.nodata = nodata
    gt.tile_width = tile_width
    gt.tile_height = tile_height

    transform = Affine(scale, 0, 0, 0, -scale, height * scale)
    gt.transform = transform
    gt.res = (scale, scale)

    crs_mock = MagicMock()
    crs_mock.to_epsg.return_value = crs_epsg
    gt.crs = crs_mock

    gt.bounds = (0, 0, width * scale, height * scale)
    gt.overviews = []

    return gt
