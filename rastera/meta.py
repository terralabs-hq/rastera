from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from affine import Affine
import numpy as np

from .geo import BBox, Window, bounds_from_transform

@dataclass
class Profile:
    """Core geospatial metadata for a GeoTIFF-like dataset.

    This mirrors the minimal subset of information we need from the TIFF IFD
    to perform pixel/coordinate conversions and windowing.
    """

    width: int
    height: int
    count: int  # number of bands (samples per pixel)
    dtype: np.dtype
    transform: Affine  # maps (col,row) -> (x,y)
    res: tuple[float, float]  # pixel size (xres, yres) in CRS units
    crs_epsg: int | None
    bounds: BBox
    nodata: int | float | None = None
    overviews: list[tuple[int, int]] | None = None
    tile_width: int | None = None
    tile_height: int | None = None

    def __repr__(self) -> str:
        parts = [
            f"width={self.width}",
            f"height={self.height}",
            f"count={self.count}",
            f"dtype={self.dtype!r}",
            f"res={self.res}",
            f"crs_epsg={self.crs_epsg}",
            f"nodata={self.nodata}",
            f"overviews={self.overviews}",
            f"bounds={self.bounds}",
            f"transform={self.transform!r}",
        ]
        return f"Profile({', '.join(parts)})"

    def copy(self) -> Profile:
        return replace(self)

    @classmethod
    def for_bbox(
        cls,
        bbox: BBox,
        res: float,
        crs_epsg: int | None,
        count: int,
        dtype: np.dtype,
        nodata: int | float | None = None,
        tile_width: int | None = None,
        tile_height: int | None = None,
    ) -> Profile:
        """Build a Profile for a regular grid covering *bbox* at the given resolution."""
        width = max(1, math.ceil(bbox.width / res))
        height = max(1, math.ceil(bbox.height / res))
        transform = Affine(res, 0, bbox.minx, 0, -res, bbox.maxy)
        bounds = BBox(
            bbox.minx,
            bbox.maxy - height * res,
            bbox.minx + width * res,
            bbox.maxy,
        )
        return cls(
            width=width,
            height=height,
            count=count,
            dtype=dtype,
            transform=transform,
            res=(res, res),
            crs_epsg=crs_epsg,
            bounds=bounds,
            nodata=nodata,
            tile_width=tile_width,
            tile_height=tile_height,
        )

    def adjust_to_window(self, window: Window) -> Profile:
        window_transform = self.transform * Affine.translation(
            window.col_min, window.row_min
        )
        width = window.win_width
        height = window.win_height
        return replace(
            self,
            width=width,
            height=height,
            transform=window_transform,
            bounds=bounds_from_transform(window_transform, width, height),
        )

    @classmethod
    def from_geotiff(cls, gt: Any) -> Profile:
        """Construct a Profile from an ``async_geotiff.GeoTIFF`` instance."""
        dtype = gt.dtype
        nodata = _coerce_nodata(gt.nodata, dtype)
        bounds = gt.bounds  # (minx, miny, maxx, maxy)
        return cls(
            width=gt.width,
            height=gt.height,
            count=gt.count,
            dtype=dtype,
            transform=gt.transform,
            res=gt.res,
            crs_epsg=gt.crs.to_epsg(),
            bounds=BBox(*bounds),
            nodata=nodata,
            tile_width=gt.tile_width,
            tile_height=gt.tile_height,
        )

    @classmethod
    def from_overview(cls, gt: Any, overview: Any) -> Profile:
        """Construct a Profile for an ``async_geotiff.Overview``."""
        dtype = gt.dtype
        nodata = _coerce_nodata(gt.nodata, dtype)
        bounds = gt.bounds  # overviews share the base image bounds
        return cls(
            width=overview.width,
            height=overview.height,
            count=gt.count,
            dtype=dtype,
            transform=overview.transform,
            res=overview.res,
            crs_epsg=gt.crs.to_epsg(),
            bounds=BBox(*bounds),
            nodata=nodata,
            tile_width=overview.tile_width,
            tile_height=overview.tile_height,
        )


def _coerce_nodata(nodata: float | None, dtype: np.dtype) -> int | float | None:
    """Coerce nodata from async-geotiff (always float) to match the raster dtype."""
    if nodata is None:
        return None
    kind = np.dtype(dtype).kind
    if kind in ("i", "u"):
        return None if math.isnan(nodata) else int(nodata)
    return float(nodata)
