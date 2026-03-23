"""Unit tests for Profile construction from async-geotiff objects."""

import math
from unittest.mock import MagicMock

import numpy as np
import pytest
from affine import Affine

from rastera.meta import Profile, _coerce_nodata
from tests.conftest import make_mock_geotiff


# ── _coerce_nodata ──────────────────────────────────────────────────────


class TestCoerceNodata:
    def test_none_returns_none(self):
        assert _coerce_nodata(None, np.dtype("f4")) is None

    def test_float_for_float_dtype(self):
        result = _coerce_nodata(-9999.0, np.dtype("f4"))
        assert result == -9999.0
        assert isinstance(result, float)

    def test_float_coerced_to_int_for_int_dtype(self):
        result = _coerce_nodata(255.0, np.dtype("u1"))
        assert result == 255
        assert isinstance(result, int)

    def test_nan_returns_none_for_int_dtype(self):
        assert _coerce_nodata(float("nan"), np.dtype("u2")) is None

    def test_nan_preserved_for_float_dtype(self):
        result = _coerce_nodata(float("nan"), np.dtype("f4"))
        assert math.isnan(result)

    def test_zero_nodata_int(self):
        result = _coerce_nodata(0.0, np.dtype("u1"))
        assert result == 0
        assert isinstance(result, int)

    def test_negative_nodata_int(self):
        result = _coerce_nodata(-32768.0, np.dtype("i2"))
        assert result == -32768
        assert isinstance(result, int)


# ── Profile.from_geotiff ───────────────────────────────────────────────


class TestProfileFromGeotiff:
    def test_basic_profile(self):
        gt = make_mock_geotiff(width=256, height=512, scale=10.0, count=3)
        p = Profile.from_geotiff(gt)
        assert p.width == 256
        assert p.height == 512
        assert p.count == 3
        assert p.dtype == np.dtype("u2")
        assert p.crs_epsg == 32632
        assert p.res == (10.0, 10.0)
        assert p.tile_width == 256
        assert p.tile_height == 256

    def test_nodata_populated(self):
        gt = make_mock_geotiff(nodata=-9999.0)
        p = Profile.from_geotiff(gt)
        assert p.nodata == -9999

    def test_nodata_none(self):
        gt = make_mock_geotiff(nodata=None)
        p = Profile.from_geotiff(gt)
        assert p.nodata is None

    def test_nodata_nan_int_dtype(self):
        gt = make_mock_geotiff(nodata=float("nan"), dtype=np.dtype("u2"))
        p = Profile.from_geotiff(gt)
        assert p.nodata is None

    def test_nodata_nan_float_dtype(self):
        gt = make_mock_geotiff(nodata=float("nan"), dtype=np.dtype("f4"))
        p = Profile.from_geotiff(gt)
        assert math.isnan(p.nodata)

    def test_bounds(self):
        gt = make_mock_geotiff(width=100, height=200, scale=10.0)
        p = Profile.from_geotiff(gt)
        assert p.bounds.minx == 0
        assert p.bounds.miny == 0
        assert p.bounds.maxx == 1000
        assert p.bounds.maxy == 2000

    def test_transform(self):
        gt = make_mock_geotiff(width=100, height=100, scale=10.0)
        p = Profile.from_geotiff(gt)
        assert p.transform == Affine(10, 0, 0, 0, -10, 1000)

    def test_crs_none(self):
        gt = make_mock_geotiff()
        gt.crs.to_epsg.return_value = None
        p = Profile.from_geotiff(gt)
        assert p.crs_epsg is None


# ── Profile.from_overview ──────────────────────────────────────────────


class TestProfileFromOverview:
    def test_overview_profile(self):
        gt = make_mock_geotiff(width=1000, height=1000, scale=10.0, count=3)
        gt.bounds = (0, 0, 10000, 10000)

        overview = MagicMock()
        overview.width = 500
        overview.height = 500
        overview.transform = Affine(20, 0, 0, 0, -20, 10000)
        overview.res = (20.0, 20.0)
        overview.tile_width = 256
        overview.tile_height = 256

        p = Profile.from_overview(gt, overview)
        assert p.width == 500
        assert p.height == 500
        assert p.count == 3
        assert p.res == (20.0, 20.0)
        assert p.bounds.maxx == 10000
        assert p.tile_width == 256

    def test_overview_inherits_nodata(self):
        gt = make_mock_geotiff(nodata=-9999.0)
        gt.bounds = (0, 0, 1000, 1000)

        overview = MagicMock()
        overview.width = 50
        overview.height = 50
        overview.transform = Affine(20, 0, 0, 0, -20, 1000)
        overview.res = (20.0, 20.0)
        overview.tile_width = 128
        overview.tile_height = 128

        p = Profile.from_overview(gt, overview)
        assert p.nodata == -9999
