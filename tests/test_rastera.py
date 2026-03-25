"""Unit tests for AsyncGeoTIFF."""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from affine import Affine

import rastera
from rastera.reader import AsyncGeoTIFF, _extract_key
from async_geotiff import Window

from rastera.geo import BBox
from rastera.meta import Profile
from tests.conftest import make_mock_geotiff


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_read_result(shape, dtype=np.uint16, fill=1):
    """Create a mock async-geotiff Array result."""
    result = MagicMock()
    result.data = np.full(shape, fill, dtype=dtype)
    return result


# ── Construction & properties ────────────────────────────────────────────


class TestAsyncGeoTIFFInit:
    def test_construction(self):
        gt = make_mock_geotiff()
        obj = AsyncGeoTIFF("s3://bucket/key.tif", gt)
        assert obj.uri == "s3://bucket/key.tif"
        assert isinstance(obj.profile, Profile)

    def test_repr(self):
        gt = make_mock_geotiff()
        obj = AsyncGeoTIFF("s3://bucket/key.tif", gt)
        r = repr(obj)
        assert "AsyncGeoTIFF" in r
        assert "s3://bucket/key.tif" in r

    def test_profile_matches_geotiff(self):
        gt = make_mock_geotiff(width=200, height=150, count=4)
        obj = AsyncGeoTIFF("s3://b/k.tif", gt)
        assert obj.profile.width == 200
        assert obj.profile.height == 150
        assert obj.profile.count == 4

    def test_overviews_populated(self):
        gt = make_mock_geotiff()
        ovr = MagicMock()
        ovr.width = 50
        ovr.height = 50
        gt.overviews = [ovr]
        obj = AsyncGeoTIFF("s3://b/k.tif", gt)
        assert obj.overviews == [(50, 50)]


# ── open() classmethod ──────────────────────────────────────────────────


class TestOpen:
    @pytest.mark.asyncio
    @patch("rastera.reader.GeoTIFF")
    @patch("rastera.reader.from_url")
    async def test_open_auto_store(self, mock_from_url, mock_geotiff_cls):
        """Without an explicit store, from_url builds one from the URI."""
        gt = make_mock_geotiff()
        mock_store = MagicMock()
        mock_from_url.return_value = mock_store
        mock_geotiff_cls.open = AsyncMock(return_value=gt)

        obj = await AsyncGeoTIFF.open(
            "s3://bucket/key.tif", skip_signature=True
        )

        mock_from_url.assert_called_once_with(
            "s3://bucket/key.tif", skip_signature=True, region="us-west-2"
        )
        mock_geotiff_cls.open.assert_awaited_once_with(
            "", store=mock_store, prefetch=32768
        )
        assert obj.uri == "s3://bucket/key.tif"
        assert isinstance(obj, AsyncGeoTIFF)

    @pytest.mark.asyncio
    @patch("rastera.reader.GeoTIFF")
    async def test_open_with_store(self, mock_geotiff_cls):
        """With an explicit store, from_url is NOT called; key is extracted from URI."""
        gt = make_mock_geotiff()
        mock_geotiff_cls.open = AsyncMock(return_value=gt)
        existing_store = MagicMock()

        obj = await AsyncGeoTIFF.open(
            "s3://bucket/path/to/key.tif", store=existing_store
        )

        mock_geotiff_cls.open.assert_awaited_once_with(
            "path/to/key.tif", store=existing_store, prefetch=32768
        )
        assert obj.uri == "s3://bucket/path/to/key.tif"

    @pytest.mark.asyncio
    async def test_open_multi_uri_cross_bucket_raises(self):
        """Passing URIs from different buckets without an explicit store should raise."""
        with pytest.raises(ValueError, match="same bucket/host"):
            await rastera.open([
                "s3://bucket-a/file1.tif",
                "s3://bucket-b/file2.tif",
            ])


# ── read() ───────────────────────────────────────────────────────────────


class TestRead:
    @pytest.mark.asyncio
    async def test_read_bbox_and_window_raises(self):
        gt = make_mock_geotiff()
        obj = AsyncGeoTIFF("s3://b/k.tif", gt)
        with pytest.raises(ValueError, match="Cannot specify both"):
            await obj.read(bbox=(0, 0, 100, 100), bbox_crs=32632, window=Window(col_off=0, row_off=0, width=10, height=10))

    @pytest.mark.asyncio
    async def test_read_full_image(self):
        """Read with no bbox/window should use full image bounds."""
        gt = make_mock_geotiff(width=16, height=16, scale=1.0, count=1, tile_width=16, tile_height=16)
        obj = AsyncGeoTIFF("s3://b/k.tif", gt)

        result = _make_read_result((1, 16, 16), dtype=np.uint16)
        gt.read = AsyncMock(return_value=result)

        data, profile = await obj.read()
        assert data.shape == (1, 16, 16)
        assert data.dtype == np.uint16
        assert profile.width == 16
        assert profile.height == 16
        np.testing.assert_array_equal(data, 1)

    @pytest.mark.asyncio
    async def test_read_with_window(self):
        gt = make_mock_geotiff(width=32, height=32, scale=1.0, count=2, tile_width=32, tile_height=32)
        obj = AsyncGeoTIFF("s3://b/k.tif", gt)

        result = _make_read_result((2, 16, 16), dtype=np.uint16, fill=42)
        gt.read = AsyncMock(return_value=result)

        window = Window(col_off=4, row_off=4, width=16, height=16)
        data, profile = await obj.read(window=window)
        assert data.shape == (2, 16, 16)
        np.testing.assert_array_equal(data, 42)

    @pytest.mark.asyncio
    async def test_read_band_indices(self):
        gt = make_mock_geotiff(width=16, height=16, scale=1.0, count=3, tile_width=16, tile_height=16)
        obj = AsyncGeoTIFF("s3://b/k.tif", gt)

        arr = np.arange(3 * 16 * 16, dtype=np.uint16).reshape(3, 16, 16)
        result = MagicMock()
        result.data = arr
        gt.read = AsyncMock(return_value=result)

        data, profile = await obj.read(band_indices=[1, 3])
        assert data.shape == (2, 16, 16)
        # band_indices [1, 3] → 0-based [0, 2]
        np.testing.assert_array_equal(data[0], arr[0])
        np.testing.assert_array_equal(data[1], arr[2])

    @pytest.mark.asyncio
    async def test_read_band_index_zero_raises(self):
        gt = make_mock_geotiff(width=16, height=16, scale=1.0, count=3)
        obj = AsyncGeoTIFF("s3://b/k.tif", gt)

        with pytest.raises(ValueError, match="1-based"):
            await obj.read(band_indices=[0])
