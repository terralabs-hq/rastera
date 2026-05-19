"""Unit tests for ``VRTProcessedDataset`` (LUT-based DISPLAY VRTs)."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from affine import Affine
from async_geotiff import RasterArray

from rastera.reader import AsyncGeoTIFF
from rastera.vrt import (
    _LUT_SIZE,
    _compile_lut,
    _open_vrt,
    _parse_vrt_xml,
    _VRTProcessedDataset,
    _VRTProcessedSpec,
)
from tests.conftest import make_mock_geotiff

# ── fixtures ────────────────────────────────────────────────────────────────


def _short_lut(scale: float = 1.0) -> str:
    """``input:output`` control points; output range 0..255 by ``scale``."""
    return ",".join(f"{i * 100.0}:{int(i * 25.5 * scale)}" for i in range(11))


def _processed_vrt(
    *,
    n_bands: int = 2,
    input_path: str = "tile.xml",
    relative: bool = True,
    src_nodata: int = 0,
    dst_nodata: int = 0,
    extra_args: str = "",
) -> bytes:
    rel = "1" if relative else "0"
    luts = "\n      ".join(
        f'<Argument name="lut_{i + 1}">{_short_lut(scale=1.0 + i * 0.1)}</Argument>'
        for i in range(n_bands)
    )
    band_decls = "\n  ".join(
        f'<VRTRasterBand dataType="Byte" band="{i + 1}" subClass="VRTProcessedRasterBand">'
        f"<Description>B{i + 1}</Description></VRTRasterBand>"
        for i in range(n_bands)
    )
    return f"""<VRTDataset subClass="VRTProcessedDataset">
  <Input>
    <SourceFilename relativeToVRT="{rel}">{input_path}</SourceFilename>
  </Input>
  <OutputBands dataType="Byte" count="FROM_LAST_STEP" />
  <ProcessingSteps>
    <Step name="ReflectanceToDisplay">
      <Algorithm>LUT</Algorithm>
      {luts}
      <Argument name="src_nodata">{src_nodata}</Argument>
      <Argument name="dst_nodata">{dst_nodata}</Argument>
      {extra_args}
    </Step>
  </ProcessingSteps>
  {band_decls}
</VRTDataset>""".encode()


def _src_array(
    shape: tuple[int, int, int], *, dtype: Any = np.uint16
) -> RasterArray:
    data = np.zeros(shape, dtype=dtype)
    # Stripe distinct values across bands so we can verify per-band LUT routing.
    for i in range(shape[0]):
        data[i] = (i + 1) * 100
    geotiff = MagicMock()
    geotiff.nodata = 0
    return RasterArray(
        data=data,
        mask=None,
        width=shape[2],
        height=shape[1],
        count=shape[0],
        transform=Affine(1, 0, 0, 0, -1, shape[1]),
        _alpha_band_idx=None,
        _geotiff=geotiff,
    )


# ── LUT compiler ────────────────────────────────────────────────────────────


class TestCompileLut:
    def test_endpoints_and_interpolation(self):
        lut = _compile_lut("0.0:0,100.0:50,200.0:255", src_nodata=0, dst_nodata=0)
        assert lut.dtype == np.uint8
        assert lut.shape == (_LUT_SIZE,)
        assert lut[0] == 0
        assert lut[100] == 50
        assert lut[200] == 255
        # Linear midpoints between control points
        assert lut[50] == 25  # halfway between 0 and 50
        # Beyond the table: clamp to the last output value.
        assert lut[65535] == 255
        # Below the table is impossible (uint indexing), but the first
        # control point at 0 already pins lut[0] = 0.

    def test_clamping_at_edges(self):
        """Inputs outside [first, last] clamp, not extrapolate."""
        lut = _compile_lut("10.0:5,20.0:200", src_nodata=0, dst_nodata=0)
        # Below the first x: clamped to y0 (then overridden for nodata=0)
        assert lut[5] == 5
        # Above the last x: clamped to y_last
        assert lut[1000] == 200

    def test_src_nodata_forced_to_dst_nodata(self):
        """A LUT that maps src_nodata to something nonzero must still
        emit dst_nodata for src_nodata, so the nodata invariant survives
        odd control-point tables."""
        lut = _compile_lut("0.0:42,100.0:200", src_nodata=0, dst_nodata=0)
        assert lut[0] == 0
        # Non-nodata values are unaffected by the override.
        assert lut[100] == 200

    def test_rejects_decreasing_x(self):
        with pytest.raises(ValueError, match="non-decreasing"):
            _compile_lut("100.0:1,50.0:2", src_nodata=0, dst_nodata=0)

    def test_rejects_malformed_pair(self):
        with pytest.raises(ValueError, match="malformed"):
            _compile_lut("100.0,200:5", src_nodata=0, dst_nodata=0)

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="empty"):
            _compile_lut("", src_nodata=0, dst_nodata=0)


# ── XML parser ──────────────────────────────────────────────────────────────


class TestParseProcessedVRT:
    def test_parses_minimal(self):
        spec = _parse_vrt_xml(_processed_vrt(n_bands=2), "s3://b/x.vrt")
        assert isinstance(spec, _VRTProcessedSpec)
        assert spec.output_count == 2
        assert spec.luts.shape == (2, _LUT_SIZE)
        assert spec.luts.dtype == np.uint8
        assert spec.src_nodata == 0
        assert spec.dst_nodata == 0
        # relativeToVRT="1" is resolved against the VRT's parent.
        assert spec.input_uri == "s3://b/tile.xml"

    def test_absolute_input_uri_passes_through(self):
        spec = _parse_vrt_xml(
            _processed_vrt(input_path="/vsis3/other/tile.xml", relative=False),
            "s3://b/x.vrt",
        )
        assert isinstance(spec, _VRTProcessedSpec)
        assert spec.input_uri == "s3://other/tile.xml"

    def test_per_band_luts_are_distinct(self):
        """Each ``lut_N`` is compiled into its own row, in band order."""
        spec = _parse_vrt_xml(_processed_vrt(n_bands=3), "s3://b/x.vrt")
        assert isinstance(spec, _VRTProcessedSpec)
        # The fixture uses different ``scale`` per band, so the LUTs differ.
        assert not np.array_equal(spec.luts[0], spec.luts[1])
        assert not np.array_equal(spec.luts[1], spec.luts[2])

    def test_rejects_missing_input(self):
        xml = _processed_vrt().replace(
            b"<Input>\n    <SourceFilename"
            b' relativeToVRT="1">tile.xml</SourceFilename>\n  </Input>',
            b"",
        )
        with pytest.raises(ValueError, match="<Input>"):
            _parse_vrt_xml(xml, "s3://b/x.vrt")

    def test_rejects_unknown_subclass(self):
        xml = b'<VRTDataset subClass="VRTSomethingElse"></VRTDataset>'
        with pytest.raises(NotImplementedError, match="subClass="):
            _parse_vrt_xml(xml, "s3://b/x.vrt")

    def test_rejects_multi_step(self):
        extra_step = (
            "<Step><Algorithm>LUT</Algorithm>"
            '<Argument name="lut_1">0.0:0,1.0:1</Argument></Step>'
        )
        xml = _processed_vrt(n_bands=1).replace(
            b"</ProcessingSteps>", extra_step.encode() + b"</ProcessingSteps>"
        )
        with pytest.raises(NotImplementedError, match="<Step>"):
            _parse_vrt_xml(xml, "s3://b/x.vrt")

    def test_rejects_non_lut_algorithm(self):
        xml = _processed_vrt().replace(
            b"<Algorithm>LUT</Algorithm>",
            b"<Algorithm>BandAffineCombination</Algorithm>",
        )
        with pytest.raises(NotImplementedError, match="BandAffineCombination"):
            _parse_vrt_xml(xml, "s3://b/x.vrt")

    def test_rejects_non_byte_output(self):
        xml = _processed_vrt().replace(
            b'<VRTRasterBand dataType="Byte" band="1"',
            b'<VRTRasterBand dataType="UInt16" band="1"',
        )
        with pytest.raises(NotImplementedError, match="UInt16"):
            _parse_vrt_xml(xml, "s3://b/x.vrt")

    def test_rejects_dst_nodata_out_of_byte_range(self):
        xml = _processed_vrt(n_bands=1, dst_nodata=300)
        with pytest.raises(ValueError, match="dst_nodata=300"):
            _parse_vrt_xml(xml, "s3://b/x.vrt")

    def test_rejects_missing_lut_for_band(self):
        # Strip the second band's LUT.
        xml = _processed_vrt(n_bands=2).replace(
            b'<Argument name="lut_2">', b'<Argument name="_skip_">'
        )
        with pytest.raises(ValueError, match="lut_2"):
            _parse_vrt_xml(xml, "s3://b/x.vrt")


# ── _VRTProcessedDataset reads ──────────────────────────────────────────────


def _make_processed_ds(
    *, n_bands: int = 2, src_dtype: Any = np.uint16
) -> tuple[_VRTProcessedDataset, AsyncGeoTIFF]:
    """Build a processed VRT wrapping a mock source. The source's
    ``_geotiff`` reports ``src_dtype`` and the natural band count."""
    spec = _parse_vrt_xml(_processed_vrt(n_bands=n_bands), "s3://b/x.vrt")
    assert isinstance(spec, _VRTProcessedSpec)
    src_gt = make_mock_geotiff(count=n_bands, dtype=np.dtype(src_dtype))
    source = AsyncGeoTIFF("s3://b/tile.xml", src_gt)
    return _VRTProcessedDataset("s3://b/x.vrt", spec, source), source


class TestProcessedRead:
    def test_count_reflects_output_band_count(self):
        ds, _ = _make_processed_ds(n_bands=4)
        assert ds.count == 4

    @pytest.mark.asyncio
    async def test_read_applies_per_band_lut(self):
        ds, source = _make_processed_ds(n_bands=2)
        source.read = AsyncMock(return_value=_src_array((2, 3, 3)))

        arr = await ds.read()
        assert source.read.await_count == 1
        # All output bands → forward 1-based [1, 2]
        assert source.read.await_args.kwargs["band_indices"] == [1, 2]

        data: np.ndarray[Any, Any] = arr.data  # type: ignore[reportUnknownMemberType]
        assert data.dtype == np.uint8
        assert data.shape == (2, 3, 3)
        # Band 1 source value 100, lut_1[100] precomputed:
        expected_0 = ds._spec.luts[0][100]
        expected_1 = ds._spec.luts[1][200]
        np.testing.assert_array_equal(data[0], expected_0)
        np.testing.assert_array_equal(data[1], expected_1)

    @pytest.mark.asyncio
    async def test_read_native_subset_bands(self):
        """_read_native is the internal path used by merge; 0-based indexing,
        per-band LUT selection must follow the requested bands."""
        ds, source = _make_processed_ds(n_bands=3)
        source._read_native = AsyncMock(return_value=_src_array((2, 2, 2)))

        arr = await ds._read_native(band_indices=[2, 0])
        assert source._read_native.await_args.kwargs["band_indices"] == [2, 0]

        data: np.ndarray[Any, Any] = arr.data  # type: ignore[reportUnknownMemberType]
        assert data.shape == (2, 2, 2)
        # First requested band is output band 2 → source array row 0 has fill 100,
        # apply lut_3 (index 2) to it.
        np.testing.assert_array_equal(data[0], ds._spec.luts[2][100])
        # Second requested band is output band 0 → source array row 1 fill 200,
        # apply lut_1 (index 0) to it.
        np.testing.assert_array_equal(data[1], ds._spec.luts[0][200])

    @pytest.mark.asyncio
    async def test_use_overviews_rejected(self):
        ds, _ = _make_processed_ds()
        with pytest.raises(NotImplementedError, match="use_overviews"):
            await ds.read(use_overviews=True)

    @pytest.mark.asyncio
    async def test_read_native_overview_rejected(self):
        ds, _ = _make_processed_ds()
        with pytest.raises(NotImplementedError, match="overview"):
            await ds._read_native(overview=MagicMock())

    @pytest.mark.asyncio
    async def test_rejects_float_source(self):
        ds, source = _make_processed_ds(n_bands=1, src_dtype=np.float32)
        source._read_native = AsyncMock(
            return_value=_src_array((1, 2, 2), dtype=np.float32)
        )
        with pytest.raises(NotImplementedError, match="not integer"):
            await ds._read_native()

    @pytest.mark.asyncio
    async def test_rejects_source_out_of_range(self):
        """A source value outside the LUT domain is a real error — we
        won't silently saturate."""
        ds, source = _make_processed_ds(n_bands=1)
        # int32 instead of uint16 so we can push past _LUT_SIZE.
        out_of_range = _src_array((1, 2, 2), dtype=np.int32)
        out_of_range.data[:] = _LUT_SIZE + 1
        source._read_native = AsyncMock(return_value=out_of_range)
        with pytest.raises(ValueError, match="outside"):
            await ds._read_native()


# ── _open_vrt end-to-end ────────────────────────────────────────────────────


class TestOpenProcessedVRT:
    @pytest.mark.asyncio
    async def test_opens_input_via_async_geotiff_open(self):
        """``<Input>`` resolution flows through ``AsyncGeoTIFF.open`` so the
        normal ``.xml`` → DIMAP auto-routing applies for free."""
        src_gt = make_mock_geotiff(count=2, dtype=np.dtype("uint16"))
        source = AsyncGeoTIFF("s3://b/tile.xml", src_gt)

        with (
            patch(
                "rastera.vrt._fetch_descriptor_bytes",
                new=AsyncMock(return_value=_processed_vrt(n_bands=2)),
            ),
            patch.object(
                AsyncGeoTIFF, "open", new=AsyncMock(return_value=source)
            ) as mock_open,
        ):
            ds = await _open_vrt("s3://b/x.vrt")

        assert isinstance(ds, _VRTProcessedDataset)
        assert ds._source is source
        assert ds.count == 2
        # The single Input is opened — once, against the resolved URI.
        assert mock_open.await_count == 1
        assert mock_open.await_args.args[0] == "s3://b/tile.xml"

    @pytest.mark.asyncio
    async def test_forwards_meta_overrides_to_source(self):
        """``meta_overrides`` must reach the source open so the wrapper's
        CRS and the source agree."""
        src_gt = make_mock_geotiff(count=2, dtype=np.dtype("uint16"))

        async def fake_open(uri: str, **kwargs: Any) -> AsyncGeoTIFF:
            return AsyncGeoTIFF(
                uri, src_gt, meta_overrides=kwargs.get("meta_overrides")
            )

        with (
            patch(
                "rastera.vrt._fetch_descriptor_bytes",
                new=AsyncMock(return_value=_processed_vrt(n_bands=2)),
            ),
            patch.object(AsyncGeoTIFF, "open", side_effect=fake_open) as mock_open,
        ):
            ds = await _open_vrt(
                "s3://b/x.vrt", meta_overrides={"crs": 3006}
            )

        assert mock_open.await_args.kwargs["meta_overrides"] == {"crs": 3006}
        assert ds._crs_epsg == 3006
        assert ds._source._crs_epsg == 3006
