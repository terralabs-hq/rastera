# rastera

**Async rasterio for COGs**, built on [async-geotiff](https://github.com/developmentseed/async-geotiff), no GDAL.

- `read` and `merge` (multi-file, cross-crs) with `target_crs`, `target_resolution`, `bbox`, `window`
- Parallel everywhere: concurrent file opens, tile downloads, and optimized cross-file merges for maximum network throughput
- Built on [async-geotiff](https://github.com/developmentseed/async-geotiff) handling GeoTIFF parsing, async tile fetching, request coalescing, and Rust-native decompression

**Note:** Only COGs & tiled GeoTIFFs are supported. Stripped (non-tiled) TIFFs will not work.

### Read a single COG

```python
import rastera

uri = "s3://my-bucket/my-cog.tif"
src = await rastera.open(uri)

# Full image
data, profile = await src.read()

# Spatial subset with reprojection
data, profile = await src.read(
    bbox=(minx, miny, maxx, maxy),
    bbox_crs=32633,
    target_crs=32632,
    target_resolution=20
)
```

### Merge to mosaic

```python
uris = ["s3://bucket/tile_a.tif", "s3://bucket/tile_b.tif", ...]
sources = await rastera.open(uris)  # concurrent opens, shared connection pool

data, profile = await rastera.merge(sources, bbox=bbox, bbox_crs=32633, target_resolution=20)
```

## Concurrency

When reading a COG, async-geotiff fires all tile HTTP range requests in parallel
with request coalescing via obspec. Big, high-resolution COGs can have hundreds of tiles.

When mosaicing via `merge`, multiple COGs are read concurrently (`max_concurrency`,
default 4) so tile fetches for the next COG overlap with the previous COG's
tile decoding processing, keeping the network busy. Tile fetching within each COG is handled
entirely by async-geotiff's internal coalescing.


## TODO maybe

- **Test merge tile throughput under load**: The old manual `tile_batch_size` batching
  (per-COG tile fetch limiting) was removed — async-geotiff's internal request coalescing
  now handles it. If merges with many large COGs cause connection exhaustion, reintroduce
  manual batching via `_geotiff._tiff.fetch_tiles()` in `_gather_and_paste()`.
- Bilinear / cubic / lanczos resampling (currently nearest-neighbor only)
- Extend current per-session wide persistent COG header cache across sessions (e.g. SQLite/diskcache)
- Basic raster stats (min, max, mean, histogram)