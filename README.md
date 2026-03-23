# rastera

**Async rasterio for COGs**, build on [async-tiff](https://github.com/developmentseed/async-tiff), no GDAL.

- `read` and multi-file, cross-crs `merge` with `target_crs`, `target_resolution`, `bbox`, `window`
- Parallel everywhere: concurrent file opens, tile downloads, and cross-file merges for maximum throughput
- Built on [async-tiff](https://github.com/developmentseed/async-tiff) handling async tile fetching, batched range requests, and Rust-native decompression
- In-memory COG header cache (TODO: Between sessions?)

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

## Concurrency and tile batching

When merging, rastera reads multiple COGs concurrently (`max_concurrency`,
default 6). Each COG fires all its tile HTTP range requests in parallel via
`async_tiff.fetch_tiles()`. High-resolution COGs can require 100-200+ tiles,
so without batching, `max_concurrency` COGs each firing ~150 requests = ~900
simultaneous connections, which overwhelms reqwest's connection pool (client-side
limit, not S3 throttling).

To prevent this, tile fetches are batched (`batch_size`, default 32) per COG.
Peak concurrent requests = `max_concurrency × batch_size` (e.g. 6 × 32 = 192).
Concurrent COGs still help throughput, because while one COG awaits a batch response,
others fetch in parallel.

Tuning:
```python
import rastera

# Adjust tile batch size (default 32) — higher = more concurrent HTTP requests
# per COG, e.g. on aws, lower = safer but potentially slower for single-COG reads
rastera.set_tile_fetch_batch_size(64)
```

## TODO maybe

- Bilinear / cubic / lanczos resampling (currently nearest-neighbor only)
- Persistent COG header cache across sessions (e.g. SQLite/diskcache)
- Basic raster stats (min, max, mean, histogram)