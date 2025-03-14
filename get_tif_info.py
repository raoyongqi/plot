import rasterio
from rasterio.transform import rowcol

lon, lat = 104.87, 31.79

with rasterio.open('data/HWSD_1247/tif/AWC_CLASS.tif') as src:
    crs = src.crs
    print(f"CRS: {crs}")

    transform = src.transform

    row, col = rowcol(transform, lon, lat)

    print(f"Row: {row}, Column: {col}")

    band_data = src.read(1)

    value = band_data[row, col]

    print(f"Value at ({lon}, {lat}): {value}")
