import xarray as xr
import matplotlib.pyplot as plt

input_file = 'data/HWSD_1247/data/AWT_SOC.nc4'

ds = xr.open_dataset(input_file)


print("Variables in the dataset:")
print(ds.data_vars)

variable = list(ds.data_vars.keys())[0]
data = ds[variable]

print(f"Data dimensions: {data.dims}")
print(f"Data coordinates: {data.coords}")


if len(data.dims) == 2:
    plt.figure(figsize=(10, 6))
    data.plot()
    plt.title(f'{variable} - 2D Plot')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()


elif len(data.dims) == 3:

    data_2d = data.isel(time=0)
    plt.figure(figsize=(10, 6))
    data_2d.plot()
    plt.title(f'{variable} - Time Step 0')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

elif len(data.dims) == 4:

    data_2d = data.isel(time=0, level=0)
    plt.figure(figsize=(10, 6))
    data_2d.plot()
    plt.title(f'{variable} - Time Step 0, Level 0')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

else:
    print("Unsupported data dimensions for plotting.")
