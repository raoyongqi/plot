import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import geopandas as gpd

# Load your GeoJSON data (replace 'your_data.json' with your actual file path)
gdf = gpd.read_file('china.json')

# Create a figure and an axis with a projection
fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

# Define the projection (PlateCarree is common for lat-lon coordinates)
ax = plt.axes(projection=ccrs.PlateCarree())

# Plot your GeoJSON data
gdf.plot(ax=ax, transform=ccrs.PlateCarree(), color='red', marker='o')

# Optionally, add a label for a specific point or feature (adjust coordinates)
ax.text(-74.0060 + 1, 40.7128 + 1, 'New York City', color='blue', transform=ccrs.PlateCarree())

# Show the map without coastlines
plt.show()
