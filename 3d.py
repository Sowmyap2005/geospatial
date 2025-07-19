import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from shapely.geometry import shape
from scipy.interpolate import RegularGridInterpolator
import os

# === TILE SETUP ===
tile_files = {}
for lat in range(11, 19):
    for lon in range(75, 79):
        tile_files[(lat, lon)] = f"N{lat:02d}E{lon:03d}.hgt"

geojson_file = "new_predictions.geojson"

def read_hgt(file_path):
    with open(file_path, 'rb') as f:
        return np.fromfile(f, np.dtype('>i2'), 1201 * 1201).reshape((1201, 1201))

# === READ ELEVATION DATA ===
tiles = {}
for (lat, lon), filename in tile_files.items():
    try:
        if os.path.exists(filename):
            tiles[(lat, lon)] = read_hgt(filename)
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        print(f"Missing file: {filename}, using synthetic flat 300m")
        tiles[(lat, lon)] = np.full((1201, 1201), 300, dtype=np.int16)  # synthetic flat terrain

# === MERGE ALL ===
rows = []
for lat in range(18, 10, -1):  # from 18 down to 11
    row = []
    for lon in range(75, 79):  # from 75 to 78
        row.append(tiles[(lat, lon)])
    rows.append(np.hstack(row))
elevation_data = np.vstack(rows)

# === FIX EDGE IF NEEDED ===
if np.isnan(elevation_data[-1201:, :]).all():
    print("Bottom edge (lat=11) is NaN, filling with lat=12 values")
    elevation_data[-1201:, :] = elevation_data[-(2 * 1201):-1201, :]

# === GRID SETUP ===
lat = np.linspace(18, 11, elevation_data.shape[0])
lon = np.linspace(75, 78, elevation_data.shape[1])
lon_grid, lat_grid = np.meshgrid(lon, lat)

# === LOAD ANOMALIES ===
with open(geojson_file) as f:
    geojson_data = json.load(f)

anomaly_lons = []
anomaly_lats = []

for feature in geojson_data["features"]:
    if feature.get("properties", {}).get("predicted_anomaly") == 1:
        centroid = shape(feature["geometry"]).centroid
        if 75 <= centroid.x <= 78 and 11 <= centroid.y <= 18:
            anomaly_lons.append(centroid.x)
            anomaly_lats.append(centroid.y)

# === INTERPOLATE ELEVATION ===
interp = RegularGridInterpolator((lat[::-1], lon), elevation_data, bounds_error=False, fill_value=np.nan)
anomaly_points = np.array([[lat, lon] for lat, lon in zip(anomaly_lats, anomaly_lons)])
anomaly_base_elev = interp(anomaly_points)

# === FIX NaNs USING NEIGHBOR ESTIMATES ===
for i, (lat_val, lon_val, elev) in enumerate(zip(anomaly_lats, anomaly_lons, anomaly_base_elev)):
    if np.isnan(elev):
        neighbors = [
            [lat_val + 0.01, lon_val],
            [lat_val - 0.01, lon_val],
            [lat_val, lon_val + 0.01],
            [lat_val, lon_val - 0.01]
        ]
        neighbor_elevs = []
        for pt in neighbors:
            try:
                e = interp(pt)
                if not np.isnan(e): neighbor_elevs.append(e)
            except:
                pass
        if neighbor_elevs:
            anomaly_base_elev[i] = np.mean(neighbor_elevs)
        else:
            print(f"Could not estimate elevation for anomaly at ({lat_val:.4f}, {lon_val:.4f})")

# === FILTER VALID ELEVATIONS ===
valid = ~np.isnan(anomaly_base_elev)
anomaly_lats = np.array(anomaly_lats)[valid]
anomaly_lons = np.array(anomaly_lons)[valid]
anomaly_base_elev = anomaly_base_elev[valid]

# === 3D PLOTTING ===
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Terrain surface
surf = ax.plot_surface(
    lon_grid, lat_grid, elevation_data,
    cmap='terrain', linewidth=0, antialiased=False, alpha=1.0
)

# Plot red cylinders for anomalies
cylinder_height = 3000
cylinder_radius = 0.05

for lon_val, lat_val, base_z in zip(anomaly_lons, anomaly_lats, anomaly_base_elev):
    ax.bar3d(
        x=lon_val - cylinder_radius / 2,
        y=lat_val - cylinder_radius / 2,
        z=base_z,
        dx=cylinder_radius,
        dy=cylinder_radius,
        dz=cylinder_height,
        color='red',
        alpha=1.0,
        zsort='max'
    )

# Labels and aesthetics
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_zlabel("Elevation (m)")
ax.set_title("3D Terrain with Anomaly Columns")

ax.set_zlim(0, 4000)
ax.view_init(elev=55, azim=135)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=12, label='Elevation (m)')
plt.tight_layout()
plt.show()
