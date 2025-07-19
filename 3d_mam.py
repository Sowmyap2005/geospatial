import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from shapely.geometry import Polygon
from scipy.interpolate import RegularGridInterpolator
from pyproj import Transformer
import os

# === TILE SETUP ===
tile_files = {}
for lat_val in range(11, 19):  # 11 to 18
    for lon_val in range(75, 79):  # 75 to 78
        tile_files[(lat_val, lon_val)] = f"N{lat_val:02d}E{lon_val:03d}.hgt"

def read_hgt(file_path):
    with open(file_path, 'rb') as f:
        return np.fromfile(f, np.dtype('>i2'), 1201 * 1201).reshape((1201, 1201))

# === READ ELEVATION DATA ===
tiles = {}
for (lat_val, lon_val), filename in tile_files.items():
    try:
        if os.path.exists(filename):
            tiles[(lat_val, lon_val)] = read_hgt(filename)
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        print(f"Missing file: {filename}, using synthetic flat 300m")
        tiles[(lat_val, lon_val)] = np.full((1201, 1201), 300, dtype=np.int16)

# === MERGE ALL TILES ===
rows = []
for lat_val in range(18, 10, -1):  # from 18 down to 11
    row = []
    for lon_val in range(75, 79):  # from 75 to 78
        row.append(tiles[(lat_val, lon_val)])
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

# === DEFINE ANOMALY POLYGONS (UTM - EPSG:32643) ===
polygons_utm = [
    Polygon([(772131.030566, 1577306.158207), (772131.030566, 1581306.158207),
             (768131.030566, 1581306.158207), (768131.030566, 1577306.158207),
             (772131.030566, 1577306.158207)]),

    Polygon([(780131.030566, 1577306.158207), (780131.030566, 1581306.158207),
             (776131.030566, 1581306.158207), (776131.030566, 1577306.158207),
             (780131.030566, 1577306.158207)]),

    Polygon([(780131.030566, 1593306.158207), (780131.030566, 1597306.158207),
             (776131.030566, 1597306.158207), (776131.030566, 1593306.158207),
             (780131.030566, 1593306.158207)])
]

# === TRANSFORM UTM TO LAT/LON (EPSG:32643 ➝ EPSG:4326) ===
transformer = Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True)
anomaly_lons = []
anomaly_lats = []

for poly in polygons_utm:
    centroid = poly.centroid
    lon_c, lat_c = transformer.transform(centroid.x, centroid.y)
    if 75 <= lon_c <= 78 and 11 <= lat_c <= 18:
        anomaly_lons.append(lon_c)
        anomaly_lats.append(lat_c)

# === INTERPOLATE ELEVATION ===
interp = RegularGridInterpolator((lat[::-1], lon), elevation_data, bounds_error=False, fill_value=np.nan)
anomaly_points = np.array([[lat, lon] for lat, lon in zip(anomaly_lats, anomaly_lons)])
anomaly_base_elev = interp(anomaly_points)

# === HANDLE NaNs USING NEIGHBOR ESTIMATES ===
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

# === 3D PLOT ===
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Terrain surface
surf = ax.plot_surface(
    lon_grid, lat_grid, elevation_data,
    cmap='terrain', linewidth=0, antialiased=False, alpha=1.0
)

# Plot red cylinders for anomaly centroids
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

# Aesthetics
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_zlabel("Elevation (m)")
ax.set_title("3D Terrain with Anomaly Columns")

ax.set_zlim(0, 4000)
ax.view_init(elev=55, azim=135)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=12, label='Elevation (m)')
plt.tight_layout()
plt.show()
