# import geopandas as gpd
# import pandas as pd
# import os
# from shapely.geometry import box
# from geopandas.tools import sjoin

# # ==== CONFIGURATION ====
# data_path = r"C:\CDAC_project\GeoSpatial\preprocess_layerbylayer_9June"  # Make sure this path is correct
# dataset_names = [
#     "dyke.geojson", "fold.geojson", "geochemical_clean.geojson", "merged_dataset.geojson",
#     "merged_lineament_1.geojson", "oriented_structure_line.geojson", "oriented_structure_plane_gdf.geojson",
#     "shear_zone.geojson", "combined_NGPM_gravity.csv",
#     "Physical_properties_rock_samples.csv", "combined_lithology.geojson",
#     "combined_data.geojson", "combined_mines_quarries.geojson"
# ]
# output_file = "preprocessed_grid_with_features.geojson"
# cell_size = 0.1  # in degrees (~10km if using EPSG:4326)

# # ==== STEP 1: Load and Preprocess Each Layer ====
# def load_layer(filename):
#     full_path = os.path.join(data_path, filename)
#     print(f"🔍 Loading: {full_path}")

#     if filename.endswith(".geojson"):
#         gdf = gpd.read_file(full_path)
#         gdf = gdf.to_crs(epsg=4326)
#         gdf = gdf.dropna(subset=["geometry"])
#     elif filename.endswith(".csv"):
#         df = pd.read_csv(full_path)

#         # Normalize column names
#         df.columns = [col.strip().lower() for col in df.columns]

#         if "x" in df.columns and "y" in df.columns:
#             lon_col, lat_col = "x", "y"
#         elif "longitude" in df.columns and "latitude" in df.columns:
#             lon_col, lat_col = "longitude", "latitude"
#         elif "lon" in df.columns and "lat" in df.columns:
#             lon_col, lat_col = "lon", "lat"
#         else:
#             raise ValueError(f"❌ Could not detect coordinate columns in {filename}")

#         df = df.dropna(subset=[lon_col, lat_col])
#         geometry = gpd.points_from_xy(df[lon_col], df[lat_col])
#         gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
#     else:
#         raise ValueError(f"Unsupported file format: {filename}")

#     return gdf


# # ==== STEP 2: Create Bounding Grid ====
# def create_grid(gdfs, cell_size=0.1):
#     bounds = [gdf.total_bounds for gdf in gdfs]
#     xmin = min(b[0] for b in bounds)
#     ymin = min(b[1] for b in bounds)
#     xmax = max(b[2] for b in bounds)
#     ymax = max(b[3] for b in bounds)

#     grid_cells = []
#     x = xmin
#     while x < xmax:
#         y = ymin
#         while y < ymax:
#             grid_cells.append(box(x, y, x + cell_size, y + cell_size))
#             y += cell_size
#         x += cell_size

#     grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs="EPSG:4326")
#     return grid

# # ==== STEP 3: Join Layer to Grid and Count Features ====
# def join_layer_to_grid(grid, layer_gdf, label):
#     joined = gpd.sjoin(grid, layer_gdf, how='left', predicate='intersects')

#     if 'index_left' not in joined.columns:
#         grid[label] = 0
#     else:
#         counts = joined.groupby('index_left').size()
#         grid[label] = grid.index.map(counts).fillna(0).astype(int)

#     return grid

# # ==== STEP 4: Apply Each Dataset Layer by Layer ====
# if __name__ == "__main__":
#     print("🔄 Loading datasets and preparing grid...")
#     all_gdfs = [load_layer(name) for name in dataset_names]
#     grid = create_grid(all_gdfs, cell_size=cell_size)

#     for name, gdf in zip(dataset_names, all_gdfs):
#         label = name.replace(".geojson", "").replace(".csv", "")
#         print(f"🧱 Applying layer: {label}")
#         grid = join_layer_to_grid(grid, gdf, label)

#     print("💾 Saving output...")
#     grid.to_file(output_file, driver='GeoJSON')
#     print(f"✅ Grid with all features saved to: {output_file}")

import geopandas as gpd
import pandas as pd
import os
from shapely.geometry import box
from geopandas.tools import sjoin

# ==== CONFIGURATION ====
#data_path = r"C:\CDAC_project\GeoSpatial\preprocess_9June"  # Your folder path (use raw string or double backslash)
data_path = r"C:\sowmya\CDAC_project\geospatial_new\23JUNE\10June"
dataset_names = [
    "dyke.geojson", "fold.geojson", "geochemical_clean.geojson", "merged_dataset.geojson",
    "merged_lineament_1.geojson", "oriented_structure_line.geojson", "oriented_structure_plane_gdf.geojson",
    "shear_zone.geojson", "combined_NGPM_gravity.csv",
    "Physical_properties_rock_samples.csv", "combined_lithology.geojson", 'combined_data.geojson', 'combined_mines_quarries.geojson'
]
output_file = "preprocessed_grid_with_features_check.geojson"
cell_size = 0.1  # in degrees (~10km if using EPSG:4326)

# ==== HELPER FUNCTION TO LOAD EACH LAYER ====
def load_layer(filename):
    full_path = os.path.join(data_path, filename)
    ext = os.path.splitext(filename)[1].lower()

    if ext in ['.geojson', '.shp']:
        gdf = gpd.read_file(full_path)
        gdf = gdf.to_crs(epsg=4326)
        gdf = gdf.dropna(subset=["geometry"])
    elif ext == '.csv':
        df = pd.read_csv(full_path)

        # Special handling for gravity CSV with 'X' and 'Y' columns for coords
        if 'X' in df.columns and 'Y' in df.columns:
            gdf = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df['X'], df['Y']),
                crs="EPSG:4326"
            )
        # Special handling for Physical_properties_rock_samples.csv with 'Latitude' and 'Longitude'
        elif 'Latitude' in df.columns and 'Longitude' in df.columns:
            gdf = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']),
                crs="EPSG:4326"
            )
        else:
            raise ValueError(f"❌ Could not detect coordinate columns in {filename}")
    else:
        raise ValueError(f"❌ Unsupported file extension: {ext}")

    return gdf

# ==== STEP 1: Load All Datasets ====
print("🔄 Loading datasets and preparing grid...")
all_gdfs = [load_layer(name) for name in dataset_names]

# ==== STEP 2: Create Bounding Grid ====
def create_grid(gdfs, cell_size=0.1):
    bounds = [gdf.total_bounds for gdf in gdfs]
    xmin = min(b[0] for b in bounds)
    ymin = min(b[1] for b in bounds)
    xmax = max(b[2] for b in bounds)
    ymax = max(b[3] for b in bounds)

    grid_cells = []
    x = xmin
    while x < xmax:
        y = ymin
        while y < ymax:
            grid_cells.append(box(x, y, x + cell_size, y + cell_size))
            y += cell_size
        x += cell_size

    grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs="EPSG:4326")
    return grid

grid = create_grid(all_gdfs, cell_size=cell_size)

# Reset index to preserve grid cell indices for joins
grid = grid.reset_index(drop=False).rename(columns={'index': 'grid_index'})

# ==== STEP 3: Apply Each Dataset Layer-by-Layer on Grid ====
def join_and_count(grid, gdf, label):
    joined = sjoin(grid, gdf, how='left', predicate='intersects')
    counts = joined.groupby('grid_index').size()
    grid[label] = grid['grid_index'].map(counts).fillna(0).astype(int)
    return grid

for name, gdf in zip(dataset_names, all_gdfs):
    label = os.path.splitext(name)[0]
    print(f"Processing layer: {label}")
    grid = join_and_count(grid, gdf, label)

# ==== STEP 4: Assign Target Variable 'commodity' from combined_data.geojson ====
# Load combined_data.geojson separately for target
combined_data_path = os.path.join(data_path, 'combined_data.geojson')
combined_gdf = gpd.read_file(combined_data_path).to_crs(epsg=4326)
combined_gdf = combined_gdf.dropna(subset=["geometry"])

# Spatial join grid with combined_data.geojson points
joined_min = sjoin(grid, combined_gdf[['commodity', 'geometry']], how='left', predicate='intersects')

# Group by grid_index and get first commodity per cell (or could do majority voting)
commodity_per_cell = joined_min.groupby('grid_index')['commodity'].first()

# Map commodity to grid
grid['commodity'] = grid['grid_index'].map(commodity_per_cell).fillna('None')

# ==== STEP 5: Save output ====
grid.to_file(output_file, driver='GeoJSON')
print(f"✅ Saved preprocessed grid with features and target to: {output_file}")

#--------------------------------
# import geopandas as gpd
# import pandas as pd
# import os
# from shapely.geometry import box
# from geopandas.tools import sjoin

# # ==== CONFIGURATION ====
# data_path = r"C:\CDAC_project\GeoSpatial\preprocess_layerbylayer_9June"
# dataset_names = [
#     "dyke.geojson", "fold.geojson", "geochemical_clean.geojson", "merged_dataset.geojson",
#     "merged_lineament_1.geojson", "oriented_structure_line.geojson", "oriented_structure_plane_gdf.geojson",
#     "shear_zone.geojson", "combined_NGPM_gravity.csv",
#     "Physical_properties_rock_samples.csv", "combined_lithology.geojson", 'combined_data.geojson', 'combined_mines_quarries.geojson'
# ]
# output_file = "preprocessed_grid_with_features.geojson"
# cell_size = 0.1  # in degrees (~10km)

# # ==== Function to Load Each Layer ====
# def load_layer(filename):
#     full_path = os.path.join(data_path, filename)
#     ext = os.path.splitext(filename)[1].lower()

#     if ext in ['.geojson', '.shp']:
#         gdf = gpd.read_file(full_path).to_crs(epsg=4326)
#     elif ext == '.csv':
#         df = pd.read_csv(full_path)
#         if 'X' in df.columns and 'Y' in df.columns:
#             gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['X'], df['Y']), crs="EPSG:4326")
#         elif 'Latitude' in df.columns and 'Longitude' in df.columns:
#             gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']), crs="EPSG:4326")
#         else:
#             raise ValueError(f"❌ Coordinate columns not found in: {filename}")
#     else:
#         raise ValueError(f"❌ Unsupported file type: {ext}")

#     return gdf.dropna(subset=["geometry"])

# # ==== Create Grid ====
# def create_grid(gdfs, cell_size=0.1):
#     bounds = [gdf.total_bounds for gdf in gdfs]
#     xmin = min(b[0] for b in bounds)
#     ymin = min(b[1] for b in bounds)
#     xmax = max(b[2] for b in bounds)
#     ymax = max(b[3] for b in bounds)

#     grid_cells = []
#     x = xmin
#     while x < xmax:
#         y = ymin
#         while y < ymax:
#             grid_cells.append(box(x, y, x + cell_size, y + cell_size))
#             y += cell_size
#         x += cell_size

#     return gpd.GeoDataFrame({'geometry': grid_cells}, crs="EPSG:4326").reset_index(drop=False).rename(columns={'index': 'grid_index'})

# # ==== Preprocess (One-Hot Encode) and Join ====
# def preprocess_and_join(grid, gdf, label_prefix):
#     if gdf.empty:
#         return grid

#     gdf = gdf.copy()

#     # Identify categorical columns for one-hot encoding,
#     # but skip columns with high cardinality or unwanted patterns
#     categorical_cols = [
#         col for col in gdf.columns
#         if col != 'geometry' and gdf[col].dtype == 'object'
#         and gdf[col].nunique() <= 50  # threshold for cardinality
#         and 'sampleno' not in col.lower()  # skip columns with 'sampleno'
#     ]

#     # One-hot encode filtered categorical columns
#     dummies_list = []
#     for col in categorical_cols:
#         dummies = pd.get_dummies(gdf[col], prefix=f"{label_prefix}_{col}", dtype=int)
#         dummies_list.append(dummies)

#     if dummies_list:
#         dummies_combined = pd.concat(dummies_list, axis=1)
#         gdf = pd.concat([gdf.drop(columns=categorical_cols), dummies_combined], axis=1)
#         gdf = gdf.copy()

#     feature_dict = {}

#     # For numeric columns, aggregate sum per grid cell
#     for col in gdf.columns:
#         if col == 'geometry':
#             continue
#         if not pd.api.types.is_numeric_dtype(gdf[col]):
#             print(f"⚠️ Skipping non-numeric column: {col}")
#             continue

#         try:
#             joined = sjoin(grid, gdf[[col, 'geometry']], how='left', predicate='intersects')
#             counts = joined.groupby('grid_index')[col].sum()
#             feature_dict[col] = grid['grid_index'].map(counts).fillna(0)
#             print(f"✅ Joined {len(joined)} rows for feature '{col}'")
#         except Exception as e:
#             print(f"⚠️ Skipping column '{col}' due to error: {e}")

#     features_df = pd.DataFrame(feature_dict, index=grid.index)

#     grid = grid.drop(columns=features_df.columns, errors='ignore')
#     grid = pd.concat([grid, features_df], axis=1)
#     grid = grid.copy()

#     return grid


# # ==== Main Pipeline ====

# print("🔄 Loading datasets...")
# all_gdfs = [load_layer(name) for name in dataset_names]

# grid = create_grid(all_gdfs, cell_size)
# print(f"🗺️ Created grid with {len(grid)} cells")

# print("⚙️ Preprocessing and joining layers to grid...")
# for name, gdf in zip(dataset_names, all_gdfs):
#     label = os.path.splitext(name)[0]
#     print(f"🔹 Processing: {label} with {len(gdf)} features")
#     grid = preprocess_and_join(grid, gdf, label)
#     print(f"    Grid now has {len(grid.columns)} columns")

# # Add target 'commodity'
# print("🎯 Assigning target label: 'commodity'")
# combined_data_path = os.path.join(data_path, 'combined_data.geojson')
# combined_gdf = gpd.read_file(combined_data_path).to_crs(epsg=4326).dropna(subset=["geometry"])

# joined_min = sjoin(grid, combined_gdf[['commodity', 'geometry']], how='left', predicate='intersects')

# # Debug:
# print(f"    Joined for commodity label: {len(joined_min)} rows")

# commodity_per_cell = joined_min.groupby('grid_index')['commodity'].first()
# commodity_per_cell = commodity_per_cell.reindex(grid['grid_index'], fill_value='None')

# grid['commodity'] = commodity_per_cell

# grid = grid.copy()
# grid.set_crs(epsg=4326, inplace=True)

# print(f"✅ Saving final dataset with {len(grid)} rows and {len(grid.columns)} columns to: {output_file}")
# grid.to_file(output_file, driver='GeoJSON')


# # ==== Main Pipeline ====
# print("🔄 Loading datasets...")
# all_gdfs = [load_layer(name) for name in dataset_names]
# grid = create_grid(all_gdfs, cell_size)

# print("⚙️ Preprocessing and joining layers to grid...")
# for name, gdf in zip(dataset_names, all_gdfs):
#     label = os.path.splitext(name)[0]
#     print(f"🔹 Processing: {label}")
#     grid = preprocess_and_join(grid, gdf, label)

# # ==== Add 'commodity' from combined_data.geojson as Target ====
# print("🎯 Assigning target label: 'commodity'")
# combined_data_path = os.path.join(data_path, 'combined_data.geojson')
# combined_gdf = gpd.read_file(combined_data_path).to_crs(epsg=4326).dropna(subset=["geometry"])

# joined_min = sjoin(grid, combined_gdf[['commodity', 'geometry']], how='left', predicate='intersects')
# commodity_per_cell = joined_min.groupby('grid_index')['commodity'].first()
# grid['commodity'] = grid['grid_index'].map(commodity_per_cell).fillna('None')

# # Defragment final grid
# grid = grid.copy()

# # ==== Save Output ====
# grid.to_file(output_file, driver='GeoJSON')
# print(f"✅ Saved final dataset with features and labels: {output_file}")

# import geopandas as gpd
# import pandas as pd
# import os
# from shapely.geometry import box
# from geopandas.tools import sjoin

# # ==== CONFIGURATION ====
# data_path = r"C:\CDAC_project\GeoSpatial\preprocess_layerbylayer_9June"
# dataset_names = [
#     "dyke.geojson", "fold.geojson", "geochemical_clean.geojson", "merged_dataset.geojson",
#     "merged_lineament_1.geojson", "oriented_structure_line.geojson", "oriented_structure_plane_gdf.geojson",
#     "shear_zone.geojson", "combined_NGPM_gravity.csv",
#     "Physical_properties_rock_samples.csv", "combined_lithology.geojson",
#     "combined_data.geojson", "combined_mines_quarries.geojson"
# ]
# output_file = "preprocessed_grid_with_features.geojson"
# cell_size = 0.1  # in degrees (~10km if using EPSG:4326)

# # ==== HELPER FUNCTION TO LOAD EACH LAYER ====
# def load_layer(filename):
#     full_path = os.path.join(data_path, filename)
#     ext = os.path.splitext(filename)[1].lower()

#     if ext in ['.geojson', '.shp']:
#         gdf = gpd.read_file(full_path).to_crs(epsg=4326)
#         gdf = gdf.dropna(subset=["geometry"])
#     elif ext == '.csv':
#         df = pd.read_csv(full_path)
#         df.columns = [col.strip().lower() for col in df.columns]  # Normalize column names

#         if 'x' in df.columns and 'y' in df.columns:
#             gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['x'], df['y']), crs="EPSG:4326")
#         elif 'longitude' in df.columns and 'latitude' in df.columns:
#             gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], df['latitude']), crs="EPSG:4326")
#         else:
#             raise ValueError(f"❌ Could not detect coordinate columns in {filename}")
#     else:
#         raise ValueError(f"❌ Unsupported file extension: {ext}")

#     return gdf

# # ==== STEP 1: Load All Datasets ====
# print("🔄 Loading datasets and preparing grid...")
# all_gdfs = [load_layer(name) for name in dataset_names]

# # ==== STEP 2: Create Bounding Grid ====
# def create_grid(gdfs, cell_size=0.1):
#     bounds = [gdf.total_bounds for gdf in gdfs]
#     xmin = min(b[0] for b in bounds)
#     ymin = min(b[1] for b in bounds)
#     xmax = max(b[2] for b in bounds)
#     ymax = max(b[3] for b in bounds)

#     grid_cells = []
#     x = xmin
#     while x < xmax:
#         y = ymin
#         while y < ymax:
#             grid_cells.append(box(x, y, x + cell_size, y + cell_size))
#             y += cell_size
#         x += cell_size

#     grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs="EPSG:4326")
#     return grid

# grid = create_grid(all_gdfs, cell_size=cell_size)
# grid = grid.reset_index(drop=False).rename(columns={'index': 'grid_index'})

# # ==== STEP 3: Apply Each Dataset Layer-by-Layer on Grid ====
# def join_and_count(grid, gdf, label):
#     joined = sjoin(grid, gdf, how='left', predicate='intersects')
#     counts = joined.groupby('grid_index').size()
#     grid[label] = grid['grid_index'].map(counts).fillna(0).astype(int)
#     return grid

# for name, gdf in zip(dataset_names, all_gdfs):
#     label = os.path.splitext(name)[0]
#     print(f"Processing layer: {label}")
#     grid = join_and_count(grid, gdf, label)

# # ==== STEP 4: Assign Target Variable 'commodity' and One-Hot Encode ====
# combined_data_path = os.path.join(data_path, 'combined_data.geojson')
# combined_gdf = gpd.read_file(combined_data_path).to_crs(epsg=4326)
# combined_gdf = combined_gdf.dropna(subset=["geometry"])

# # Spatial join grid with combined_data.geojson points
# joined_min = sjoin(grid, combined_gdf[['commodity', 'geometry']], how='left', predicate='intersects')

# # Assign commodities and one-hot encode
# commodity_per_cell = joined_min.groupby('grid_index')['commodity'].first().fillna('None')
# commodity_encoded = pd.get_dummies(commodity_per_cell, prefix="commodity")

# # Merge encoded commodity data into grid
# grid = grid.merge(commodity_encoded, left_on='grid_index', right_index=True, how="left").fillna(0)

# # ==== STEP 5: Save Output ====
# grid.to_file(output_file, driver='GeoJSON')
# print(f"✅ Saved preprocessed grid with features and one-hot encoded commodities to: {output_file}")