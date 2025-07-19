import geopandas as gpd
import pandas as pd
import numpy as np
import folium
import joblib
import matplotlib.pyplot as plt
from folium.plugins import MarkerCluster

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay,
    precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap
import os
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)


# === Step 1: Load and preprocess data ===
grid = gpd.read_file("preprocessed_grid_with_features_check.geojson")
grid['commodity'] = grid['commodity'].fillna('None')
grid['is_anomalous'] = grid['commodity'].apply(lambda x: 0 if str(x).strip().lower() == 'none' else 1)

# === Step 2: Feature preparation ===
feature_cols = [col for col in grid.columns if col not in ['geometry', 'grid_index', 'commodity', 'is_anomalous']]
X = grid[feature_cols].fillna(0)
y = grid['is_anomalous']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=SEED

)

# === Step 3: SMOTE Oversampling ===
smote = SMOTE(random_state=SEED)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# === Step 4: XGBoost Tuning ===
param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [4, 6],
    'learning_rate': [0.01, 0.05],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
    'gamma': [0, 1],
    'reg_alpha': [0.1],
    'reg_lambda': [1.5]
}

xgb_base = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=SEED,
    n_jobs=-1
)

search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_grid,
    scoring='f1',
    n_iter=10,
    cv=3,
    n_jobs=-1,
    verbose=1
)

search.fit(X_resampled, y_resampled)
xgb_best = search.best_estimator_

# === Step 5: Stacked Model Training ===
stacked_model = StackingClassifier(
    estimators=[
        ('xgb', xgb_best),
        ('rf', RandomForestClassifier(n_estimators=150, random_state=SEED
))
    ],
    final_estimator=LogisticRegression(),
    cv=3,
    n_jobs=-1
)

stacked_model.fit(X_resampled, y_resampled)

# === Step 6: Evaluation ===
y_probs = stacked_model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
f1 = 2 * precision * recall / (precision + recall + 1e-10)


best_idx = np.argmax(f1)

best_thresh = thresholds[best_idx]

print(f"\n🔧 Best threshold for F1: {best_thresh:.2f}")



y_pred_opt = (y_probs >= best_thresh).astype(int)

print("=== Classification Report (Optimized Threshold) ===")
print(classification_report(y_test, y_pred_opt, digits=4))
print(f"Accuracy: {accuracy_score(y_test, y_pred_opt):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_opt):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_opt):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred_opt):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_probs):.4f}")

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_opt)
plt.title("Confusion Matrix (Optimized Threshold)")
plt.show()

# === Step 7: SHAP feature importance (with full feature names) ===
X_train_df = pd.DataFrame(X_train, columns=feature_cols)
explainer = shap.Explainer(xgb_best, X_train_df)
shap_values = explainer(X_train_df)

plt.figure(figsize=(12, 6))  # Wide figure to avoid cropping
shap.plots.bar(shap_values, show=False)
plt.tight_layout()
plt.savefig("shap_feature_importance.png", dpi=300, bbox_inches='tight')
plt.show()

# === Step 8: Predict on Full Dataset ===
X_full_scaled = scaler.transform(X)
grid['predicted_anomaly'] = (stacked_model.predict_proba(X_full_scaled)[:, 1] >= best_thresh).astype(int)
grid['predicted_label'] = np.where(grid['predicted_anomaly'] == 1, grid['commodity'], 'No Minerals')

grid.to_file("predicted_anomalies_xgb.geojson", driver="GeoJSON")
grid[grid['predicted_anomaly'] == 1].to_file("predicted_anomalies_only.geojson", driver="GeoJSON")
grid[grid['predicted_anomaly'] == 0].to_file("predicted_non_anomalies.geojson", driver="GeoJSON")

joblib.dump(stacked_model, "stacked_model.pkl")
joblib.dump(scaler, "stacked_scaler.pkl")

def visualize_mineral_predictions_simple(grid, filename="predicted_minerals_map_simple.html"):
    grid = grid.to_crs(epsg=32643)  # Ensure projected CRS before centroid computation
    grid['centroid'] = grid.geometry.centroid
    grid = grid.set_geometry('centroid').to_crs(epsg=4326)
    grid['centroid_x'] = grid.geometry.x
    grid['centroid_y'] = grid.geometry.y
    grid.set_geometry('geometry', inplace=True)  # Reset to original geometry

    m = folium.Map(location=[grid['centroid_y'].mean(), grid['centroid_x'].mean()], zoom_start=7)
    marker_cluster = MarkerCluster().add_to(m)

    for _, row in grid.iterrows():
        if pd.notnull(row['predicted_label']) and row['predicted_label'] != 'No Minerals':
            folium.Marker(
                location=[row['centroid_y'], row['centroid_x']],
                popup=f"Cell ID: {int(row.name)}<br>Predicted: {row['predicted_label']}<br>True: {row.get('commodity', 'N/A')}",
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(marker_cluster)

    m.save(filename)
    print(f"✅ Mineral prediction map saved: {filename}")



# === Call Visualization ===
visualize_mineral_predictions_simple(grid)
