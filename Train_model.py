import kagglehub
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle

# ---------------- LOAD DATA ----------------
path = kagglehub.dataset_download("hanaksoy/customer-purchasing-behaviors")
data = pd.read_csv(os.path.join(path, 'Customer Purchasing Behaviors.csv'))

print(data.columns)

# ---------------- CLEAN DATA ----------------
data = data.select_dtypes(include=['int64', 'float64'])

# ---------------- SCALE ----------------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# ---------------- MODEL ----------------
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(scaled_data)

# ---------------- SAVE ----------------
pickle.dump(kmeans, open("kmeans_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("✅ Model & scaler saved successfully")