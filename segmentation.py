# =====================================================
# Customer Segmentation for E-Commerce Personalization
# Complete End-to-End Pipeline with 9 Advanced Features
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------------------------------
# 1. Load and Inspect Dataset
# -----------------------------------------------------
# Ensure 'Mall_Customers.csv' is in your project directory
data = pd.read_csv("Mall_Customers.csv")
print("Dataset Loaded Successfully! Shape:", data.shape)

# Display options for full clarity
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("\nMissing Values:\n", data.isnull().sum())

# -----------------------------------------------------
# 2. Advanced Feature Engineering (9 Total Features)
# -----------------------------------------------------
np.random.seed(42)

# Engineer custom and simulated e-commerce metrics
data['Income_to_Spending_Ratio'] = (
    data['Annual Income (k$)'] / (data['Spending Score (1-100)'] + 1)
)
data['Total_Orders'] = np.random.randint(1, 20, size=len(data))
data['Average_Order_Value'] = (data['Annual Income (k$)'] * 500) / (data['Total_Orders'] + 1)
data['Discount_Sensitivity'] = np.random.uniform(0.0, 1.0, size=len(data))
data['Cart_Abandonment_Rate'] = np.random.uniform(0.1, 0.9, size=len(data))
data['Tenure_Months'] = np.random.randint(1, 48, size=len(data))

print("\nAdvanced Feature Engineering Completed: 9 Features Engineered.")

# -----------------------------------------------------
# 3. Feature Selection & Scaling (All 9 Features)
# -----------------------------------------------------
X = data[[
    'Age',
    'Annual Income (k$)',
    'Spending Score (1-100)',
    'Income_to_Spending_Ratio',
    'Total_Orders',
    'Average_Order_Value',
    'Discount_Sensitivity',
    'Cart_Abandonment_Rate',
    'Tenure_Months'
]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------------------------
# 4. Model Selection Analysis (Elbow & Silhouette)
# -----------------------------------------------------
# Elbow Method (WCSS)
wcss = []
for i in range(1, 11):
    kmeans_temp = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans_temp.fit(X_scaled)
    wcss.append(kmeans_temp.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(range(1, 11), wcss, marker='o', color='b')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method for Optimal k")
plt.show()

# Silhouette Score Comparison across k
scores = {}
for k in range(2, 8):
    model_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_temp = model_temp.fit_predict(X_scaled)
    scores[k] = silhouette_score(X_scaled, labels_temp)

silhouette_df = pd.DataFrame.from_dict(
    scores, orient='index', columns=['Silhouette Score']
)
print("\nSilhouette Score Comparison:")
print(silhouette_df)

# -----------------------------------------------------
# 5. Apply Optimal K-Means Clustering (k=5)
# -----------------------------------------------------
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

data['Cluster'] = clusters
print("\nClustering Completed with k =", optimal_k)

# Final Silhouette Evaluation
sil_score = silhouette_score(X_scaled, clusters)
print(f"Final Silhouette Score: {sil_score:.4f}")

# -----------------------------------------------------
# 6. Visualizations (2D, 3D, and PCA)
# -----------------------------------------------------
# 2D View: Annual Income vs Spending Score
plt.figure(figsize=(6, 4))
plt.scatter(
    data['Annual Income (k$)'],
    data['Spending Score (1-100)'],
    c=data['Cluster'],
    cmap='viridis',
    edgecolor='k',
    s=50
)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation (2D View)")
plt.show()

# 3D View: Age, Annual Income, Spending Score
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111, projection='3d')
scatter_3d = ax.scatter(
    data['Age'],
    data['Annual Income (k$)'],
    data['Spending Score (1-100)'],
    c=data['Cluster'],
    cmap='viridis',
    s=40
)
ax.set_xlabel("Age")
ax.set_ylabel("Annual Income")
ax.set_zlabel("Spending Score")
ax.set_title("Customer Segmentation (3D View)")
plt.show()

# PCA View (Dimensionality Reduction to 2 components)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(6, 4))
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=data['Cluster'],
    cmap='viridis',
    edgecolor='k',
    s=50
)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Clusters Visualized using PCA")
plt.show()

# -----------------------------------------------------
# 7. Cluster Interpretation & Business Naming
# -----------------------------------------------------
numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
cluster_summary = data[numeric_cols].groupby('Cluster').mean().reset_index()

print("\nCluster Characteristics (Numeric Means):")
print(cluster_summary)

segment_labels = {
    0: "High Value Customers",
    1: "Budget Buyers",
    2: "Impulse Shoppers",
    3: "Potential Loyalists",
    4: "Low Engagement Customers"
}

data['Segment_Name'] = data['Cluster'].map(segment_labels)
print("\nAssigned Segment Names (Preview):")
print(data[['Cluster', 'Segment_Name']].head())

# -----------------------------------------------------
# 8. Hierarchical Clustering Dendrogram Comparison
# -----------------------------------------------------
linked = linkage(X_scaled, method='ward')

plt.figure(figsize=(10, 5))
dendrogram(linked, truncate_mode='lastp', p=30)
plt.title("Hierarchical Clustering Dendrogram (Truncated)")
plt.xlabel("Cluster Size / Customers")
plt.ylabel("Distance")
plt.show()

# -----------------------------------------------------
# 9. Deployment Style: Predict Segment for New Customer
# -----------------------------------------------------
# Feature Order: [Age, Annual Income, Spending Score, Income_to_Spending_Ratio, 
#                 Total_Orders, Average_Order_Value, Discount_Sensitivity, 
#                 Cart_Abandonment_Rate, Tenure_Months]
new_customer = [[25, 60, 80, 60 / 81, 5, 3000, 0.4, 0.2, 12]]
new_customer_scaled = scaler.transform(new_customer)
predicted_cluster = kmeans.predict(new_customer_scaled)[0]

print("\n--- New Customer Inference Test ---")
print("Predicted Cluster:", predicted_cluster)
print("Assigned E-Commerce Segment:", segment_labels[predicted_cluster])

# -----------------------------------------------------
# 10. Save Final Segmented Dataset
# -----------------------------------------------------
data.to_csv("Customer_Segments_Final.csv", index=False)
print("\nFinal segmented dataset saved successfully as 'Customer_Segments_Final.csv'!")