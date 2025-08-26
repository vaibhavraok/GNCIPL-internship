Internship Projects:
This repository contains the projects completed during my internship at GLOBAL NEXT CONSULTING INDIA PRIVATE LIMITED (GNCIPL).

🏢 Company: GLOBAL NEXT CONSULTING INDIA PVT. LTD.
🧑‍💻 Intern: Vaibhav K
📅 Internship Period: [Insert Duration – 23, July 2025]
🌐 Website: www.gncipl.com

📁 Projects Included
1. Disease Diagnosis Accuracy (week 1)

Domain: Medical Science
Tools Used: Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
Dataset: UCI Machine Learning Repository – Disease Diagnosis Accuracy

Description:
Evaluated diagnostic accuracy of various medical tests using classification and evaluation metrics. The project focused on quantifying false positives, false negatives, sensitivity, and specificity for disease diagnosis.

Highlights:

Preprocessed patient medical test results.

Built confusion matrix to evaluate predictions.

Generated ROC curves to compare test effectiveness.

Analyzed false positive and false negative rates.

Derived insights for improving diagnostic reliability.

2. Credit Card Fraud Detection (week 2)

Domain: Finance / Cybersecurity
Tools Used: Python, Pandas, Scikit-learn, Matplotlib, Seaborn, PCA
Dataset: Kaggle – Credit Card Fraud Detection (Anonymized Transaction Data)

Description:
Performed exploratory data analysis (EDA) on a highly imbalanced transaction dataset to detect fraudulent activities. Focused on anomaly detection, outlier patterns, and clustering of suspicious transactions.

Highlights:

Cleaned anonymized credit card transaction data.

Explored outlier trends and correlations in fraud vs non-fraud transactions.

Applied PCA for dimensionality reduction and visualization.

Used heatmaps, histograms, and boxplots to identify fraud-related anomalies.

Derived actionable insights into fraudulent behavior patterns.

3. E-commerce User Behavior Segmentation (week 3)

Domain: E-commerce / Customer Analytics
Tools Used: Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Hierarchical Clustering
Dataset: UCI Online Retail Dataset

Description:
Implemented customer segmentation for an e-commerce platform by analyzing user purchase behavior. The project aimed to identify different shopper types for better personalization and marketing strategies.

Highlights:

Preprocessed online retail transaction logs (removed duplicates, handled missing values).

Applied Hierarchical Clustering to group users based on purchase frequency, recency, and monetary value (RFM analysis).

Visualized dendrograms to determine optimal cluster separation.

Compared user clusters to identify loyal customers, discount-seekers, and one-time buyers.

Derived insights for targeted promotions and recommendation systems.


4. Smart City Sensor – Air Quality Clustering (Week 4)

Domain: Environmental Data Science / Unsupervised Machine Learning
Tools Used: Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Plotly, GeoPandas, PCA, K-Means, t-SNE
Dataset: Kaggle – Air Quality Dataset

Description:
Developed an unsupervised learning pipeline to analyze air pollution sensor data in a smart city setup. The project focused on preprocessing, dimensionality reduction, clustering, and visualization of pollution patterns to create sensor-based zones.

Highlights:

Cleaned raw dataset (handled missing values -200, NaNs, duplicates).

Applied StandardScaler + PCA (90% variance retained).

Used K-Means clustering with Elbow & Silhouette methods to determine optimal cluster count.

Evaluated clusters using Silhouette Score & Davies–Bouldin Index.

Visualized clusters with PCA (2D), t-SNE (optional), and interactive Plotly.

Geo-mapped sensor clusters using GeoPandas when latitude/longitude were available.

Saved outputs:

smart_city_sensor_clusters.csv → full dataset with cluster labels

smart_city_sensor_cluster_summary.csv → cluster-wise mean pollutant levels

Cluster Interpretation:

Cluster 0: Low pollution zones (cleaner air, safer for residential).

Cluster 1: High pollution zones (industrial/traffic-heavy areas).
