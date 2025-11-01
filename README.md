# Internship Projects

This repository contains the projects completed during my internship at **GLOBAL NEXT CONSULTING INDIA PRIVATE LIMITED (GNCIPL)**.

🏢 **Company:** GLOBAL NEXT CONSULTING INDIA PVT. LTD.

🧑‍💻 **Intern:** Vaibhav K

📅 **Internship Period:** [Insert Duration – 24, July 2025]

🌐 **Website:** www.gncipl.com

# 📁 Projects Included
# **1. Disease Diagnosis Accuracy (Week 1)**

**Domain:** Medical Science

**Tools Used:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

**Dataset:** UCI Machine Learning Repository – Disease Diagnosis Accuracy

**Description:**
Evaluated diagnostic accuracy of various medical tests using classification and evaluation metrics. The project focused on quantifying false positives, false negatives, sensitivity, and specificity for disease diagnosis.

**Highlights:**

- Preprocessed patient medical test results

- Built confusion matrix to evaluate predictions

- Generated ROC curves to compare test effectiveness

- Analyzed false positive and false negative rates

- Derived insights for improving diagnostic reliability

# **2. Credit Card Fraud Detection (Week 2)**

**Domain:** Finance / Cybersecurity

**Tools Used:** Python, Pandas, Scikit-learn, Matplotlib, Seaborn, PCA

**Dataset:** Kaggle – Credit Card Fraud Detection (Anonymized Transaction Data)

**Description:**
Performed exploratory data analysis (EDA) on a highly imbalanced transaction dataset to detect fraudulent activities. Focused on anomaly detection, outlier patterns, and clustering of suspicious transactions.

**Highlights:**

- Cleaned anonymized credit card transaction data

- Explored outlier trends and correlations in fraud vs non-fraud transactions

- Applied PCA for dimensionality reduction and visualization

- Used heatmaps, histograms, and boxplots to identify fraud-related anomalies

- Derived actionable insights into fraudulent behavior patterns

# **3. E-commerce User Behavior Segmentation (Week 3)**

**Domain:** E-commerce / Customer Analytics

**Tools Used:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Hierarchical Clustering

**Dataset:** UCI Online Retail Dataset

**Description:**
Implemented customer segmentation for an e-commerce platform by analyzing user purchase behavior. The project aimed to identify different shopper types for better personalization and marketing strategies.

**Highlights:**

- Preprocessed online retail transaction logs (removed duplicates, handled missing values)

- Applied Hierarchical Clustering to group users based on purchase frequency, recency, and monetary value (RFM analysis)

- Visualized dendrograms to determine optimal cluster separation

- Compared user clusters to identify loyal customers, discount-seekers, and one-time buyers

- Derived insights for targeted promotions and recommendation systems

# **4. Smart City Sensor – Air Quality Clustering (Week 4)**

**Domain:** Environmental Data Science / Unsupervised Machine Learning

**Tools Used:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Plotly, GeoPandas, PCA, K-Means, t-SNE

**Dataset:** Kaggle – Air Quality Dataset

**Description:**
Developed an unsupervised learning pipeline to analyze air pollution sensor data in a smart city setup. The project focused on preprocessing, dimensionality reduction, clustering, and visualization of pollution patterns to create sensor-based zones.

**Highlights:**

- Cleaned raw dataset (handled missing values -200, NaNs, duplicates)

- Applied StandardScaler + PCA (90% variance retained)

- Used K-Means clustering with Elbow & Silhouette methods to determine optimal cluster count

- Evaluated clusters using Silhouette Score & Davies–Bouldin Index

- Visualized clusters with PCA (2D), t-SNE (optional), and interactive Plotly

- Geo-mapped sensor clusters using GeoPandas when latitude/longitude were available

**Saved Outputs:**

- smart_city_sensor_clusters.csv → full dataset with cluster labels

- smart_city_sensor_cluster_summary.csv → cluster-wise mean pollutant levels

**Cluster Interpretation:**

- Cluster 0: Low pollution zones (cleaner air, safer for residential)

- Cluster 1: High pollution zones (industrial/traffic-heavy areas)

# **5. MNIST Digit Classifier (Week 5)**

**Domain:** Computer Vision / Deep Learning

**Tools Used:** Python, NumPy, Keras, TensorFlow, Matplotlib

**Dataset:** MNIST Handwritten Digits (60,000 train, 10,000 test images)

**Description:**
Built an Artificial Neural Network (ANN) model to classify handwritten digits (0–9) using the MNIST dataset. The project demonstrated fundamental deep learning concepts including forward propagation, backpropagation, and softmax activation.

**Highlights:**

- Preprocessed grayscale images (28×28 pixels).

- Implemented ANN with input → hidden → output layers using Keras Sequential API.

- Used Softmax activation for multi-class classification.

- Trained with backpropagation & optimized using Adam.

- Achieved >97% accuracy on the test dataset.

- Visualized predictions with confusion matrix & sample digit outputs.
## 6. Final Project – Enhancing Sentiment Analysis with Synthetic Text Generated by GPT (week 6)

**Domain:** Social Media / Natural Language Processing (NLP)  
**Tools Used:** Python, Hugging Face Transformers, Scikit-learn, Matplotlib, Gradio  
**Dataset:** IMDB Sentiment Dataset (50k reviews, balanced classes)

**Description:**  
This project focused on improving sentiment analysis performance by augmenting real-world datasets with **synthetic reviews generated by GPT**. Traditional datasets are often limited and imbalanced, which reduces model accuracy. By generating synthetic samples and combining them with real data, the system achieved better generalization and prediction performance. A Gradio demo app was built for real-time sentiment prediction.

**Highlights:**
- Collected and analyzed IMDB sentiment dataset (EDA on review length and class distribution)  
- Built a **baseline model** using TF-IDF + Logistic Regression  
- Generated **200 synthetic reviews** (100 positive, 100 negative) using distilGPT2  
- Trained augmented model on real + synthetic data → **accuracy & F1-score improved**  
- Built an **interactive Gradio app** for real-time sentiment prediction  
- Allowed model comparison: **Baseline vs Augmented**

📄 **Project Documentation:** [View Documentation](https://docs.google.com/document/d/1iomBIAV42uMKoIjJoQYkBMfgK4_UMRg1adfi_47YiDQ/edit?usp=sharing)  
📊 **Presentation (PPT):** [View Slides](https://docs.google.com/presentation/d/1G3LO-rnCzu57rDETwwbXUHczOfbgp7iF/edit?usp=sharing&ouid=114709268787572312826&rtpof=true&sd=true)  
🚀 **Live Demo:** [Try it Here](https://huggingface.co/spaces/VaibhavRaoK/Vaibhavs-sentiment-analysis-gpt2-augmentation)

